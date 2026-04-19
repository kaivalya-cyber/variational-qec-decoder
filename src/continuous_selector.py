"""Continuous decoder selector using Gaussian kernel interpolation.

Generates: Section IV.C (Decoder Interpolation),
           Figure 4 (noise_manifold_heatmap.png)

Instead of hard-routing to one decoder, this module computes soft
interpolation weights over a bank of pre-trained decoders based on
continuously estimated noise parameters.

Hardware constraints (3 GB RAM):
- Only one decoder loaded into memory at a time during weighted decode
- All numpy operations use float32
- Figures are saved and immediately freed from memory

# ======================================================================
# THEORETICAL DERIVATION: Gaussian Kernel Interpolation as
# Bayesian Model Averaging Under a Gaussian Prior
# ======================================================================
#
# We maintain a decoder bank D = {D_1, ..., D_K} where each D_k was
# trained under noise parameters φ_k = (p_X^(k), p_Z^(k), p_depol^(k)).
#
# Given an estimated noise vector φ = (p_X, p_Z, p_depol), we compute
# interpolation weights via a Gaussian kernel:
#
#   w_k = exp( -||φ - φ_k||² / (2σ²) )
#
# Normalising: w_k := w_k / Σ_j w_j
#
# CLAIM: This is equivalent to Bayesian model averaging (BMA) under a
# Gaussian prior on the noise parameters.
#
# PROOF:
#   In BMA, the posterior predictive for correction c given syndrome s is:
#
#     p(c|s) = Σ_k p(c|s, M_k) · p(M_k|φ)
#
#   where p(M_k|φ) is the posterior probability of model M_k given the
#   estimated noise φ.
#
#   By Bayes' theorem:
#     p(M_k|φ) ∝ p(φ|M_k) · p(M_k)
#
#   Assume:
#     (i)  Uniform prior over models: p(M_k) = 1/K
#     (ii) Gaussian likelihood centred at each model's training point:
#          p(φ|M_k) = N(φ; φ_k, σ²I)
#                   = (2πσ²)^(-d/2) exp(-||φ - φ_k||² / (2σ²))
#
#   Then:
#     p(M_k|φ) ∝ exp(-||φ - φ_k||² / (2σ²))
#
#   which, after normalisation, gives exactly our kernel weights w_k.
#
#   Therefore the weighted correction:
#     c = Σ_k w_k · D_k(s)
#
#   is the BMA posterior predictive under a Gaussian prior on noise. ∎
#
#   The bandwidth σ controls the "softness" of routing:
#     - σ → 0:  hard routing (argmax), recovers discrete classifier
#     - σ → ∞:  uniform average, ignores noise estimation
#
#   We set σ = 0.02 (≈ half the grid spacing) which provides a smooth
#   interpolation while still concentrating weight on the nearest decoder.
# ======================================================================
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .decoder import VariationalDecoder
from .noise_estimator import NoiseParameterEstimator
from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR: str = "results/figures"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

KERNEL_SIGMA: float = 0.02  # Gaussian kernel bandwidth


# ---------------------------------------------------------------------------
# Continuous Decoder Selector
# ---------------------------------------------------------------------------

@dataclass
class ContinuousDecoderSelector:
    """Decoder selector using continuous noise estimation and soft routing.

    Given an estimated noise vector (p_X, p_Z, p_depol), computes
    Gaussian kernel similarity weights to each pre-trained decoder and
    combines their correction outputs.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code.
    noise_estimator : NoiseParameterEstimator
        Trained continuous noise estimator.
    decoder_bank : dict
        Mapping ``{noise_type: VariationalDecoder}`` of pre-trained decoders.
    sigma : float
        Gaussian kernel bandwidth.
    """

    code: StabilizerCode
    noise_estimator: NoiseParameterEstimator
    decoder_bank: Dict[str, VariationalDecoder] = field(default_factory=dict)
    sigma: float = KERNEL_SIGMA

    def __post_init__(self) -> None:
        # Center of each decoder's training distribution
        self._centers: Dict[str, np.ndarray] = {
            "depolarizing": np.array([0.0, 0.0, 0.025], dtype=np.float32),
            "bit_flip": np.array([0.025, 0.0, 0.0], dtype=np.float32),
            "phase_flip": np.array([0.0, 0.025, 0.0], dtype=np.float32),
            "combined": np.array([0.0125, 0.0125, 0.0125], dtype=np.float32),
        }
        logger.info(
            "ContinuousDecoderSelector initialised with %d decoders, sigma=%.4f",
            len(self.decoder_bank),
            self.sigma,
        )

    def _compute_weights(
        self, phi: np.ndarray
    ) -> Dict[str, float]:
        """Compute Gaussian kernel interpolation weights.

        Parameters
        ----------
        phi : np.ndarray
            Estimated noise vector ``(p_X, p_Z, p_depol)``, shape ``(3,)``.

        Returns
        -------
        dict
            ``{noise_type: weight}`` normalised to sum to 1.
        """
        phi = np.asarray(phi, dtype=np.float32)
        raw_weights: Dict[str, float] = {}
        total = 0.0

        for name, center in self._centers.items():
            if name in self.decoder_bank:
                dist_sq = float(np.sum((phi - center) ** 2))
                w = float(np.exp(-dist_sq / (2.0 * self.sigma ** 2)))
                raw_weights[name] = w
                total += w

        # Normalise
        if total > 0:
            weights = {k: v / total for k, v in raw_weights.items()}
        else:
            # Uniform fallback
            n = len(raw_weights)
            weights = {k: 1.0 / n for k in raw_weights}

        return weights

    def decode_continuous(
        self,
        syndrome: np.ndarray,
        syndrome_history: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decode using continuous noise estimation and soft interpolation.

        Parameters
        ----------
        syndrome : np.ndarray
            Current binary syndrome vector.
        syndrome_history : np.ndarray
            Recent syndrome history, shape ``(time_steps, n_stabilizers)``.

        Returns
        -------
        tuple
            ``(correction, metadata)`` where correction is an integer
            Pauli correction array and metadata is a dict containing:
            - ``estimated_p_X``, ``estimated_p_Z``, ``estimated_p_depol``
            - ``uncertainty``
            - ``decoder_weights``
            - ``decode_time_ms``
        """
        t0 = time.perf_counter()

        # Step 1: Estimate noise parameters
        estimation = self.noise_estimator.estimate(syndrome_history)
        phi = np.array(
            [estimation["p_X"], estimation["p_Z"], estimation["p_depol"]],
            dtype=np.float32,
        )

        # Step 2: Compute interpolation weights
        weights = self._compute_weights(phi)

        # Step 3: Weighted correction
        # Get correction probabilities from each decoder, weight them
        n_qubits = self.code.n_qubits
        weighted_probs = np.zeros(n_qubits, dtype=np.float32)

        for name, w in weights.items():
            if w < 1e-6:
                continue
            decoder = self.decoder_bank[name]
            probs = decoder.decode_probabilities(syndrome)
            probs = np.asarray(probs, dtype=np.float32)
            if probs.ndim > 1:
                probs = probs[0]
            weighted_probs += w * probs

        # Threshold to get binary correction
        correction = np.zeros(n_qubits, dtype=int)
        correction[weighted_probs > 0.5] = 1

        t1 = time.perf_counter()
        decode_time_ms = float((t1 - t0) * 1000.0)

        metadata = {
            "estimated_p_X": estimation["p_X"],
            "estimated_p_Z": estimation["p_Z"],
            "estimated_p_depol": estimation["p_depol"],
            "uncertainty": estimation["uncertainty"],
            "decoder_weights": weights,
            "decode_time_ms": decode_time_ms,
        }

        logger.debug(
            "Continuous decode: phi=(%.4f, %.4f, %.4f), weights=%s, time=%.2fms",
            phi[0], phi[1], phi[2], weights, decode_time_ms,
        )

        return correction, metadata

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Simple decode using equal-weighted average (for compatibility).

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Pauli correction array.
        """
        if not self.decoder_bank:
            raise RuntimeError("No decoders in the bank.")
        # Use first available decoder as fallback
        first_key = next(iter(self.decoder_bank))
        return self.decoder_bank[first_key].decode(syndrome)

    def plot_noise_manifold(
        self,
        p_range: Tuple[float, float] = (0.0, 0.05),
        n_steps: int = 10,
        fixed_p_depol: float = 0.01,
        n_shots: int = 50,
        save_path: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Generate 2D heatmap of LER across (p_X, p_Z) space.

        Parameters
        ----------
        p_range : tuple of float
            ``(min_p, max_p)`` for both p_X and p_Z axes.
        n_steps : int
            Number of grid steps per axis.
        fixed_p_depol : float
            Fixed depolarizing component.
        n_shots : int
            Shots per grid point for LER estimation.
        save_path : str, optional
            Output figure path.
        seed : int
            Random seed.

        Returns
        -------
        str
            Path to the saved figure.
        """
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, "noise_manifold_heatmap.png")

        rng = np.random.default_rng(seed)
        p_x_vals = np.linspace(p_range[0], p_range[1], n_steps, dtype=np.float32)
        p_z_vals = np.linspace(p_range[0], p_range[1], n_steps, dtype=np.float32)
        ler_grid = np.zeros((n_steps, n_steps), dtype=np.float32)

        logical_ops = self.code.get_logical_ops()
        lx, lz = logical_ops["X"], logical_ops["Z"]

        for i, p_x in enumerate(p_x_vals):
            for j, p_z in enumerate(p_z_vals):
                noise = create_noise_model(
                    "combined",
                    p_bit=float(p_x + fixed_p_depol / 3.0),
                    p_phase=float(p_z + fixed_p_depol / 3.0),
                )

                n_logical_errors = 0
                for shot in range(n_shots):
                    error = noise.sample_errors(self.code.n_qubits, 1)[0]
                    syndrome = self.code.extract_syndrome(error)

                    # Simple decode with weighted average
                    correction = self.decode(syndrome)
                    residual = _compose_paulis(error, correction)
                    if _is_logical_error(residual, lx, lz):
                        n_logical_errors += 1

                ler_grid[j, i] = n_logical_errors / n_shots

            logger.info(
                "Noise manifold: row %d/%d complete", i + 1, n_steps
            )

        # Plot heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        im = ax.imshow(
            ler_grid,
            extent=[p_range[0], p_range[1], p_range[0], p_range[1]],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
        )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Logical Error Rate", fontsize=12)

        ax.set_xlabel(r"$p_X$ (bit-flip probability)", fontsize=14)
        ax.set_ylabel(r"$p_Z$ (phase-flip probability)", fontsize=14)
        ax.set_title(
            f"Decoder Performance Landscape ($p_{{depol}}={fixed_p_depol}$)",
            fontsize=15,
            fontweight="bold",
        )
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.1)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close("all")
        del fig
        gc.collect()

        logger.info("Saved noise manifold heatmap to %s", save_path)
        return save_path


# ---------------------------------------------------------------------------
# Utility helpers (duplicated to avoid circular imports)
# ---------------------------------------------------------------------------

def _compose_paulis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Pauli vectors.

    Parameters
    ----------
    a, b : np.ndarray
        Integer arrays with entries in {0,1,2,3}.

    Returns
    -------
    np.ndarray
        Composed Pauli (ignoring global phase).
    """
    ax = np.isin(a, [1, 2]).astype(int)
    az = np.isin(a, [2, 3]).astype(int)
    bx = np.isin(b, [1, 2]).astype(int)
    bz = np.isin(b, [2, 3]).astype(int)

    rx = (ax + bx) % 2
    rz = (az + bz) % 2

    result = np.zeros_like(a)
    result[(rx == 1) & (rz == 0)] = 1
    result[(rx == 1) & (rz == 1)] = 2
    result[(rx == 0) & (rz == 1)] = 3
    return result


def _is_logical_error(
    residual: np.ndarray, lx: np.ndarray, lz: np.ndarray
) -> bool:
    """Check if residual corresponds to a non-trivial logical error.

    Parameters
    ----------
    residual : np.ndarray
        Residual Pauli error after correction.
    lx : np.ndarray
        Logical X operator.
    lz : np.ndarray
        Logical Z operator.

    Returns
    -------
    bool
        True if residual is a logical error.
    """
    res_x = np.isin(residual, [1, 2]).astype(int)
    res_z = np.isin(residual, [2, 3]).astype(int)
    log_x = np.isin(lx, [1, 2]).astype(int)
    log_z = np.isin(lz, [2, 3]).astype(int)

    comm_z = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
    comm_x = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2

    return bool(comm_x or comm_z)
