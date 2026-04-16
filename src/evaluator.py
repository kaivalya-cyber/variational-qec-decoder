"""Evaluation utilities for QEC decoders.

Provides threshold scanning, decoder comparison, plotting, and
wall-clock overhead benchmarking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR: str = "results/figures"
DEFAULT_N_SHOTS: int = 1000
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"

# Ensure output directory exists
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Threshold scan
# ---------------------------------------------------------------------------

def threshold_scan(
    decoder: Any,
    code: StabilizerCode,
    noise_model_factory: Callable[[float], NoiseModel],
    p_values: np.ndarray,
    n_shots_per_p: int = DEFAULT_N_SHOTS,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Scan logical error rate vs. physical error rate.

    Parameters
    ----------
    decoder : object
        A decoder with a ``decode(syndrome)`` method.
    code : StabilizerCode
        The QEC code.
    noise_model_factory : callable
        Function ``p -> NoiseModel`` for varying the error rate.
    p_values : np.ndarray
        Array of physical error rates to scan.
    n_shots_per_p : int
        Number of Monte Carlo shots per error rate.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``{'p_values': array, 'logical_error_rates': array}``.
    """
    logical_error_rates = np.zeros(len(p_values))
    logical_ops = code.get_logical_ops()
    logical_x = logical_ops["X"]

    for idx, p in enumerate(p_values):
        noise = noise_model_factory(p)
        errors = noise.sample_errors(code.n_qubits, n_shots_per_p, seed=seed + idx)
        n_logical_errors = 0

        for i in range(n_shots_per_p):
            syndrome = code.extract_syndrome(errors[i])
            correction = decoder.decode(syndrome)
            residual = _compose_paulis(errors[i], correction)
            if _is_logical_error(residual, logical_x):
                n_logical_errors += 1

        logical_error_rates[idx] = n_logical_errors / n_shots_per_p
        logger.info(
            "p=%.4f: logical error rate = %.6f (%d/%d)",
            p,
            logical_error_rates[idx],
            n_logical_errors,
            n_shots_per_p,
        )

    return {"p_values": p_values, "logical_error_rates": logical_error_rates}


# ---------------------------------------------------------------------------
# Decoder comparison
# ---------------------------------------------------------------------------

def compare_decoders(
    decoders_dict: Dict[str, Any],
    code: StabilizerCode,
    noise_model_factory: Callable[[float], NoiseModel],
    p_values: np.ndarray,
    n_shots: int = DEFAULT_N_SHOTS,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Benchmark multiple decoders across noise levels.

    Parameters
    ----------
    decoders_dict : dict
        ``{name: decoder}`` mapping.
    code : StabilizerCode
        The QEC code.
    noise_model_factory : callable
        ``p -> NoiseModel``.
    p_values : np.ndarray
        Physical error rates.
    n_shots : int
        Shots per error rate per decoder.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``{decoder_name: {'p_values': array, 'logical_error_rates': array}}``.
    """
    results = {}
    for name, decoder in decoders_dict.items():
        logger.info("Evaluating decoder: %s", name)
        results[name] = threshold_scan(
            decoder, code, noise_model_factory, p_values, n_shots, seed
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_curve(
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "Logical vs Physical Error Rate",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot threshold curves for multiple decoders.

    Parameters
    ----------
    results : dict
        Output of ``compare_decoders``.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure. Defaults to ``results/figures/threshold.png``.
    show : bool
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="whitegrid", palette="deep")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    markers = ["o", "s", "^", "D", "v", "<", ">"]
    for i, (name, data) in enumerate(results.items()):
        ax.semilogy(
            data["p_values"],
            data["logical_error_rates"],
            marker=markers[i % len(markers)],
            label=name,
            linewidth=2,
            markersize=6,
        )

    # Reference line: p_logical = p_physical
    p_vals = list(results.values())[0]["p_values"]
    ax.semilogy(p_vals, p_vals, "--", color="gray", alpha=0.5, label="p_L = p")

    ax.set_xlabel("Physical Error Rate (p)", fontsize=13)
    ax.set_ylabel("Logical Error Rate (p_L)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "threshold.png")
    fig.savefig(save_path, dpi=150)
    logger.info("Saved threshold plot to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training metrics.

    Parameters
    ----------
    history : dict
        Training history dictionary with keys 'losses',
        'logical_error_rates', 'gradient_norms'.
    save_path : str, optional
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["losses"]) + 1)

    # Loss
    axes[0].plot(epochs, history["losses"], linewidth=2, color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    # Logical error rate
    axes[1].semilogy(
        epochs, history["logical_error_rates"], linewidth=2, color="tab:red"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Logical Error Rate")
    axes[1].set_title("Logical Error Rate")

    # Gradient norms
    axes[2].plot(epochs, history["gradient_norms"], linewidth=2, color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("||∇||")
    axes[2].set_title("Gradient Norm")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "training_history.png")
    fig.savefig(save_path, dpi=150)
    logger.info("Saved training history plot to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Decoder overhead
# ---------------------------------------------------------------------------

def compute_decoder_overhead(
    decoder: Any,
    code: StabilizerCode,
    noise_model: NoiseModel,
    n_shots: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Measure wall-clock decoding time per shot.

    Parameters
    ----------
    decoder : object
        Decoder with a ``decode`` method.
    code : StabilizerCode
        The QEC code.
    noise_model : NoiseModel
        Noise model for generating test syndromes.
    n_shots : int
        Number of decoding calls to average over.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{'mean_time_ms': float, 'std_time_ms': float,
          'total_time_s': float, 'n_shots': int}``.
    """
    errors = noise_model.sample_errors(code.n_qubits, n_shots, seed=seed)

    times: List[float] = []
    for i in range(n_shots):
        syndrome = code.extract_syndrome(errors[i])
        t0 = time.perf_counter()
        _ = decoder.decode(syndrome)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    result = {
        "mean_time_ms": float(np.mean(times)),
        "std_time_ms": float(np.std(times)),
        "total_time_s": float(np.sum(times) / 1000.0),
        "n_shots": n_shots,
    }
    logger.info(
        "Decoder overhead: %.3f ± %.3f ms per decode (%d shots)",
        result["mean_time_ms"],
        result["std_time_ms"],
        n_shots,
    )
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _compose_paulis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Pauli vectors.

    Parameters
    ----------
    a, b : np.ndarray

    Returns
    -------
    np.ndarray
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


def _is_logical_error(residual: np.ndarray, logical_x: np.ndarray) -> bool:
    """Check if residual matches logical X.

    Parameters
    ----------
    residual, logical_x : np.ndarray

    Returns
    -------
    bool
    """
    res_x = np.isin(residual, [1, 2]).astype(int)
    log_x = np.isin(logical_x, [1, 2]).astype(int)
    return bool(np.array_equal(res_x, log_x))
