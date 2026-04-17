"""Adaptive decoder selector — NOVEL RESEARCH COMPONENT 2.

Combines the noise classifier with noise-specific variational decoders
to adaptively select the best ansatz at decode time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ansatz import AdaptiveAnsatz, Ansatz
from .decoder import VariationalDecoder
from .noise_classifier import NoiseClassifier, NOISE_LABELS
from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptive Decoder Selector
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveDecoderSelector:
    """Adaptive noise-aware decoder selector.

    On each decoding call:
    1. Classify noise from recent syndrome history.
    2. Select the best pre-trained decoder for that noise type.
    3. Decode using the selected decoder.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code.
    noise_classifier : NoiseClassifier
        Trained noise classifier.
    """

    code: StabilizerCode
    noise_classifier: NoiseClassifier

    def __post_init__(self) -> None:
        self._decoder_bank: Dict[str, VariationalDecoder] = {}
        self._default_noise_type = "depolarizing"
        logger.info(
            "AdaptiveDecoderSelector initialised for code d=%d", self.code.d
        )

    def register_decoder(
        self, noise_type: str, decoder: VariationalDecoder
    ) -> None:
        """Register a pre-trained decoder for a specific noise type.

        Parameters
        ----------
        noise_type : str
            Noise type string (e.g., 'depolarizing', 'bit_flip').
        decoder : VariationalDecoder
            Pre-trained variational decoder.
        """
        if noise_type not in NOISE_LABELS:
            raise ValueError(
                f"Unknown noise type '{noise_type}'. Must be one of {NOISE_LABELS}"
            )
        self._decoder_bank[noise_type] = decoder
        logger.info("Registered decoder for noise type: %s", noise_type)

    def build_decoder_bank(
        self,
        noise_models: Optional[Dict[str, NoiseModel]] = None,
        n_layers: int = 4,
        seed: int = 42,
    ) -> None:
        """Build a bank of decoders, one per noise type.

        Parameters
        ----------
        noise_models : dict, optional
            ``{noise_type: NoiseModel}``.  If None, creates defaults.
        n_layers : int
            Number of ansatz layers.
        seed : int
            Random seed for parameter initialisation.
        """
        if noise_models is None:
            noise_models = {
                "depolarizing": create_noise_model("depolarizing", p=0.05),
                "bit_flip": create_noise_model("bit_flip", p=0.05),
                "phase_flip": create_noise_model("phase_flip", p=0.05),
                "combined": create_noise_model(
                    "combined", p_bit=0.05, p_phase=0.05
                ),
            }

        for noise_type in NOISE_LABELS:
            ansatz = AdaptiveAnsatz(
                n_qubits=self.code.n_qubits,
                noise_type=noise_type,
                n_layers=n_layers,
            )
            noise = noise_models.get(
                noise_type,
                create_noise_model("depolarizing", p=0.05),
            )
            decoder = VariationalDecoder(
                code=self.code,
                ansatz=ansatz,
                noise_model=noise,
            )
            # Initialize with specific seed per noise type
            decoder.params = ansatz.init_params(
                seed=seed + NOISE_LABELS.index(noise_type)
            )
            self._decoder_bank[noise_type] = decoder
            logger.info("Built decoder for noise type: %s", noise_type)

    def decode_adaptive(
        self,
        syndrome: np.ndarray,
        syndrome_history: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adaptively decode a syndrome.

        Parameters
        ----------
        syndrome : np.ndarray
            Current binary syndrome vector.
        syndrome_history : np.ndarray
            Recent syndrome history, shape
            ``(time_steps, n_stabilizers)``.

        Returns
        -------
        tuple
            ``(correction, metadata)`` where correction is a Pauli
            correction array and metadata is a dict with:
            - ``detected_noise_type``: classified noise type
            - ``confidence``: classifier confidence score
            - ``ansatz_used``: name of the ansatz used
            - ``decode_time_ms``: decoding wall-clock time
        """
        t0 = time.perf_counter()

        # Step 1: Classify noise from syndrome history
        noise_type, confidence = self.noise_classifier.classify(
            syndrome_history
        )

        # Step 2: Select decoder
        if noise_type in self._decoder_bank:
            decoder = self._decoder_bank[noise_type]
            ansatz_name = f"AdaptiveAnsatz({noise_type})"
        else:
            # Fallback to default
            logger.warning(
                "No decoder for noise type '%s', using '%s'",
                noise_type,
                self._default_noise_type,
            )
            decoder = self._decoder_bank[self._default_noise_type]
            ansatz_name = f"AdaptiveAnsatz({self._default_noise_type})"

        # Step 3: Decode
        correction = decoder.decode(syndrome)

        t1 = time.perf_counter()
        decode_time_ms = (t1 - t0) * 1000.0

        metadata = {
            "detected_noise_type": noise_type,
            "confidence": confidence,
            "ansatz_used": ansatz_name,
            "decode_time_ms": decode_time_ms,
        }

        logger.debug(
            "Adaptive decode: noise=%s (conf=%.3f), ansatz=%s, time=%.2fms",
            noise_type,
            confidence,
            ansatz_name,
            decode_time_ms,
        )

        return correction, metadata

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Simple decode using the default decoder (for compatibility).

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome.

        Returns
        -------
        np.ndarray
            Pauli correction.
        """
        if self._default_noise_type in self._decoder_bank:
            return self._decoder_bank[self._default_noise_type].decode(syndrome)
        elif self._decoder_bank:
            first_key = next(iter(self._decoder_bank))
            return self._decoder_bank[first_key].decode(syndrome)
        else:
            raise RuntimeError("No decoders registered in the bank")

    def evaluate_adaptive(
        self,
        noise_model: NoiseModel,
        n_shots: int = 1000,
        time_steps: int = 10,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Evaluate the adaptive decoder under a specific noise model.

        Parameters
        ----------
        noise_model : NoiseModel
            Noise model for generating test errors.
        n_shots : int
            Number of evaluation shots.
        time_steps : int
            Number of syndrome rounds for the history.
        seed : int
            Random seed.

        Returns
        -------
        dict
            Evaluation metrics including logical error rate,
            classification accuracy, and timing.
        """
        np.random.seed(seed)
        logical_ops = self.code.get_logical_ops()
        logical_x = logical_ops["X"]

        n_logical_errors = 0
        noise_type_counts: Dict[str, int] = {nt: 0 for nt in NOISE_LABELS}
        total_decode_time = 0.0
        confidences: List[float] = []

        for i in range(n_shots):
            # Generate syndrome history
            history = np.zeros(
                (time_steps, self.code.n_stabilizers), dtype=float
            )
            for t in range(time_steps):
                err = noise_model.sample_errors(self.code.n_qubits, 1)
                history[t] = self.code.extract_syndrome(err[0])

            # Generate current error + syndrome
            error = noise_model.sample_errors(
                self.code.n_qubits, 1, seed=seed + i
            )[0]
            syndrome = self.code.extract_syndrome(error)

            # Adaptive decode
            correction, metadata = self.decode_adaptive(syndrome, history)
            noise_type_counts[metadata["detected_noise_type"]] += 1
            total_decode_time += metadata["decode_time_ms"]
            confidences.append(metadata["confidence"])

            # Check logical error
            residual = _compose_paulis(error, correction)
            lx, lz = logical_ops["X"], logical_ops["Z"]
            if _is_logical_error(residual, lx, lz):
                n_logical_errors += 1

        ler = n_logical_errors / n_shots
        mean_time = total_decode_time / n_shots
        mean_conf = float(np.mean(confidences))

        results = {
            "logical_error_rate": ler,
            "n_logical_errors": n_logical_errors,
            "n_shots": n_shots,
            "noise_type_distribution": noise_type_counts,
            "mean_decode_time_ms": mean_time,
            "mean_confidence": mean_conf,
        }

        logger.info(
            "Adaptive evaluation: LER=%.6f, mean_time=%.2fms, "
            "mean_confidence=%.3f",
            ler,
            mean_time,
            mean_conf,
        )

        return results


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


def _is_logical_error(
    residual: np.ndarray, lx: np.ndarray, lz: np.ndarray
) -> bool:
    """Check if residual matches logical X or Z using commutation."""
    res_x = np.isin(residual, [1, 2]).astype(int)
    res_z = np.isin(residual, [2, 3]).astype(int)
    log_x = np.isin(lx, [1, 2]).astype(int)
    log_z = np.isin(lz, [2, 3]).astype(int)

    comm_z = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
    comm_x = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2
    
    return bool(comm_x or comm_z)
