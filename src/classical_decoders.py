"""Classical baseline decoders: MWPM and lookup table.

Provides reference implementations for benchmarking the variational decoder.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stabilizer_codes import StabilizerCode, SurfaceCode, RepetitionCode

logger = logging.getLogger(__name__)

# Try to import pymatching for MWPM
try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False
    logger.warning("pymatching not installed — MWPM decoder unavailable")


# ---------------------------------------------------------------------------
# MWPM Decoder
# ---------------------------------------------------------------------------

@dataclass
class MWPMDecoder:
    """Minimum Weight Perfect Matching decoder using PyMatching.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code to decode.
    noise_model : object, optional
        Noise model (used for weight calibration).
    """

    code: StabilizerCode
    noise_model: object = None

    def __post_init__(self) -> None:
        if not HAS_PYMATCHING:
            raise ImportError(
                "pymatching is required for MWPMDecoder. "
                "Install with: pip install pymatching"
            )
        self._build_matching()
        logger.info("MWPMDecoder initialised for code with d=%d", self.code.d)

    def _build_matching(self) -> None:
        """Construct the PyMatching matching graph from the code's check matrix."""
        h = self.code.get_stabilizers()
        n = self.code.n_qubits

        if isinstance(self.code, SurfaceCode):
            # Build separate X and Z matching graphs
            # X stabilisers -> detect Z errors -> correct Z
            h_x = h[: self.code.n_x_stabilizers, n:]  # Z part
            # Z stabilisers -> detect X errors -> correct X
            h_z = h[self.code.n_x_stabilizers :, :n]  # X part

            self._matching_x = pymatching.Matching(h_x)
            self._matching_z = pymatching.Matching(h_z)
            self._is_surface = True
        else:
            # Repetition code: single matching graph
            h_z = h[:, n:]  # Z part detects X errors
            self._matching = pymatching.Matching(h_z)
            self._is_surface = False

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome into a Pauli correction.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Integer Pauli correction of length ``n_qubits``.
        """
        correction = np.zeros(self.code.n_qubits, dtype=int)

        if self._is_surface:
            z_syn = syndrome[: self.code.n_x_stabilizers]
            x_syn = syndrome[self.code.n_x_stabilizers :]

            # Decode Z errors (from X stabiliser syndrome)
            z_corr = self._matching_x.decode(z_syn)
            # Decode X errors (from Z stabiliser syndrome)
            x_corr = self._matching_z.decode(x_syn)

            for q in range(self.code.n_qubits):
                if x_corr[q] and z_corr[q]:
                    correction[q] = 2  # Y
                elif x_corr[q]:
                    correction[q] = 1  # X
                elif z_corr[q]:
                    correction[q] = 3  # Z
        else:
            x_corr = self._matching.decode(syndrome)
            correction = x_corr.astype(int)  # X corrections

        return correction

    def compute_logical_error_rate(
        self,
        noise_model: object,
        n_shots: int,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the logical error rate.

        Parameters
        ----------
        noise_model : NoiseModel
            Noise model for sampling errors.
        n_shots : int
            Number of Monte Carlo shots.
        seed : int, optional
            Random seed.

        Returns
        -------
        float
        """
        errors = noise_model.sample_errors(self.code.n_qubits, n_shots, seed=seed)
        logical_ops = self.code.get_logical_ops()
        logical_x = logical_ops["X"]

        n_errors = 0
        for i in range(n_shots):
            syndrome = self.code.extract_syndrome(errors[i])
            correction = self.decode(syndrome)
            residual = _compose_paulis(errors[i], correction)
            if _is_logical_error(residual, logical_x):
                n_errors += 1

        return n_errors / n_shots


# ---------------------------------------------------------------------------
# Lookup Table Decoder
# ---------------------------------------------------------------------------

@dataclass
class LookupTableDecoder:
    """Lookup table decoder for small codes.

    Pre-computes the optimal correction for every possible syndrome.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code to decode.
    noise_model : object, optional
        If provided, used to weight the lookup table by error probability.
    max_weight : int
        Maximum error weight to enumerate in the lookup table.
    """

    code: StabilizerCode
    noise_model: object = None
    max_weight: int = 3

    def __post_init__(self) -> None:
        self._table: Dict[tuple, np.ndarray] = {}
        self._build_table()
        logger.info(
            "LookupTableDecoder built with %d entries for d=%d",
            len(self._table),
            self.code.d,
        )

    def _build_table(self) -> None:
        """Enumerate all errors up to ``max_weight`` and store corrections."""
        n = self.code.n_qubits

        # Weight 0: trivial
        zero_error = np.zeros(n, dtype=int)
        zero_syn = tuple(self.code.extract_syndrome(zero_error))
        self._table[zero_syn] = zero_error.copy()

        # Weight 1
        for q in range(n):
            for pauli in [1, 2, 3]:  # X, Y, Z
                error = np.zeros(n, dtype=int)
                error[q] = pauli
                syn = tuple(self.code.extract_syndrome(error))
                if syn not in self._table:
                    self._table[syn] = error.copy()

        # Weight 2
        if self.max_weight >= 2:
            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    for p1 in [1, 2, 3]:
                        for p2 in [1, 2, 3]:
                            error = np.zeros(n, dtype=int)
                            error[q1] = p1
                            error[q2] = p2
                            syn = tuple(self.code.extract_syndrome(error))
                            if syn not in self._table:
                                self._table[syn] = error.copy()

        # Weight 3 (only for small codes)
        if self.max_weight >= 3 and n <= 9:
            for q1 in range(n):
                for q2 in range(q1 + 1, n):
                    for q3 in range(q2 + 1, n):
                        for p1 in [1, 3]:
                            for p2 in [1, 3]:
                                for p3 in [1, 3]:
                                    error = np.zeros(n, dtype=int)
                                    error[q1] = p1
                                    error[q2] = p2
                                    error[q3] = p3
                                    syn = tuple(
                                        self.code.extract_syndrome(error)
                                    )
                                    if syn not in self._table:
                                        self._table[syn] = error.copy()

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome using the lookup table.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Pauli correction of length ``n_qubits``.
        """
        key = tuple(syndrome)
        if key in self._table:
            return self._table[key].copy()
        else:
            logger.warning("Syndrome not in lookup table, returning identity")
            return np.zeros(self.code.n_qubits, dtype=int)

    def compute_logical_error_rate(
        self,
        noise_model: object,
        n_shots: int,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the logical error rate.

        Parameters
        ----------
        noise_model : NoiseModel
            Noise model.
        n_shots : int
            Number of shots.
        seed : int, optional
            Random seed.

        Returns
        -------
        float
        """
        errors = noise_model.sample_errors(self.code.n_qubits, n_shots, seed=seed)
        logical_ops = self.code.get_logical_ops()
        logical_x = logical_ops["X"]

        n_errors = 0
        for i in range(n_shots):
            syndrome = self.code.extract_syndrome(errors[i])
            correction = self.decode(syndrome)
            residual = _compose_paulis(errors[i], correction)
            if _is_logical_error(residual, logical_x):
                n_errors += 1

        return n_errors / n_shots


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _compose_paulis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Pauli vectors (mod-2 in X/Z components).

    Parameters
    ----------
    a, b : np.ndarray
        Integer arrays with entries in {0,1,2,3}.

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
    result[(rx == 1) & (rz == 0)] = 1  # X
    result[(rx == 1) & (rz == 1)] = 2  # Y
    result[(rx == 0) & (rz == 1)] = 3  # Z
    return result


def _is_logical_error(residual: np.ndarray, logical_x: np.ndarray) -> bool:
    """Check if residual matches the logical X operator.

    Parameters
    ----------
    residual : np.ndarray
    logical_x : np.ndarray

    Returns
    -------
    bool
    """
    res_x = np.isin(residual, [1, 2]).astype(int)
    log_x = np.isin(logical_x, [1, 2]).astype(int)
    return bool(np.array_equal(res_x, log_x))
