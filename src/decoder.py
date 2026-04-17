"""Variational decoder core using PennyLane.

Wraps a parameterized ansatz into a full decode pipeline with manual
parameter-shift-rule gradients for quantum-aware optimisation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml

from .ansatz import Ansatz
from .noise_models import NoiseModel
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARAMETER_SHIFT_DELTA: float = np.pi / 2


# ---------------------------------------------------------------------------
# Variational Decoder
# ---------------------------------------------------------------------------

@dataclass
class VariationalDecoder:
    """Variational quantum error correction decoder.

    Combines a stabiliser code, a variational ansatz, and a noise model
    to produce a trainable decoding pipeline.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code being decoded.
    ansatz : Ansatz
        The variational circuit ansatz.
    noise_model : NoiseModel
        The noise channel used to generate training data.
    n_qubits : int, optional
        Number of circuit qubits.  Defaults to code's qubit count.
    """

    code: StabilizerCode
    ansatz: Ansatz
    noise_model: NoiseModel
    n_qubits: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_qubits is None:
            self.n_qubits = self.code.n_qubits
        self.params: np.ndarray = self.ansatz.init_params(seed=42)
        self._device = qml.device("lightning.qubit", wires=self.n_qubits)
        self._build_circuit()
        logger.info(
            "VariationalDecoder initialised: %d qubits, %d params",
            self.n_qubits,
            self.ansatz.n_params,
        )

    def _build_circuit(self) -> None:
        """Build the PennyLane QNode for the decoding circuit."""
        ansatz = self.ansatz
        n_qubits = self.n_qubits

        # Use adjoint method for fast gradients with lightning.qubit
        diff_method = "adjoint"

        @qml.qnode(self._device, interface="autograd", diff_method=diff_method)
        def circuit(params: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
            ansatz.forward(params, syndrome)
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self._circuit = circuit

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a syndrome into a Pauli correction operator.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Integer array of length ``n_qubits`` with entries in {0,1,2,3}
            representing the proposed Pauli correction.
        """
        expectations = self._circuit(self.params, syndrome)
        expectations = np.array(expectations)

        # Convert expectation values to Pauli corrections
        # Negative expectation -> qubit likely flipped
        correction = np.zeros(self.n_qubits, dtype=int)
        for q in range(self.n_qubits):
            if expectations[q] < 0:
                correction[q] = 1  # X correction
        return correction

    def decode_probabilities(
        self, syndrome: np.ndarray, params: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Return the raw flip probabilities for each qubit."""
        if params is None:
            params = self.params
        expectations = qml.math.stack(self._circuit(params, syndrome))
        if len(qml.math.shape(expectations)) > 1:
            expectations = qml.math.T(expectations)
        # Map [-1, 1] to [1, 0] (flip prob)
        return (1.0 - expectations) / 2.0

    def compute_loss(
        self, 
        syndromes: np.ndarray, 
        errors: np.ndarray, 
        params: Optional[np.ndarray] = None
    ) -> float:
        """Compute the average binary cross-entropy loss using vectorized execution."""
        if params is None:
            params = self.params
            
        # Vectorized probabilities: returns (batch_size, n_qubits)
        probs = self.decode_probabilities(syndromes, params=params)
        
        # target_x: 1 if error is X or Y, 0 if I or Z
        targets = qml.math.cast(np.isin(errors, [1, 2]), "float64")

        # Vectorized Binary Cross Entropy
        eps = 1e-8
        loss = -qml.math.sum(
            targets * qml.math.log(probs + eps) + 
            (1.0 - targets) * qml.math.log(1.0 - probs + eps)
        )

        return loss / len(syndromes)

    def compute_logical_error_rate(
        self,
        n_shots: int,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Estimate the logical error rate via Monte Carlo.

        Parameters
        ----------
        n_shots : int
            Number of error instances to sample.
        p : float, optional
            Override noise level for sampling.
        seed : int, optional
            Random seed.

        Returns
        -------
        float
            Fraction of shots where the correction causes a logical error.
        """
        errors = self.noise_model.sample_errors(
            self.code.n_qubits, n_shots, seed=seed
        )
        logical_ops = self.code.get_logical_ops()
        lx, lz = logical_ops["X"], logical_ops["Z"]

        n_logical_errors = 0
        for i in range(n_shots):
            error = errors[i]
            syndrome = self.code.extract_syndrome(error)
            correction = self.decode(syndrome)
            residual = self._compose_paulis(error, correction)

            if self._is_logical_error(residual, lx, lz):
                n_logical_errors += 1

        return n_logical_errors / n_shots

    @staticmethod
    def _compose_paulis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compose two Pauli error vectors (mod 2 in X and Z parts).

        Parameters
        ----------
        a, b : np.ndarray
            Integer arrays with entries in {0,1,2,3}.

        Returns
        -------
        np.ndarray
            Composed Pauli (ignoring global phase).
        """
        # Decompose into X and Z bits
        ax = np.isin(a, [1, 2]).astype(int)
        az = np.isin(a, [2, 3]).astype(int)
        bx = np.isin(b, [1, 2]).astype(int)
        bz = np.isin(b, [2, 3]).astype(int)

        rx = (ax + bx) % 2
        rz = (az + bz) % 2

        # Recombine: 0=I, 1=X, 2=Y, 3=Z
        return rx + 2 * (rx & rz) + 3 * rz - 2 * (rx & rz)
        # Simplification: rx*(1-rz) + 2*rx*rz + 3*rz*(1-rx)

    @staticmethod
    def _is_logical_error(
        residual: np.ndarray, lx: np.ndarray, lz: np.ndarray
    ) -> bool:
        """Check if residual corresponds to a non-trivial logical error."""
        res_x = np.isin(residual, [1, 2]).astype(int)
        res_z = np.isin(residual, [2, 3]).astype(int)
        log_x = np.isin(lx, [1, 2]).astype(int)
        log_z = np.isin(lz, [2, 3]).astype(int)

        comm_z = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
        comm_x = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2
        
        return bool(comm_x or comm_z)
