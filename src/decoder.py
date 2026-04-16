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
        self._device = qml.device("default.qubit", wires=self.n_qubits)
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

        @qml.qnode(self._device, interface="autograd")
        def circuit(params: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
            """Execute the variational decode circuit.

            Parameters
            ----------
            params : np.ndarray
                Variational parameters.
            syndrome : np.ndarray
                Syndrome input.

            Returns
            -------
            np.ndarray
                Expectation values of PauliZ on each qubit.
            """
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

    def decode_probabilities(self, syndrome: np.ndarray) -> np.ndarray:
        """Return the raw flip probabilities for each qubit.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Float array of shape ``(n_qubits,)`` with values in [0, 1].
        """
        expectations = np.array(self._circuit(self.params, syndrome))
        # Map [-1, 1] to [1, 0] (flip prob)
        return (1.0 - expectations) / 2.0

    def compute_loss(
        self,
        syndromes: np.ndarray,
        errors: np.ndarray,
    ) -> float:
        """Compute the cross-entropy loss over a batch.

        Parameters
        ----------
        syndromes : np.ndarray
            Shape ``(batch_size, n_stabilizers)`` binary syndromes.
        errors : np.ndarray
            Shape ``(batch_size, n_qubits)`` integer error labels.

        Returns
        -------
        float
            Mean cross-entropy loss.
        """
        batch_size = syndromes.shape[0]
        total_loss = 0.0

        for i in range(batch_size):
            probs = self.decode_probabilities(syndromes[i])
            # Binary target: 1 if error is X or Y (has X component)
            targets = np.isin(errors[i], [1, 2]).astype(float)

            # Cross entropy: -[t*log(p) + (1-t)*log(1-p)]
            eps = 1e-8
            probs_clipped = np.clip(probs, eps, 1.0 - eps)
            loss = -(
                targets * np.log(probs_clipped)
                + (1 - targets) * np.log(1 - probs_clipped)
            )
            total_loss += np.mean(loss)

        return total_loss / batch_size

    def parameter_shift_gradient(
        self,
        syndromes: np.ndarray,
        errors: np.ndarray,
    ) -> np.ndarray:
        """Compute gradients using the parameter shift rule.

        For each parameter θ_k, the gradient is computed as:
            ∂L/∂θ_k = [L(θ_k + π/2) - L(θ_k - π/2)] / 2

        Parameters
        ----------
        syndromes : np.ndarray
            Shape ``(batch_size, n_stabilizers)``.
        errors : np.ndarray
            Shape ``(batch_size, n_qubits)``.

        Returns
        -------
        np.ndarray
            Gradient vector of shape ``(n_params,)``.
        """
        gradients = np.zeros(self.ansatz.n_params)
        original_params = self.params.copy()

        for k in range(self.ansatz.n_params):
            # Forward shift
            shifted_plus = original_params.copy()
            shifted_plus[k] += PARAMETER_SHIFT_DELTA
            self.params = shifted_plus
            loss_plus = self.compute_loss(syndromes, errors)

            # Backward shift
            shifted_minus = original_params.copy()
            shifted_minus[k] -= PARAMETER_SHIFT_DELTA
            self.params = shifted_minus
            loss_minus = self.compute_loss(syndromes, errors)

            gradients[k] = (loss_plus - loss_minus) / (
                2.0 * np.sin(PARAMETER_SHIFT_DELTA)
            )

        # Restore original parameters
        self.params = original_params
        return gradients

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
        logical_x = logical_ops["X"]

        n_logical_errors = 0
        for i in range(n_shots):
            error = errors[i]
            syndrome = self.code.extract_syndrome(error)
            correction = self.decode(syndrome)

            # Combined error after correction
            residual = self._compose_paulis(error, correction)

            # Check if residual is a non-trivial logical operator
            if self._is_logical_error(residual, logical_x):
                n_logical_errors += 1

        rate = n_logical_errors / n_shots
        logger.debug(
            "Logical error rate: %d / %d = %.6f", n_logical_errors, n_shots, rate
        )
        return rate

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
        residual: np.ndarray, logical_x: np.ndarray
    ) -> bool:
        """Check if the residual error is equivalent to a logical X.

        Parameters
        ----------
        residual : np.ndarray
            Residual Pauli after correction.
        logical_x : np.ndarray
            Logical X operator.

        Returns
        -------
        bool
        """
        # Check if X component of residual matches logical X
        res_x = np.isin(residual, [1, 2]).astype(int)
        log_x = np.isin(logical_x, [1, 2]).astype(int)
        return bool(np.array_equal(res_x, log_x))
