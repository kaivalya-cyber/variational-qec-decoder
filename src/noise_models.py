"""Noise models for quantum error correction simulations.

Implements depolarizing, bit-flip, phase-flip, combined, and amplitude
damping noise channels with support for PennyLane circuit integration,
Monte Carlo error sampling, and Kraus/channel matrix representations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOISE_TYPES: List[str] = ["depolarizing", "bit_flip", "phase_flip", "combined"]

# Pauli matrices
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class NoiseModel(ABC):
    """Abstract base class for all noise models.

    Every concrete noise model must implement three methods:

    * ``apply`` — insert noise operations into a PennyLane circuit.
    * ``sample_errors`` — draw Monte Carlo Pauli error samples.
    * ``get_channel_matrix`` — return the (superoperator) channel matrix.
    """

    @abstractmethod
    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply the noise channel to *qubits* inside *circuit*.

        Parameters
        ----------
        circuit : qml.tape.QuantumTape
            The PennyLane quantum tape / circuit being constructed.
        qubits : list of int
            Indices of the qubits to apply noise to.
        """

    @abstractmethod
    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample Pauli errors via Monte Carlo.

        Parameters
        ----------
        n_qubits : int
            Number of data qubits.
        n_shots : int
            Number of independent error samples.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Integer array of shape ``(n_shots, n_qubits)`` with entries in
            ``{0, 1, 2, 3}`` representing ``{I, X, Y, Z}``.
        """

    @abstractmethod
    def get_channel_matrix(self) -> np.ndarray:
        """Return the single-qubit superoperator (Pauli transfer) matrix.

        Returns
        -------
        np.ndarray
            A 4×4 real matrix acting on the Pauli-vector representation
            ``[I, X, Y, Z]``.
        """


# ---------------------------------------------------------------------------
# Depolarizing noise
# ---------------------------------------------------------------------------

@dataclass
class DepolarizingNoise(NoiseModel):
    """Depolarizing channel: applies X, Y, Z each with probability ``p/3``.

    Parameters
    ----------
    p : float
        Total depolarizing probability (0 ≤ p ≤ 1).
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        logger.info("DepolarizingNoise initialised with p=%.4f", self.p)

    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply depolarizing noise to each qubit in *qubits*."""
        for q in qubits:
            qml.DepolarizingChannel(self.p, wires=q)

    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample depolarizing errors.

        Returns
        -------
        np.ndarray
            Shape ``(n_shots, n_qubits)``, entries in {0,1,2,3}.
        """
        rng = np.random.default_rng(seed)
        p_each = self.p / 3.0
        probs = np.array([1.0 - self.p, p_each, p_each, p_each], dtype=np.float64)
        probs /= probs.sum()  # Force sum to 1.0 to avoid floating point errors
        return rng.choice(4, size=(n_shots, n_qubits), p=probs)

    def get_channel_matrix(self) -> np.ndarray:
        """Pauli transfer matrix for the depolarizing channel."""
        lam = 1.0 - 4.0 * self.p / 3.0
        return np.diag([1.0, lam, lam, lam])


# ---------------------------------------------------------------------------
# Bit-flip noise
# ---------------------------------------------------------------------------

@dataclass
class BitFlipNoise(NoiseModel):
    """Bit-flip channel: applies X with probability *p*.

    Parameters
    ----------
    p : float
        Bit-flip probability (0 ≤ p ≤ 1).
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        logger.info("BitFlipNoise initialised with p=%.4f", self.p)

    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply bit-flip noise to each qubit in *qubits*."""
        for q in qubits:
            qml.BitFlip(self.p, wires=q)

    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample bit-flip errors (0=I, 1=X)."""
        rng = np.random.default_rng(seed)
        flips = rng.random((n_shots, n_qubits)) < self.p
        return flips.astype(int)  # 0 -> I, 1 -> X

    def get_channel_matrix(self) -> np.ndarray:
        """Pauli transfer matrix for the bit-flip channel."""
        return np.diag([1.0, 1.0 - 2.0 * self.p, 1.0 - 2.0 * self.p, 1.0])


# ---------------------------------------------------------------------------
# Phase-flip noise
# ---------------------------------------------------------------------------

@dataclass
class PhaseFlipNoise(NoiseModel):
    """Phase-flip channel: applies Z with probability *p*.

    Parameters
    ----------
    p : float
        Phase-flip probability (0 ≤ p ≤ 1).
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        logger.info("PhaseFlipNoise initialised with p=%.4f", self.p)

    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply phase-flip noise to each qubit in *qubits*."""
        for q in qubits:
            qml.PhaseFlip(self.p, wires=q)

    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample phase-flip errors (0=I, 3=Z)."""
        rng = np.random.default_rng(seed)
        flips = rng.random((n_shots, n_qubits)) < self.p
        return (flips.astype(int)) * 3  # 0 -> I, 3 -> Z

    def get_channel_matrix(self) -> np.ndarray:
        """Pauli transfer matrix for the phase-flip channel."""
        return np.diag([1.0, 1.0 - 2.0 * self.p, 1.0 - 2.0 * self.p, 1.0])


# ---------------------------------------------------------------------------
# Combined bit + phase flip noise
# ---------------------------------------------------------------------------

@dataclass
class CombinedNoise(NoiseModel):
    """Combined independent bit-flip and phase-flip channel.

    Parameters
    ----------
    p_bit : float
        Bit-flip probability.
    p_phase : float
        Phase-flip probability.
    """

    p_bit: float
    p_phase: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_bit <= 1.0:
            raise ValueError(f"p_bit must be in [0, 1], got {self.p_bit}")
        if not 0.0 <= self.p_phase <= 1.0:
            raise ValueError(f"p_phase must be in [0, 1], got {self.p_phase}")
        logger.info(
            "CombinedNoise initialised with p_bit=%.4f, p_phase=%.4f",
            self.p_bit,
            self.p_phase,
        )

    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply independent bit-flip then phase-flip noise."""
        for q in qubits:
            qml.BitFlip(self.p_bit, wires=q)
            qml.PhaseFlip(self.p_phase, wires=q)

    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Sample combined bit+phase errors.

        Each qubit independently receives an X error with prob ``p_bit``
        and a Z error with prob ``p_phase``.  The resulting Pauli is
        the product: I, X, Z, or Y = iXZ  (encoded as 0,1,3,2).
        """
        rng = np.random.default_rng(seed)
        x_err = rng.random((n_shots, n_qubits)) < self.p_bit
        z_err = rng.random((n_shots, n_qubits)) < self.p_phase
        # Encode: 0=I, 1=X, 2=Y, 3=Z
        errors = x_err.astype(int) + 3 * z_err.astype(int)
        # Fix the Y case: X (1) + Z (3) = 4 -> should be Y (2)
        errors[errors == 4] = 2
        return errors

    def get_channel_matrix(self) -> np.ndarray:
        """Pauli transfer matrix for the combined channel."""
        bx = 1.0 - 2.0 * self.p_bit
        pz = 1.0 - 2.0 * self.p_phase
        return np.diag([1.0, bx * pz, bx * pz, pz * bx])


# ---------------------------------------------------------------------------
# Amplitude damping noise
# ---------------------------------------------------------------------------

@dataclass
class AmplitudeDamping(NoiseModel):
    """Amplitude damping channel (non-Pauli).

    Parameters
    ----------
    gamma : float
        Damping probability (0 ≤ γ ≤ 1).
    """

    gamma: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        logger.info("AmplitudeDamping initialised with gamma=%.4f", self.gamma)

    def apply(self, circuit: qml.tape.QuantumTape, qubits: List[int]) -> None:
        """Apply amplitude damping channel to each qubit."""
        for q in qubits:
            qml.AmplitudeDamping(self.gamma, wires=q)

    def sample_errors(
        self, n_qubits: int, n_shots: int, seed: Optional[int] = None
    ) -> np.ndarray:
        """Approximate amplitude damping errors with Pauli twirl.

        Amplitude damping is non-Pauli, so we use the Pauli-twirl
        approximation for sampling:
        - I with prob 1 - gamma/2
        - X with prob gamma/4
        - Y with prob gamma/4
        - Z with prob 0  (to first order)

        For exact simulation use the Kraus representation via ``apply``.
        """
        rng = np.random.default_rng(seed)
        p_x = self.gamma / 4.0
        p_y = self.gamma / 4.0
        p_z = 0.0
        p_i = 1.0 - p_x - p_y - p_z
        probs = [p_i, p_x, p_y, p_z]
        return rng.choice(4, size=(n_shots, n_qubits), p=probs)

    def get_channel_matrix(self) -> np.ndarray:
        """Pauli transfer matrix (Pauli-twirl approximation)."""
        g = self.gamma
        sqrt_g = np.sqrt(1.0 - g)
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, sqrt_g, 0.0, 0.0],
                [0.0, 0.0, sqrt_g, 0.0],
                [g, 0.0, 0.0, 1.0 - g],
            ]
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_noise_model(
    noise_type: str,
    p: float = 0.01,
    p_bit: float = 0.01,
    p_phase: float = 0.01,
    gamma: float = 0.01,
) -> NoiseModel:
    """Create a noise model from a string identifier.

    Parameters
    ----------
    noise_type : str
        One of ``'depolarizing'``, ``'bit_flip'``, ``'phase_flip'``,
        ``'combined'``, ``'amplitude_damping'``.
    p : float
        Error probability for single-parameter models.
    p_bit : float
        Bit-flip probability for the combined model.
    p_phase : float
        Phase-flip probability for the combined model.
    gamma : float
        Damping parameter for amplitude damping.

    Returns
    -------
    NoiseModel
        An instance of the requested noise model.
    """
    noise_type = noise_type.lower().strip()
    if noise_type == "depolarizing":
        return DepolarizingNoise(p=p)
    elif noise_type == "bit_flip":
        return BitFlipNoise(p=p)
    elif noise_type == "phase_flip":
        return PhaseFlipNoise(p=p)
    elif noise_type == "combined":
        return CombinedNoise(p_bit=p_bit, p_phase=p_phase)
    elif noise_type == "amplitude_damping":
        return AmplitudeDamping(gamma=gamma)
    else:
        raise ValueError(
            f"Unknown noise type: '{noise_type}'. "
            f"Supported: depolarizing, bit_flip, phase_flip, combined, amplitude_damping"
        )
