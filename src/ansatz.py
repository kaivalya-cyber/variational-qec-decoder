"""Parameterized quantum circuit ansätze for variational QEC decoding.

Implements hardware-efficient, symmetry-preserving, and adaptive ansatz
circuits using PennyLane.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pennylane as qml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_N_LAYERS: int = 4
SUPPORTED_NOISE_TYPES: List[str] = ["depolarizing", "bit_flip", "phase_flip", "combined"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Ansatz(ABC):
    """Abstract base class for variational ansätze."""

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of trainable parameters.

        Returns
        -------
        int
        """

    @abstractmethod
    def init_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize parameters (small random values).

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            1-D array of shape ``(n_params,)``.
        """

    @abstractmethod
    def forward(self, params: np.ndarray, syndrome_input: np.ndarray) -> None:
        """Apply the parameterized circuit to the current PennyLane context.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector of length ``n_params``.
        syndrome_input : np.ndarray
            Binary syndrome vector used to condition the circuit.
        """


# ---------------------------------------------------------------------------
# Hardware-Efficient Ansatz
# ---------------------------------------------------------------------------

@dataclass
class HardwareEfficientAnsatz(Ansatz):
    """RY/RZ + CNOT brickwork ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    n_layers : int
        Number of variational layers.
    """

    n_qubits: int
    n_layers: int = DEFAULT_N_LAYERS

    def __post_init__(self) -> None:
        logger.info(
            "HardwareEfficientAnsatz: n_qubits=%d, n_layers=%d",
            self.n_qubits,
            self.n_layers,
        )

    @property
    def n_params(self) -> int:
        """Each layer has 2 parameters per qubit (RY + RZ)."""
        return self.n_layers * self.n_qubits * 2

    def init_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Small random initialisation.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Parameter vector of shape ``(n_params,)``.
        """
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.1, size=self.n_params)

    def forward(self, params: np.ndarray, syndrome_input: np.ndarray) -> None:
        """Apply the hardware-efficient circuit.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        syndrome_input : np.ndarray
            Binary syndrome; used for input encoding via RX rotations.
        """
        # Encode syndrome into the circuit
        for i, s in enumerate(syndrome_input):
            if i < self.n_qubits:
                qml.RX(float(s) * np.pi, wires=i)

        # Variational layers
        idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                qml.RY(params[idx], wires=q)
                idx += 1
                qml.RZ(params[idx], wires=q)
                idx += 1
            # CNOT brickwork
            start = layer % 2
            for q in range(start, self.n_qubits - 1, 2):
                qml.CNOT(wires=[q, q + 1])


# ---------------------------------------------------------------------------
# Symmetry-Preserving Ansatz
# ---------------------------------------------------------------------------

@dataclass
class SymmetryPreservingAnsatz(Ansatz):
    """Ansatz that preserves the Z₂ symmetry of stabiliser codes.

    Uses parity-preserving gates (XX + YY interactions) so that the
    circuit output stays within the code space.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    """

    n_qubits: int
    n_layers: int = DEFAULT_N_LAYERS

    def __post_init__(self) -> None:
        logger.info(
            "SymmetryPreservingAnsatz: n_qubits=%d, n_layers=%d",
            self.n_qubits,
            self.n_layers,
        )

    @property
    def n_params(self) -> int:
        """Each layer: 1 RZ per qubit + 1 param per pair for IsingXX."""
        n_pairs = max(self.n_qubits - 1, 0)
        return self.n_layers * (self.n_qubits + n_pairs)

    def init_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Small random initialisation preserving symmetry bias.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
        """
        rng = np.random.default_rng(seed)
        return rng.normal(0, 0.05, size=self.n_params)

    def forward(self, params: np.ndarray, syndrome_input: np.ndarray) -> None:
        """Apply the symmetry-preserving circuit.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        syndrome_input : np.ndarray
            Binary syndrome for input encoding.
        """
        # Syndrome encoding
        for i, s in enumerate(syndrome_input):
            if i < self.n_qubits:
                qml.RX(float(s) * np.pi, wires=i)

        idx = 0
        n_pairs = max(self.n_qubits - 1, 0)
        for _layer in range(self.n_layers):
            # RZ rotations
            for q in range(self.n_qubits):
                qml.RZ(params[idx], wires=q)
                idx += 1
            # Parity-preserving IsingXX entanglement
            for q in range(n_pairs):
                qml.IsingXX(params[idx], wires=[q, q + 1])
                idx += 1


# ---------------------------------------------------------------------------
# Adaptive Ansatz (NOVEL COMPONENT)
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveAnsatz(Ansatz):
    """Noise-type-aware adaptive ansatz — NOVEL COMPONENT.

    Selects the circuit structure based on the detected noise type:

    * **depolarizing**: deeper hardware-efficient (more layers, balanced gates)
    * **bit_flip**: X-rotation heavy with ZZ entanglement
    * **phase_flip**: Z-rotation heavy with XX entanglement
    * **combined**: mixture of both strategies

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    noise_type : str
        One of ``'depolarizing'``, ``'bit_flip'``, ``'phase_flip'``,
        ``'combined'``.
    n_layers : int
        Base number of layers (adjusted per noise type).
    """

    n_qubits: int
    noise_type: str = "depolarizing"
    n_layers: int = DEFAULT_N_LAYERS

    def __post_init__(self) -> None:
        if self.noise_type not in SUPPORTED_NOISE_TYPES:
            raise ValueError(
                f"Unsupported noise_type '{self.noise_type}'. "
                f"Choose from {SUPPORTED_NOISE_TYPES}"
            )
        self._effective_layers = self._compute_effective_layers()
        self._params_per_layer = self._compute_params_per_layer()
        logger.info(
            "AdaptiveAnsatz: n_qubits=%d, noise_type='%s', "
            "effective_layers=%d, params_per_layer=%d",
            self.n_qubits,
            self.noise_type,
            self._effective_layers,
            self._params_per_layer,
        )

    def _compute_effective_layers(self) -> int:
        """Compute the effective number of layers based on noise type."""
        multipliers = {
            "depolarizing": 1.5,
            "bit_flip": 1.0,
            "phase_flip": 1.0,
            "combined": 1.25,
        }
        return max(1, int(self.n_layers * multipliers[self.noise_type]))

    def _compute_params_per_layer(self) -> int:
        """Compute parameters per layer based on noise type."""
        if self.noise_type == "depolarizing":
            # RY + RZ per qubit
            return self.n_qubits * 2
        elif self.noise_type == "bit_flip":
            # RX + RY per qubit
            return self.n_qubits * 2
        elif self.noise_type == "phase_flip":
            # RZ + RY per qubit
            return self.n_qubits * 2
        else:  # combined
            # RX + RY + RZ per qubit
            return self.n_qubits * 3

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        return self._effective_layers * self._params_per_layer

    def init_params(self, seed: Optional[int] = None) -> np.ndarray:
        """Initialize with noise-aware strategy.

        Parameters
        ----------
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
        """
        rng = np.random.default_rng(seed)
        scale = 0.1 if self.noise_type != "combined" else 0.05
        return rng.normal(0, scale, size=self.n_params)

    def forward(self, params: np.ndarray, syndrome_input: np.ndarray) -> None:
        """Apply the noise-adapted circuit.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector.
        syndrome_input : np.ndarray
            Binary syndrome for input encoding.
        """
        # Syndrome encoding
        for i, s in enumerate(syndrome_input):
            if i < self.n_qubits:
                qml.RX(float(s) * np.pi, wires=i)

        idx = 0
        for _layer in range(self._effective_layers):
            if self.noise_type == "depolarizing":
                idx = self._depolarizing_layer(params, idx)
            elif self.noise_type == "bit_flip":
                idx = self._bit_flip_layer(params, idx)
            elif self.noise_type == "phase_flip":
                idx = self._phase_flip_layer(params, idx)
            else:
                idx = self._combined_layer(params, idx)

    def _depolarizing_layer(self, params: np.ndarray, idx: int) -> int:
        """Balanced RY/RZ layer with CNOT brickwork."""
        for q in range(self.n_qubits):
            qml.RY(params[idx], wires=q)
            idx += 1
            qml.RZ(params[idx], wires=q)
            idx += 1
        for q in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        return idx

    def _bit_flip_layer(self, params: np.ndarray, idx: int) -> int:
        """X-rotation heavy layer with ZZ entanglement."""
        for q in range(self.n_qubits):
            qml.RX(params[idx], wires=q)
            idx += 1
            qml.RY(params[idx], wires=q)
            idx += 1
        for q in range(0, self.n_qubits - 1, 2):
            qml.IsingZZ(0.25 * np.pi, wires=[q, q + 1])
        return idx

    def _phase_flip_layer(self, params: np.ndarray, idx: int) -> int:
        """Z-rotation heavy layer with XX entanglement."""
        for q in range(self.n_qubits):
            qml.RZ(params[idx], wires=q)
            idx += 1
            qml.RY(params[idx], wires=q)
            idx += 1
        for q in range(0, self.n_qubits - 1, 2):
            qml.IsingXX(0.25 * np.pi, wires=[q, q + 1])
        return idx

    def _combined_layer(self, params: np.ndarray, idx: int) -> int:
        """Full rotation layer (RX + RY + RZ) with alternating entanglement."""
        for q in range(self.n_qubits):
            qml.RX(params[idx], wires=q)
            idx += 1
            qml.RY(params[idx], wires=q)
            idx += 1
            qml.RZ(params[idx], wires=q)
            idx += 1
        for q in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        for q in range(1, self.n_qubits - 1, 2):
            qml.CNOT(wires=[q, q + 1])
        return idx
