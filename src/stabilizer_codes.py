"""Stabilizer codes for quantum error correction.

Implements the repetition code and rotated surface code with full
syndrome extraction, logical operator construction, and syndrome-graph
utilities compatible with PyMatching and Stim.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import stim
except ImportError:
    stim = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StabilizerCode(ABC):
    """Abstract base class for stabilizer codes."""

    @abstractmethod
    def get_stabilizers(self) -> np.ndarray:
        """Return the stabilizer check matrix (binary symplectic).

        Returns
        -------
        np.ndarray
            Shape ``(n_stabilizers, 2 * n_qubits)`` binary matrix where
            the first ``n_qubits`` columns represent X components and the
            last ``n_qubits`` columns represent Z components.
        """

    @abstractmethod
    def extract_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Extract the syndrome for a given Pauli error.

        Parameters
        ----------
        error : np.ndarray
            Integer array of length ``n_qubits`` with entries in {0,1,2,3}
            encoding I, X, Y, Z.

        Returns
        -------
        np.ndarray
            Binary syndrome vector of length ``n_stabilizers``.
        """

    @abstractmethod
    def get_logical_ops(self) -> Dict[str, np.ndarray]:
        """Return the logical X and Z operators.

        Returns
        -------
        dict
            ``{'X': array, 'Z': array}`` each of length ``n_qubits``
            with entries in {0,1,2,3}.
        """

    @abstractmethod
    def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """Minimum-weight lookup decode.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Proposed Pauli correction of length ``n_qubits``.
        """


# ---------------------------------------------------------------------------
# Repetition Code
# ---------------------------------------------------------------------------

@dataclass
class RepetitionCode(StabilizerCode):
    """Distance-*d* repetition code (bit-flip only).

    The code encodes 1 logical qubit into *d* physical qubits.
    Stabiliser generators are Z_i Z_{i+1} for i = 0, ..., d-2.

    Parameters
    ----------
    d : int
        Code distance (≥ 3, odd recommended).
    """

    d: int

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError(f"Code distance must be >= 2, got {self.d}")
        self.n_qubits: int = self.d
        self.n_stabilizers: int = self.d - 1
        logger.info("RepetitionCode initialised with d=%d", self.d)

    # -- stabiliser check matrix -------------------------------------------

    def get_stabilizers(self) -> np.ndarray:
        """Return the parity check matrix in binary symplectic form.

        Returns
        -------
        np.ndarray
            Shape ``(d-1, 2*d)`` binary matrix.
        """
        h = np.zeros((self.n_stabilizers, 2 * self.n_qubits), dtype=int)
        for i in range(self.n_stabilizers):
            # Z_i Z_{i+1} -> set the Z-part (second half) columns
            h[i, self.n_qubits + i] = 1
            h[i, self.n_qubits + i + 1] = 1
        return h

    # -- syndrome extraction -----------------------------------------------

    def extract_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Extract syndrome for a Pauli error vector.

        Parameters
        ----------
        error : np.ndarray
            Length-*d* integer array, entries in {0,1,2,3}.

        Returns
        -------
        np.ndarray
            Binary syndrome of length ``d - 1``.
        """
        # X part of the error (stabilisers are ZZ so they anticommute with X)
        x_component = np.isin(error, [1, 2]).astype(int)  # X or Y
        syndrome = np.zeros(self.n_stabilizers, dtype=int)
        for i in range(self.n_stabilizers):
            syndrome[i] = (x_component[i] + x_component[i + 1]) % 2
        return syndrome

    # -- logical operators --------------------------------------------------

    def get_logical_ops(self) -> Dict[str, np.ndarray]:
        """Return logical X (all X) and logical Z (single Z).

        Returns
        -------
        dict
            ``{'X': array, 'Z': array}`` of length ``d``.
        """
        logical_x = np.ones(self.n_qubits, dtype=int)        # X on all qubits
        logical_z = np.zeros(self.n_qubits, dtype=int)
        logical_z[0] = 3  # Z on qubit 0
        return {"X": logical_x, "Z": logical_z}

    # -- simple decoder -----------------------------------------------------

    def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode via minimum-weight matching on the 1-D chain.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome of length ``d - 1``.

        Returns
        -------
        np.ndarray
            Pauli correction (length ``d``), entries in {0, 1}.
        """
        correction = np.zeros(self.n_qubits, dtype=int)
        defect_positions = np.where(syndrome == 1)[0]

        if len(defect_positions) == 0:
            return correction

        # Pair adjacent defects greedily
        i = 0
        while i < len(defect_positions):
            if i + 1 < len(defect_positions):
                left = defect_positions[i]
                right = defect_positions[i + 1]
                # Flip qubits between left and right+1
                for q in range(left + 1, right + 1):
                    correction[q] = 1  # X correction
                i += 2
            else:
                # Unpaired defect — connect to boundary
                pos = defect_positions[i]
                if pos < self.n_qubits // 2:
                    for q in range(0, pos + 1):
                        correction[q] = 1
                else:
                    for q in range(pos + 1, self.n_qubits):
                        correction[q] = 1
                i += 1

        return correction

    # -- stim circuit (for fast sampling) ------------------------------------

    def get_stim_circuit(self, p: float) -> "stim.Circuit":
        """Build a Stim circuit for fast syndrome sampling.

        Parameters
        ----------
        p : float
            Physical error rate (bit-flip).

        Returns
        -------
        stim.Circuit
        """
        if stim is None:
            raise ImportError("stim is required for get_stim_circuit")
        circuit = stim.Circuit()
        # Data qubits: 0..d-1
        # Apply X errors
        for q in range(self.n_qubits):
            circuit.append("X_ERROR", [q], p)
        # Syndrome extraction via ZZ measurements
        for i in range(self.n_stabilizers):
            circuit.append("M", [i, i + 1])
        # Detectors
        for i in range(self.n_stabilizers):
            circuit.append(
                "DETECTOR",
                [stim.target_rec(-2 * self.n_stabilizers + 2 * i),
                 stim.target_rec(-2 * self.n_stabilizers + 2 * i + 1)],
            )
        return circuit


# ---------------------------------------------------------------------------
# Rotated Surface Code
# ---------------------------------------------------------------------------

@dataclass
class SurfaceCode(StabilizerCode):
    """Rotated surface code of distance *d*.

    Arranges ``d * d`` data qubits on a grid with X and Z stabilisers
    following the standard rotated layout.

    Parameters
    ----------
    d : int
        Code distance (≥ 3, odd recommended).
    """

    d: int

    def __post_init__(self) -> None:
        if self.d < 2:
            raise ValueError(f"Code distance must be >= 2, got {self.d}")
        self.n_qubits: int = self.d * self.d
        self._build_stabilizers()
        logger.info(
            "SurfaceCode initialised with d=%d (%d data qubits, %d X-stabs, %d Z-stabs)",
            self.d,
            self.n_qubits,
            len(self._x_stabilizers),
            len(self._z_stabilizers),
        )

    # -- internal construction -----------------------------------------------

    def _qubit_index(self, row: int, col: int) -> int:
        """Convert (row, col) to flat qubit index."""
        return row * self.d + col

    def _build_stabilizers(self) -> None:
        """Build X and Z stabiliser lists for the rotated surface code.

        X-stabilisers live on "white" plaquettes, Z-stabilisers on "black".
        Each stabiliser is a list of qubit indices it acts on.
        """
        self._x_stabilizers: List[List[int]] = []
        self._z_stabilizers: List[List[int]] = []

        for row in range(self.d - 1):
            for col in range(self.d - 1):
                plaquette = [
                    self._qubit_index(row, col),
                    self._qubit_index(row, col + 1),
                    self._qubit_index(row + 1, col),
                    self._qubit_index(row + 1, col + 1),
                ]
                if (row + col) % 2 == 0:
                    self._x_stabilizers.append(plaquette)
                else:
                    self._z_stabilizers.append(plaquette)

        # Boundary stabilisers (weight-2)
        for col in range(0, self.d - 1, 2):
            self._x_stabilizers.append(
                [self._qubit_index(0, col), self._qubit_index(0, col + 1)]
            )
        for col in range(1, self.d - 1, 2):
            self._x_stabilizers.append(
                [
                    self._qubit_index(self.d - 1, col),
                    self._qubit_index(self.d - 1, col + 1),
                ]
            )
        for row in range(0, self.d - 1, 2):
            self._z_stabilizers.append(
                [self._qubit_index(row, 0), self._qubit_index(row + 1, 0)]
            )
        for row in range(1, self.d - 1, 2):
            self._z_stabilizers.append(
                [
                    self._qubit_index(row, self.d - 1),
                    self._qubit_index(row + 1, self.d - 1),
                ]
            )

        self.n_x_stabilizers: int = len(self._x_stabilizers)
        self.n_z_stabilizers: int = len(self._z_stabilizers)
        self.n_stabilizers: int = self.n_x_stabilizers + self.n_z_stabilizers

    # -- public interface ---------------------------------------------------

    def get_stabilizers(self) -> np.ndarray:
        """Return the full stabiliser check matrix (binary symplectic).

        Returns
        -------
        np.ndarray
            Shape ``(n_stabilizers, 2 * n_qubits)``.
        """
        n = self.n_qubits
        h = np.zeros((self.n_stabilizers, 2 * n), dtype=int)

        for i, stab in enumerate(self._x_stabilizers):
            for q in stab:
                h[i, q] = 1  # X part

        for j, stab in enumerate(self._z_stabilizers):
            for q in stab:
                h[self.n_x_stabilizers + j, n + q] = 1  # Z part

        return h

    def extract_syndrome(self, error: np.ndarray) -> np.ndarray:
        """Extract X and Z syndromes from a Pauli error.

        Parameters
        ----------
        error : np.ndarray
            Length ``d*d`` integer array, entries in {0,1,2,3}.

        Returns
        -------
        np.ndarray
            Binary syndrome of length ``n_stabilizers``.
        """
        # Decompose into X and Z components
        x_component = np.isin(error, [1, 2]).astype(int)
        z_component = np.isin(error, [2, 3]).astype(int)

        syndrome = np.zeros(self.n_stabilizers, dtype=int)

        # X stabilisers detect Z errors
        for i, stab in enumerate(self._x_stabilizers):
            syndrome[i] = np.sum(z_component[stab]) % 2

        # Z stabilisers detect X errors
        for j, stab in enumerate(self._z_stabilizers):
            syndrome[self.n_x_stabilizers + j] = np.sum(x_component[stab]) % 2

        return syndrome

    def get_logical_ops(self) -> Dict[str, np.ndarray]:
        """Return representative logical X and Z operators.
        
        For a rotated surface code, logical X connects Top and Bottom borders.
        Logical Z connects Left and Right borders.
        """
        logical_x = np.zeros(self.n_qubits, dtype=int)
        logical_z = np.zeros(self.n_qubits, dtype=int)

        # Logical X: Column 0 connects Top and Bottom X-boundaries
        for row in range(self.d):
            logical_x[self._qubit_index(row, 0)] = 1

        # Logical Z: Row 0 connects Left and Right Z-boundaries
        for col in range(self.d):
            logical_z[self._qubit_index(0, col)] = 3

        return {"X": logical_x, "Z": logical_z}

    def decode_syndrome(self, syndrome: np.ndarray) -> np.ndarray:
        """Simple lookup / greedy decode for small codes.

        For production use, prefer PyMatching via ``classical_decoders.py``.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.

        Returns
        -------
        np.ndarray
            Pauli correction (length ``d*d``).
        """
        correction = np.zeros(self.n_qubits, dtype=int)
        z_syn = syndrome[: self.n_x_stabilizers]
        x_syn = syndrome[self.n_x_stabilizers :]

        # Greedy: find stabiliser with syndrome 1 and flip first qubit in it
        for i, val in enumerate(z_syn):
            if val == 1 and i < len(self._x_stabilizers):
                correction[self._x_stabilizers[i][0]] ^= 3  # Z correction

        for j, val in enumerate(x_syn):
            if val == 1 and j < len(self._z_stabilizers):
                correction[self._z_stabilizers[j][0]] ^= 1  # X correction

        return correction

    def get_syndrome_graph(self) -> Dict[str, object]:
        """Return the syndrome graph structure for MWPM decoding.

        Returns
        -------
        dict
            Contains ``'x_stabilizers'``, ``'z_stabilizers'``,
            ``'edges'`` and ``'boundary_nodes'``.
        """
        edges_x: List[Tuple[int, int, float]] = []
        edges_z: List[Tuple[int, int, float]] = []

        # Build adjacency for X stabilisers (sharing data qubits)
        for i in range(len(self._x_stabilizers)):
            for j in range(i + 1, len(self._x_stabilizers)):
                shared = set(self._x_stabilizers[i]) & set(self._x_stabilizers[j])
                if shared:
                    edges_x.append((i, j, 1.0))

        # Build adjacency for Z stabilisers
        n_off = self.n_x_stabilizers
        for i in range(len(self._z_stabilizers)):
            for j in range(i + 1, len(self._z_stabilizers)):
                shared = set(self._z_stabilizers[i]) & set(self._z_stabilizers[j])
                if shared:
                    edges_z.append((n_off + i, n_off + j, 1.0))

        return {
            "x_stabilizers": self._x_stabilizers,
            "z_stabilizers": self._z_stabilizers,
            "edges_x": edges_x,
            "edges_z": edges_z,
            "n_x_stabs": self.n_x_stabilizers,
            "n_z_stabs": self.n_z_stabilizers,
        }

    def get_stim_circuit(self, p: float) -> "stim.Circuit":
        """Build a Stim circuit for fast syndrome sampling.

        Parameters
        ----------
        p : float
            Physical depolarizing error rate.

        Returns
        -------
        stim.Circuit
        """
        if stim is None:
            raise ImportError("stim is required for get_stim_circuit")

        circuit = stim.Circuit()

        # Apply depolarizing noise to all data qubits
        for q in range(self.n_qubits):
            circuit.append("DEPOLARIZE1", [q], p)

        # Measure X stabilisers
        for stab in self._x_stabilizers:
            for q in stab:
                circuit.append("H", [q])
            if len(stab) >= 2:
                for i in range(len(stab) - 1):
                    circuit.append("CNOT", [stab[i], stab[i + 1]])
            for q in stab:
                circuit.append("H", [q])
            circuit.append("M", stab[-1:])

        # Measure Z stabilisers
        for stab in self._z_stabilizers:
            if len(stab) >= 2:
                for i in range(len(stab) - 1):
                    circuit.append("CNOT", [stab[i], stab[i + 1]])
            circuit.append("M", stab[-1:])

        return circuit
