"""Online learning and real-time decoder adaptation.

Generates: Section V (Online Adaptation),
           Figures 5-6 (online_sudden_switch.png, online_gradual_drift.png)

Implements an online fine-tuning loop that continuously updates decoder
parameters using recent syndrome data during deployment.  The decoder
literally learns and adapts as hardware noise drifts in real time.

Hardware constraints (3 GB RAM):
- Experience replay buffer: hard cap 200 experiences (deque maxlen=200)
- Each experience stores only: syndrome bits (uint8), correction bits (uint8),
  outcome (bool) — no full tensors, no gradients
- Minibatch size for online updates: 8 samples maximum
- Drift simulator generates noise samples one at a time (no pre-generation)
- numpy float32 throughout
"""

from __future__ import annotations

import collections
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .decoder import VariationalDecoder
from .noise_models import (
    NoiseModel,
    DepolarizingNoise,
    BitFlipNoise,
    PhaseFlipNoise,
    CombinedNoise,
    create_noise_model,
)
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OnlineConfig:
    """Configuration for online learning.

    Attributes
    ----------
    buffer_size : int
        Maximum experience replay buffer size.
    minibatch_size : int
        Number of samples drawn per gradient update.
    update_freq : int
        Steps between gradient updates.
    lr : float
        Learning rate for online parameter updates.
    parameter_shift_delta : float
        Shift value for parameter-shift rule.
    """

    buffer_size: int = 200
    minibatch_size: int = 8
    update_freq: int = 50
    lr: float = 0.01
    parameter_shift_delta: float = np.pi / 2


ONLINE_CONFIG = OnlineConfig()

FIGURES_DIR: str = "results/figures"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Experience (memory-efficient)
# ---------------------------------------------------------------------------

class Experience(NamedTuple):
    """Single experience tuple stored in compact format.

    Attributes
    ----------
    syndrome : np.ndarray
        Syndrome bits as uint8.
    correction : np.ndarray
        Applied correction bits as uint8.
    outcome : bool
        True if the correction resulted in a logical error.
    """

    syndrome: np.ndarray  # uint8
    correction: np.ndarray  # uint8
    outcome: bool


# ---------------------------------------------------------------------------
# Online Learner
# ---------------------------------------------------------------------------

@dataclass
class OnlineLearner:
    """Online fine-tuning loop for variational decoders.

    Continuously updates decoder parameters using recent syndrome data
    during deployment via experience replay and the parameter-shift rule.

    Parameters
    ----------
    decoder : VariationalDecoder
        The variational decoder to fine-tune online.
    config : OnlineConfig
        Online learning configuration.
    """

    decoder: VariationalDecoder
    config: OnlineConfig = field(default_factory=OnlineConfig)

    def __post_init__(self) -> None:
        self._buffer: Deque[Experience] = collections.deque(
            maxlen=self.config.buffer_size
        )
        self._step_count: int = 0
        self._param_history: List[float] = []
        self._loss_history: List[float] = []
        logger.info(
            "OnlineLearner initialised: buffer_size=%d, update_freq=%d, "
            "minibatch=%d, lr=%.4f",
            self.config.buffer_size,
            self.config.update_freq,
            self.config.minibatch_size,
            self.config.lr,
        )

    def update(
        self,
        syndrome: np.ndarray,
        correction: np.ndarray,
        logical_outcome: bool,
    ) -> float:
        """Process one decoding experience and optionally update parameters.

        Parameters
        ----------
        syndrome : np.ndarray
            Binary syndrome vector.
        correction : np.ndarray
            Applied Pauli correction.
        logical_outcome : bool
            True if the correction resulted in a logical error.

        Returns
        -------
        float
            Current loss value (0.0 if no update performed this step).
        """
        # Store experience in compact format
        exp = Experience(
            syndrome=syndrome.astype(np.uint8),
            correction=correction.astype(np.uint8),
            outcome=logical_outcome,
        )
        self._buffer.append(exp)
        self._step_count += 1

        loss = 0.0

        # Perform gradient update every update_freq steps
        if (
            self._step_count % self.config.update_freq == 0
            and len(self._buffer) >= self.config.minibatch_size
        ):
            loss = self._gradient_step()
            self._loss_history.append(loss)

        # Track parameter norm for adaptation rate
        param_norm = float(np.linalg.norm(self.decoder.params))
        self._param_history.append(param_norm)

        return loss

    def _gradient_step(self) -> float:
        """Perform one gradient update using parameter-shift rule.

        Samples a minibatch from the replay buffer and computes gradients
        using the parameter-shift rule on the quantum circuit.

        Returns
        -------
        float
            Loss value after the update.
        """
        # Sample minibatch
        indices = np.random.choice(
            len(self._buffer),
            size=min(self.config.minibatch_size, len(self._buffer)),
            replace=False,
        )
        batch = [self._buffer[i] for i in indices]

        syndromes = np.array([e.syndrome for e in batch], dtype=np.float32)
        errors = np.array([e.correction for e in batch], dtype=np.float32)
        outcomes = np.array([e.outcome for e in batch], dtype=np.float32)

        # Compute loss: fraction of logical errors in minibatch
        current_loss = float(np.mean(outcomes))

        # Parameter-shift gradient computation
        params = self.decoder.params.copy()
        n_params = len(params)
        gradients = np.zeros(n_params, dtype=np.float32)
        delta = self.config.parameter_shift_delta

        for i in range(n_params):
            # Shifted+ parameters
            params_plus = params.copy()
            params_plus[i] += delta

            # Shifted- parameters
            params_minus = params.copy()
            params_minus[i] -= delta

            # Evaluate loss at shifted parameters
            loss_plus = self._evaluate_minibatch(
                syndromes, errors, params_plus
            )
            loss_minus = self._evaluate_minibatch(
                syndromes, errors, params_minus
            )

            gradients[i] = (loss_plus - loss_minus) / (2.0 * np.sin(delta))

        # Adam-like update (simplified: just SGD with momentum)
        self.decoder.params = params - self.config.lr * gradients

        # Memory cleanup
        del syndromes, errors, outcomes, batch
        gc.collect()

        return current_loss

    def _evaluate_minibatch(
        self,
        syndromes: np.ndarray,
        errors: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Evaluate the decoder on a minibatch with given parameters.

        Parameters
        ----------
        syndromes : np.ndarray
            Shape ``(batch, n_stabilizers)``.
        errors : np.ndarray
            Shape ``(batch, n_qubits)``.
        params : np.ndarray
            Parameter vector to evaluate.

        Returns
        -------
        float
            Average loss over the minibatch.
        """
        original_params = self.decoder.params.copy()
        self.decoder.params = params

        n_errors = 0
        logical_ops = self.decoder.code.get_logical_ops()
        lx, lz = logical_ops["X"], logical_ops["Z"]

        for i in range(len(syndromes)):
            correction = self.decoder.decode(syndromes[i])
            residual = _compose_paulis(
                errors[i].astype(int), correction
            )
            if _is_logical_error(residual, lx, lz):
                n_errors += 1

        self.decoder.params = original_params
        return n_errors / len(syndromes)

    def get_adaptation_rate(self) -> float:
        """Measure how quickly parameters are changing.

        Returns
        -------
        float
            Norm of parameter change per step (averaged over last 10 steps).
        """
        if len(self._param_history) < 2:
            return 0.0
        recent = self._param_history[-10:]
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return float(np.mean(diffs)) if diffs else 0.0

    def get_loss_history(self) -> List[float]:
        """Return the loss history.

        Returns
        -------
        list of float
        """
        return self._loss_history.copy()

    def reset_buffer(self) -> None:
        """Clear the experience replay buffer.

        This is useful when the noise environment changes drastically.
        """
        self._buffer.clear()
        self._step_count = 0
        logger.info("Online learner buffer reset")


# ---------------------------------------------------------------------------
# Drift Simulator
# ---------------------------------------------------------------------------

@dataclass
class DriftSimulator:
    """Simulates realistic hardware noise drift.

    Supports four noise schedules:
    - ``'sudden_switch'``: abrupt change at a specified step
    - ``'gradual_drift'``: Ornstein-Uhlenbeck process with mean reversion
    - ``'periodic'``: periodic switching between noise types
    - ``'random_walk'``: bounded random walk of noise parameters

    Parameters
    ----------
    code : StabilizerCode
        The QEC code.
    schedule : str
        One of ``'sudden_switch'``, ``'gradual_drift'``, ``'periodic'``,
        ``'random_walk'``.
    p_base : float
        Base physical error rate.
    seed : int
        Random seed.
    """

    code: StabilizerCode
    schedule: str = "sudden_switch"
    p_base: float = 0.03
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        # Internal state for OU process
        self._ou_state = np.array(
            [self.p_base, 0.0, 0.0], dtype=np.float32
        )  # (p_depol, p_x_bias, p_z_bias)
        self._ou_theta = np.float32(0.1)  # mean reversion rate
        self._ou_sigma = np.float32(0.005)  # volatility
        self._ou_mu = np.array(
            [self.p_base, 0.0, 0.0], dtype=np.float32
        )  # long-term mean
        # Random walk state
        self._rw_state = np.array(
            [self.p_base, 0.0, 0.0], dtype=np.float32
        )
        logger.info(
            "DriftSimulator initialised: schedule='%s', p_base=%.4f",
            self.schedule,
            self.p_base,
        )

    def get_noise_at_step(self, t: int) -> NoiseModel:
        """Generate noise model for step t (no pre-generation).

        Parameters
        ----------
        t : int
            Current step index.

        Returns
        -------
        NoiseModel
            Noise model for this step.
        """
        if self.schedule == "sudden_switch":
            return self._sudden_switch(t)
        elif self.schedule == "gradual_drift":
            return self._gradual_drift(t)
        elif self.schedule == "periodic":
            return self._periodic(t)
        elif self.schedule == "random_walk":
            return self._random_walk(t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _sudden_switch(self, t: int) -> NoiseModel:
        """Abrupt switch from depolarizing to bit-flip at step 500.

        Parameters
        ----------
        t : int
            Current step.

        Returns
        -------
        NoiseModel
        """
        if t < 500:
            return DepolarizingNoise(p=np.float32(self.p_base))
        else:
            return BitFlipNoise(p=np.float32(self.p_base))

    def _gradual_drift(self, t: int) -> NoiseModel:
        """Ornstein-Uhlenbeck process for continuous drift.

        The noise parameters evolve as:
            dX = θ(μ - X)dt + σ dW

        Parameters
        ----------
        t : int
            Current step (used to seed deterministic noise per step).

        Returns
        -------
        NoiseModel
        """
        dt = np.float32(1.0)
        noise = self._rng.standard_normal(3).astype(np.float32)
        self._ou_state += (
            self._ou_theta * (self._ou_mu - self._ou_state) * dt
            + self._ou_sigma * np.sqrt(dt) * noise
        )
        # Clamp to valid range
        self._ou_state = np.clip(
            self._ou_state, np.float32(0.001), np.float32(0.08)
        )

        p_depol = float(self._ou_state[0])
        p_x_bias = float(max(0, self._ou_state[1]))
        p_z_bias = float(max(0, self._ou_state[2]))

        return CombinedNoise(
            p_bit=p_depol / 3.0 + p_x_bias,
            p_phase=p_depol / 3.0 + p_z_bias,
        )

    def _periodic(self, t: int) -> NoiseModel:
        """Cycle through noise types every 250 steps.

        Parameters
        ----------
        t : int
            Current step.

        Returns
        -------
        NoiseModel
        """
        cycle = (t // 250) % 4
        p = np.float32(self.p_base)
        if cycle == 0:
            return DepolarizingNoise(p=p)
        elif cycle == 1:
            return BitFlipNoise(p=p)
        elif cycle == 2:
            return PhaseFlipNoise(p=p)
        else:
            return CombinedNoise(p_bit=p, p_phase=p)

    def _random_walk(self, t: int) -> NoiseModel:
        """Bounded random walk of noise parameters.

        Parameters
        ----------
        t : int
            Current step.

        Returns
        -------
        NoiseModel
        """
        step = self._rng.standard_normal(3).astype(np.float32) * np.float32(0.002)
        self._rw_state += step
        self._rw_state = np.clip(
            self._rw_state, np.float32(0.001), np.float32(0.08)
        )

        return CombinedNoise(
            p_bit=float(self._rw_state[0] / 3.0 + max(0, self._rw_state[1])),
            p_phase=float(self._rw_state[0] / 3.0 + max(0, self._rw_state[2])),
        )

    def plot_drift_trajectory(
        self,
        n_steps: int = 2000,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot the noise parameter evolution over time.

        Parameters
        ----------
        n_steps : int
            Number of steps to simulate.
        save_path : str, optional
            Output figure path.

        Returns
        -------
        str
            Path to the saved figure.
        """
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, "drift_trajectory.png")

        # Reset state for clean plot
        self._ou_state = np.array(
            [self.p_base, 0.0, 0.0], dtype=np.float32
        )
        self._rw_state = np.array(
            [self.p_base, 0.0, 0.0], dtype=np.float32
        )

        p_bit_history = []
        p_phase_history = []

        for t in range(n_steps):
            noise = self.get_noise_at_step(t)
            if hasattr(noise, "p_bit"):
                p_bit_history.append(noise.p_bit)
                p_phase_history.append(noise.p_phase)
            elif hasattr(noise, "p"):
                p_bit_history.append(noise.p)
                p_phase_history.append(noise.p)
            else:
                p_bit_history.append(0.0)
                p_phase_history.append(0.0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(
            range(n_steps), p_bit_history,
            label=r"$p_{bit}$", color="#0173b2", linewidth=1.5,
        )
        ax.plot(
            range(n_steps), p_phase_history,
            label=r"$p_{phase}$", color="#de8f05", linewidth=1.5,
        )
        ax.set_xlabel("Step", fontsize=14)
        ax.set_ylabel("Error Probability", fontsize=14)
        ax.set_title(
            f"Noise Drift Trajectory ({self.schedule})",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(fontsize=12)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", alpha=0.1)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close("all")
        del fig
        gc.collect()

        logger.info("Saved drift trajectory to %s", save_path)
        return save_path


# ---------------------------------------------------------------------------
# Utility helpers
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
    """
    res_x = np.isin(residual, [1, 2]).astype(int)
    res_z = np.isin(residual, [2, 3]).astype(int)
    log_x = np.isin(lx, [1, 2]).astype(int)
    log_z = np.isin(lz, [2, 3]).astype(int)

    comm_z = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
    comm_x = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2

    return bool(comm_x or comm_z)
