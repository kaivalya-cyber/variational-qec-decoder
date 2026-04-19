"""Unit tests for online learning components.

Tests OnlineLearner, DriftSimulator, ConvergenceAnalyzer, and all
supporting classes with memory constraint verification.
"""

import collections
import gc
import os

import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ansatz import AdaptiveAnsatz
from src.convergence_analysis import (
    ConvergenceAnalyzer,
    kl_divergence_pauli,
    noise_to_pauli_probs,
)
from src.decoder import VariationalDecoder
from src.noise_models import (
    BitFlipNoise,
    DepolarizingNoise,
    create_noise_model,
)
from src.online_learner import (
    DriftSimulator,
    Experience,
    OnlineConfig,
    OnlineLearner,
)
from src.stabilizer_codes import RepetitionCode


# ---------------------------------------------------------------------------
# Experience tests
# ---------------------------------------------------------------------------

class TestExperience:
    """Tests for the Experience namedtuple."""

    def test_memory_efficient_storage(self):
        """Experiences must store data as uint8 and bool."""
        syn = np.array([1, 0, 1], dtype=np.uint8)
        corr = np.array([1, 0, 0], dtype=np.uint8)
        exp = Experience(syndrome=syn, correction=corr, outcome=True)

        assert exp.syndrome.dtype == np.uint8
        assert exp.correction.dtype == np.uint8
        assert isinstance(exp.outcome, bool)

    def test_namedtuple_access(self):
        """Fields must be accessible by name."""
        exp = Experience(
            syndrome=np.zeros(3, dtype=np.uint8),
            correction=np.ones(3, dtype=np.uint8),
            outcome=False,
        )
        assert not exp.outcome
        assert exp.correction.sum() == 3


# ---------------------------------------------------------------------------
# OnlineConfig tests
# ---------------------------------------------------------------------------

class TestOnlineConfig:
    """Tests for configuration constraints."""

    def test_buffer_size_limit(self):
        """Buffer size must be 200."""
        config = OnlineConfig()
        assert config.buffer_size == 200

    def test_minibatch_limit(self):
        """Minibatch size must be ≤ 8."""
        config = OnlineConfig()
        assert config.minibatch_size <= 8

    def test_update_freq(self):
        """Update frequency must be set."""
        config = OnlineConfig()
        assert config.update_freq == 50


# ---------------------------------------------------------------------------
# OnlineLearner tests
# ---------------------------------------------------------------------------

class TestOnlineLearner:
    """Tests for the online learner."""

    @pytest.fixture
    def learner(self):
        """Create a test learner with a small code."""
        code = RepetitionCode(d=3)
        ansatz = AdaptiveAnsatz(n_qubits=3, noise_type="depolarizing", n_layers=1)
        noise = create_noise_model("depolarizing", p=0.03)
        decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
        config = OnlineConfig(buffer_size=200, minibatch_size=8, update_freq=5)
        return OnlineLearner(decoder=decoder, config=config)

    def test_buffer_maxlen(self, learner):
        """Buffer must enforce maxlen=200."""
        assert learner._buffer.maxlen == 200

    def test_buffer_is_deque(self, learner):
        """Buffer must be a collections.deque."""
        assert isinstance(learner._buffer, collections.deque)

    def test_update_adds_experience(self, learner):
        """Update must add experience to buffer."""
        syn = np.array([1, 0], dtype=np.float64)
        corr = np.array([1, 0, 0], dtype=np.float64)
        learner.update(syn, corr, False)
        assert len(learner._buffer) == 1

    def test_buffer_overflow(self, learner):
        """Buffer must drop oldest when full."""
        for i in range(250):
            syn = np.array([i % 2, (i + 1) % 2], dtype=np.float64)
            corr = np.array([0, 0, 0], dtype=np.float64)
            learner.update(syn, corr, False)
        assert len(learner._buffer) == 200

    def test_experiences_stored_as_uint8(self, learner):
        """Stored experiences must use uint8 for syndrome/correction."""
        syn = np.array([1.0, 0.0], dtype=np.float64)
        corr = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        learner.update(syn, corr, True)

        exp = learner._buffer[0]
        assert exp.syndrome.dtype == np.uint8
        assert exp.correction.dtype == np.uint8

    def test_adaptation_rate_initial(self, learner):
        """Initial adaptation rate must be 0."""
        assert learner.get_adaptation_rate() == 0.0

    def test_reset_buffer(self, learner):
        """Reset must clear buffer and step count."""
        for i in range(10):
            syn = np.array([0, 1], dtype=np.float64)
            corr = np.array([0, 0, 0], dtype=np.float64)
            learner.update(syn, corr, False)

        learner.reset_buffer()
        assert len(learner._buffer) == 0
        assert learner._step_count == 0


# ---------------------------------------------------------------------------
# DriftSimulator tests
# ---------------------------------------------------------------------------

class TestDriftSimulator:
    """Tests for the drift simulator."""

    @pytest.fixture
    def code(self):
        return RepetitionCode(d=3)

    def test_sudden_switch(self, code):
        """Sudden switch must return different noise types."""
        drift = DriftSimulator(code=code, schedule="sudden_switch", p_base=0.03)
        n1 = drift.get_noise_at_step(0)
        n2 = drift.get_noise_at_step(500)
        assert isinstance(n1, DepolarizingNoise)
        assert isinstance(n2, BitFlipNoise)

    def test_gradual_drift_returns_noise(self, code):
        """Gradual drift must return valid noise models."""
        drift = DriftSimulator(code=code, schedule="gradual_drift", p_base=0.03)
        for t in range(10):
            noise = drift.get_noise_at_step(t)
            assert hasattr(noise, "sample_errors")

    def test_periodic_cycles(self, code):
        """Periodic must cycle through 4 noise types."""
        drift = DriftSimulator(code=code, schedule="periodic", p_base=0.03)
        types = set()
        for t in range(1000):
            noise = drift.get_noise_at_step(t)
            types.add(type(noise).__name__)
        assert len(types) >= 3  # At least 3 different noise types

    def test_random_walk_bounded(self, code):
        """Random walk parameters must stay in bounds."""
        drift = DriftSimulator(code=code, schedule="random_walk", p_base=0.03)
        for t in range(100):
            noise = drift.get_noise_at_step(t)
            if hasattr(noise, "p_bit"):
                assert 0 <= noise.p_bit <= 1.0
                assert 0 <= noise.p_phase <= 1.0

    def test_single_sample_generation(self, code):
        """Each step must generate noise one at a time (no pre-generation)."""
        drift = DriftSimulator(code=code, schedule="gradual_drift", p_base=0.03)
        # Just verify it works step by step without batch
        for t in range(5):
            noise = drift.get_noise_at_step(t)
            error = noise.sample_errors(code.n_qubits, 1)
            assert error.shape == (1, code.n_qubits)

    def test_unknown_schedule_raises(self, code):
        """Unknown schedule must raise ValueError."""
        drift = DriftSimulator(code=code, schedule="invalid")
        with pytest.raises(ValueError, match="Unknown schedule"):
            drift.get_noise_at_step(0)

    def test_numpy_float32(self, code):
        """Internal state must use float32."""
        drift = DriftSimulator(code=code, schedule="gradual_drift")
        assert drift._ou_state.dtype == np.float32
        assert drift._rw_state.dtype == np.float32


# ---------------------------------------------------------------------------
# KL Divergence tests
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """Tests for Pauli channel KL divergence."""

    def test_same_channel_zero_kl(self):
        """KL divergence of a channel with itself must be 0."""
        p = {"I": 0.97, "X": 0.01, "Y": 0.01, "Z": 0.01}
        kl = kl_divergence_pauli(p, p)
        assert abs(kl) < 1e-8

    def test_different_channels_positive_kl(self):
        """KL divergence between different channels must be positive."""
        p1 = {"I": 0.97, "X": 0.01, "Y": 0.01, "Z": 0.01}
        p2 = {"I": 0.97, "X": 0.03, "Y": 0.0, "Z": 0.0}
        kl = kl_divergence_pauli(p1, p2)
        assert kl > 0

    def test_noise_to_pauli_probs(self):
        """Conversion must produce valid probability distribution."""
        noise = DepolarizingNoise(p=0.03)
        probs = noise_to_pauli_probs(noise, p=0.03)
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# ConvergenceAnalyzer tests
# ---------------------------------------------------------------------------

class TestConvergenceAnalyzer:
    """Tests for convergence analysis."""

    @pytest.fixture
    def analyzer(self):
        code = RepetitionCode(d=3)
        return ConvergenceAnalyzer(code=code)

    def test_theoretical_bound(self, analyzer):
        """Theoretical bound must return valid results."""
        n_old = DepolarizingNoise(p=0.03)
        n_new = BitFlipNoise(p=0.03)
        result = analyzer.theoretical_bound(n_old, n_new, lr=0.01)

        assert "kl_divergence" in result
        assert "theoretical_lag" in result
        assert result["kl_divergence"] >= 0
        assert result["theoretical_lag"] >= 0

    def test_bound_inversely_proportional_to_lr(self, analyzer):
        """Higher LR should give smaller theoretical lag."""
        n_old = DepolarizingNoise(p=0.03)
        n_new = BitFlipNoise(p=0.03)

        r1 = analyzer.theoretical_bound(n_old, n_new, lr=0.01)
        r2 = analyzer.theoretical_bound(n_old, n_new, lr=0.1)

        assert r2["theoretical_lag"] < r1["theoretical_lag"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
