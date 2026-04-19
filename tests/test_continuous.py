"""Unit tests for continuous noise estimation components.

Tests NoiseParameterEstimator, ContinuousDecoderSelector, and all
supporting classes with memory constraint verification.
"""

import gc
import os
import tempfile

import numpy as np
import pytest
import torch

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.noise_estimator import (
    EstimatorConfig,
    MCDropout,
    NoiseEstimatorNet,
    NoiseParameterEstimator,
    SyndromeDataset,
)
from src.noise_models import create_noise_model
from src.stabilizer_codes import RepetitionCode, SurfaceCode


# ---------------------------------------------------------------------------
# MCDropout tests
# ---------------------------------------------------------------------------

class TestMCDropout:
    """Tests for always-on MC dropout."""

    def test_dropout_active_in_eval(self):
        """MC dropout must remain active even in eval mode."""
        drop = MCDropout(p=0.5)
        drop.eval()
        x = torch.ones(100, 100)
        y = drop(x)
        # Some values should be zeroed out even in eval mode
        assert (y == 0).any(), "MCDropout should be active in eval mode"

    def test_dropout_active_in_train(self):
        """MC dropout must be active in train mode."""
        drop = MCDropout(p=0.5)
        drop.train()
        x = torch.ones(100, 100)
        y = drop(x)
        assert (y == 0).any(), "MCDropout should be active in train mode"


# ---------------------------------------------------------------------------
# NoiseEstimatorNet tests
# ---------------------------------------------------------------------------

class TestNoiseEstimatorNet:
    """Tests for the estimator network architecture."""

    def test_parameter_count_under_limit(self):
        """Model must have ≤50,000 parameters."""
        config = EstimatorConfig(max_params=50_000)
        net = NoiseEstimatorNet(n_stabilizers=8, config=config)
        total = sum(p.numel() for p in net.parameters())
        assert total <= 50_000, f"Model has {total} params, limit is 50,000"

    def test_parameter_count_enforcement(self):
        """Model with too many params must raise ValueError."""
        config = EstimatorConfig(
            max_params=10,  # impossibly small
            conv1_channels=32,
            conv2_channels=64,
        )
        with pytest.raises(ValueError, match="exceeding the hard limit"):
            NoiseEstimatorNet(n_stabilizers=8, config=config)

    def test_output_shape(self):
        """Forward pass must output (batch, 3)."""
        config = EstimatorConfig()
        net = NoiseEstimatorNet(n_stabilizers=8, config=config)
        x = torch.randn(4, 20, 8)
        y = net(x)
        assert y.shape == (4, 3), f"Expected (4, 3), got {y.shape}"

    def test_output_range(self):
        """Output must be in [0, p_max]."""
        config = EstimatorConfig(p_max=0.15)
        net = NoiseEstimatorNet(n_stabilizers=8, config=config)
        x = torch.randn(10, 20, 8)
        y = net(x)
        assert (y >= 0).all(), "Output has negative values"
        assert (y <= 0.15 + 1e-5).all(), "Output exceeds p_max"


# ---------------------------------------------------------------------------
# NoiseParameterEstimator tests
# ---------------------------------------------------------------------------

class TestNoiseParameterEstimator:
    """Tests for the full estimator class."""

    @pytest.fixture
    def estimator(self):
        """Create a test estimator with a small code."""
        code = RepetitionCode(d=3)
        config = EstimatorConfig(
            train_batch_size=4,
            train_epochs=2,
            max_syndromes_in_ram=50,
        )
        return NoiseParameterEstimator(code=code, config=config)

    def test_float16_weights(self, estimator):
        """Model weights must be stored in float16."""
        for param in estimator.model.parameters():
            assert param.dtype == torch.float16, (
                f"Expected float16, got {param.dtype}"
            )

    def test_generate_training_data(self, estimator):
        """Training data generation must work and save to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = estimator.generate_training_data(
                n_samples=20, save_dir=tmpdir, seed=42
            )
            assert os.path.exists(path)
            data = np.load(path)
            assert "X" in data
            assert "y" in data
            assert data["X"].shape[0] == 20
            assert data["y"].shape == (20, 3)

    def test_train_and_estimate(self, estimator):
        """Full train→estimate pipeline must work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = estimator.generate_training_data(
                n_samples=20, save_dir=tmpdir, seed=42
            )
            history = estimator.train(data_path=path, n_epochs=2)
            assert "losses" in history
            assert len(history["losses"]) == 2

            # Test estimation
            history_arr = np.random.rand(20, estimator._n_stabilizers).astype(
                np.float32
            )
            result = estimator.estimate(history_arr)
            assert "p_X" in result
            assert "p_Z" in result
            assert "p_depol" in result
            assert "uncertainty" in result
            assert result["p_X"] >= 0
            assert result["uncertainty"] >= 0

    def test_save_and_load(self, estimator):
        """Save/load round-trip must preserve model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pt")
            estimator.save(save_path)
            assert os.path.exists(save_path)

            estimator.load(save_path)
            assert estimator._trained is False  # not trained yet

    def test_batch_size_constraint(self, estimator):
        """Training batch size must never exceed 16."""
        assert estimator.config.train_batch_size <= 16


# ---------------------------------------------------------------------------
# SyndromeDataset tests
# ---------------------------------------------------------------------------

class TestSyndromeDataset:
    """Tests for the streaming dataset."""

    def test_dataset_loads(self):
        """Dataset must load without reading full file into RAM."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            X = np.random.rand(100, 20, 8).astype(np.float32)
            y = np.random.rand(100, 3).astype(np.float32)
            np.savez(path, X=X, y=y)

            ds = SyndromeDataset(path, max_in_ram=10)
            assert len(ds) == 100

            x_sample, y_sample = ds[0]
            assert x_sample.dtype == torch.float16
            assert y_sample.dtype == torch.float16

    def test_max_in_ram_parameter(self):
        """max_in_ram must be respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            X = np.random.rand(50, 10, 4).astype(np.float32)
            y = np.random.rand(50, 3).astype(np.float32)
            np.savez(path, X=X, y=y)

            ds = SyndromeDataset(path, max_in_ram=10)
            assert ds._max_in_ram == 10


# ---------------------------------------------------------------------------
# ContinuousDecoderSelector tests
# ---------------------------------------------------------------------------

class TestContinuousDecoderSelector:
    """Tests for the continuous selector."""

    def test_weight_computation(self):
        """Gaussian kernel weights must sum to 1."""
        from src.continuous_selector import ContinuousDecoderSelector
        from src.ansatz import AdaptiveAnsatz
        from src.decoder import VariationalDecoder

        code = RepetitionCode(d=3)
        config = EstimatorConfig(train_epochs=1, train_batch_size=4)
        estimator = NoiseParameterEstimator(code=code, config=config)

        # Build minimal decoder bank
        bank = {}
        for nt in ["depolarizing", "bit_flip"]:
            ansatz = AdaptiveAnsatz(n_qubits=3, noise_type=nt, n_layers=1)
            noise = create_noise_model(nt, p=0.03)
            dec = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
            bank[nt] = dec

        selector = ContinuousDecoderSelector(
            code=code,
            noise_estimator=estimator,
            decoder_bank=bank,
        )

        phi = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        weights = selector._compute_weights(phi)

        assert abs(sum(weights.values()) - 1.0) < 1e-5, "Weights must sum to 1"
        assert all(w >= 0 for w in weights.values()), "Weights must be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
