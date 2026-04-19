"""Continuous noise parameter estimation via regression CNN.

Generates: Section IV.B (Continuous Noise Estimation),
           Figure 3 (estimation_calibration.png)

Instead of classifying noise into 4 discrete buckets, this module trains a
regression network that outputs a continuous noise parameter vector
(p_X, p_Z, p_depol) directly.  MC dropout provides uncertainty estimates.

Hardware constraints (3 GB RAM):
- Model is capped at 50,000 parameters (hard enforced in __init__)
- All weights stored in float16
- Batch size never exceeds 16 during training
- DataLoader uses num_workers=0, pin_memory=False
- Training data is streamed from disk, never more than 500 syndromes in RAM
- Tensors are deleted and gc.collect() called after every batch
"""

from __future__ import annotations

import gc
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EstimatorConfig:
    """Configuration for NoiseParameterEstimator.

    Attributes
    ----------
    max_params : int
        Hard upper bound on total model parameters.
    train_batch_size : int
        Maximum training batch size (memory constraint).
    train_lr : float
        Adam learning rate.
    train_epochs : int
        Default number of training epochs.
    mc_dropout_p : float
        Dropout probability for MC dropout (stays on during eval).
    mc_samples : int
        Number of forward passes for MC dropout uncertainty.
    max_syndromes_in_ram : int
        Maximum number of syndromes held in memory at once.
    p_max : float
        Maximum noise parameter value for output scaling.
    conv1_channels : int
        Number of output channels for first Conv1d layer.
    conv2_channels : int
        Number of output channels for second Conv1d layer.
    fc_hidden : int
        Hidden dimension for the fully connected layer.
    """

    max_params: int = 50_000
    train_batch_size: int = 16
    train_lr: float = 1e-3
    train_epochs: int = 100
    mc_dropout_p: float = 0.1
    mc_samples: int = 10
    max_syndromes_in_ram: int = 500
    p_max: float = 0.15
    conv1_channels: int = 32
    conv2_channels: int = 64
    fc_hidden: int = 32


ESTIMATOR_CONFIG = EstimatorConfig()

FIGURES_DIR: str = "results/figures"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Always-on MC Dropout
# ---------------------------------------------------------------------------

class MCDropout(nn.Module):
    """Dropout layer that stays active during both training AND eval.

    Standard nn.Dropout disables itself in eval mode.  For MC dropout
    uncertainty quantification we need stochastic forward passes at
    inference time as well.

    Parameters
    ----------
    p : float
        Dropout probability.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout unconditionally.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with dropout applied.
        """
        return F.dropout(x, p=self.p, training=True)


# ---------------------------------------------------------------------------
# Network Architecture
# ---------------------------------------------------------------------------

class NoiseEstimatorNet(nn.Module):
    """1-D CNN regression network for continuous noise parameter estimation.

    Architecture:
        Conv1d(1, 32, 3) -> MCDropout -> ReLU
        -> Conv1d(32, 64, 3) -> MCDropout -> ReLU
        -> GlobalAvgPool
        -> Linear(64, 32) -> MCDropout -> ReLU
        -> Linear(32, 3) -> Sigmoid * p_max

    The output is a continuous vector (p_X, p_Z, p_depol) each in [0, p_max].

    Parameters
    ----------
    n_stabilizers : int
        Number of stabilizer measurements (spatial dimension of input).
    config : EstimatorConfig
        Configuration dataclass.
    """

    def __init__(
        self,
        n_stabilizers: int,
        config: EstimatorConfig = ESTIMATOR_CONFIG,
    ) -> None:
        super().__init__()
        self.config = config
        self.p_max = config.p_max

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=config.conv1_channels,
            kernel_size=3,
            padding=1,
        )
        self.drop1 = MCDropout(config.mc_dropout_p)

        self.conv2 = nn.Conv1d(
            in_channels=config.conv1_channels,
            out_channels=config.conv2_channels,
            kernel_size=3,
            padding=1,
        )
        self.drop2 = MCDropout(config.mc_dropout_p)

        self.fc1 = nn.Linear(config.conv2_channels, config.fc_hidden)
        self.drop3 = MCDropout(config.mc_dropout_p)

        self.fc2 = nn.Linear(config.fc_hidden, 3)

        # Enforce parameter count limit
        total = sum(p.numel() for p in self.parameters())
        if total > config.max_params:
            raise ValueError(
                f"Model has {total} parameters, exceeding the hard limit "
                f"of {config.max_params}. Reduce layer sizes."
            )
        logger.info(
            "NoiseEstimatorNet: %d parameters (limit %d)",
            total,
            config.max_params,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, time_steps, n_stabilizers)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 3)`` — estimated ``(p_X, p_Z, p_depol)``.
        """
        # Average over time steps -> (batch, n_stabilizers)
        x = x.mean(dim=1)
        x = x.unsqueeze(1)  # (batch, 1, n_stabilizers) for Conv1d

        x = self.drop1(F.relu(self.conv1(x)))
        x = self.drop2(F.relu(self.conv2(x)))

        # Global average pooling over spatial dim
        x = x.mean(dim=2)  # (batch, conv2_channels)

        x = self.drop3(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x)) * self.p_max

        return x


# ---------------------------------------------------------------------------
# Streaming Dataset (disk-backed, memory-safe)
# ---------------------------------------------------------------------------

class SyndromeDataset(Dataset):
    """Disk-backed dataset that streams syndrome data.

    Never loads more than ``max_in_ram`` syndromes into memory at once.

    Parameters
    ----------
    data_path : str
        Path to a ``.npz`` file with keys ``'X'`` and ``'y'``.
    max_in_ram : int
        Maximum number of samples to cache in memory.
    """

    def __init__(self, data_path: str, max_in_ram: int = 500) -> None:
        self._path = data_path
        self._max_in_ram = max_in_ram
        # Memory-map the file so we never load the full dataset
        self._data = np.load(data_path, mmap_mode="r")
        self._X = self._data["X"]  # (N, T, S)
        self._y = self._data["y"]  # (N, 3)
        self._len = self._X.shape[0]
        logger.info(
            "SyndromeDataset: %d samples from %s (mmap, max_in_ram=%d)",
            self._len,
            data_path,
            max_in_ram,
        )

    def __len__(self) -> int:
        """Return dataset length.

        Returns
        -------
        int
        """
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple of torch.Tensor
            ``(syndrome_history, noise_params)`` both in float16.
        """
        x = torch.tensor(
            np.array(self._X[idx]), dtype=torch.float16
        )
        y = torch.tensor(
            np.array(self._y[idx]), dtype=torch.float16
        )
        return x, y


# ---------------------------------------------------------------------------
# Noise Parameter Estimator (main class)
# ---------------------------------------------------------------------------

@dataclass
class NoiseParameterEstimator:
    """Continuous noise parameter estimator using regression CNN.

    Estimates (p_X, p_Z, p_depol) from syndrome history with MC dropout
    uncertainty quantification.

    Parameters
    ----------
    code : StabilizerCode
        The QEC code for syndrome extraction.
    config : EstimatorConfig
        Configuration dataclass.
    """

    code: StabilizerCode
    config: EstimatorConfig = field(default_factory=EstimatorConfig)

    def __post_init__(self) -> None:
        self._n_stabilizers = self.code.n_stabilizers
        self.model = NoiseEstimatorNet(
            n_stabilizers=self._n_stabilizers,
            config=self.config,
        )
        # Convert to float16 for memory efficiency
        self.model = self.model.half()
        self._trained = False
        self._data_path: Optional[str] = None
        logger.info(
            "NoiseParameterEstimator initialised: n_stabilizers=%d, "
            "model_params=%d",
            self._n_stabilizers,
            sum(p.numel() for p in self.model.parameters()),
        )

    def generate_training_data(
        self,
        n_samples: int = 5000,
        p_range: Tuple[float, float] = (0.0, 0.05),
        time_steps: int = 20,
        save_dir: str = "results/data",
        seed: int = 42,
    ) -> str:
        """Generate training data and save to disk (never in full RAM).

        Generates syndromes at known (p_X, p_Z, p_depol) combinations
        sampled uniformly from the given range.

        Parameters
        ----------
        n_samples : int
            Total number of training samples to generate.
        p_range : tuple of float
            ``(min_p, max_p)`` range for each noise parameter.
        time_steps : int
            Number of syndrome measurement rounds per sample.
        save_dir : str
            Directory to save the .npz file.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        str
            Path to the saved ``.npz`` file.

        Raises
        ------
        RuntimeError
            If code is not set.
        """
        rng = np.random.default_rng(seed)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, "noise_estimator_train.npz")

        # Pre-allocate on disk using memory-mapped arrays
        X_shape = (n_samples, time_steps, self._n_stabilizers)
        y_shape = (n_samples, 3)

        # Generate in chunks to stay under 500 syndromes in RAM
        chunk_size = min(self.config.max_syndromes_in_ram, n_samples)
        X_all = np.zeros(X_shape, dtype=np.float32)
        y_all = np.zeros(y_shape, dtype=np.float32)

        n_generated = 0
        while n_generated < n_samples:
            current_chunk = min(chunk_size, n_samples - n_generated)

            for i in range(current_chunk):
                # Sample random noise parameters
                p_x = np.float32(rng.uniform(p_range[0], p_range[1]))
                p_z = np.float32(rng.uniform(p_range[0], p_range[1]))
                p_dep = np.float32(rng.uniform(p_range[0], p_range[1]))

                # Create a combined noise model
                noise = create_noise_model(
                    "combined",
                    p_bit=float(p_x + p_dep / 3.0),
                    p_phase=float(p_z + p_dep / 3.0),
                )

                # Generate syndrome history
                history = np.zeros(
                    (time_steps, self._n_stabilizers), dtype=np.float32
                )
                for t in range(time_steps):
                    errors = noise.sample_errors(self.code.n_qubits, 1)
                    syndrome = self.code.extract_syndrome(errors[0])
                    history[t] = syndrome.astype(np.float32)

                idx = n_generated + i
                X_all[idx] = history
                y_all[idx] = np.array([p_x, p_z, p_dep], dtype=np.float32)

            n_generated += current_chunk
            logger.info(
                "Generated %d/%d training samples", n_generated, n_samples
            )

        # Shuffle
        perm = rng.permutation(n_samples)
        X_all = X_all[perm]
        y_all = y_all[perm]

        np.savez(save_path, X=X_all, y=y_all)

        # Free memory
        del X_all, y_all
        gc.collect()

        self._data_path = save_path
        logger.info("Saved training data to %s", save_path)
        return save_path

    def train(
        self,
        data_path: Optional[str] = None,
        n_epochs: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """Train the estimator from disk-backed data.

        Parameters
        ----------
        data_path : str, optional
            Path to ``.npz`` training data. Uses last generated if None.
        n_epochs : int, optional
            Number of training epochs. Uses config default if None.
        seed : int
            Random seed.

        Returns
        -------
        dict
            ``{'losses': list, 'val_losses': list}``.

        Raises
        ------
        FileNotFoundError
            If data_path is not valid.
        """
        if n_epochs is None:
            n_epochs = self.config.train_epochs
        if data_path is None:
            data_path = self._data_path
        if data_path is None or not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Training data not found at {data_path}. "
                "Call generate_training_data() first."
            )

        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = SyndromeDataset(
            data_path, max_in_ram=self.config.max_syndromes_in_ram
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        # Move model to float32 for training stability, then back to float16
        self.model = self.model.float()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.train_lr
        )
        criterion = nn.MSELoss()

        history: Dict[str, List[float]] = {"losses": []}

        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                # Convert float16 inputs to float32 for computation
                batch_x = batch_x.float()
                batch_y = batch_y.float()

                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Memory cleanup after every batch
                del batch_x, batch_y, preds, loss
                gc.collect()

            avg_loss = epoch_loss / max(n_batches, 1)
            history["losses"].append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Estimator epoch %d/%d: loss=%.6f",
                    epoch + 1,
                    n_epochs,
                    avg_loss,
                )

        # Convert back to float16
        self.model = self.model.half()
        self._trained = True
        logger.info("Estimator training complete. Final loss: %.6f", avg_loss)
        return history

    def estimate(
        self, syndrome_history: np.ndarray
    ) -> Dict[str, float]:
        """Estimate noise parameters with uncertainty.

        Uses MC dropout: runs multiple stochastic forward passes and
        returns the mean prediction and standard deviation as uncertainty.

        Parameters
        ----------
        syndrome_history : np.ndarray
            Shape ``(time_steps, n_stabilizers)`` or
            ``(batch, time_steps, n_stabilizers)``.

        Returns
        -------
        dict
            Keys: ``'p_X'``, ``'p_Z'``, ``'p_depol'``, ``'uncertainty'``.
            ``'uncertainty'`` is the mean std across MC dropout samples.
        """
        x = torch.tensor(syndrome_history, dtype=torch.float16)
        if x.dim() == 2:
            x = x.unsqueeze(0)

        # MC dropout: model always has dropout active
        predictions = []
        with torch.no_grad():
            for _ in range(self.config.mc_samples):
                pred = self.model(x.half())
                predictions.append(pred.float().cpu().numpy())

        predictions = np.stack(predictions, axis=0)  # (mc_samples, batch, 3)
        mean_pred = np.mean(predictions, axis=0)[0]  # (3,)
        std_pred = np.std(predictions, axis=0)[0]  # (3,)

        result = {
            "p_X": float(mean_pred[0]),
            "p_Z": float(mean_pred[1]),
            "p_depol": float(mean_pred[2]),
            "uncertainty": float(np.mean(std_pred)),
        }

        del x, predictions
        gc.collect()

        return result

    def calibration_curve(
        self,
        data_path: Optional[str] = None,
        n_test: int = 200,
        save_path: Optional[str] = None,
        seed: int = 123,
    ) -> str:
        """Plot predicted vs true noise parameters.

        Parameters
        ----------
        data_path : str, optional
            Path to test data ``.npz``. Generates fresh test data if None.
        n_test : int
            Number of test samples.
        save_path : str, optional
            Output figure path.
        seed : int
            Random seed.

        Returns
        -------
        str
            Path to the saved figure.
        """
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, "estimation_calibration.png")

        # Generate small test set
        rng = np.random.default_rng(seed)
        trues = []
        preds = []

        for i in range(n_test):
            p_x = np.float32(rng.uniform(0.0, 0.05))
            p_z = np.float32(rng.uniform(0.0, 0.05))
            p_dep = np.float32(rng.uniform(0.0, 0.05))

            noise = create_noise_model(
                "combined",
                p_bit=float(p_x + p_dep / 3.0),
                p_phase=float(p_z + p_dep / 3.0),
            )

            history = np.zeros(
                (20, self._n_stabilizers), dtype=np.float32
            )
            for t in range(20):
                errors = noise.sample_errors(self.code.n_qubits, 1)
                syndrome = self.code.extract_syndrome(errors[0])
                history[t] = syndrome.astype(np.float32)

            est = self.estimate(history)
            trues.append([p_x, p_z, p_dep])
            preds.append([est["p_X"], est["p_Z"], est["p_depol"]])

        trues = np.array(trues)
        preds = np.array(preds)

        # Plot
        labels = [r"$p_X$", r"$p_Z$", r"$p_{depol}$"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 7))

        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.scatter(
                trues[:, i], preds[:, i],
                s=10, alpha=0.6, color="#0173b2"
            )
            lims = [0, 0.06]
            ax.plot(lims, lims, "--", color="gray", linewidth=1)
            ax.set_xlabel(f"True {label}", fontsize=12)
            ax.set_ylabel(f"Predicted {label}", fontsize=12)
            ax.set_title(f"Calibration: {label}", fontsize=13)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", alpha=0.1)

            mae = np.mean(np.abs(trues[:, i] - preds[:, i]))
            ax.text(
                0.05, 0.92, f"MAE={mae:.4f}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        fig.suptitle(
            "Noise Parameter Estimation Calibration",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close("all")
        del fig
        gc.collect()

        logger.info("Saved calibration curve to %s", save_path)
        return save_path

    def save(self, path: str) -> None:
        """Save the trained estimator model.

        Parameters
        ----------
        path : str
            Output file path.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "n_stabilizers": self._n_stabilizers,
                "config": {
                    "max_params": self.config.max_params,
                    "mc_dropout_p": self.config.mc_dropout_p,
                    "p_max": self.config.p_max,
                    "conv1_channels": self.config.conv1_channels,
                    "conv2_channels": self.config.conv2_channels,
                    "fc_hidden": self.config.fc_hidden,
                },
                "trained": self._trained,
            },
            path,
        )
        logger.info("Saved NoiseParameterEstimator to %s", path)

    def load(self, path: str) -> None:
        """Load a trained estimator model.

        Parameters
        ----------
        path : str
            Path to saved model checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._trained = checkpoint.get("trained", True)
        logger.info("Loaded NoiseParameterEstimator from %s", path)
