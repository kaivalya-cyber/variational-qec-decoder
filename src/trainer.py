"""Training loop for the variational QEC decoder.

Implements Adam-based optimisation with parameter-shift gradients,
early stopping, checkpoint saving, and comprehensive logging.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import torch

from .decoder import VariationalDecoder
from .noise_models import NoiseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR: str = "results/checkpoints"
DEFAULT_PATIENCE: int = 50
DEFAULT_LR: float = 0.01
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_EPOCHS: int = 500


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """Container for training metrics.

    Attributes
    ----------
    losses : list of float
        Loss value per epoch.
    logical_error_rates : list of float
        Logical error rate per epoch.
    gradient_norms : list of float
        L2 norm of the gradient per epoch.
    learning_rates : list of float
        Learning rate per epoch.
    epoch_times : list of float
        Wall-clock time per epoch (seconds).
    """

    losses: List[float] = field(default_factory=list)
    logical_error_rates: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert history to a serializable dictionary."""
        return {
            "losses": self.losses,
            "logical_error_rates": self.logical_error_rates,
            "gradient_norms": self.gradient_norms,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class Trainer:
    """Full training class for the variational decoder.

    Parameters
    ----------
    decoder : VariationalDecoder
        The variational decoder to train.
    lr : float
        Initial learning rate for Adam.
    patience : int
        Number of epochs with no improvement before early stopping.
    checkpoint_dir : str
        Directory for saving checkpoints.
    """

    decoder: VariationalDecoder
    lr: float = DEFAULT_LR
    patience: int = DEFAULT_PATIENCE
    checkpoint_dir: str = RESULTS_DIR
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        # PyTorch-based Adam optimiser acting on numpy params
        self._params_tensor = torch.tensor(
            self.decoder.params, dtype=torch.float64, requires_grad=False
        )
        self._optimizer = torch.optim.Adam(
            [self._params_tensor], lr=self.lr
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode="min", factor=0.5, patience=10
        )
        self.history = TrainingHistory()

        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "Trainer initialised: lr=%.4f, patience=%d, checkpoint_dir='%s'",
            self.lr,
            self.patience,
            self.checkpoint_dir,
        )

    def train(
        self,
        n_epochs: int = DEFAULT_N_EPOCHS,
        batch_size: Optional[int] = None,
        p_range: Tuple[float, float] = (0.01, 0.1),
        eval_shots: int = 500,
        seed: int = 42,
    ) -> TrainingHistory:
        """Train the variational decoder.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Number of error samples per batch.
        p_range : tuple of float
            (min, max) physical error rate range for sampling.
        eval_shots : int
            Number of shots for logical error rate evaluation.
        seed : int
            Base random seed.

        Returns
        -------
        TrainingHistory
            Training metrics.
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        np.random.seed(seed)
        torch.manual_seed(seed)

        best_loss = float("inf")
        patience_counter = 0
        best_params = self.decoder.params.copy()

        logger.info(
            "Starting training: %d epochs, batch_size=%d, p_range=%s",
            n_epochs,
            batch_size,
            p_range,
        )

        for epoch in range(n_epochs):
            t0 = time.time()

            # Sample noise level
            p = np.random.uniform(p_range[0], p_range[1])

            # Generate training data
            errors = self.decoder.noise_model.sample_errors(
                self.decoder.code.n_qubits, batch_size, seed=seed + epoch
            )
            syndromes = np.array(
                [self.decoder.code.extract_syndrome(e) for e in errors]
            )

            # PennyLane autograd for gradients
            def cost_fn(params):
                return self.decoder.compute_loss(syndromes, errors, params=params)
            
            # Use pennylane.grad to compute gradients
            # Ensure decoder params have requires_grad=True
            params_pnp = pnp.array(self.decoder.params, requires_grad=True)
            
            grad_fn = qml.grad(cost_fn)
            gradients = grad_fn(params_pnp)
            
            # The PennyLane grad returns a tuple if params is complex or multiple args,
            # but here self.decoder.params is a single numpy array.
            
            # Update via Optimizer
            self._params_tensor.data = torch.tensor(
                self.decoder.params, dtype=torch.float64
            )
            # gradients might be a numpy-wrapped array, convert to standard numpy then torch
            self._params_tensor.grad = torch.tensor(np.array(gradients), dtype=torch.float64)
            self._optimizer.step()
            self._optimizer.zero_grad()

            # Sync back
            self.decoder.params = self._params_tensor.data.numpy().copy()
            loss = float(cost_fn(self.decoder.params))
            grad_norm = float(np.linalg.norm(np.array(gradients)))

            # Schedule
            self._scheduler.step(loss)
            current_lr = self._optimizer.param_groups[0]["lr"]

            # Evaluate LER
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                ler = self.decoder.compute_logical_error_rate(
                    eval_shots, seed=seed + epoch + 10000
                )
            else:
                ler = self.history.logical_error_rates[-1] if self.history.logical_error_rates else float("nan")

            elapsed = time.time() - t0
            self.history.losses.append(loss)
            self.history.logical_error_rates.append(ler)
            self.history.gradient_norms.append(grad_norm)
            self.history.learning_rates.append(current_lr)
            self.history.epoch_times.append(elapsed)

            logger.info(
                "Epoch %3d/%d | loss=%.6f | LER=%.6f | |∇|=%.4f | lr=%.6f | %.2fs",
                epoch + 1, n_epochs, float(loss), ler, grad_norm, current_lr, elapsed
            )

            # Early stopping check
            if loss < best_loss:
                best_loss = loss
                best_params = self.decoder.params.copy()
                patience_counter = 0
                self._save_checkpoint("best", epoch, loss, ler)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch + 1,
                    self.patience,
                )
                break

            # Periodic checkpoint
            if (epoch + 1) % 20 == 0:
                self._save_checkpoint("periodic", epoch, loss, ler)

        # Restore best parameters
        self.decoder.params = best_params
        self._save_checkpoint("final", n_epochs - 1, best_loss, ler)
        self._save_history()

        logger.info("Training complete. Best loss: %.6f", best_loss)
        return self.history

    def _save_checkpoint(
        self, tag: str, epoch: int, loss: float, ler: float
    ) -> None:
        """Save a checkpoint.

        Parameters
        ----------
        tag : str
            Checkpoint identifier (e.g. 'best', 'periodic', 'final').
        epoch : int
            Current epoch number.
        loss : float
            Current loss.
        ler : float
            Current logical error rate.
        """
        path = Path(self.checkpoint_dir) / f"checkpoint_{tag}.npz"
        np.savez(
            path,
            params=self.decoder.params,
            epoch=epoch,
            loss=loss,
            logical_error_rate=ler,
        )
        logger.debug("Saved checkpoint: %s", path)

    def _save_history(self) -> None:
        """Save training history to JSON."""
        path = Path(self.checkpoint_dir) / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history.to_dict(), f, indent=2)
        logger.debug("Saved training history: %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load parameters from a checkpoint.

        Parameters
        ----------
        path : str
            Path to a ``.npz`` checkpoint file.
        """
        data = np.load(path)
        self.decoder.params = data["params"]
        logger.info(
            "Loaded checkpoint from %s (epoch=%d, loss=%.6f)",
            path,
            int(data["epoch"]),
            float(data["loss"]),
        )
