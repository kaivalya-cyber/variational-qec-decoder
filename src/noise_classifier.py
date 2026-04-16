"""Noise classifier — NOVEL RESEARCH COMPONENT 1.

A lightweight classical CNN that identifies the noise type from syndrome
measurement histories.  This is original work; no published paper has
combined noise classification with ansatz selection for QEC decoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOISE_LABELS: List[str] = ["depolarizing", "bit_flip", "phase_flip", "combined"]
DEFAULT_TIME_STEPS: int = 10
DEFAULT_TRAIN_EPOCHS: int = 50
DEFAULT_TRAIN_LR: float = 0.001
DEFAULT_TRAIN_BATCH: int = 64


# ---------------------------------------------------------------------------
# CNN Architecture
# ---------------------------------------------------------------------------

class NoiseClassifierNet(nn.Module):
    """Lightweight 1-D CNN for noise-type classification.

    Architecture:
        Conv1d(1, 16, 3) → ReLU → Conv1d(16, 32, 3) → GlobalAvgPool → Linear(32, 4)

    Parameters
    ----------
    n_stabilizers : int
        Number of stabiliser measurements (spatial dimension).
    n_classes : int
        Number of noise types to classify.
    """

    def __init__(self, n_stabilizers: int, n_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, time_steps, n_stabilizers)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_classes)``.
        """
        # Average over time steps to get (batch, n_stabilizers)
        x = x.mean(dim=1)  # (batch, n_stabilizers)
        x = x.unsqueeze(1)  # (batch, 1, n_stabilizers) for Conv1d

        x = F.relu(self.conv1(x))  # (batch, 16, n_stabilizers)
        x = F.relu(self.conv2(x))  # (batch, 32, n_stabilizers)

        # Global average pooling over spatial dimension
        x = x.mean(dim=2)  # (batch, 32)

        logits = self.fc(x)  # (batch, n_classes)
        return logits


# ---------------------------------------------------------------------------
# Noise Classifier wrapper
# ---------------------------------------------------------------------------

@dataclass
class NoiseClassifier:
    """Classifies noise type from syndrome measurement history.

    Parameters
    ----------
    n_qubits : int
        Number of data qubits in the code.
    code : StabilizerCode, optional
        The QEC code (needed for syndrome extraction during training).
    time_steps : int
        Number of consecutive syndrome measurements.
    """

    n_qubits: int
    code: Optional[StabilizerCode] = None
    time_steps: int = DEFAULT_TIME_STEPS

    def __post_init__(self) -> None:
        if self.code is not None:
            self._n_stabilizers = self.code.n_stabilizers
        else:
            self._n_stabilizers = self.n_qubits - 1  # fallback heuristic

        self.model = NoiseClassifierNet(
            n_stabilizers=self._n_stabilizers, n_classes=len(NOISE_LABELS)
        )
        self._trained = False
        logger.info(
            "NoiseClassifier initialised: n_qubits=%d, n_stabilizers=%d, "
            "time_steps=%d",
            self.n_qubits,
            self._n_stabilizers,
            self.time_steps,
        )

    def generate_training_data(
        self,
        n_samples_per_class: int = 500,
        p: float = 0.05,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic syndrome training data.

        Parameters
        ----------
        n_samples_per_class : int
            Number of syndrome histories per noise type.
        p : float
            Physical error rate for data generation.
        seed : int
            Random seed.

        Returns
        -------
        tuple of torch.Tensor
            ``(X, y)`` where X has shape
            ``(n_total, time_steps, n_stabilizers)`` and y has shape
            ``(n_total,)``.
        """
        if self.code is None:
            raise RuntimeError("Code must be set to generate training data")

        np.random.seed(seed)
        all_x: List[np.ndarray] = []
        all_y: List[int] = []

        for label_idx, noise_type in enumerate(NOISE_LABELS):
            if noise_type == "combined":
                noise = create_noise_model(noise_type, p_bit=p, p_phase=p)
            else:
                noise = create_noise_model(noise_type, p=p)

            for _ in range(n_samples_per_class):
                # Generate a syndrome history (multiple rounds)
                history = np.zeros(
                    (self.time_steps, self._n_stabilizers), dtype=float
                )
                for t in range(self.time_steps):
                    errors = noise.sample_errors(self.code.n_qubits, 1)
                    syndrome = self.code.extract_syndrome(errors[0])
                    history[t] = syndrome

                all_x.append(history)
                all_y.append(label_idx)

        X = torch.tensor(np.array(all_x), dtype=torch.float32)
        y = torch.tensor(np.array(all_y), dtype=torch.long)

        # Shuffle
        perm = torch.randperm(len(y))
        X = X[perm]
        y = y[perm]

        logger.info("Generated training data: X.shape=%s, y.shape=%s", X.shape, y.shape)
        return X, y

    def train_classifier(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int = DEFAULT_TRAIN_EPOCHS,
        batch_size: int = DEFAULT_TRAIN_BATCH,
        lr: float = DEFAULT_TRAIN_LR,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        """Train the noise classifier.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(n_samples, time_steps, n_stabilizers)``.
        y : torch.Tensor
            Shape ``(n_samples,)`` integer class labels.
        n_epochs : int
            Training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        seed : int
            Random seed.

        Returns
        -------
        dict
            ``{'losses': list, 'accuracies': list}``.
        """
        torch.manual_seed(seed)

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        history: Dict[str, List[float]] = {"losses": [], "accuracies": []}

        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

            avg_loss = epoch_loss / total
            accuracy = correct / total
            history["losses"].append(avg_loss)
            history["accuracies"].append(accuracy)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Classifier epoch %d/%d: loss=%.4f, accuracy=%.4f",
                    epoch + 1,
                    n_epochs,
                    avg_loss,
                    accuracy,
                )

        self._trained = True
        logger.info("Classifier training complete. Final accuracy: %.4f", accuracy)
        return history

    def classify(
        self, syndrome_history: np.ndarray
    ) -> Tuple[str, float]:
        """Classify the noise type from a syndrome history.

        Parameters
        ----------
        syndrome_history : np.ndarray
            Shape ``(time_steps, n_stabilizers)`` or
            ``(batch, time_steps, n_stabilizers)``.

        Returns
        -------
        tuple
            ``(noise_type_str, confidence)`` where confidence is the
            softmax probability of the predicted class.
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(syndrome_history, dtype=torch.float32)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dim

            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)

            noise_type = NOISE_LABELS[pred_idx[0].item()]
            conf = confidence[0].item()

        logger.debug("Classified noise: %s (confidence=%.4f)", noise_type, conf)
        return noise_type, conf

    def save(self, path: str) -> None:
        """Save the trained model.

        Parameters
        ----------
        path : str
            Output file path.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "n_stabilizers": self._n_stabilizers,
                "trained": self._trained,
            },
            path,
        )
        logger.info("Saved noise classifier to %s", path)

    def load(self, path: str) -> None:
        """Load a trained model.

        Parameters
        ----------
        path : str
            Path to saved model.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._trained = checkpoint.get("trained", True)
        logger.info("Loaded noise classifier from %s", path)
