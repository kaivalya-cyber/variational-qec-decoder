"""Hardware Noise Fingerprinting for personalized QEC decoding.

Generates: Section VII.A (Noise Fingerprinting), 
           Figure 12 (fingerprint_heatmap.png), Figure 13 (temporal_autocorr.png)

Novel Contribution: A systematic framework to extract device-specific noise 
biases from raw syndrome streams to personalize the decoding ansatz.

Learns spatial and temporal correlations that are unique to specific QPUs.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

class HardwareNoiseFingerprinter:
    """Learns device-specific noise characteristics from syndrome data.

    Tracks per-stabilizer flip rates and temporal correlations to build 
    a unique 'fingerprint' of the quantum hardware's noise floor.

    Parameters
    ----------
    n_qubits : int
        Number of physical data qubits.
    n_stabilizers : int
        Number of stabilizer measurements.
    """

    def __init__(self, n_qubits: int, n_stabilizers: int) -> None:
        self.n_qubits = n_qubits
        self.n_stabilizers = n_stabilizers
        
        # Internal state
        self.counts = 0
        self.stabilizer_sum = np.zeros(n_stabilizers, dtype=np.float64)
        self.history = [] # For temporal autocorrelation
        self.max_history = 500
        
        # Spatial noise map (estimated from stabilizers)
        self.qubit_error_rates = np.zeros(n_qubits, dtype=np.float64)
        
        logger.info(
            "HardwareNoiseFingerprinter initialised for %d qubits and %d stabilizers",
            n_qubits, n_stabilizers
        )

    def update(self, syndrome_batch: np.ndarray) -> None:
        """Update the fingerprint with a new batch of syndromes.

        Parameters
        ----------
        syndrome_batch : np.ndarray
            Shape (batch_size, n_stabilizers).
        """
        batch_size = syndrome_batch.shape[0]
        self.counts += batch_size
        self.stabilizer_sum += np.sum(syndrome_batch, axis=0)
        
        # Append to history for temporal analysis
        for syn in syndrome_batch:
            self.history.append(syn)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def get_fingerprint(self) -> Dict[str, np.ndarray]:
        """Compute and return the current noise fingerprint."""
        if self.counts == 0:
            return {"flip_rates": np.zeros(self.n_stabilizers)}
            
        flip_rates = self.stabilizer_sum / self.counts
        
        # Simple spatial correlation (covariance of stabilizers)
        history_arr = np.array(self.history)
        if len(history_arr) > 1:
            spatial_corr = np.corrcoef(history_arr.T)
        else:
            spatial_corr = np.zeros((self.n_stabilizers, self.n_stabilizers))

        return {
            "flip_rates": flip_rates,
            "spatial_correlation": spatial_corr,
            "n_samples": self.counts
        }

    def plot_spatial_heatmap(
        self, 
        d: int, 
        save_path: str = "results/figures/fingerprint_heatmap.png"
    ) -> None:
        """Visualize stabilizer flip rates on the code lattice."""
        fp = self.get_fingerprint()
        rates = fp["flip_rates"]
        
        # Reshape to lattice (simplified for square codes)
        # For a d=3 surface code, there are d(d-1) stabilizers of each type
        # We'll plot them on a grid
        grid_size = int(np.ceil(np.sqrt(self.n_stabilizers)))
        grid = np.zeros((grid_size, grid_size))
        grid.flat[:self.n_stabilizers] = rates
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(grid, annot=True, cmap="YlOrRd", fmt=".4f")
        plt.title(f"Hardware Noise Fingerprint: Stabilizer Flip Rates (d={d})")
        plt.xlabel("Lattice X")
        plt.ylabel("Lattice Y")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved spatial heatmap to %s", save_path)

    def plot_temporal_autocorrelation(
        self, 
        max_lag: int = 20,
        save_path: str = "results/figures/temporal_autocorr.png"
    ) -> None:
        """Plot autocorrelation of stabilizers over time."""
        if len(self.history) < max_lag * 2:
            logger.warning("Not enough history for temporal autocorrelation")
            return
            
        history_arr = np.array(self.history)
        autocorrs = []
        
        # Compute mean autocorrelation across all stabilizers
        for lag in range(1, max_lag + 1):
            corrs = []
            for i in range(self.n_stabilizers):
                series = history_arr[:, i]
                c = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                corrs.append(c if not np.isnan(c) else 0.0)
            autocorrs.append(np.mean(corrs))
            
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, max_lag + 1), autocorrs, color="#de8f05")
        plt.xlabel("Time Lag (Syndrome Rounds)")
        plt.ylabel("Mean Autocorrelation")
        plt.title("Hardware Temporal Noise Fingerprint")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info("Saved temporal autocorrelation plot to %s", save_path)
