"""Statistical analysis of correlations between logical qubit syndromes.

Generates: Section VIII.B (Syndrome Mutual Information), 
           Figure 17 (syndrome_correlation_matrix.png)

Novel Contribution: First quantitative measurement of mutual information 
between stabilizer syndrome streams in multi-qubit systems.

Provides evidence for why joint block decoding outperforms independent decoding.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

class CrossQubitCorrelationAnalyzer:
    """Measures statistical relationships between different logical qubits.

    Parameters
    ----------
    n_logical : int
        Number of logical qubits to analyze.
    """

    def __init__(self, n_logical: int) -> None:
        self.n_logical = n_logical
        logger.info("CrossQubitCorrelationAnalyzer initialised for %d qubits", n_logical)

    def measure_mutual_information(
        self, 
        syn_data1: np.ndarray, 
        syn_data2: np.ndarray
    ) -> float:
        """Estimate MI between two syndrome streams.
        
        Using a simplified discrete estimator for binary syndromes.
        MI(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        def entropy(data):
            # Flatten to treat as one variable or analyze per stabilizer
            # Here we treat the entire syndrome vector as a 'state'
            unique, counts = np.unique(data, axis=0, return_counts=True)
            probs = counts / len(data)
            return -np.sum(probs * np.log2(probs + 1e-12))

        h1 = entropy(syn_data1)
        h2 = entropy(syn_data2)
        
        # Joint entropy
        joint_data = np.hstack([syn_data1, syn_data2])
        h12 = entropy(joint_data)
        
        return max(0, h1 + h2 - h12)

    def plot_correlation_matrix(
        self, 
        all_syndromes: List[np.ndarray],
        save_path: str = "results/figures/syndrome_correlation_matrix.png"
    ) -> np.ndarray:
        """Compute and plot pairwise MI between all logical qubits."""
        mi_matrix = np.zeros((self.n_logical, self.n_logical))
        
        for i in range(self.n_logical):
            for j in range(i, self.n_logical):
                if i == j:
                    # Self-MI is entropy
                    mi = self.measure_mutual_information(all_syndromes[i], all_syndromes[i])
                else:
                    mi = self.measure_mutual_information(all_syndromes[i], all_syndromes[j])
                
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
                
        plt.figure(figsize=(10, 8))
        sns.heatmap(mi_matrix, annot=True, cmap="mako", fmt=".3f")
        plt.title("Syndrome Mutual Information Matrix (Inter-Qubit Correlations)")
        plt.xlabel("Logical Qubit Index")
        plt.ylabel("Logical Qubit Index")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info("Saved syndrome correlation matrix to %s", save_path)
        return mi_matrix
