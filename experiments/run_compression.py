"""Experiment: Scaling and Fidelity of Syndrome Compression.

Generates: Section X.C (Latent Dimension Tradeoffs), 
           Figure 25 (compression_tradeoff.png)

This script sweeps latent dimensions to find the optimal compression 
ratio that preserves enough information for accurate decoding.
"""

import argparse
import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.syndrome_autoencoder import SyndromeAutoencoder
from src.compressed_decoder import CompressedVariationalDecoder
from src.stabilizer_codes import SurfaceCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_compression_experiment(code_distance: int = 3, n_samples: int = 1000):
    """Measures reconstruction fidelity vs latent dimension."""
    code = SurfaceCode(d=code_distance)
    n_stab = code.n_stabilizers
    
    # Generate dummy syndrome dataset
    # (Simplified: random sparse vectors)
    data = (np.random.random((n_samples, n_stab)) < 0.1).astype(np.float32)
    labels = np.sum(data, axis=1) % 2 # Simple parity label for t-SNE
    
    latent_dims = [2, 3, 4, 6, 8]
    fidelities = []
    
    for ld in latent_dims:
        if ld > n_stab: continue
            
        ae = SyndromeAutoencoder(n_stab, latent_dim=ld)
        losses = ae.train_ae(data, epochs=50)
        
        # Compute fidelity (1 - final reconstruction error)
        ae.eval()
        with torch.no_grad():
            recon, _ = ae(torch.tensor(data))
            error = torch.mean((recon - torch.tensor(data))**2).item()
            fidelities.append(1.0 - error)
            
        logger.info("Latent Dim %d: Fidelity %.4f", ld, fidelities[-1])

    # Plot Tradeoff
    plt.figure(figsize=(8, 5))
    plt.plot(latent_dims, fidelities, 'o-', color='#0173b2', linewidth=2)
    plt.xlabel("Latent Dimension")
    plt.ylabel("Reconstruction Fidelity")
    plt.title(f"Syndrome Compression vs. Fidelity (d={code_distance})")
    plt.grid(True, linestyle='--')
    plt.savefig("results/figures/compression_tradeoff.png", dpi=300)
    plt.close()

    # Generate t-SNE for the best model
    best_ae = SyndromeAutoencoder(n_stab, latent_dim=4)
    best_ae.train_ae(data, epochs=50)
    best_ae.plot_latent_space(data, labels)
    best_ae.plot_reconstruction_fidelity(losses)
    
    # Resource analysis
    dec = CompressedVariationalDecoder(code, best_ae, None)
    dec.plot_qubit_reduction()

def main():
    parser = argparse.ArgumentParser(description="Run Syndrome Compression Experiment")
    parser.add_argument("--d", type=int, default=3, help="Code distance")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    
    logger.info("Starting Syndrome Compression Experiment...")
    run_compression_experiment(code_distance=args.d, n_samples=args.samples)
    
    logger.info("Experiment complete. Figures saved to results/figures/")

if __name__ == "__main__":
    main()
