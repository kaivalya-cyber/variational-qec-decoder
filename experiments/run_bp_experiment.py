"""Experiment: Impact of BP Preprocessing on Decoding Performance.

Generates: Section VI.C (Scaling of BP Gains), 
           Figure 11 (bp_time_tradeoff.png)

This script sweeps BP iterations and damping factors to find the 
optimal balance between classical overhead and quantum logical gain.
"""

import argparse
import logging
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.belief_propagation import BeliefPropagator
from src.bp_enhanced_decoder import BPEnhancedVariationalDecoder
from src.stabilizer_codes import RepetitionCode, SurfaceCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_bp_scaling_experiment(code_distance: int = 3, n_shots: int = 100):
    """Measures LER improvement vs BP iteration count."""
    code = SurfaceCode(d=code_distance)
    H = code.get_parity_check_matrix()
    
    iter_counts = [1, 2, 5, 10, 20, 50]
    lers = []
    times = []
    
    p = 0.05
    
    # Generate syndromes once to ensure fair comparison
    errors = (np.random.random((n_shots, code.n_qubits)) < p).astype(np.uint8)
    syndromes = [(H @ e) % 2 for e in errors]
    
    for max_iter in iter_counts:
        bp = BeliefPropagator(H, max_iterations=max_iter, damping=0.5)
        
        start_time = time.time()
        n_errors = 0
        for i in range(n_shots):
            soft_probs, converged, _ = bp.run_bp(syndromes[i], p)
            corr = (soft_probs > 0.5).astype(np.uint8)
            residual = (errors[i] + corr) % 2
            if np.any(residual):
                n_errors += 1
        
        elapsed = (time.time() - start_time) / n_shots
        times.append(elapsed * 1000) # ms
        lers.append(n_errors / n_shots)
        
        logger.info("Iter %d: LER=%.4f, Time=%.4f ms", max_iter, lers[-1], times[-1])

    # Plot 1: LER vs Iterations
    plt.figure(figsize=(8, 5))
    plt.plot(iter_counts, lers, 'o-', color='#0173b2', linewidth=2)
    plt.xlabel("BP Max Iterations")
    plt.ylabel("Logical Error Rate")
    plt.title(f"BP Error Suppression vs Computation (d={code_distance}, p={p})")
    plt.grid(True, linestyle='--')
    plt.savefig("results/figures/bp_improvement_scaling.png", dpi=300)
    plt.close()

    # Plot 2: Time Tradeoff
    plt.figure(figsize=(8, 5))
    plt.scatter(times, lers, s=100, c=iter_counts, cmap='viridis')
    plt.colorbar(label="BP Iterations")
    plt.xlabel("Average BP Time per Syndrome (ms)")
    plt.ylabel("Logical Error Rate")
    plt.title("Performance vs. Latency Tradeoff")
    plt.grid(True, linestyle='--')
    plt.savefig("results/figures/bp_time_tradeoff.png", dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run BP Preprocessing Experiment")
    parser.add_argument("--d", type=int, default=3, help="Code distance")
    parser.add_argument("--shots", type=int, default=100, help="Number of shots")
    args = parser.parse_args()

    os.makedirs("results/figures", exist_ok=True)
    
    logger.info("Starting BP Preprocessing Experiment...")
    run_bp_scaling_experiment(code_distance=args.d, n_shots=args.shots)
    
    # Run structural plots
    code = SurfaceCode(d=args.d)
    bp = BeliefPropagator(code.get_parity_check_matrix())
    bp.plot_tanner_graph()
    bp.convergence_analysis(n_syndromes=args.shots)
    
    logger.info("Experiment complete. Figures saved to results/figures/")

if __name__ == "__main__":
    main()
