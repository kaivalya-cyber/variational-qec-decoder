"""High-Precision experiment runner for QEC decoding.

This script targets d=3 and d=4 for variational experiments with 
very high statistics and resolution, staying strictly within 6GB RAM.
Also runs classical baselines up to d=11.
"""

import logging
import os
import subprocess
import time
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HighPrecisionRunner")

def run_command(cmd: list):
    """Run a command and monitor execution."""
    logger.info("="*60)
    logger.info("RUNNING: %s", " ".join(cmd))
    logger.info("="*60)
    
    start_time = time.time()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        # d=4 fits easily, so we can use more threads if available
        subprocess.run(cmd, check=True, env=env)
        logger.info("SUCCESS: Command completed.")
    except subprocess.CalledProcessError as e:
        logger.error("FAILED: Command failed with error %s", e)
    
    elapsed = time.time() - start_time
    logger.info("Time taken: %.2f seconds", elapsed)
    gc.collect()

def main():
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. High-Precision Variational (d=3, 4)
    # 20 p-values, 10,000 shots for "Gold Standard" curves
    for d in [3, 4]:
        run_command([
            "python", "experiments/run_variational.py",
            "--d", str(d),
            "--ansatz", "hardware_efficient",
            "--epochs", "200",
            "--shots", "10000",
            "--batch_size", "32",
            "--p_steps", "20"
        ])

    # 2. High-Distance Baselines (d=7, 9, 11)
    # These are fast and memory-efficient
    for d in [7, 9, 11]:
        run_command([
            "python", "experiments/run_baseline.py",
            "--d", str(d),
            "--shots", "20000",
            "--noise", "depolarizing",
            "--p_steps", "20"
        ])

    # 3. Generate combined plots
    run_command(["python", "experiments/plot_results.py"])

    logger.info("="*60)
    logger.info("HIGH PRECISION EXPERIMENTS COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
