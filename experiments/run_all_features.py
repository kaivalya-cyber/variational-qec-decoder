"""Master runner for all advanced research features.

Executes all feature-specific experiments sequentially while monitoring 
RAM usage to ensure compliance with 4GB hardware constraints.
"""

import gc
import logging
import os
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureRunner")

def check_ram():
    """Check RAM usage on macOS (simplified)."""
    try:
        # Use ps to get memory usage of current process
        pid = os.getpid()
        out = subprocess.check_output(["ps", "-o", "rss", "-p", str(pid)])
        rss_kb = int(out.split()[1])
        logger.info("Current process RAM: %.2f MB", rss_kb / 1024.0)
    except Exception:
        pass

def run_script(script_path: str, args: list = []):
    """Run a feature experiment script."""
    logger.info("="*60)
    logger.info("RUNNING: %s", script_path)
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        cmd = ["python", script_path] + args
        subprocess.run(cmd, check=True)
        
        logger.info("SUCCESS: %s", script_path)
    except subprocess.CalledProcessError as e:
        logger.error("FAILED: %s with error %s", script_path, e)
    
    elapsed = time.time() - start_time
    logger.info("Time taken: %.2f seconds", elapsed)
    
    # Cleanup between runs
    gc.collect()
    check_ram()
    time.sleep(2) # Cooldown

def main():
    os.makedirs("results/figures", exist_ok=True)
    
    experiments = [
        ("experiments/run_bp_experiment.py", ["--shots", "50"]),
        ("experiments/run_fingerprinting.py", ["--shots", "200"]),
        ("experiments/run_block_decoder.py", ["--shots", "200"]),
        ("experiments/run_calibration.py", ["--samples", "500"]),
        ("experiments/run_compression.py", ["--samples", "500"]),
    ]
    
    total_start = time.time()
    
    for script, args in experiments:
        run_script(script, args)
        
    total_elapsed = time.time() - total_start
    logger.info("="*60)
    logger.info("ALL EXPERIMENTS COMPLETE in %.2f seconds", total_elapsed)
    logger.info("Check results/figures/ for generated paper content.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
