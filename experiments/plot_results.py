import glob
import json
import logging
import os
import sys

# For matplotlib to not require X11
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluator import plot_threshold_curve, FIGURES_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def main():
    data_files = glob.glob("results/data/*.json")
    if not data_files:
        logging.error("No JSON results found in results/data/. Please run experiments first.")
        return

    combined_results = {}
    for fpath in data_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                for k, v in data.items():
                    # converting lists back to np.arrays for the plotting function
                    combined_results[k] = {
                        "p_values": np.array(v["p_values"]),
                        "logical_error_rates": np.array(v["logical_error_rates"])
                    }
        except Exception as e:
            logging.warning(f"Failed to load {fpath}: {e}")

    if combined_results:
        out_path = os.path.join(FIGURES_DIR, "combined_thresholds.png")
        plot_threshold_curve(
            combined_results,
            title="Logical Error Rate Comparison",
            save_path=out_path
        )
        logging.info(f"Successfully generated plot at {out_path}")
    else:
        logging.info("No valid results found to plot.")

if __name__ == "__main__":
    main()
