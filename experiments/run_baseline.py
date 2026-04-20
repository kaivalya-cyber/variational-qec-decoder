import argparse
import json
import logging
import os
import sys

import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classical_decoders import LookupTableDecoder, MWPMDecoder, HAS_PYMATCHING
from src.evaluator import compare_decoders
from src.noise_models import create_noise_model
from src.stabilizer_codes import SurfaceCode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run classical baseline decoders.")
    parser.add_argument("--d", type=int, default=3, help="Surface code distance")
    parser.add_argument("--shots", type=int, default=5000, help="Number of Monte Carlo shots")
    parser.add_argument("--noise", type=str, default="depolarizing", help="Noise model type")
    parser.add_argument("--p_steps", type=int, default=10, help="Number of p-values in threshold scan")
    args = parser.parse_args()

    os.makedirs("results/data", exist_ok=True)

    code = SurfaceCode(d=args.d)
    
    decoders = {}
    
    # Lookup table decoder is exponentially expensive, only use for d=3
    if args.d == 3:
        decoders["LookupTable"] = LookupTableDecoder(code=code, max_weight=2)

    if HAS_PYMATCHING:
        decoders["MWPM"] = MWPMDecoder(code=code)
    else:
        logging.warning("Skipping MWPM as pymatching is not installed.")

    p_values = np.logspace(-3, -1, args.p_steps)
    
    # Generate partial function for noise model factory
    def noise_factory(p):
        if args.noise == "combined":
            return create_noise_model(args.noise, p_bit=p, p_phase=p)
        return create_noise_model(args.noise, p=p)

    results = compare_decoders(
        decoders_dict=decoders,
        code=code,
        noise_model_factory=noise_factory,
        p_values=p_values,
        n_shots=args.shots,
    )

    # Convert results to serializable format
    serializable_results = {}
    for name, data in results.items():
        serializable_results[f"{name}_d{args.d}"] = {
            "p_values": data["p_values"].tolist(),
            "logical_error_rates": data["logical_error_rates"].tolist()
        }

    out_file = f"results/data/baseline_d{args.d}_{args.noise}.json"
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Saved baseline results to {out_file}")

if __name__ == "__main__":
    main()
