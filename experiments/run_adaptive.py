import argparse
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adaptive_selector import AdaptiveDecoderSelector
from src.evaluator import threshold_scan
from src.noise_classifier import NoiseClassifier
from src.noise_models import create_noise_model
from src.stabilizer_codes import SurfaceCode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Run Adaptive Decoder Novel Experiment")
    parser.add_argument("--d", type=int, default=3, help="Surface code distance")
    parser.add_argument("--shots", type=int, default=5000, help="Number of shots per noise type")
    parser.add_argument("--epochs", type=int, default=500, help="Classifier training epochs")
    parser.add_argument("--layers", type=int, default=2, help="Decoder bank ansatz layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    logging.info(f"Running Adaptive Decoder Novel Experiment for d={args.d}")
    
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    code = SurfaceCode(d=args.d)
    
    # 1. Train Noise Classifier
    classifier = NoiseClassifier(n_qubits=code.n_qubits, code=code, time_steps=10)
    logging.info("Generating training data for Classifier...")
    X, y = classifier.generate_training_data(n_samples_per_class=1000, p=0.05)
    
    logging.info("Training Noise Classifier...")
    history = classifier.train_classifier(X, y, n_epochs=args.epochs)
    classifier.save(f"results/models/noise_classifier_d{args.d}.pt")
    
    # 2. Build Adaptive Selector (builds Bank of Decoders implicitly)
    logging.info("Building Decoder Bank for Adaptive Selector... (this may take a while to initialize)")
    selector = AdaptiveDecoderSelector(code=code, noise_classifier=classifier)
    selector.build_decoder_bank(n_layers=args.layers)

    # 3. Evaluate on Mixed Environment
    # We will test in an environment where the noise type changes randomly
    p_values = np.logspace(-3, -1, 10)
    time_steps = 10
    
    logical_rates = []
    
    for p in p_values:
        n_errors = 0
        total_shots = args.shots * 4 # Test each noise type n_shots times
        
        for noise_type in ["depolarizing", "bit_flip", "phase_flip", "combined"]:
            b_p, p_p = (p, p) if noise_type == "combined" else (p, 0)
            noise_model = create_noise_model(noise_type, p=p, p_bit=b_p, p_phase=p_p)
            
            res = selector.evaluate_adaptive(noise_model, n_shots=args.shots, time_steps=time_steps, seed=42)
            n_errors += res["n_logical_errors"]
            
        logical_rates.append(n_errors / total_shots)
        logging.info(f"Mixed Env d={args.d} p={p:.4f}: LER={logical_rates[-1]:.6f}")

    serializable_results = {
        f"Adaptive_Novel_d{args.d}": {
            "p_values": p_values.tolist(),
            "logical_error_rates": logical_rates
        }
    }

    out_file = f"results/data/adaptive_d{args.d}_mixed.json"
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
        
    logging.info(f"Saved adaptive results to {out_file}")

if __name__ == "__main__":
    main()
