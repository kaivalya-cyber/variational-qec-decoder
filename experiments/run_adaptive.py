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
    logging.info("Running Adaptive Decoder Novel Experiment")
    
    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    d = 3
    code = SurfaceCode(d=d)
    
    # 1. Train Noise Classifier
    classifier = NoiseClassifier(n_qubits=code.n_qubits, code=code, time_steps=10)
    logging.info("Generating training data for Classifier...")
    X, y = classifier.generate_training_data(n_samples_per_class=1000, p=0.05)
    
    logging.info("Training Noise Classifier...")
    history = classifier.train_classifier(X, y, n_epochs=30)
    classifier.save("results/models/noise_classifier.pt")
    
    # 2. Build Adaptive Selector (builds Bank of Decoders implicitly)
    logging.info("Building Decoder Bank for Adaptive Selector... (this may take a while to initialize)")
    selector = AdaptiveDecoderSelector(code=code, noise_classifier=classifier)
    selector.build_decoder_bank(n_layers=2) # Small for demonstration

    # 3. Evaluate on Mixed Environment
    # We will test in an environment where the noise type changes randomly
    p_values = np.logspace(-3, -1, 10)
    n_shots = 200
    time_steps = 10
    
    logical_rates = []
    
    for p in p_values:
        n_errors = 0
        total_shots = n_shots * 4 # Test each noise type n_shots times
        
        for noise_type in ["depolarizing", "bit_flip", "phase_flip", "combined"]:
            b_p, p_p = (p, p) if noise_type == "combined" else (p, 0)
            noise_model = create_noise_model(noise_type, p=p, p_bit=b_p, p_phase=p_p)
            
            # Using the evaluate_adaptive feature we built
            res = selector.evaluate_adaptive(noise_model, n_shots=n_shots, time_steps=time_steps, seed=42)
            n_errors += res["n_logical_errors"]
            
        logical_rates.append(n_errors / total_shots)
        logging.info(f"Mixed Env p={p:.4f}: LER={logical_rates[-1]:.6f}")

    serializable_results = {
        "Adaptive_Novel": {
            "p_values": p_values.tolist(),
            "logical_error_rates": logical_rates
        }
    }

    out_file = f"results/data/adaptive_d{d}_mixed.json"
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
        
    logging.info(f"Saved adaptive results to {out_file}")

if __name__ == "__main__":
    main()
