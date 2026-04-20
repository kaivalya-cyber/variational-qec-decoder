import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ansatz import HardwareEfficientAnsatz, SymmetryPreservingAnsatz
from src.decoder import VariationalDecoder
from src.evaluator import threshold_scan, plot_training_history
from src.noise_models import create_noise_model
from src.stabilizer_codes import SurfaceCode
from src.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Variational Decoder.")
    parser.add_argument("--d", type=int, default=3, help="Surface code distance")
    parser.add_argument("--ansatz", type=str, default="hardware_efficient", choices=["hardware_efficient", "symmetry_preserving"])
    parser.add_argument("--layers", type=int, default=4, help="Ansatz layers")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--noise", type=str, default="depolarizing", help="Noise model type")
    parser.add_argument("--shots", type=int, default=5000, help="Evaluation shots")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--p_steps", type=int, default=10, help="Number of p-values in threshold scan")
    args = parser.parse_args()

    os.makedirs("results/data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    code = SurfaceCode(d=args.d)
    
    if args.ansatz == "hardware_efficient":
        ansatz = HardwareEfficientAnsatz(n_qubits=code.n_qubits, n_layers=args.layers)
    else:
        ansatz = SymmetryPreservingAnsatz(n_qubits=code.n_qubits, n_layers=args.layers)

    train_noise = create_noise_model(args.noise, p=0.05)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=train_noise)
    
    trainer = Trainer(decoder=decoder, lr=0.01, batch_size=args.batch_size)
    
    logging.info(f"Training {args.ansatz} Decoder for {args.epochs} epochs...")
    history = trainer.train(n_epochs=args.epochs, eval_shots=args.shots)
    
    plot_training_history(history.to_dict(), save_path=f"results/figures/train_d{args.d}_{args.ansatz}.png")

    logging.info("Evaluating threshold curve...")
    p_values = np.logspace(-3, -1, args.p_steps)
    
    def noise_factory(p):
        return create_noise_model(args.noise, p=p)

    results = threshold_scan(
        decoder=decoder,
        code=code,
        noise_model_factory=noise_factory,
        p_values=p_values,
        n_shots_per_p=args.shots,
    )

    serializable_results = {
        f"Variational_{args.ansatz}_d{args.d}": {
            "p_values": results["p_values"].tolist(),
            "logical_error_rates": results["logical_error_rates"].tolist()
        }
    }

    out_file = f"results/data/variational_d{args.d}_{args.ansatz}.json"
    with open(out_file, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    logging.info(f"Saved variational results to {out_file}")

if __name__ == "__main__":
    main()
