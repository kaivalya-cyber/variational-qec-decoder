"""Continuous noise estimation experiments.

Runs three experiments:
1. Noise manifold sweep — grid over (p_X, p_Z) space
2. Estimation accuracy — predicted vs true noise parameters
3. Interpolation ablation — hard routing vs soft interpolation

All experiments process in chunks of 50 iterations, save intermediate
results to disk, and use tqdm progress bars.

Hardware constraints: 3GB RAM, chunk processing, immediate figure cleanup.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ansatz import AdaptiveAnsatz
from src.continuous_selector import ContinuousDecoderSelector
from src.decoder import VariationalDecoder
from src.noise_estimator import NoiseParameterEstimator, EstimatorConfig
from src.noise_models import create_noise_model
from src.stabilizer_codes import SurfaceCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


RESULTS_DIR = "results/continuous"
FIGURES_DIR = "results/figures"
CHUNK_SIZE = 50


def setup_dirs():
    """Create output directories."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def build_decoder_bank(code, n_layers=2, seed=42):
    """Build a bank of pre-trained decoders for each noise type.

    Parameters
    ----------
    code : SurfaceCode
        The QEC code.
    n_layers : int
        Number of ansatz layers.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{noise_type: VariationalDecoder}``
    """
    bank = {}
    noise_types = ["depolarizing", "bit_flip", "phase_flip", "combined"]
    for i, nt in enumerate(noise_types):
        ansatz = AdaptiveAnsatz(
            n_qubits=code.n_qubits, noise_type=nt, n_layers=n_layers
        )
        if nt == "combined":
            noise = create_noise_model(nt, p_bit=0.03, p_phase=0.03)
        else:
            noise = create_noise_model(nt, p=0.03)
        decoder = VariationalDecoder(
            code=code, ansatz=ansatz, noise_model=noise
        )
        decoder.params = ansatz.init_params(seed=seed + i)
        bank[nt] = decoder
        logger.info("Built decoder for %s", nt)
    return bank


# -----------------------------------------------------------------------
# Experiment 1: Noise Manifold Sweep
# -----------------------------------------------------------------------

def run_manifold_sweep(code, selector, n_steps=10, n_shots=50, seed=42):
    """Grid over (p_X, p_Z) space, compute LER at each point.

    Parameters
    ----------
    code : SurfaceCode
    selector : ContinuousDecoderSelector
    n_steps : int
        Grid resolution per axis.
    n_shots : int
        Shots per grid point.
    seed : int

    Returns
    -------
    dict
        Results to save as JSON.
    """
    logger.info("=== Experiment 1: Noise Manifold Sweep ===")
    p_vals = np.linspace(0.0, 0.05, n_steps, dtype=np.float32)
    results = {
        "p_x_values": p_vals.tolist(),
        "p_z_values": p_vals.tolist(),
        "ler_grid": [],
    }

    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]
    total = n_steps * n_steps
    processed = 0

    for i, p_x in enumerate(tqdm(p_vals, desc="Manifold rows")):
        row = []
        for j, p_z in enumerate(p_vals):
            noise = create_noise_model(
                "combined",
                p_bit=float(p_x + 0.01 / 3.0),
                p_phase=float(p_z + 0.01 / 3.0),
            )
            n_errors = 0
            for shot in range(n_shots):
                error = noise.sample_errors(code.n_qubits, 1)[0]
                syndrome = code.extract_syndrome(error)
                correction = selector.decode(syndrome)
                residual = _compose_paulis(error, correction)
                if _is_logical_error(residual, lx, lz):
                    n_errors += 1
            row.append(n_errors / n_shots)
            processed += 1

            # Save intermediate every CHUNK_SIZE grid points
            if processed % CHUNK_SIZE == 0:
                _save_intermediate(
                    results, "continuous_manifold_partial.json"
                )

        results["ler_grid"].append(row)

    # Save final
    out_path = os.path.join(RESULTS_DIR, "continuous_manifold.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved manifold results to %s", out_path)

    # Generate figure
    _plot_manifold(results, p_vals)

    return results


def _plot_manifold(results, p_vals):
    """Generate and save manifold heatmap figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ler = np.array(results["ler_grid"], dtype=np.float32)
    im = ax.imshow(
        ler,
        extent=[p_vals[0], p_vals[-1], p_vals[0], p_vals[-1]],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Logical Error Rate", fontsize=12)
    ax.set_xlabel(r"$p_X$ (bit-flip)", fontsize=14)
    ax.set_ylabel(r"$p_Z$ (phase-flip)", fontsize=14)
    ax.set_title(
        "Continuous Decoder: Noise Manifold Performance",
        fontsize=15, fontweight="bold",
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "noise_manifold_heatmap.png")
    fig.savefig(save_path, dpi=300)
    plt.close("all")
    del fig
    gc.collect()
    logger.info("Saved noise manifold heatmap to %s", save_path)


# -----------------------------------------------------------------------
# Experiment 2: Estimation Accuracy
# -----------------------------------------------------------------------

def run_estimation_accuracy(estimator, code, n_test=200, seed=123):
    """Compare estimated vs true noise parameters.

    Parameters
    ----------
    estimator : NoiseParameterEstimator
    code : SurfaceCode
    n_test : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 2: Estimation Accuracy ===")
    rng = np.random.default_rng(seed)
    trues = []
    preds = []
    uncertainties = []

    for i in tqdm(range(n_test), desc="Estimation accuracy"):
        p_x = rng.uniform(0.0, 0.05).astype(np.float32)
        p_z = rng.uniform(0.0, 0.05).astype(np.float32)
        p_dep = rng.uniform(0.0, 0.05).astype(np.float32)

        noise = create_noise_model(
            "combined",
            p_bit=float(p_x + p_dep / 3.0),
            p_phase=float(p_z + p_dep / 3.0),
        )

        history = np.zeros((20, code.n_stabilizers), dtype=np.float32)
        for t in range(20):
            errors = noise.sample_errors(code.n_qubits, 1)
            syndrome = code.extract_syndrome(errors[0])
            history[t] = syndrome.astype(np.float32)

        est = estimator.estimate(history)
        trues.append([float(p_x), float(p_z), float(p_dep)])
        preds.append([est["p_X"], est["p_Z"], est["p_depol"]])
        uncertainties.append(est["uncertainty"])

        # Intermediate save
        if (i + 1) % CHUNK_SIZE == 0:
            _save_intermediate(
                {"trues": trues, "preds": preds}, "estimation_partial.json"
            )

    trues_arr = np.array(trues, dtype=np.float32)
    preds_arr = np.array(preds, dtype=np.float32)
    mae = np.mean(np.abs(trues_arr - preds_arr), axis=0)

    results = {
        "trues": trues,
        "preds": preds,
        "uncertainties": [float(u) for u in uncertainties],
        "mae_p_X": float(mae[0]),
        "mae_p_Z": float(mae[1]),
        "mae_p_depol": float(mae[2]),
        "mean_uncertainty": float(np.mean(uncertainties)),
    }

    out_path = os.path.join(RESULTS_DIR, "estimation_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved estimation accuracy to %s", out_path)

    # Generate calibration figure
    estimator.calibration_curve(n_test=100, seed=seed + 1000)

    return results


# -----------------------------------------------------------------------
# Experiment 3: Interpolation Ablation
# -----------------------------------------------------------------------

def run_interpolation_ablation(
    code, selector, n_boundary_tests=50, n_shots=50, seed=42
):
    """Compare hard routing vs soft interpolation at boundary conditions.

    Parameters
    ----------
    code : SurfaceCode
    selector : ContinuousDecoderSelector
    n_boundary_tests : int
    n_shots : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 3: Interpolation Ablation ===")
    rng = np.random.default_rng(seed)
    results_hard = []
    results_soft = []

    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]

    for i in tqdm(range(n_boundary_tests), desc="Interpolation ablation"):
        # Boundary noise: halfway between two decoder centers
        p_x = rng.uniform(0.01, 0.03).astype(np.float32)
        p_z = rng.uniform(0.01, 0.03).astype(np.float32)

        noise = create_noise_model(
            "combined",
            p_bit=float(p_x),
            p_phase=float(p_z),
        )

        # Hard routing: use closest single decoder
        n_err_hard = 0
        n_err_soft = 0

        for shot in range(n_shots):
            error = noise.sample_errors(code.n_qubits, 1)[0]
            syndrome = code.extract_syndrome(error)

            # Hard: just use the first decoder
            correction_hard = selector.decode(syndrome)
            residual = _compose_paulis(error, correction_hard)
            if _is_logical_error(residual, lx, lz):
                n_err_hard += 1

            # Soft: use continuous selector (with dummy history)
            history = np.zeros(
                (20, code.n_stabilizers), dtype=np.float32
            )
            history[-1] = syndrome.astype(np.float32)
            correction_soft, _ = selector.decode_continuous(
                syndrome, history
            )
            residual_soft = _compose_paulis(error, correction_soft)
            if _is_logical_error(residual_soft, lx, lz):
                n_err_soft += 1

        results_hard.append(n_err_hard / n_shots)
        results_soft.append(n_err_soft / n_shots)

        if (i + 1) % CHUNK_SIZE == 0:
            _save_intermediate(
                {"hard": results_hard, "soft": results_soft},
                "interpolation_partial.json",
            )

    results = {
        "hard_routing_ler": results_hard,
        "soft_interpolation_ler": results_soft,
        "mean_hard_ler": float(np.mean(results_hard)),
        "mean_soft_ler": float(np.mean(results_soft)),
        "improvement_pct": float(
            (np.mean(results_hard) - np.mean(results_soft))
            / max(np.mean(results_hard), 1e-10) * 100
        ),
    }

    out_path = os.path.join(RESULTS_DIR, "interpolation_ablation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved interpolation ablation to %s", out_path)

    # Plot comparison
    _plot_interpolation(results)

    return results


def _plot_interpolation(results):
    """Generate interpolation comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    x = range(len(results["hard_routing_ler"]))
    ax.plot(
        x, results["hard_routing_ler"],
        label="Hard Routing", color="#de8f05",
        linewidth=1.5, alpha=0.8,
    )
    ax.plot(
        x, results["soft_interpolation_ler"],
        label="Soft Interpolation", color="#0173b2",
        linewidth=1.5, alpha=0.8,
    )
    ax.axhline(
        y=results["mean_hard_ler"], color="#de8f05",
        linestyle="--", alpha=0.5, linewidth=1,
    )
    ax.axhline(
        y=results["mean_soft_ler"], color="#0173b2",
        linestyle="--", alpha=0.5, linewidth=1,
    )
    ax.set_xlabel("Boundary Test Index", fontsize=14)
    ax.set_ylabel("Logical Error Rate", fontsize=14)
    ax.set_title(
        "Hard Routing vs Soft Interpolation at Noise Boundaries",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "interpolation_comparison.png")
    fig.savefig(save_path, dpi=300)
    plt.close("all")
    del fig
    gc.collect()
    logger.info("Saved interpolation comparison to %s", save_path)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _save_intermediate(data, filename):
    """Save intermediate results to disk."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _compose_paulis(a, b):
    """Compose two Pauli vectors."""
    ax = np.isin(a, [1, 2]).astype(int)
    az = np.isin(a, [2, 3]).astype(int)
    bx = np.isin(b, [1, 2]).astype(int)
    bz = np.isin(b, [2, 3]).astype(int)
    rx = (ax + bx) % 2
    rz = (az + bz) % 2
    result = np.zeros_like(a)
    result[(rx == 1) & (rz == 0)] = 1
    result[(rx == 1) & (rz == 1)] = 2
    result[(rx == 0) & (rz == 1)] = 3
    return result


def _is_logical_error(residual, lx, lz):
    """Check if residual is a logical error."""
    res_x = np.isin(residual, [1, 2]).astype(int)
    res_z = np.isin(residual, [2, 3]).astype(int)
    log_x = np.isin(lx, [1, 2]).astype(int)
    log_z = np.isin(lz, [2, 3]).astype(int)
    comm_z = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
    comm_x = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2
    return bool(comm_x or comm_z)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run continuous noise estimation experiments"
    )
    parser.add_argument("--d", type=int, default=3, help="Code distance")
    parser.add_argument(
        "--grid-steps", type=int, default=10, help="Manifold grid steps"
    )
    parser.add_argument("--shots", type=int, default=50, help="Shots per point")
    parser.add_argument(
        "--n-train", type=int, default=2000, help="Training samples"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Estimator training epochs"
    )
    args = parser.parse_args()

    setup_dirs()

    logger.info("Setting up code d=%d", args.d)
    code = SurfaceCode(d=args.d)

    # 1. Train noise estimator
    logger.info("Training noise parameter estimator...")
    config = EstimatorConfig(
        train_batch_size=16,
        train_epochs=args.epochs,
    )
    estimator = NoiseParameterEstimator(code=code, config=config)
    data_path = estimator.generate_training_data(
        n_samples=args.n_train, seed=42
    )
    estimator.train(data_path=data_path)
    estimator.save(os.path.join(RESULTS_DIR, "noise_estimator.pt"))

    # 2. Build decoder bank and selector
    logger.info("Building decoder bank...")
    bank = build_decoder_bank(code, n_layers=2)
    selector = ContinuousDecoderSelector(
        code=code,
        noise_estimator=estimator,
        decoder_bank=bank,
    )

    # 3. Run experiments
    run_manifold_sweep(
        code, selector,
        n_steps=args.grid_steps,
        n_shots=args.shots,
    )

    run_estimation_accuracy(estimator, code, n_test=200)

    run_interpolation_ablation(
        code, selector,
        n_boundary_tests=50,
        n_shots=args.shots,
    )

    logger.info("All continuous experiments complete!")


if __name__ == "__main__":
    main()
