"""Online learning experiments.

Runs four experiments:
1. Sudden switch adaptation — noise switches abruptly at step 500
2. Gradual drift — OU process over 5000 steps
3. Convergence analysis — adaptation lag for 10 random switches
4. Learning rate sensitivity — sweep η from 1e-4 to 1e-1

All experiments process in chunks of 50, save intermediate results,
and use tqdm progress bars.

Hardware constraints: 3GB RAM, experience replay maxlen=200, uint8 storage,
minibatch=8, single sample noise generation, immediate figure cleanup.
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
from src.convergence_analysis import ConvergenceAnalyzer
from src.decoder import VariationalDecoder
from src.noise_models import (
    DepolarizingNoise,
    BitFlipNoise,
    create_noise_model,
)
from src.online_learner import (
    DriftSimulator,
    OnlineConfig,
    OnlineLearner,
    _compose_paulis,
    _is_logical_error,
)
from src.stabilizer_codes import SurfaceCode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results/online"
FIGURES_DIR = "results/figures"
CHUNK_SIZE = 50


def setup_dirs():
    """Create output directories."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


def build_decoder(code, noise_type="depolarizing", n_layers=2, seed=42):
    """Build a variational decoder for a given noise type.

    Parameters
    ----------
    code : SurfaceCode
    noise_type : str
    n_layers : int
    seed : int

    Returns
    -------
    VariationalDecoder
    """
    ansatz = AdaptiveAnsatz(
        n_qubits=code.n_qubits, noise_type=noise_type, n_layers=n_layers
    )
    noise = create_noise_model(noise_type, p=0.03)
    decoder = VariationalDecoder(
        code=code, ansatz=ansatz, noise_model=noise
    )
    decoder.params = ansatz.init_params(seed=seed)
    return decoder


def compute_rolling_ler(outcomes, window_size=50):
    """Compute rolling logical error rate.

    Parameters
    ----------
    outcomes : list of int
        1 if logical error, 0 otherwise.
    window_size : int

    Returns
    -------
    list of float
    """
    ler = []
    for i in range(len(outcomes)):
        start = max(0, i - window_size + 1)
        ler.append(float(np.mean(outcomes[start : i + 1])))
    return ler


# -----------------------------------------------------------------------
# Experiment 1: Sudden Switch Adaptation
# -----------------------------------------------------------------------

def run_sudden_switch(code, n_steps=2000, switch_step=500, seed=42):
    """Compare frozen vs online vs oracle decoder after abrupt switch.

    Parameters
    ----------
    code : SurfaceCode
    n_steps : int
    switch_step : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 1: Sudden Switch Adaptation ===")

    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]

    # Build decoders
    frozen_decoder = build_decoder(code, "depolarizing", seed=seed)
    online_decoder = build_decoder(code, "depolarizing", seed=seed)
    oracle_decoder = build_decoder(code, "bit_flip", seed=seed + 1)

    online_config = OnlineConfig(
        buffer_size=200, minibatch_size=8, update_freq=50, lr=0.01
    )
    learner = OnlineLearner(decoder=online_decoder, config=online_config)

    drift = DriftSimulator(
        code=code, schedule="sudden_switch", p_base=0.03, seed=seed
    )

    frozen_outcomes = []
    online_outcomes = []
    oracle_outcomes = []
    results_partial = {"frozen": [], "online": [], "oracle": []}

    for t in tqdm(range(n_steps), desc="Sudden switch"):
        noise = drift.get_noise_at_step(t)
        error = noise.sample_errors(code.n_qubits, 1)[0]
        syndrome = code.extract_syndrome(error)

        # Frozen decoder
        corr_frozen = frozen_decoder.decode(syndrome)
        res_frozen = _compose_paulis(error, corr_frozen)
        is_err_frozen = _is_logical_error(res_frozen, lx, lz)
        frozen_outcomes.append(int(is_err_frozen))

        # Online decoder
        corr_online = online_decoder.decode(syndrome)
        res_online = _compose_paulis(error, corr_online)
        is_err_online = _is_logical_error(res_online, lx, lz)
        online_outcomes.append(int(is_err_online))
        learner.update(syndrome, corr_online, is_err_online)

        # Oracle decoder
        corr_oracle = oracle_decoder.decode(syndrome)
        res_oracle = _compose_paulis(error, corr_oracle)
        is_err_oracle = _is_logical_error(res_oracle, lx, lz)
        oracle_outcomes.append(int(is_err_oracle))

        # Intermediate save
        if (t + 1) % CHUNK_SIZE == 0:
            results_partial["frozen"] = frozen_outcomes
            results_partial["online"] = online_outcomes
            results_partial["oracle"] = oracle_outcomes
            _save_intermediate(results_partial, "sudden_switch_partial.json")

    # Compute rolling LER
    frozen_ler = compute_rolling_ler(frozen_outcomes)
    online_ler = compute_rolling_ler(online_outcomes)
    oracle_ler = compute_rolling_ler(oracle_outcomes)

    results = {
        "frozen_ler": frozen_ler,
        "online_ler": online_ler,
        "oracle_ler": oracle_ler,
        "switch_step": switch_step,
        "n_steps": n_steps,
    }

    out_path = os.path.join(RESULTS_DIR, "online_sudden_switch.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved sudden switch results to %s", out_path)

    _plot_sudden_switch(results)
    return results


def _plot_sudden_switch(results):
    """Plot sudden switch comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    steps = range(len(results["frozen_ler"]))

    ax.plot(
        steps, results["frozen_ler"],
        label="Frozen Decoder", color="#de8f05",
        linewidth=1.5, alpha=0.8,
    )
    ax.plot(
        steps, results["online_ler"],
        label="Online Learner", color="#0173b2",
        linewidth=2,
    )
    ax.plot(
        steps, results["oracle_ler"],
        label="Oracle Decoder", color="#029e73",
        linewidth=1.5, linestyle="--",
    )
    ax.axvline(
        x=results["switch_step"], color="red",
        linestyle=":", alpha=0.7, linewidth=1.5, label="Noise Switch",
    )
    ax.set_xlabel("Decoding Step", fontsize=14)
    ax.set_ylabel("Logical Error Rate (rolling avg)", fontsize=14)
    ax.set_title(
        "Sudden Noise Switch: Online vs Frozen Decoder",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "online_sudden_switch.png")
    fig.savefig(save_path, dpi=300)
    plt.close("all")
    del fig
    gc.collect()
    logger.info("Saved sudden switch figure to %s", save_path)


# -----------------------------------------------------------------------
# Experiment 2: Gradual Drift
# -----------------------------------------------------------------------

def run_gradual_drift(code, n_steps=2000, seed=42):
    """Compare online vs frozen under OU drift.

    Parameters
    ----------
    code : SurfaceCode
    n_steps : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 2: Gradual Drift ===")

    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]

    frozen_decoder = build_decoder(code, "depolarizing", seed=seed)
    online_decoder = build_decoder(code, "depolarizing", seed=seed)

    online_config = OnlineConfig(
        buffer_size=200, minibatch_size=8, update_freq=50, lr=0.01
    )
    learner = OnlineLearner(decoder=online_decoder, config=online_config)

    drift = DriftSimulator(
        code=code, schedule="gradual_drift", p_base=0.03, seed=seed
    )

    frozen_outcomes = []
    online_outcomes = []

    for t in tqdm(range(n_steps), desc="Gradual drift"):
        noise = drift.get_noise_at_step(t)
        error = noise.sample_errors(code.n_qubits, 1)[0]
        syndrome = code.extract_syndrome(error)

        corr_frozen = frozen_decoder.decode(syndrome)
        res_frozen = _compose_paulis(error, corr_frozen)
        frozen_outcomes.append(int(_is_logical_error(res_frozen, lx, lz)))

        corr_online = online_decoder.decode(syndrome)
        res_online = _compose_paulis(error, corr_online)
        is_err = _is_logical_error(res_online, lx, lz)
        online_outcomes.append(int(is_err))
        learner.update(syndrome, corr_online, is_err)

        if (t + 1) % CHUNK_SIZE == 0:
            _save_intermediate(
                {"frozen": frozen_outcomes, "online": online_outcomes},
                "gradual_drift_partial.json",
            )

    frozen_ler = compute_rolling_ler(frozen_outcomes)
    online_ler = compute_rolling_ler(online_outcomes)

    results = {
        "frozen_ler": frozen_ler,
        "online_ler": online_ler,
        "n_steps": n_steps,
    }

    out_path = os.path.join(RESULTS_DIR, "online_gradual_drift.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved gradual drift results to %s", out_path)

    _plot_gradual_drift(results)
    return results


def _plot_gradual_drift(results):
    """Plot gradual drift comparison figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    steps = range(len(results["frozen_ler"]))
    ax.plot(
        steps, results["frozen_ler"],
        label="Frozen Decoder", color="#de8f05",
        linewidth=1.5, alpha=0.8,
    )
    ax.plot(
        steps, results["online_ler"],
        label="Online Learner", color="#0173b2",
        linewidth=2,
    )
    ax.set_xlabel("Decoding Step", fontsize=14)
    ax.set_ylabel("Logical Error Rate (rolling avg)", fontsize=14)
    ax.set_title(
        "Gradual Noise Drift: Online vs Frozen Decoder",
        fontsize=15, fontweight="bold",
    )
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "online_gradual_drift.png")
    fig.savefig(save_path, dpi=300)
    plt.close("all")
    del fig
    gc.collect()
    logger.info("Saved gradual drift figure to %s", save_path)


# -----------------------------------------------------------------------
# Experiment 3: Convergence Analysis
# -----------------------------------------------------------------------

def run_convergence(code, n_trials=5, n_steps=1000, seed=42):
    """Measure adaptation lag for multiple random switches.

    Parameters
    ----------
    code : SurfaceCode
    n_trials : int
    n_steps : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 3: Convergence Analysis ===")

    decoder = build_decoder(code, "depolarizing", seed=seed)
    analyzer = ConvergenceAnalyzer(code=code)

    # Measure lag with online learning
    online_result = analyzer.measure_adaptation_lag(
        decoder=decoder, n_trials=n_trials, n_steps=n_steps, seed=seed
    )

    # Measure lag without online learning (frozen)
    frozen_decoder = build_decoder(code, "depolarizing", seed=seed)
    frozen_result = analyzer.measure_adaptation_lag(
        decoder=frozen_decoder, n_trials=n_trials, n_steps=n_steps, seed=seed
    )

    # Theoretical bound
    noise_old = DepolarizingNoise(p=0.03)
    noise_new = BitFlipNoise(p=0.03)
    theory = analyzer.theoretical_bound(noise_old, noise_new, lr=0.01)

    results = {
        "online_lag_mean": online_result["lag_steps"],
        "online_lag_std": online_result["lag_std"],
        "online_steady_ler": online_result["steady_state_ler"],
        "frozen_lag_mean": frozen_result["lag_steps"],
        "frozen_lag_std": frozen_result["lag_std"],
        "frozen_steady_ler": frozen_result["steady_state_ler"],
        "theoretical_lag": theory["theoretical_lag"],
        "kl_divergence": theory["kl_divergence"],
    }

    out_path = os.path.join(RESULTS_DIR, "online_convergence.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved convergence results to %s", out_path)

    # Plot
    analyzer.plot_convergence_curves(
        ler_curves_online=online_result["ler_curves"],
        ler_curves_frozen=frozen_result["ler_curves"],
        switch_step=500,
    )

    return results


# -----------------------------------------------------------------------
# Experiment 4: Learning Rate Sensitivity
# -----------------------------------------------------------------------

def run_lr_sweep(code, n_steps=1000, seed=42):
    """Sweep learning rate and measure convergence vs steady-state tradeoff.

    Parameters
    ----------
    code : SurfaceCode
    n_steps : int
    seed : int

    Returns
    -------
    dict
    """
    logger.info("=== Experiment 4: Learning Rate Sweep ===")

    lr_values = np.logspace(-4, -1, 8).tolist()
    convergence_speeds = []
    steady_lers = []

    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]

    for lr in tqdm(lr_values, desc="LR sweep"):
        decoder = build_decoder(code, "depolarizing", seed=seed)
        config = OnlineConfig(
            buffer_size=200, minibatch_size=8, update_freq=50, lr=lr
        )
        learner = OnlineLearner(decoder=decoder, config=config)

        drift = DriftSimulator(
            code=code, schedule="sudden_switch", p_base=0.03, seed=seed
        )

        outcomes = []
        for t in range(n_steps):
            noise = drift.get_noise_at_step(t)
            error = noise.sample_errors(code.n_qubits, 1)[0]
            syndrome = code.extract_syndrome(error)
            correction = decoder.decode(syndrome)
            residual = _compose_paulis(error, correction)
            is_err = _is_logical_error(residual, lx, lz)
            outcomes.append(int(is_err))
            learner.update(syndrome, correction, is_err)

        ler_curve = compute_rolling_ler(outcomes)

        # Convergence speed: steps after switch until LER below threshold
        post_switch = ler_curve[500:]
        if len(post_switch) > 50:
            steady = float(np.mean(post_switch[-50:]))
            steady_lers.append(steady)
            threshold = steady * 1.1
            speed = n_steps - 500  # default: never converged
            for i, val in enumerate(post_switch):
                if val <= threshold:
                    speed = i
                    break
            convergence_speeds.append(speed)
        else:
            convergence_speeds.append(n_steps)
            steady_lers.append(float(np.mean(ler_curve)))

        gc.collect()

    results = {
        "lr_values": lr_values,
        "convergence_speeds": convergence_speeds,
        "steady_state_lers": steady_lers,
    }

    out_path = os.path.join(RESULTS_DIR, "online_lr_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved LR sweep results to %s", out_path)

    _plot_lr_sweep(results)
    return results


def _plot_lr_sweep(results):
    """Plot LR sensitivity figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    ax1.semilogx(
        results["lr_values"], results["convergence_speeds"],
        "o-", color="#0173b2", linewidth=2, markersize=7,
    )
    ax1.set_xlabel("Learning Rate η", fontsize=14)
    ax1.set_ylabel("Convergence Speed (steps)", fontsize=14)
    ax1.set_title("Convergence Speed vs η", fontsize=13, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", alpha=0.1)

    ax2.semilogx(
        results["lr_values"], results["steady_state_lers"],
        "s-", color="#de8f05", linewidth=2, markersize=7,
    )
    ax2.set_xlabel("Learning Rate η", fontsize=14)
    ax2.set_ylabel("Steady-State LER", fontsize=14)
    ax2.set_title("Steady-State LER vs η", fontsize=13, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.minorticks_on()
    ax2.grid(True, which="minor", linestyle=":", alpha=0.1)

    fig.suptitle(
        "Learning Rate Sensitivity Analysis",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "online_lr_sweep.png")
    fig.savefig(save_path, dpi=300)
    plt.close("all")
    del fig
    gc.collect()
    logger.info("Saved LR sweep figure to %s", save_path)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _save_intermediate(data, filename):
    """Save intermediate results to disk."""
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run online learning experiments"
    )
    parser.add_argument("--d", type=int, default=3, help="Code distance")
    parser.add_argument(
        "--steps", type=int, default=2000, help="Steps per experiment"
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Trials for convergence"
    )
    args = parser.parse_args()

    setup_dirs()

    logger.info("Setting up code d=%d", args.d)
    code = SurfaceCode(d=args.d)

    run_sudden_switch(code, n_steps=args.steps, seed=42)
    run_gradual_drift(code, n_steps=args.steps, seed=42)
    run_convergence(code, n_trials=args.trials, n_steps=1000, seed=42)
    run_lr_sweep(code, n_steps=1000, seed=42)

    logger.info("All online learning experiments complete!")


if __name__ == "__main__":
    main()
