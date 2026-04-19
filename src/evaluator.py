"""Evaluation utilities for QEC decoders.

Provides threshold scanning, decoder comparison, plotting, and
wall-clock overhead benchmarking.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .noise_models import NoiseModel, create_noise_model
from .stabilizer_codes import StabilizerCode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIGURES_DIR: str = "results/figures"
DEFAULT_N_SHOTS: int = 1000
PLOT_STYLE: str = "seaborn-v0_8-whitegrid"

# Ensure output directory exists
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Threshold scan
# ---------------------------------------------------------------------------

def threshold_scan(
    decoder: Any,
    code: StabilizerCode,
    noise_model_factory: Callable[[float], NoiseModel],
    p_values: np.ndarray,
    n_shots_per_p: int = DEFAULT_N_SHOTS,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Scan logical error rate vs. physical error rate.

    Parameters
    ----------
    decoder : object
        A decoder with a ``decode(syndrome)`` method.
    code : StabilizerCode
        The QEC code.
    noise_model_factory : callable
        Function ``p -> NoiseModel`` for varying the error rate.
    p_values : np.ndarray
        Array of physical error rates to scan.
    n_shots_per_p : int
        Number of Monte Carlo shots per error rate.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``{'p_values': array, 'logical_error_rates': array}``.
    """
    logical_error_rates = np.zeros(len(p_values))
    logical_ops = code.get_logical_ops()
    lx, lz = logical_ops["X"], logical_ops["Z"]

    for idx, p in enumerate(p_values):
        noise = noise_model_factory(p)
        errors = noise.sample_errors(code.n_qubits, n_shots_per_p, seed=seed + idx)
        n_logical_errors = 0

        for i in range(n_shots_per_p):
            syndrome = code.extract_syndrome(errors[i])
            correction = decoder.decode(syndrome)
            residual = _compose_paulis(errors[i], correction)
            if _is_logical_error(residual, lx, lz):
                n_logical_errors += 1

        logical_error_rates[idx] = n_logical_errors / n_shots_per_p
        logger.info(
            "p=%.4f: logical error rate = %.6f (%d/%d)",
            p,
            logical_error_rates[idx],
            n_logical_errors,
            n_shots_per_p,
        )

    return {"p_values": p_values, "logical_error_rates": logical_error_rates}


# ---------------------------------------------------------------------------
# Decoder comparison
# ---------------------------------------------------------------------------

def compare_decoders(
    decoders_dict: Dict[str, Any],
    code: StabilizerCode,
    noise_model_factory: Callable[[float], NoiseModel],
    p_values: np.ndarray,
    n_shots: int = DEFAULT_N_SHOTS,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Benchmark multiple decoders across noise levels.

    Parameters
    ----------
    decoders_dict : dict
        ``{name: decoder}`` mapping.
    code : StabilizerCode
        The QEC code.
    noise_model_factory : callable
        ``p -> NoiseModel``.
    p_values : np.ndarray
        Physical error rates.
    n_shots : int
        Shots per error rate per decoder.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``{decoder_name: {'p_values': array, 'logical_error_rates': array}}``.
    """
    results = {}
    for name, decoder in decoders_dict.items():
        logger.info("Evaluating decoder: %s", name)
        results[name] = threshold_scan(
            decoder, code, noise_model_factory, p_values, n_shots, seed
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_threshold_curve(
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "Logical Error Rate Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
    thresholds: Optional[List[float]] = None,
) -> plt.Figure:
    """Plot threshold curves for multiple decoders with professional styling.

    Parameters
    ----------
    results : dict
        Output of ``compare_decoders``.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot.
    thresholds : list of float, optional
        List of threshold values to plot as vertical lines.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Colorblind safe palette
    # Blue: Adaptive, Orange: HEA, Black: MWPM, Purple: Lookup
    color_map = {
        "Adaptive": "#0173b2",
        "Variational": "#de8f05",
        "MWPM": "black",
        "LookupTable": "#cc78bc",
        "p_L = p": "gray"
    }
    
    def get_color(name):
        for k, v in color_map.items():
            if k in name: return v
        return None

    # Sort results to have baselines first for better layering
    sorted_names = sorted(results.keys(), key=lambda x: ("MWPM" in x, "Lookup" in x), reverse=True)
    
    active_curves = {}
    
    for name in sorted_names:
        data = results[name]
        p_vals = np.array(data["p_values"])
        ler_vals = np.array(data["logical_error_rates"])
        
        # Filter out zero LER values before plotting on log scale
        valid_idx = ler_vals > 0
        p_plot = p_vals[valid_idx]
        ler_plot = ler_vals[valid_idx]

        if len(p_plot) == 0:
            continue

        color = get_color(name)
        ls = "--" if "Lookup" in name else "-"
        m = "o" if "Adaptive" in name else "s" if "Variational" in name else "D"
        lw = 2.5 if "Adaptive" in name or "Variational" in name else 1.5
        z = 10 if "Adaptive" in name else 5

        line, = ax.semilogy(
            p_plot,
            ler_plot,
            label=name,
            color=color,
            linestyle=ls,
            marker=m,
            linewidth=lw,
            markersize=7,
            zorder=z,
            alpha=0.9
        )
        active_curves[name] = (p_plot, ler_plot, color)

    # Shaded region between Adaptive and HEA if both exist for same distance
    adaptive_keys = [k for k in results.keys() if "Adaptive" in k]
    hea_keys = [k for k in results.keys() if "Variational" in k]
    
    for a_key in adaptive_keys:
        # Extract distance string (e.g., 'd3') from key
        d_str = a_key.split("_")[-1]
        for h_key in hea_keys:
            if d_str in h_key and a_key in active_curves and h_key in active_curves:
                p_a, ler_a, _ = active_curves[a_key]
                p_h, ler_h, _ = active_curves[h_key]
                # Assuming p_values are aligned
                ax.fill_between(p_a, ler_a, ler_h, color=color_map["Adaptive"], alpha=0.1, zorder=1)

    # Reference line: p_logical = p_physical
    p_ref = list(results.values())[0]["p_values"]
    ax.semilogy(p_ref, p_ref, "--", color=color_map["p_L = p"], alpha=0.6, label="$p_L = p$", zorder=0)

    # Add vertical lines for thresholds if provided
    if thresholds:
        for th in thresholds:
            if 0 < th <= 0.11:
                ax.axvline(x=th, color="red", linestyle=":", alpha=0.5, linewidth=1.5)
                ax.text(th + 0.002, ax.get_ylim()[1]*0.7, f"$p_{{th}}={th}$", 
                        rotation=90, color="red", alpha=0.7, fontsize=11, fontweight="bold")

    ax.set_xlabel("Physical Error Rate (p)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Logical Error Rate ($p_L$)", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    
    ax.set_xlim(0, 0.11)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.1)
    
    ax.legend(frameon=True, fontsize=10, loc="lower right", shadow=True)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "combined_thresholds.png")
    fig.savefig(save_path, dpi=300)
    logger.info("Saved enhanced threshold plot to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training metrics.

    Parameters
    ----------
    history : dict
        Training history dictionary with keys 'losses',
        'logical_error_rates', 'gradient_norms'.
    save_path : str, optional
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["losses"]) + 1)

    # Loss
    axes[0].plot(epochs, history["losses"], linewidth=2, color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

    # Logical error rate
    axes[1].semilogy(
        epochs, history["logical_error_rates"], linewidth=2, color="tab:red"
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Logical Error Rate")
    axes[1].set_title("Logical Error Rate")

    # Gradient norms
    axes[2].plot(epochs, history["gradient_norms"], linewidth=2, color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("||∇||")
    axes[2].set_title("Gradient Norm")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "training_history.png")
    fig.savefig(save_path, dpi=150)
    logger.info("Saved training history plot to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# Decoder overhead
# ---------------------------------------------------------------------------

def compute_decoder_overhead(
    decoder: Any,
    code: StabilizerCode,
    noise_model: NoiseModel,
    n_shots: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Measure wall-clock decoding time per shot.

    Parameters
    ----------
    decoder : object
        Decoder with a ``decode`` method.
    code : StabilizerCode
        The QEC code.
    noise_model : NoiseModel
        Noise model for generating test syndromes.
    n_shots : int
        Number of decoding calls to average over.
    seed : int
        Random seed.

    Returns
    -------
    dict
        ``{'mean_time_ms': float, 'std_time_ms': float,
          'total_time_s': float, 'n_shots': int}``.
    """
    errors = noise_model.sample_errors(code.n_qubits, n_shots, seed=seed)

    times: List[float] = []
    for i in range(n_shots):
        syndrome = code.extract_syndrome(errors[i])
        t0 = time.perf_counter()
        _ = decoder.decode(syndrome)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms

    result = {
        "mean_time_ms": float(np.mean(times)),
        "std_time_ms": float(np.std(times)),
        "total_time_s": float(np.sum(times) / 1000.0),
        "n_shots": n_shots,
    }
    logger.info(
        "Decoder overhead: %.3f ± %.3f ms per decode (%d shots)",
        result["mean_time_ms"],
        result["std_time_ms"],
        n_shots,
    )
    return result


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _compose_paulis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose two Pauli vectors.

    Parameters
    ----------
    a, b : np.ndarray

    Returns
    -------
    np.ndarray
    """
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


def _is_logical_error(residual: np.ndarray, lx: np.ndarray, lz: np.ndarray) -> bool:
    """Check if residual corresponds to a non-trivial logical error.
    
    A residual error is logical if it anticommutes with the logical operators.
    In the symplectic basis, commutation is (x1*z2 + z1*x2) % 2.
    """
    res_x = np.isin(residual, [1, 2]).astype(int)
    res_z = np.isin(residual, [2, 3]).astype(int)
    log_x = np.isin(lx, [1, 2]).astype(int)
    log_z = np.isin(lz, [2, 3]).astype(int)

    # Check if residual anticommutes with logical Z (signifies X-type logical error)
    # Check if residual anticommutes with logical X (signifies Z-type logical error)
    comm_x = np.sum(res_x * log_z + res_z * np.isin(lz, [1, 2]).astype(int)) % 2
    comm_z = np.sum(res_x * np.isin(lx, [2, 3]).astype(int) + res_z * log_x) % 2
    
    return bool(comm_x or comm_z)


# ---------------------------------------------------------------------------
# Extended comparison (includes continuous selector and online learner)
# ---------------------------------------------------------------------------

def compare_all_decoders(
    decoders_dict: Dict[str, Any],
    code: StabilizerCode,
    noise_model_factory: Callable[[float], NoiseModel],
    p_values: np.ndarray,
    n_shots: int = DEFAULT_N_SHOTS,
    seed: int = 42,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Benchmark all decoders including continuous and online variants.

    This extends ``compare_decoders`` to include continuous selector and
    online learner results alongside existing decoders.

    Parameters
    ----------
    decoders_dict : dict
        ``{name: decoder}`` mapping. Decoders must have a ``decode(syndrome)``
        method.  Continuous selectors should have ``decode_continuous`` and
        online learners should have a ``decoder`` attribute.
    code : StabilizerCode
        The QEC code.
    noise_model_factory : callable
        ``p -> NoiseModel``.
    p_values : np.ndarray
        Physical error rates to scan.
    n_shots : int
        Shots per error rate.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``{decoder_name: {'p_values': array, 'logical_error_rates': array}}``.
    """
    results = {}
    for name, decoder in decoders_dict.items():
        logger.info("Evaluating decoder: %s", name)
        results[name] = threshold_scan(
            decoder, code, noise_model_factory, p_values, n_shots, seed
        )
    return results


def plot_combined_figure(
    results: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate final combined paper figure with all decoder variants.

    Shows standard decoders, continuous selector, and online learner
    on the same threshold plot for direct comparison.

    Parameters
    ----------
    results : dict
        Output of ``compare_all_decoders``.
    save_path : str, optional
        Output path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import gc

    if save_path is None:
        save_path = os.path.join(FIGURES_DIR, "combined_all_decoders.png")

    fig = plot_threshold_curve(
        results,
        title="All Decoders: Logical Error Rate Comparison",
        save_path=save_path,
    )

    plt.close("all")
    del fig
    gc.collect()

    logger.info("Saved combined all-decoders figure to %s", save_path)
    return None
