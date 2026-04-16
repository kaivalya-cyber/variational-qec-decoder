# Variational Quantum Error Correction Decoder

[![Python Tests](https://github.com/kaivalya-cyber/variational-qec-decoder/actions/workflows/python-tests.yml/badge.svg?branch=main)](https://github.com/kaivalya-cyber/variational-qec-decoder/actions/workflows/python-tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

This repository provides a production-grade Python implementation of a Variational QEC Decoder, focusing on adaptive noise-aware decoding strategies. It integrates classical machine learning (CNNs for noise classification) with parameterized quantum circuits (ansätze) via PennyLane and PyTorch to dynamically adapt to varying physical noise channels.

## Project Description

Quantum error correction (QEC) is essential for fault-tolerant quantum computing. Traditional decoders like Minimum Weight Perfect Matching (MWPM) are highly effective but often assume a static, well-characterized noise model (e.g., pure depolarizing noise). 

This project explores a novel hybrid approach:
1. **Syndrome History Analysis**: A classical CNN analyzes the history of syndrome measurements to classify the dominant noise channel (depolarizing, bit-flip, phase-flip, or combined) on the fly.
2. **Adaptive Variational Decoding**: Once the noise type is identified, the system dynamically swaps to a noise-specific variational ansatz designed to correct that specific error profile optimally.
3. **Quantum-Aware Training**: The ansätze are trained using the parameter-shift rule directly on the abstract quantum device via PennyLane, using cross-entropy against logical error bounds.

### Novel Contributions
* **`noise_classifier.py`**: A 1D-CNN tracking syndrome signatures over time.
* **`adaptive_selector.py`**: A dynamic decoder bank that switches the underlying decoding strategy mid-operation based on classical heuristics.

## Installation

We recommend using a conda or virtual environment for Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/variational-qec/variational_qec_decoder.git
cd variational_qec_decoder

# Create and activate environment 
python -m venv vqec_env
source vqec_env/bin/activate

# Install requirements
pip install -r requirements.txt
pip install -e .
```

*Note: The MWPM baseline requires `pymatching>=2.0` and fast stabilizer simulation requires `stim>=1.12.*`

## Usage and Experiments

All experiments are located in the `experiments/` directory. Outputs and models are saved to `results/data/`, `results/figures/`, and `results/models/`.

### 1. Classical Baselines
Train and evaluate MWPM and Lookup Table decoders.
```bash
python experiments/run_baseline.py --d 3 --shots 1000 --noise depolarizing
```

### 2. Variational Decoder
Train a fixed variational decoder using hardware-efficient or symmetry-preserving ansätze.
```bash
python experiments/run_variational.py --d 3 --ansatz hardware_efficient --epochs 50
```

### 3. Adaptive Noise-Aware Decoding (Novel Experiment)
Run the full adaptive pipeline in a dynamically changing noise environment. This experiment trains the CNN classifier, initializes the decoder bank, and evaluates logical error rates across fluctuating noise contexts.
```bash
python experiments/run_adaptive.py
```

### 4. Plotting Results
Generate figures combining data from all completed runs.
```bash
python experiments/plot_results.py
```

## Citation

If you use this codebase in your research, please cite the framework using the following placeholder (details to be updated upon publication):

```bibtex
@misc{VQEC2026,
  author = {Kaivalya Sinhg},
  title = {Adaptive Noise-Aware Variational Decoding for Quantum Error Correction},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/variational-qec/variational_qec_decoder}}
}
```
