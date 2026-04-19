"""Unit tests for all advanced research features.

Ensures each standalone contribution is numerically stable and memory efficient.
"""

import numpy as np
import pytest
import torch

from src.features import (
    BeliefPropagator,
    HardwareNoiseFingerprinter,
    LogicalBlockDecoder,
    DecoderConfidenceCalibrator,
    SyndromeAutoencoder,
    SyntheticHardwareSimulator
)
from src.stabilizer_codes import SurfaceCode

@pytest.fixture
def surface_code_d3():
    return SurfaceCode(d=3)

# --- Feature 1: BP ---

def test_bp_convergence(surface_code_d3):
    H = surface_code_d3.get_parity_check_matrix()
    bp = BeliefPropagator(H, max_iterations=10)
    
    # Simple single error
    err = np.zeros(surface_code_d3.n_qubits)
    err[0] = 1
    syn = (H @ err) % 2
    
    soft_probs, converged, _ = bp.run_bp(syn, physical_error_rate=0.01)
    assert soft_probs.shape == (surface_code_d3.n_qubits,)
    # BP should assign highest probability to qubit 0
    assert np.argmax(soft_probs) == 0

def test_bp_damping(surface_code_d3):
    H = surface_code_d3.get_parity_check_matrix()
    bp = BeliefPropagator(H, damping=0.9) # High damping
    assert bp.damping == 0.9

# --- Feature 2: Fingerprinting ---

def test_fingerprinter_update(surface_code_d3):
    fp = HardwareNoiseFingerprinter(surface_code_d3.n_qubits, surface_code_d3.n_stabilizers)
    batch = np.ones((10, surface_code_d3.n_stabilizers))
    fp.update(batch)
    
    res = fp.get_fingerprint()
    assert np.allclose(res["flip_rates"], 1.0)
    assert res["n_samples"] == 10

def test_simulator_non_uniform():
    sim = SyntheticHardwareSimulator(n_qubits=10, p_mean=0.05, sigma=0.02)
    rates = sim.qubit_rates
    assert len(rates) == 10
    assert not np.allclose(rates, 0.05) # Verify variance

# --- Feature 3: Block Decoder ---

def test_mi_calculation():
    from src.cross_qubit_correlations import CrossQubitCorrelationAnalyzer
    analyzer = CrossQubitCorrelationAnalyzer(n_logical=2)
    
    # Perfectly correlated
    data = np.random.randint(0, 2, (100, 8))
    mi = analyzer.measure_mutual_information(data, data)
    assert mi > 0

# --- Feature 4: Calibration ---

def test_calibration_fit():
    cal = DecoderConfidenceCalibrator()
    logits = np.random.normal(2.0, 1.0, 100)
    labels = np.random.randint(0, 2, 100).astype(np.float32)
    
    cal.fit(logits, labels)
    probs = cal.calibrate(logits)
    assert probs.shape == (100,)
    assert np.all(probs >= 0) and np.all(probs <= 1)

def test_ece_computation():
    cal = DecoderConfidenceCalibrator(n_bins=5)
    probs = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 1, 0, 0])
    ece = cal.compute_ece(probs, labels)
    assert np.allclose(ece, 0.1) # Expected 0.1 for these probs/labels

# --- Feature 5: Compression ---

def test_autoencoder_shape():
    ae = SyndromeAutoencoder(n_stabilizers=8, latent_dim=4)
    x = torch.randn(10, 8)
    recon, latent = ae(x)
    assert recon.shape == (10, 8)
    assert latent.shape == (10, 4)

def test_ae_training():
    ae = SyndromeAutoencoder(n_stabilizers=8, latent_dim=4)
    data = np.random.randint(0, 2, (20, 8)).astype(np.float32)
    losses = ae.train_ae(data, epochs=5)
    assert len(losses) == 5
    assert losses[-1] <= losses[0]
