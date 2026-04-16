import numpy as np
import pennylane as qml
import pytest

from src.noise_models import (
    DepolarizingNoise,
    BitFlipNoise,
    PhaseFlipNoise,
    CombinedNoise,
    AmplitudeDamping,
    create_noise_model
)

def test_depolarizing_noise():
    noise = DepolarizingNoise(p=0.1)
    errors = noise.sample_errors(n_qubits=5, n_shots=1000)
    
    assert errors.shape == (1000, 5)
    
    # Check probabilities rough estimate for first qubit
    q0_errs = errors[:, 0]
    p_err = np.mean(q0_errs != 0)
    assert 0.05 < p_err < 0.15 # should be around 0.1

def test_bit_flip_noise():
    noise = BitFlipNoise(p=0.2)
    errors = noise.sample_errors(n_qubits=3, n_shots=500)
    
    assert errors.shape == (500, 3)
    # Bit flips should only produce 0 (I) or 1 (X)
    assert set(np.unique(errors)).issubset({0, 1})

def test_phase_flip_noise():
    noise = PhaseFlipNoise(p=0.2)
    errors = noise.sample_errors(n_qubits=3, n_shots=500)
    
    assert errors.shape == (500, 3)
    # Phase flips should only produce 0 (I) or 3 (Z)
    assert set(np.unique(errors)).issubset({0, 3})

def test_create_noise_model_factory():
    noise = create_noise_model("depolarizing", p=0.05)
    assert isinstance(noise, DepolarizingNoise)
    assert noise.p == 0.05

    noise = create_noise_model("combined", p_bit=0.01, p_phase=0.02)
    assert isinstance(noise, CombinedNoise)
    assert noise.p_bit == 0.01
    assert noise.p_phase == 0.02

def test_pennylane_integration():
    noise = DepolarizingNoise(p=0.1)
    
    with qml.tape.QuantumTape() as tape:
        qml.RX(0.1, wires=0)
        noise.apply(tape, qubits=[0, 1])
        
    assert len(tape.operations) == 3 # RX + 2 Depolarizing channels
