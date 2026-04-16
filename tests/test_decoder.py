import numpy as np
import pytest

from src.ansatz import HardwareEfficientAnsatz
from src.decoder import VariationalDecoder
from src.noise_models import DepolarizingNoise
from src.stabilizer_codes import RepetitionCode

def test_variational_decoder_initialization():
    code = RepetitionCode(d=3)
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    
    assert decoder.n_qubits == 3
    assert len(decoder.params) == 12 # 2 layers * 3 qubits * 2 per qubit (RY+RZ)

def test_decoder_forward_pass():
    code = RepetitionCode(d=3)
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    
    syndrome = np.array([1, 0])
    correction = decoder.decode(syndrome)
    
    assert len(correction) == 3
    assert set(np.unique(correction)).issubset({0, 1}) # Returns X corrections for now
    
def test_decoder_parameter_shift():
    code = RepetitionCode(d=3)
    # Minimal ansatz to test gradients quickly
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    
    # Fake batch
    syndromes = np.array([[1, 0], [0, 1]])
    errors = np.array([[1, 0, 0], [0, 0, 1]])
    
    grads = decoder.parameter_shift_gradient(syndromes, errors)
    
    assert len(grads) == ansatz.n_params
    assert not np.all(grads == 0.0) # Should have some non-zero gradients
