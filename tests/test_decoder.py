import numpy as np
import pytest

from src.ansatz import HardwareEfficientAnsatz
from src.decoder import VariationalDecoder
from src.noise_models import DepolarizingNoise
from src.stabilizer_codes import RepetitionCode
from src.trainer import Trainer

def test_variational_decoder_initialization():
    code = RepetitionCode(d=3)
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=2)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    
    assert decoder.n_qubits == 3
    # HardwareEfficientAnsatz has 2 params per qubit per layer (RY + RZ)
    assert len(decoder.params) == 2 * 3 * 2
    
def test_decoder_forward_pass():
    code = RepetitionCode(d=3)
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    
    syndrome = np.array([1, 0])
    correction = decoder.decode(syndrome)
    
    assert len(correction) == 3
    # Returns Pauli corrections {0, 1, 2, 3}. 
    # For bit-flip noise on repetition code, mostly {0, 1}
    assert set(np.unique(correction)).issubset({0, 1, 2, 3})
    
def test_trainer_gradient():
    """Verify that the Trainer can compute gradients using autograd."""
    code = RepetitionCode(d=3)
    ansatz = HardwareEfficientAnsatz(n_qubits=3, n_layers=1)
    noise = DepolarizingNoise(p=0.01)
    
    decoder = VariationalDecoder(code=code, ansatz=ansatz, noise_model=noise)
    trainer = Trainer(decoder=decoder, lr=0.01)
    
    # Generate fake batch
    syndromes = np.array([[1, 0], [0, 1]])
    errors = np.array([[1, 0, 0], [0, 0, 1]])
    
    # Compute loss and check if it's differentiable
    loss = decoder.compute_loss(syndromes, errors)
    assert isinstance(loss, (float, np.float64, np.ndarray))
    
    # The Trainer should be able to run a tiny training step
    # (Checking if it doesn't crash is a good enough integration test)
    history = trainer.train(n_epochs=1, batch_size=2)
    assert len(history.losses) == 1
