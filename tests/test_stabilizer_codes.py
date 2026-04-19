import numpy as np
import pytest

from src.stabilizer_codes import RepetitionCode, SurfaceCode

def test_repetition_code_distance_3():
    code = RepetitionCode(d=3)
    assert code.n_qubits == 3
    assert code.n_stabilizers == 2
    
    H = code.get_stabilizers()
    assert H.shape == (2, 6)
    
    # Test syndrome extraction for X error on middle qubit (idx 1)
    # Quibit 1 is in both weight-2 stabilizers Z0Z1 and Z1Z2
    error = np.array([0, 1, 0])
    syndrome = code.extract_syndrome(error)
    assert np.array_equal(syndrome, np.array([1, 1]))
    
    # Logical ops test
    l_ops = code.get_logical_ops()
    assert np.array_equal(l_ops["X"], np.array([1, 1, 1]))

def test_surface_code_distance_3():
    code = SurfaceCode(d=3)
    assert code.n_qubits == 9
    assert code.n_stabilizers == 8
    
    H = code.get_stabilizers()
    assert H.shape == (8, 18)
    
    # Z error in middle should trigger two adjacent X stabilizers
    error = np.zeros(9, dtype=int)
    error[4] = 3 # Z error in center
    syndrome = code.extract_syndrome(error)
    assert np.sum(syndrome) == 2 # 2 matching stabilizers
    
    l_ops = code.get_logical_ops()
    # Weight d logical operators
    assert np.sum(l_ops["X"] != 0) == 3 
    assert np.sum(l_ops["Z"] != 0) == 3

def test_surface_code_greedy_decoder():
    code = SurfaceCode(d=3)
    syndrome = np.zeros(8, dtype=int)
    
    # Fake syndrome indicating an error we can greedily decode
    # The middle X stabilizer (index might vary based on construction, let's just test it returns *some* correction)
    syndrome[0] = 1 
    
    correction = code.decode_syndrome(syndrome)
    assert len(correction) == 9
    assert np.sum(correction) > 0 # It tried to correct
