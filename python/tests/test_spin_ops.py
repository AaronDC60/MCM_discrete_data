import numpy as np
import pytest
from python.utils import spin_ops

def test_complete_model():
    with pytest.raises(TypeError):
        # n must be an integer
        spin_ops.generate_all_ops(3.5)
    with pytest.raises(TypeError):
        # q must be an integer
        spin_ops.generate_all_ops(3, 2.5)
    # All operators base 2
    assert(np.all(spin_ops.generate_all_ops(3) == range(1, 2**3)))
    # All operators base 3
    assert(np.all(spin_ops.generate_all_ops(2, 3) == range(1, 3**2)))

def test_model_upto_order_n():
    with pytest.raises(TypeError):
        # Invalid argument for n_inter
        spin_ops.generate_ops_upto_order_n(2, 3.5)
    with pytest.raises(TypeError):
        # Invalid argument for n_var
        spin_ops.generate_ops_upto_order_n(3.5, 2)
    with pytest.raises(ValueError):
        # n_inter > n_var
        spin_ops.generate_ops_upto_order_n(2, 3)
    # 3 variables only empty operator
    assert(len(spin_ops.generate_ops_upto_order_n(3, 0)) == 0)
    # 3 variables only pairwise interactions
    assert(np.all(spin_ops.generate_ops_upto_order_n(3, 2) == [1, 2, 3, 4, 5, 6]))
    # 4 variables only single-body terms
    assert(np.all(spin_ops.generate_ops_upto_order_n(4, 1) == [1, 2, 4, 8]))

def test_construct_s_matrix():
    w = np.exp(2j * np.pi / 3)     
    with pytest.raises(TypeError):
        # n must be an integer
        spin_ops.construct_s_matrix(3.5,2)
    with pytest.raises(TypeError):
        # q must be an integer
        spin_ops.construct_s_matrix(3, 2.5)
    # 1 spin, 2 states
    assert(np.allclose(spin_ops.construct_s_matrix(1,2), np.array([[1,1], [1,-1]])))
    # 2 spins, 2 states
    assert(np.allclose(spin_ops.construct_s_matrix(2,2), np.array([[1,1,1,1], [1,-1,1,-1], [1,1,-1,-1], [1,-1,-1,1]])))
    # 1 spin, 3 states
    assert(np.allclose(spin_ops.construct_s_matrix(1,3), np.array([[1,1,1], [1,w,w**2], [1,w**2, w]])))
    # 2 spins, 3 states
    assert(np.allclose(spin_ops.construct_s_matrix(2,3), np.array([[1,1,1,1,1,1,1,1,1],[1,w,w**2,1,w,w**2,1,w,w**2],[1,w**2,w,1,w**2,w,1,w**2,w],
                                                                   [1,1,1,w,w,w,w**2,w**2,w**2],[1,w,w**2,w,w**2,1,w**2,1,w],[1,w**2,w,w,1,w**2,w**2,w,1],
                                                                   [1,1,1,w**2,w**2,w**2,w,w,w],[1,w,w**2,w**2,1,w,w,w**2,1],[1,w**2,w,w**2,w,1,w,1,w**2]])))
    # 1 spin, 4 states
    assert(np.allclose(spin_ops.construct_s_matrix(1,4), np.array([[1,1,1,1],[1,1j,-1,-1j],[1,-1,1,-1],[1,-1j,-1,1j]])))

def test_get_spin_op_indices():
    with pytest.raises(TypeError):
        # q must be an integer
        spin_ops.get_spin_op_indices(2.5, [0,1], [1,1])
    with pytest.raises(TypeError):
        # vars must be an array
        spin_ops.get_spin_op_indices(2, 1, [1,1])
    with pytest.raises(TypeError):
        # color must be an array
        spin_ops.get_spin_op_indices(2, [0,1], 1)
    with pytest.raises(ValueError):
        # vars and color must have the same length
        spin_ops.get_spin_op_indices(2, [0,1,2], [1,1])
    with pytest.raises(ValueError):
        # values of color must be between 1 and q-1
        spin_ops.get_spin_op_indices(2, [0,1], [0,1])
    with pytest.raises(ValueError):
        # values of color must be between 1 and q-1
        spin_ops.get_spin_op_indices(2, [0,1], [1,2])

    # 2 States
        
    # First-order interaction
    assert(spin_ops.get_spin_op_indices(2, [0], [1]) == (1,1))
    assert(spin_ops.get_spin_op_indices(2, [1], [1]) == (2,2))
    assert(spin_ops.get_spin_op_indices(2, [2], [1]) == (4,4))
    assert(spin_ops.get_spin_op_indices(2, [3], [1]) == (8,8))
    # Second-order interaction
    assert(spin_ops.get_spin_op_indices(2, [0,1], [1,1]) == (3,3))
    assert(spin_ops.get_spin_op_indices(2, [0,2], [1,1]) == (5,5))
    assert(spin_ops.get_spin_op_indices(2, [1,2], [1,1]) == (6,6))
    # Third-order interaction
    assert(spin_ops.get_spin_op_indices(2, [0,1,2], [1,1,1]) == (7,7))

    # 3 States

    # First-order interaction
    assert(spin_ops.get_spin_op_indices(3, [0], [1]) == (1,2))
    assert(spin_ops.get_spin_op_indices(3, [0], [2]) == (2,1))
    assert(spin_ops.get_spin_op_indices(3, [1], [1]) == (3,6))
    assert(spin_ops.get_spin_op_indices(3, [2], [1]) == (9,18))
    # Second-order interaction
    assert(spin_ops.get_spin_op_indices(3, [0,1], [1,1]) == (4,8))
    assert(spin_ops.get_spin_op_indices(3, [0,1], [2,2]) == (8,4))
    assert(spin_ops.get_spin_op_indices(3, [0,1], [1,2]) == (7,5))
    assert(spin_ops.get_spin_op_indices(3, [0,1], [2,1]) == (5,7))
    assert(spin_ops.get_spin_op_indices(3, [0,2], [1,2]) == (19,11))
    # Third-order interaction
    assert(spin_ops.get_spin_op_indices(3, [0,1,2], [1,1,1]) == (13,26))
    assert(spin_ops.get_spin_op_indices(3, [0,1,2], [1,1,2]) == (22,17))

    # 4 States

    # First-order interaction
    assert(spin_ops.get_spin_op_indices(4, [0], [1]) == (1,3))
    assert(spin_ops.get_spin_op_indices(4, [0], [2]) == (2,2))
    assert(spin_ops.get_spin_op_indices(4, [0], [3]) == (3,1))
    assert(spin_ops.get_spin_op_indices(4, [1], [1]) == (4,12))
    assert(spin_ops.get_spin_op_indices(4, [1], [2]) == (8,8))
    assert(spin_ops.get_spin_op_indices(4, [1], [3]) == (12,4))
    # Second-order interaction
    assert(spin_ops.get_spin_op_indices(4, [0,1], [1,1]) == (5,15))
    assert(spin_ops.get_spin_op_indices(4, [0,1], [1,2]) == (9,11))
    assert(spin_ops.get_spin_op_indices(4, [0,1], [1,3]) == (13,7))
