import numpy as np
import pytest
from src import utils

def test_fwht():
    a = np.array([1,0,1,0,0,1,1,0])
    assert(np.all(utils.fwht(a) == [4,2,0,-2,0,2,0,2]))

def test_complete_model():
    with pytest.raises(TypeError):
        utils.generate_all_ops(3.5)
    assert(np.all(utils.generate_all_ops(3) == range(1, 3**2 - 1)))

def test_model_upto_order_n():
    with pytest.raises(TypeError):
        # Invalid argument for n_inter
        utils.generate_ops_upto_order_n(2, 3.5)
    with pytest.raises(TypeError):
        # Invalid argument for n_var
        utils.generate_ops_upto_order_n(3.5, 2)
    with pytest.raises(ValueError):
        # n_inter > n_var
        utils.generate_ops_upto_order_n(2, 3)
    # 3 variables only empty operator
    assert(len(utils.generate_ops_upto_order_n(3, 0)) == 0)
    # 3 variables only pairwise interactions
    assert(np.all(utils.generate_ops_upto_order_n(3, 2) == [1, 2, 3, 4, 5, 6]))
    # 4 variables only single-body terms
    assert(np.all(utils.generate_ops_upto_order_n(4, 1) == [1, 2, 4, 8]))

def test_partitions():
    all_partitions = []
    utils.generate_partitions(0, 3, [],  all_partitions)
    assert(len(all_partitions) == 5)
    assert(np.all(all_partitions[0] == [[0,1,2]]))
    assert(np.all(all_partitions[1] == [[0,1],[2]]))
    assert(np.all(all_partitions[2] == [[0,2],[1]]))
    assert(np.all(all_partitions[3] == [[0],[1,2]]))
    assert(np.all(all_partitions[4] == [[0],[1],[2]]))
