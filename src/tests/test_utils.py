import numpy as np
import pytest
from src import utils

def test_fwht():
    a = np.array([1,0,1,0,0,1,1,0])
    assert(np.all(utils.fwht(a) == [4,2,0,-2,0,2,0,2]))

def test_gt_state():
    # sigma1 = s1
    # sigma2 = s1, s2
    # sigma3 = s1, s2, s3
    gt = [1,3,7]
    assert(utils.gt_state(0, gt) == 0)
    assert(utils.gt_state(4, gt) == 4)
    assert(utils.gt_state(5, gt) == 3)
    assert(utils.gt_state(7, gt) == 5)

def test_gt_model():
    # s1 = sigma1
    # s2 = sigma1, sigma2
    # s3 = sigma2, sigma3
    gt = [1,3,6]
    assert(np.all(utils.gt_model([1,2,3,4,5,6,7], gt) == [1,3,2,6,7,5,4]))

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
