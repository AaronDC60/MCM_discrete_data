import numpy as np
import pytest
from src import utils

def test_fwht():
    a = np.array([1,0,1,0,0,1,1,0])
    assert(np.all(utils.fwht(a) == [4,2,0,-2,0,2,0,2]))

def test_generate_all_ops():
    # Incorect input
    with pytest.raises(TypeError):
        utils.generate_all_operators(3.5, 2)
    with pytest.raises(TypeError):
        utils.generate_all_operators(3, 2.5)
    with pytest.raises(ValueError):
        utils.generate_all_operators(2, 3)

    # 3 variables only empty operator
    assert(utils.generate_all_operators(3, 0) == [0])
    # 4 variables only single-body terms
    assert(np.all(utils.generate_all_operators(4,1) == [0, 1, 2, 4, 8]))
    # 3 variables only pairwise interactions
    assert(np.all(utils.generate_all_operators(3, 2) == [0, 1, 2, 3, 4, 5, 6]))
    # 3 variables all interactions
    assert(np.all(utils.generate_all_operators(3, 3) == range(0, 3**2 - 1)))