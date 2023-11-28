import numpy as np
import pytest
from src.model_generator import *

def test_init():
    # Incorrect initialization
    with pytest.raises(TypeError):
        model_generator(3.5)
    generator = model_generator(10)
    # Initialization with 10 spin variables
    assert(generator.n_var == 10)

def test_complete_model():
    generator = model_generator(3)
    assert(np.all(generator.generate_all_ops() == range(1, 3**2 - 1)))
    assert(np.all(generator.generate_ops_upto_order_n(generator.n_var) == range(1, 3**2 - 1)))

def test_model_upto_order_n():
    generator = model_generator(3)
    assert(len(generator.generate_ops_upto_order_n(0)) == 0)
    assert(np.all(generator.generate_ops_upto_order_n(2) == [1, 2, 3, 4, 5, 6]))
    assert(np.all(generator.generate_pairwise_ops() == [1, 2, 3, 4, 5, 6]))
    generator = model_generator(4)
    assert(np.all(generator.generate_ops_upto_order_n(1) == [1, 2, 4, 8]))

def test_mcms():
    generator = model_generator(3)
    assert(np.all(generator.generate_mcm_ops([[0,1,2]]) == [1,2,3,4,5,6]))
    assert(np.all(generator.generate_mcm_ops([[0,2],[1]]) == [1,2,4,5]))
    assert(np.all(generator.generate_all_mcms() == generator.mcms))
    assert(np.all(generator.mcms[0] == [1,2,3,4,5,6]))
    assert(np.all(generator.mcms[1] == [1,2,3,4]))
    assert(np.all(generator.mcms[2] == [1,2,4,5]))
    assert(np.all(generator.mcms[3] == [1,2,4,6]))
    assert(np.all(generator.mcms[4] == [1,2,4]))
