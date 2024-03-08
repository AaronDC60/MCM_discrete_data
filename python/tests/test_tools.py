import numpy as np
from python.utils import tools
from python.utils import spin_ops

def test_fwht():
    a = np.array([1,0,1,0,0,1,1,0])
    assert(np.all(tools.fwht(a) == [4,2,0,-2,0,2,0,2]))

    # Should be the same as the matrix-vector product with the spin operator matrix
    a = np.random.rand(2**8)
    S = spin_ops.construct_s_matrix(8, 2)
    assert(np.allclose(tools.fwht(a), S @ a))

def test_discrete_fwht(): 
    a = np.array([2,1,0,-1])
    assert(np.allclose(tools.discrete_fwht(a, 4), np.array([2,2+2j,2,2-2j])))
    # Should be the same for as the regular fwht for q = 2
    a = np.random.rand(2**8)
    assert(np.allclose(tools.fwht(a), tools.discrete_fwht(a, 2)))

    # Should be the same as the matrix-vector product with the spin operator matrix
    a = np.random.rand(3**6)
    S = spin_ops.construct_s_matrix(6,3)
    assert(np.allclose(tools.discrete_fwht(a, 3), S @ a))

def test_partitions():
    all_partitions = []
    tools.generate_partitions(0, 3, [],  all_partitions)
    assert(len(all_partitions) == 5)
    assert(np.all(all_partitions[0] == [[0,1,2]]))
    assert(np.all(all_partitions[1] == [[0,1],[2]]))
    assert(np.all(all_partitions[2] == [[0,2],[1]]))
    assert(np.all(all_partitions[3] == [[0],[1,2]]))
    assert(np.all(all_partitions[4] == [[0],[1],[2]]))

def test_int_to_string():
    # Base 2
    assert(tools.int_to_string(0,2,5) == '00000')
    assert(tools.int_to_string(1,2,5) == '00001')
    assert(tools.int_to_string(2,2,5) == '00010')
    assert(tools.int_to_string(3,2,5) == '00011')
    assert(tools.int_to_string(4,2,5) == '00100')
    assert(tools.int_to_string(19,2,5) == '10011')

    # Base 3
    assert(tools.int_to_string(0,3,4) == '0000')
    assert(tools.int_to_string(1,3,4) == '0001')
    assert(tools.int_to_string(2,3,4) == '0002')
    assert(tools.int_to_string(3,3,4) == '0010')
    assert(tools.int_to_string(4,3,4) == '0011')
    assert(tools.int_to_string(5,3,4) == '0012')
    assert(tools.int_to_string(73,3,4) == '2201')

def test_string_to_int():
    # Base 2
    assert(tools.string_to_int('00000', 2) == 0)
    assert(tools.string_to_int('00001', 2) == 1)
    assert(tools.string_to_int('00010', 2) == 2)
    assert(tools.string_to_int('00011', 2) == 3)
    assert(tools.string_to_int('00100', 2) == 4)
    assert(tools.string_to_int('10011', 2) == 19)

    # Base 3
    assert(tools.string_to_int('0000', 3) == 0)
    assert(tools.string_to_int('0001', 3) == 1)
    assert(tools.string_to_int('0002', 3) == 2)
    assert(tools.string_to_int('0010', 3) == 3)
    assert(tools.string_to_int('0011', 3) == 4)
    assert(tools.string_to_int('0012', 3) == 5)
    assert(tools.string_to_int('2201', 3) == 73)