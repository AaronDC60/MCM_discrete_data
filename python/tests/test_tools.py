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
    assert(tools.int_to_string(1,2,5) == '10000')
    assert(tools.int_to_string(2,2,5) == '01000')
    assert(tools.int_to_string(3,2,5) == '11000')
    assert(tools.int_to_string(4,2,5) == '00100')
    assert(tools.int_to_string(19,2,5) == '11001')

    # Base 3
    assert(tools.int_to_string(0,3,4) == '0000')
    assert(tools.int_to_string(1,3,4) == '1000')
    assert(tools.int_to_string(2,3,4) == '2000')
    assert(tools.int_to_string(3,3,4) == '0100')
    assert(tools.int_to_string(4,3,4) == '1100')
    assert(tools.int_to_string(5,3,4) == '2100')
    assert(tools.int_to_string(73,3,4) == '1022')

def test_string_to_int():
    # Base 2
    assert(tools.string_to_int('00000', 2) == 0)
    assert(tools.string_to_int('10000', 2) == 1)
    assert(tools.string_to_int('01000', 2) == 2)
    assert(tools.string_to_int('11000', 2) == 3)
    assert(tools.string_to_int('00100', 2) == 4)
    assert(tools.string_to_int('11001', 2) == 19)

    # Base 3
    assert(tools.string_to_int('0000', 3) == 0)
    assert(tools.string_to_int('1000', 3) == 1)
    assert(tools.string_to_int('2000', 3) == 2)
    assert(tools.string_to_int('0100', 3) == 3)
    assert(tools.string_to_int('1100', 3) == 4)
    assert(tools.string_to_int('2100', 3) == 5)
    assert(tools.string_to_int('1022', 3) == 73)

def test_entropy():
    assert(tools.entropy([1,0]) == 0)
    assert(tools.entropy([1,0,0]) == 0)

    assert(tools.entropy([0.25, 0.25, 0.25, 0.25]) == 2)
    assert(tools.entropy([0.5, 0.5]) == 1)

    assert(np.isclose(tools.entropy([0.75, 0.25]), 0.8112781244591))

def test_mutual_inf():
    comm_1 = ['000111', '111000']
    comm_2 = ['000101', '000010', '111000']
    n = len(comm_1[0])

    comm_1 = [tools.string_to_int(comm_1[i], 2) for i in range(2)]
    comm_2 = [tools.string_to_int(comm_2[i], 2) for i in range(3)]

    # Normalized mutual information is maximal (= 1) between identical community structures
    assert(tools.mutual_information(comm_1, comm_1, n, normalize=True) == 1)
    assert(np.isclose(tools.mutual_information(comm_2, comm_2, n, normalize=True), 1))

    # Non normalized mutual information between identical community structures is equal to the unconditional entropy
    assert(tools.mutual_information(comm_1, comm_1, n) == tools.entropy([0.5, 0.5]))

    # Mutual information between different community structures
    mutual_inf = tools.mutual_information(comm_1, comm_2, n, normalize=False)
    norm_mutual_inf = tools.mutual_information(comm_1, comm_2, n, normalize=True)
    assert(np.isclose(mutual_inf, 1))
    assert(np.isclose(norm_mutual_inf, mutual_inf / (0.5 * (tools.entropy([0.5, 0.5]) + tools.entropy([1/3, 1/6, 1/2])))))
    assert(np.isclose(norm_mutual_inf, 0.8132898335036761))
