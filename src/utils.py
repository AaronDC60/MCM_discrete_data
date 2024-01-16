import numpy as np
from numba import njit

@njit(fastmath=True)
def fwht(a):
    """
    Fast Walsh-Hadamard transform of an array.

    Parameters
    ----------
    a : array
        Array for which the Walsh-Hadamard transform will be calculated
        Length should be a power of two
    
    Returns
    -------
    wht_a : array
        Walsh-Hadamard transform of the input
    """
    len_a = len(a)
    wht_a = a.copy()

    h = 1
    while h < len_a:
        for i in range(0, len_a, h*2):
            for j in range(i, i+h):
                x = wht_a[j]
                y = wht_a[j + h]

                wht_a[j] = x + y
                wht_a[j + h] = x - y
        h *= 2
    return wht_a

def gt_state(s, gt):
    """
    Calculate the Gauge transform for a given state s.

    Parameters
    ----------
    s : int
        state in s basis
    gt : array
        List with the gauge transformation
    
    Returns
    -------
    sigma : int
        state in sigma basis
    """
    sigma = 0
    for i, mu in enumerate(gt):
        if divmod((s & mu).bit_count(), 2)[1]:
            sigma += 2**i
    return sigma

def gt_model(m, gt):
    """
    Calculate the Gauge transform for a given model m.

    Parameters
    ----------
    m : array
        List with all the operators in the model
    gt : array
        List with the gauge transformation
    
    Returns
    -------
    new_m : array
        List with the gauge transform of the model
    """
    new_m = np.zeros(len(m), int)

    for i, op in enumerate(m):
        index = 0
        gt_op = 0

        while op:
            if op & 1:
                gt_op ^= gt[index]
            index += 1
            op >>= 1
        
        new_m[i] = gt_op

    return new_m

def generate_all_ops(n_var):
    """
    Generate all possible spinoperators for a system with n_var variables (complete model).
    
    Returns
    -------
    operators : array
        array containing the spinoperators
    """
    if type(n_var) != int:
        raise TypeError("The parameter 'n_var' must be an integer.")
    return np.arange(1, 2**n_var)

def generate_ops_upto_order_n(n_var, n_inter):
    """
    Generate all possible spinoperators with n_inter as highest order interaction terms in a system with n_var variables.

    Parameter
    ---------
    n_var : int
        Total number of variables in system
    n_inter : int
        Max number of spin variables in a single interaction
    
    Returns
    -------
    operators : array
        array containing the spinoperators
    """
    # Check type of the input parameters
    if type(n_var) != int:
        raise TypeError("The parameter n_var should be an integer instead of %s."%type(n_var))
    if type(n_inter) != int:
        raise TypeError("The parameter n_inter should be an integer instead of %s."%type(n_inter))
    # The max number of variables per interaction should be smaller than the total number of variables
    if n_inter > n_var:
        raise ValueError("The maximum number of variables per interaction (%i) should be smaller than the total number of variables (%i)."%(n_inter, n_var))
    
    # Start with the the all zero operator
    states = [0]
    for i in range(n_var):
        for state in states[:]:
            if state.bit_count() < n_inter:
                states.append(state + 2**i)
    return np.array(states[1:])

def generate_partitions(i, n_var, part, all_partitions):
    """
    Generate all possible partitions in a system with n_var variables.
    The variable 'all_partitions' will contain the result at the end.

    i : int
        index of the next element to add
    n_var : int
        total number of variables in the system
    part : list
        current partition (start with empty list)
    all_partitions : list   
        list that will contain all partitions (start with empty list)
    """
    if i == n_var:
        # All n variables are added to a subpartition (-> complete partition)
        # Add a copy of the partition
        all_partitions.append([subpart[:] for subpart in part])
        return
    
    for j in range(len(part)):
        # Add spinvariable to a subpartition (generate child node) and continue with the next spinvariable
        part[j].append(i)
        generate_partitions(i+1, n_var, part, all_partitions)
        # Remove spin variable from the current subpartition (going back to the parent node in the tree)
        part[j].pop()
    
    # Add spinvariable as new subpartition (generate child node) and continue with the next spinvariable 
    part.append([i])
    generate_partitions(i+1, n_var, part, all_partitions)
    # Remove the current subpartition (going back to the parent node in the tree)
    part.pop()
