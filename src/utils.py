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

def generate_all_operators(n_var, n_inter):
    """
    Generate all possible spinoperators with at most n_inter variables in a system with n_var variables.

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
