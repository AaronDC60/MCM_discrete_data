"""
Helper functions for spin models
================================
"""

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

def discrete_fwht(a, q):
    """
    Fast Walsh-Hadamard transform (for q > 2) of an array.

    Parameters
    ----------
    a : array
        Array for which the Walsh-Hadamard transform will be calculated
    
    Returns
    -------
    wht_a : array
        Walsh-Hadamard transform of the input
    """
    len_a = len(a)
    wht_a = a.copy()

    # Compute the factors (q different spin values)
    factors = np.ones(q, dtype=complex)
    for i in range(1, q):
        factors[i] = np.exp(2j * np.pi * i / q)

    h = 1
    while h < len_a:
        tmp = wht_a
        wht_a = np.zeros(len(a), dtype=complex)
        for i in range(0, len_a, h*q):
            for j in range(i, i+h):
                for k in range(q):
                    for l in range(q):
                        wht_a[j+ k*h] += tmp[j + l*h] * factors[(l*k)%q]
        h *= q
    return wht_a

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

def int_to_string(integer, q, n):
    """
    Convert integer into a string of length n in base q.

    Parameters
    ----------
    integer : int
        integer to convert
    q : int
        base of the bitstring (corresponds to the number of states)
    n : length of the string

    Returns
    -------
    state : str
        string of length n in base q representing the integer 
    """
    state = ''
    while integer:
        # Remainder is the value of the bit
        state += str(integer % q)
        # The quotient is left
        integer //= q
    
    # Add leading zeros
    state += '0'*(n-len(state))

    return state

def string_to_int(string, q):
    """
    Convert a string in base q into an integer.

    Parameters
    ----------
    string : str
        string containing discrete values in a given base
    q : int
        base in which the string is given
    
    Returns
    -------
    value : int
        integer representation of the string
    """
    value = 0

    for i, element in enumerate(string):
        value += int(element) * q**i
    
    return value

def entropy(prob_distr):
    """
    Calculate the entropy (using base 2) of a given probability distribution.

    Parameters
    ----------
    prob_distr : array
        list containing the probability distribution
    
    Returns
    -------
    value : float
        entropy corresponding to the probability distribution (using base 2)
    """
    # H(x) = - sum[ p(x) log p(x) ]
    value = 0
    for p in prob_distr:
        # Ignore zero probability (assume 0 log 0 = 0)
        if p:
            value -= (p * np.log2(p))
    return value

def mutual_information(comm_1, comm_2, n, normalize=True):
    """
    Calculate the mutual information (using base 2) between two community structures.

    Parameters
    ----------
    comm_1 : array
        list of representing the first community structure (communities represented as integers)
    comm_2 : array
        list of representing the second community structure (communities represented as integers)
    n : int
        number of variables
    normalize : boolean, default True
        option to return the normalized mutual information
    
    Returns
    -------
    mutual_inf : float
        the mutual information between two community structures
    """
    # I(x, y) = sum [P(x, y) log [ P(x,y) / P(x) P(y) ] ]
    mutual_inf = 0

    # Number of communities in both structures
    size_1 = len(comm_1)
    size_2 = len(comm_2)

    # Variables for probability distributions
    p_1 = np.zeros(size_1)
    p_2 = np.zeros(size_2)

    # Loop over all communities in the first structure
    for i in range(size_1):
        c1 = comm_1[i]
        # Probability to be in community i in first structure
        p_1[i] = c1.bit_count() / n
        # Loop over all communities in the second structure
        for j in range(size_2):
            c2 = comm_2[j]
            # Probability to be in community j in second structure
            p_2[j] = c2.bit_count() / n
            # Probability to be in community i in first structure and community j in second structure
            p_12 = (c1 & c2).bit_count() / n

            # Ignore zero probability (assume 0 log 0 = 0)
            if p_12:
                mutual_inf += (p_12 * np.log2(p_12 / (p_1[i] * p_2[j])))
    
    # Calculate the normalized mutual information
    if normalize:
        # Normalized I(x,y) = 1/2 *  I(x,y) / [H(x) + H(y)]
        entropy_1 = entropy(p_1)
        entropy_2 = entropy(p_2)
        mutual_inf /= (0.5 * (entropy_1 + entropy_2))
    
    return mutual_inf
