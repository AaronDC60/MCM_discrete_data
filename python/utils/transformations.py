"""
Functions to perform a gauge transformation of a model or state
===============================================================
"""

import numpy as np

def gt_state(s, gt):
    """
    Calculate the Gauge transform for a given binary state s.

    Parameters
    ----------
    s : int
        state in s basis
    gt : array
        list with the gauge transformation
    
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

def gt_state_discrete(state, gt, q):
    """
    Calculate the Gauge transform for a given state (q >= 2).

    Parameters
    ----------
    state : array
        state in curent basis
    gt : array
        list with the gauge transformation
    q : int
        number of different states for single variable
    
    Returns
    -------
    new_state : array
        state in new basis
    """
    new_state = np.copy(state)
    n = len(state)

    for i, mu in enumerate(gt):
        j = n-1
        value = 0
        while mu:
            value += state[j] * (mu % q)

            mu //= q
            j -= 1

        new_state[i] = value % q
    return new_state

def gt_model(m, gt):
    """
    Calculate the Gauge transform for a given model m.

    Parameters
    ----------
    m : array
        list with all the operators in the model
    gt : array
        list with the gauge transformation
    
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
