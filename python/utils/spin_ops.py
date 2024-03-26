"""
Functions related to spin operators
===================================
"""

import numpy as np
from . import tools

def generate_all_ops(n_var, q=2):
    """
    Generate all possible spinoperators for a system with n_var variables (complete model).

    Parameters
    ----------
    n_var : int
        Total number of variables in the system
    q : int
        Number of states a variable can be in
    
    Returns
    -------
    operators : array
        array containing the spinoperators
    """
    if type(n_var) != int:
        raise TypeError("The parameter 'n_var' must be an integer.")
    if type(q) != int:
        raise TypeError("The parameter 'q' must be an integer.")
    return np.arange(1, q**n_var)

def generate_ops_upto_order_n(n_var, n_inter):
    """
    Generate all possible binary spinoperators with n_inter as highest order interaction terms in a system with n_var variables.

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

def construct_s_matrix(n, q):
    """
    Construct the spin operator matrix for q number of states and n variables

    Parameters
    ----------
    n : int
        Total number of variables in the system
    q : int
        Number of states a variable can be in
    
    Returns
    -------
    S_matrix : array
        2D array representing the spin operator matrix   
    """

    if type(n) != int:
        raise TypeError("The parameter 'n_var' must be an integer.")
    if type(q) != int:
        raise TypeError("The parameter 'q' must be an integer.")
    
    # Construct the matrix for 1 spin
    S = np.ones((q, q), dtype=complex)
    for i in range(q):
        for j in range(i, q):
            S[i,j] = np.exp(((i*j)*2j*np.pi) / q)
            S[j,i] = S[i,j]

    # Construct it for n spins by taking the kronecker product repeatedly
    S_matrix = np.copy(S)
    for _ in range(n-1):
        S_matrix = np.kron(S_matrix, S)
    
    return S_matrix

def get_spin_op_indices(q, vars, color):
    """
    Get the indices (integer representation) of the spin operator and its comples conjugate between the given variables.

    The length of vars and color should be the same and indicates the order of the interaction.
    The color defines the type of interaction. For example, [1,1] or [1,2] for pairwise with three states.
    The values in the color parameter need to be between 1 and q-1.
    For q = 2, there is only 1 color (1) and the two indices will be same because spin operators are its own complex conjugate.

    Parameters
    ----------
    q : int
        Number of states the variables can be in
    vars : array
        list containing the index of the variables
    color : array
        the type of interaction (ex. [1,1] or [1,2] for pairwise with three states)
    """
    if type(q) != int:
        raise TypeError("The parameter 'q' must be an integer.")
    if np.ndim(vars) != 1:
        raise TypeError("The parameter 'vars' should be a 1D array.")
    if np.ndim(color) != 1:
        raise TypeError("The parameter 'color' should be a 1D array.")
    if len(vars) != len(color):
        raise ValueError("The length of the arguments 'vars' and 'color' are not the same.")
    if np.min(color) < 1 or np.max(color) > q-1:
        raise ValueError("The values in the color parameter need to be between 1 and q-1.")
    # Index for spin operator between the given variables and color
    index1 = 0
    # Index for the complex conjugate of the spin operator
    index2 = 0

    for i, var in enumerate(vars):
        index1 += color[i] * q**var
        index2 += (-color[i] % q) * q**var
    
    return index1, index2

def spin_value(state, op, q):
    """
    Calculate the spin value for a given state and operator.

    Parameters
    ----------
    state : array
        list with the value of every variable
    op : int
        integer representation of the operator
    q : int
        total number of states
    
    Returns
    -------
    s : int
        spin value
    """
    # s = sum(alpha_j * mu_j)
    s = 0
    # read alpha from left to right
    j = 0
    while op:
        s += state[j] * (op % q)

        op //= q
        j += 1
    return s % q

def entropy_of_operator(data, op, q):
    """
    Calculate the entropy of an operator for a given dataset.

    Parameters
    ----------
    data : array
        list of the observations represented as arrays
    op : int
        integer representation of the operator
    q : int
        total number of states

    Returns
    -------
    entropy : float
        entropy of the operator
    """
    # Variable for probability distribution
    p = np.zeros(q)

    for obs in data:
        # Determine the value of the spin operator
        value = spin_value(obs, op, q)
        # Increase number of occurences of that value by 1
        p[value] += 1
    
    p /= len(data)
    # Calculate the entropy of this distribution
    return tools.entropy(p)

def comb_ops(op_nu, op_mu, q, cc=False):
    """
    Calculate the product of two spin operators (phi_nu * phi_mu).
    
    Takes in the integer representation of two spin operators.
    If complex_conjugate (cc) is true, the product phi_nu * phi_(-mu) is also returned.

    Parameters
    ----------
    op1 : int
        integer representation of the first operator (phi_nu)
    op2 : int
        integer representation of the second operator (phi_mu)
    q : int
        total number of states
    cc : boolean, default False
        option to calculate phi_nu * phi_(-mu) as well

    Returns
    -------
    comb_op : int
        integer representation of phi_nu * phi_mu
    comb_op_cc : int, returned if cc is True
        integer representation of phi_nu * phi_(-mu)
    """
    # phi_nu * phi_mu
    comb_op = 0
    if cc:
        # phi_nu * phi_(-mu)
        comb_op_cc = 0

    # Base^(j-th bit)
    factor = 1
    while op_nu or op_mu:
        # j-th bit value is the remainder
        nu_j = op_nu % q
        mu_j = op_mu % q

        comb_op += ((nu_j + mu_j) % q) * factor
        if cc:
            comb_op_cc += ((nu_j - mu_j) % q) * factor

        # Update
        factor *= q
        op_nu //= q
        op_mu //= q
    
    if cc:
        return comb_op, comb_op_cc
    return comb_op
