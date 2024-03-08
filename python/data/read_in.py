"""
Functions to read in binary (q = 2) and discrete (q > 2) data from file and store in an array
=============================================================================================
"""

import numpy as np

def process_data_int(file):
    """
    Store (binary) data from file as integers.

    Parameters
    ----------
    file : str
        path to the file containing the data
    
    Returns
    -------
    data : numpy.ndarray
        dataset stored as array of integers
    n_var : int
        number of variables in the system
    """
    data = np.loadtxt(file, dtype=str)
    n_var = len(data[0])
    # Convert all observations to integer representation (bitwise xor with all 1s bitstring to have the map (0,1) -> (1, -1))
    data = np.array([int(i[::-1], 2) ^ int('1'*len(i), 2) for i in data])

    return data, n_var

def process_data_array(file):
    """
    Store data from file as arrays

    Parameters
    ----------
    file : str
        path to the file containing the data
    
    Returns
    -------
    data : numpy.ndarray
        dataset stored as array of arrays
    n_var : int
        number of variables in the system
    """
    # Load in the data
    data = np.loadtxt(file, dtype=str)
    data = np.array([np.fromiter(obs, dtype=int) for obs in data])
    # Determine the number of variables in the system
    n_var = len(data[0])

    return data, n_var
