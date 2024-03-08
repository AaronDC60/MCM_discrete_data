"""
Class to generate data from a stochastic block model
"""

import numpy as np
import math

from python.utils import tools
from python.utils import spin_ops

class sbm:

    def __init__(self, q, p_in, p_out, comm_structure, interactions, interaction_strengths):
        """
        Initialize a stochastic block model.

        The community structure should be an array where each element represents a community as a string.
        If the ith element in the string is 1, the element is in the community and not in the community if it is 0.
        The interactions is an array with all the interactions (as colors) that should be switched on in case of a link.
        The color defines the type of interaction. For example, [1,1] or [1,2] for pairwise with three states.
        The values in the color parameter need to be between 1 and q-1.
        Interaction strengths is an array with the same length as 'interactions' indicating the strength of the corresponding interaction as a complex number.

        Parameters
        ----------
        q : int
            number of states a variable can take
        p_in : float
            probability to form a link between components in the same community
        p_out : float
            probability to form a link between components in different communities
        comm_structure : array
            representation of the community structure
        interactions : array
            interactions that should be switched on in case of a link
        interaction_strengths: array
            array of complex numbers indicating the strength of the corresponding interaction
        """
        # Check if the arguments meet the requirements
        if type(q) != int:
            raise TypeError("The parameter 'q' must be an integer.")
        if np.ndim(comm_structure) != 1:
            raise TypeError("The parameter 'comm_structure' should be an array with strings.")
        if type(comm_structure[0]) != str:
            raise TypeError("The parameter 'comm_structure' should be an array with strings.")
        if np.ndim(interactions) != 2:
            raise TypeError("The parameter 'interactions' should be a 2D array.")
        if np.ndim(interaction_strengths) != 1:
            raise TypeError("The parameter 'interaction_strenghts' should be a 1D array.")
        if len(interaction_strengths) != len(interactions):
            raise ValueError("The length of 'interactions' does not match the length of 'interaction_strengths'.")
        # Number of components
        self.n = len(comm_structure[0])
        self.q = q
        self.p_in = p_in
        self.p_out = p_out
        # Store the community as integers
        self.comm_structure = np.empty(len(comm_structure), dtype=int)
        for i, community in enumerate(comm_structure):
            self.comm_structure[i] = tools.string_to_int(community, 2)
        # Interactions
        self.interactions = interactions
        self.interaction_strengths = interaction_strengths
        # Variable to store the value of all model parameters in a complete spin model
        self.g = np.zeros(self.q**self.n)
    
    def reset(self):
        """Remove all links (set all model parameters to zero)."""
        self.g = np.zeros(self.q**self.n)
    
    def check_in_or_out(self, vars):
        """
        Check if all given variables are in the same community or not

        Parameters
        ----------
        vars : array
            list with the indices of all the variables to consider
        
        Returns
        -------
        result : boolean
            true if all in the same community, false otherwise
        """
        # Check the community of the first variable
        comm = 0
        for i in range(len(self.comm_structure)):
            if self.comm_structure[i] & 2**vars[0]:
                comm = i
        
        # Check if the community of the other variables are the same
        for var in vars[1:]:
            if not self.comm_structure[comm] & 2**var:
                return False
        return True
    
    def find_index(self, set, index, max_value):
        """
        Find the first index from the right in set that has not reached its maximum allowed value.
        Part of algorithm to generate all increasing sequences of a given length and maximum value.

        Parameters
        ----------
        set : array
            representation of an increasing sequence (interaction)
        index : int
            starting index, is always the last element in the set
        max_value : int
            The maximum allowed value that element in the set can have
        """
        while set[index] == max_value:
            index -= 1
            max_value -= 1
        return index

    def create_links(self):
        """Create the links in the graph and switch on the corresponding model parameter."""
        for i in range(len(self.interactions)):
            # Determine the order of the interaction
            k = len(self.interactions[i])

            # Generate all possible combinations with that order (n choose k)
            n_comb = math.comb(self.n, k)
            # Set of variables in the starting interaction
            set = np.arange(0, k)
            # Start index at the last element
            index = k - 1
            max_value = self.n - 1

            for _ in range(n_comb):
                # Check the current set
                if self.check_in_or_out(set):
                    # Form link with probability p_in
                    p = self.p_in
                else:
                    # Form link with probability p_in
                    p = self.p_out
                # Generate uniform random variable to determine if link is formed
                x = np.random.rand()
                if x < p:
                    # Set the model parameters
                    indices = spin_ops.get_spin_op_indices(self.q, set, self.interactions[i])
                    self.g[indices[0]] = self.interaction_strengths[i]
                    self.g[indices[1]] = self.interaction_strengths[i].conjugate()
                
                # Update the set
                if set[index] == max_value:
                    # Increase the first element from the right that has not yet reached it max_value by 1
                    j = self.find_index(set, index, max_value)
                    set[j] += 1
                    # Reset all the value to the right from j (all increasing by 1)
                    for l in range(j+1, k):
                        set[l] = set[l-1] + 1
                else:
                    # Increase the last element
                    set[index] += 1
    
    def generate_samples(self, N, filename):
        """
        Generate N samples from the stochastic block model and store the data in a file.

        Parameters
        ----------
        N : int
            number of samples that need to be generated
        filename : str
            path to the file where the data needs to be stored
        """
        # All possible states
        states = np.arange(0,len(self.g))
        # Calculate the probability distribution
        if self.q == 2:
            # Faster FWHT algorithm
            prob_distr = np.exp(tools.fwht(self.g))
        else:
            prob_distr = np.exp(tools.discrete_fwht(self.g, self.q)).real
        # Normalize the probability distribution
        prob_distr /= np.sum(prob_distr)
        
        samples = []
        for _ in range(N):
            # Choose a state according to the calculated distribution
            state = np.random.choice(states, p=prob_distr)
            samples.append(tools.int_to_string(state, self.q, self.n))
        # Store the samples
        np.savetxt(filename, np.array(samples), fmt='%s', delimiter='')
