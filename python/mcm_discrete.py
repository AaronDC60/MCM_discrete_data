"""
Class that can be used to find the best MCM for a discrete dataset
"""

import numpy as np
import math
import copy

from python.data import read_in
from python.utils import tools
from python.utils import spin_ops
from python.utils import transformations

class mcm:

    def __init__(self, file, n_states):
        """
        Initialize an MCM model object.

        Parameters
        ----------
        file : str
            path to the file containing the data
        """
        self.file = file
        # Process data by writing observations as arrays
        self.data, self.n_var = read_in.process_data_array(file)
        # Number of states a variable can be in
        self.q = n_states

        # Generate all possible MCMs
        self.mcms = []
        tools.generate_partitions(0, self.n_var, [], self.mcms)

        # Storage for best results
        self.best_mcm = None
        self.best_evidence = 0

        # Storage for best IM
        self.best_im = np.zeros(self.n_var, dtype=int)

    def reset_data(self):
        """Transform data back to the original data."""
        self.data, _ = read_in.process_data_array(self.file)

    def transform_data(self, gt=None):
        """
        Perform a Gauge transformation on the data.
        Data is transformed in place.

        Parameters
        ----------
        gt : array
            List with the gauge transformation, default is the best IM
        """
        if gt is None:
            gt = self.best_im
        else:
            # Check if gt is valid
            if type(gt) != list and type(gt) != np.ndarray:
                raise TypeError("The parameter 'gt' should be a list.")
        
        new_data = np.empty(self.data.shape)
        # Transform every observation in the data
        for i, obs in enumerate(self.data):
            new_data[i] = transformations.gt_state_discrete(obs, gt, self.q)
        self.data = new_data
    
    def calc_log_evidence(self, mcm):
        """
        Calculate the log evidence for a given partition

        Parameters
        ----------
        mcm : list
            array that represents an mcm
        
        Returns
        -------
        evidence : float
            log evidence for the given mcm
        """
        evidence = 0
        for subpart in mcm:
            evidence += (math.lgamma(self.q**(len(subpart))/2) - math.lgamma(len(self.data) + self.q**(len(subpart))/2) )
            # Extract the part of the dataset that represents the subpartition
            part_data = self.data[:,subpart]
            # Count the number of times the different patterns occur
            counts = np.unique(part_data, return_counts=True, axis=0)[1]
            counts = counts + 0.5

            for pattern in counts:
                evidence += (math.lgamma(pattern) - 0.5 * math.log(math.pi))
        
        return evidence
    
    def find_best_im(self):
        """
        Find the best independent model (im) for the data.
        
        Returns
        -------
        best_im : array
            list with the integer representation of the n operators forming the best basis
        """
        # All combinations that can be formed with the operators in th IM (to check independence)
        all_comb = [0]

        # All possible spin operators (q^n - 1)
        all_ops = np.arange(1, self.q**self.n_var)

        # Calculate and store the entropy for all operators
        entropy = np.empty((len(all_ops), 2))
        for i, op in enumerate(all_ops):
            entropy[i, 0] = op
            entropy[i, 1] = spin_ops.entropy_of_operator(self.data, op, self.q)
        
        # Sort operators based on entropy from low to high
        operators = entropy[np.argsort(entropy[:,1])][:,0]

        # Search the n most biased operators that are independent
        i = 0
        for op_mu in operators:
            # Check if the operator is independent from the previously found operators
            combinations = []
            for op_nu in all_comb:
                # Combine the operators (also with the complex conjugate)
                new_ops = spin_ops.comb_ops(op_nu, op_mu, self.q, cc=True)
                combinations.append(new_ops[0])
                combinations.append(new_ops[1])
                # Check independence
                if 0 in combinations:
                    break

            # Go to the next operator if this one is not independent
            if 0 in combinations:
                continue
            # Otherwise, add to IM
            self.best_im[i] = op_mu
            all_comb += combinations
            i += 1

            # Stop search if n independent operators are found
            if i == self.n_var:
                break
        # Represent the data in this basis
        self.transform_data(self.best_im)

        return self.best_im

    def divide_and_conquer(self, mcm, final_mcm, print_search=False):
        """
        Finding best MCM using the divide and conquer approach.

        Parameters
        ----------
        mcm : list
            starting MCM (one component) to split up
        final_mcm : list
            variable to store the resulting MCM
        print_search : boolean, default False
            Option to print all the encounterd MCMs and their evidence
        """
        # Variable to store best MCM after split (even if no overall improvement occured)
        tmp_best_mcm = mcm
        # Variable to store best MCM (only if overall improvement occured)
        best_mcm = mcm
        best_ev = self.calc_log_evidence(mcm)
        # Print result
        if print_search:
            print('Start splitting procedure: ', mcm, best_ev)
        # Add new subparition
        mcm.append([])

        # Keep moving components to second partition
        while len(tmp_best_mcm[0]) > 2:
            # Start again from previously best split
            tmp_best_ev = -np.inf
            current_mcm = tmp_best_mcm

            # Try each one in the first subpartition as a member of the second partition
            for i in range(len(current_mcm[0])):
                # Make a hard copy
                new_mcm = copy.deepcopy(current_mcm)
                # Move the ith component from first to second partition
                new_mcm[0] = current_mcm[0][:i] + current_mcm[0][i+1:]
                new_mcm[1].append(current_mcm[0][i])
                ev = self.calc_log_evidence(new_mcm)
                # Print result
                if print_search:
                    print(new_mcm, ev)
                # Check if this is the best split
                if ev > tmp_best_ev:
                    tmp_best_mcm = new_mcm
                    tmp_best_ev = ev
                    # Print result
                    if print_search:
                        print('New intermediate best: ', tmp_best_mcm,tmp_best_ev)

            # Check if the best split is an overall improvement
            if tmp_best_ev > best_ev:
                best_mcm = tmp_best_mcm
                best_ev = tmp_best_ev
                # Print result
                if print_search:
                    print('New overall best: ', best_mcm, best_ev)
            
        if len(best_mcm[1]) == 0:
            # Starting MCM was the best option
            final_mcm.append(best_mcm[0])
        else:
            # Try to split up each of the resulting partitions
            for subpart in best_mcm:
                if len(subpart) != 1:
                    self.divide_and_conquer([subpart], final_mcm, print_search)
                else:
                    final_mcm.append(subpart)
    
    def find_best_mcm(self, method='exhaustive', print_search=False):
        """
        Find the best mcm for the data.

        Parameters
        ----------
        method : str, default 'exhaustive'
            Searching method (options: 'exhaustive', 'greedy', 'divide_and_conquer')
        print_search : boolean, default False
            Option to print all the encounterd MCMs and their evidence

        Returns
        -------
        best_mcm : list
            array that represents the mcm with the highest evidence
        """
        best_ev = -np.inf
        best_mcm = None

        if method == 'exhaustive':
            for mcm in self.mcms:
                evidence = self.calc_log_evidence(mcm)
                # Check if this is the best one yet
                if evidence > best_ev:
                    best_ev = evidence
                    best_mcm = mcm
                # Print result
                if print_search:
                    print(mcm, evidence)
            
            self.best_mcm = best_mcm
            self.best_evidence = best_ev
        
        elif method == 'greedy':
            # Start with best IM
            current_mcm = self.mcms[-1]
            best_ev = self.calc_log_evidence(current_mcm)
            # Print result
            if print_search:
                print(current_mcm, best_ev)

            # Best MCM so far (hard copy of IM)
            best_mcm = current_mcm

            # Apply hierarchical merging procedure
            while len(current_mcm) > 1:
                n_icc = len(current_mcm)
                # Generate all combinations in which 2 ICCs are merged
                for i in range(n_icc):
                    for j in range(i+1, n_icc):
                        # Start from current MCM
                        new_mcm = copy.deepcopy(current_mcm)
                        # Concatenate two ICCs
                        new_mcm[i] += new_mcm[j]
                        del new_mcm[j]

                        ev = self.calc_log_evidence(new_mcm)
                        if ev > best_ev:
                            # Update best MCM found so far
                            best_mcm = new_mcm
                            best_ev = ev
                        # Print result
                        if print_search:
                            print(new_mcm, ev)

                # Stop procedure if current MCM is the best one
                if best_mcm == current_mcm:
                    break
                else:
                    # Update current MCM
                    current_mcm = best_mcm
        
        elif method == 'divide_and_conquer':
            # Start with best all basis operators in one partition
            current_mcm = copy.deepcopy(self.mcms[0])
            best_mcm = []

            self.divide_and_conquer(current_mcm, best_mcm, print_search)
            best_ev = self.calc_log_evidence(best_mcm)
        else:
            raise NameError('Unknown method. Options are "exhaustive", "greedy" and "divide_and_conquer".')
        
        self.best_mcm = best_mcm
        self.best_evidence = best_ev
        return best_mcm
