import numpy as np
import math

from . import utils

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
        self.data, self.n_var = utils.process_data_array(file)
        # Number of states a variable can be in
        self.q = n_states

        # Generate all possible MCMs
        self.mcms = []
        utils.generate_partitions(0, self.n_var, [], self.mcms)

        # Storage for best results
        self.best_mcm = None
        self.best_evidence = 0

        # Storage for best IM
        self.best_im = None

    def reset_data(self):
        """Transform data back to the original data."""
        self.data, _ = utils.process_data_int(self.file)
    
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
            evidence += (math.lgamma(self.q**(len(subpart))/2) - math.lgamma(len(self.data) + self.q**(len(subpart))/2) - (self.q**(len(subpart))/2) * math.log(math.pi))

            # Extract the part of the dataset that represents the subpartition
            part_data = part_data = self.data[:,subpart]
            # Count the number of times the different patterns occur
            counts = np.unique(part_data, return_counts=True, axis=0)[1]
            counts = counts + 0.5

            for pattern in counts:
                evidence += math.lgamma(pattern)
        
        return evidence
    
    def find_best_mcm(self, method='exhaustive'):
        """
        Find the best mcm for the data

        Parameters
        ----------
        method : str, default 'exhaustive'
            Searching method (options: 'exhaustive', 'greedy')

        Returns
        -------
        best_mcm : list
            array that represents the mcm with the highest evidence
        """
        best_ev = -np.inf
        best_mcm = self.mcms[0]

        if method == 'exhaustive':
            for mcm in self.mcms:
                evidence = self.calc_log_evidence(mcm)
                # Check if this is the best one yet
                if evidence > best_ev:
                    best_ev = evidence
                    best_mcm = mcm
            
            self.best_mcm = best_mcm
            self.best_evidence = best_ev
        
        elif method == 'greedy':
            # Start with best IM
            current_mcm = self.mcms[-1] 
            best_ev = self.calc_log_evidence(current_mcm)

            # Best MCM so far (hard copy of IM)
            best_mcm = [current_mcm[i] for i in range(len(current_mcm))]

            # Apply hierarchical merging procedure
            while len(current_mcm) > 1:
                n_icc = len(current_mcm)
                # Generate all combinations in which 2 ICCs are merged
                for i in range(n_icc):
                    for j in range(i+1, n_icc):
                        # Start from current MCM
                        new_mcm = [current_mcm[i] for i in range(len(current_mcm))]
                        # Concatenate two ICCs
                        new_mcm[i] = [k for l in [new_mcm[i], new_mcm[j]] for k in l]
                        del new_mcm[j]

                        ev = self.calc_log_evidence(new_mcm)
                        if ev > best_ev:
                            # Update best MCM found so far
                            best_mcm = [new_mcm[i] for i in range(len(new_mcm))]
                            best_ev = ev

                # Stop procedure if current MCM is the best one
                if best_mcm == current_mcm:
                    break
                else:
                    # Update current MCM
                    current_mcm = [best_mcm[i] for i in range(len(best_mcm))]
        else:
            raise NameError('Unknown method. Options are "exhaustive" and "greedy".')
        
        self.best_mcm = best_mcm
        self.best_evidence = best_ev
        return best_mcm
