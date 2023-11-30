import numpy as np
import math

from . import utils

class mcm:

    def __init__(self, file):
        """
        Initialize an MCM model object.

        Parameters
        ----------
        file : str
            path to the file containing the data
        """
        # Load in the data
        data = np.loadtxt(file, dtype=str)
        # Convert all observations to integer representation (bitwise xor with all 1s bitstring to have the map (0,1) -> (1, -1))
        self.data = np.array([int(i[::-1], 2) ^ int('1'*len(i), 2) for i in data])
        # Determine the number of variables in the system
        self.n_var = len(data[0])

        # Generate all possible MCMs
        self.mcms = []
        utils.generate_partitions(0, self.n_var, [], self.mcms)

        # Storage for best results
        self.best_mcm = None
        self.best_evidence = 0
    
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
            evidence += (math.lgamma(2**(len(subpart)-1)) - math.lgamma(len(self.data) + 2**(len(subpart)-1)))

            # Extract the part of the dataset that represents the subpartition
            relevant_bits = np.sum([2**i for i in subpart])
            part_data = list(map(lambda x: x&relevant_bits, self.data))
            # Count the number of times the different patterns occur
            counts = np.unique(part_data, return_counts=True)[1]

            # Check if we have a value for every possible pattern
            missing = 2**len(subpart) - len(counts)
            if missing != 0:
                # Add zero contribution for missing patterns
                counts = np.append(counts, np.zeros(missing))
            counts = counts + 0.5

            for pattern in counts:
                evidence += (math.lgamma(pattern) - math.log(math.sqrt(math.pi)))
        
        return evidence
    
    def find_best_mcm(self):
        """
        Find the best mcm for the data

        Returns
        -------
        best_mcm : list
            array that represents the mcm with the highest evidence
        """
        best_ev = -np.inf
        best_mcm = self.mcms[0]

        for mcm in self.mcms:
            evidence = self.calc_log_evidence(mcm)
            print(evidence)
            # Check if this is the best one yet
            if evidence > best_ev:
                best_ev = evidence
                best_mcm = mcm
        
        self.best_mcm = best_mcm
        self.best_evidence = best_ev

        return best_mcm
        