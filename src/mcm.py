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

        # Storage for best IM
        self.best_im = None
    
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
            # Take the best IM as basis
            # Check if best IM still needs to be calculated
            if self.best_im is None:
                self.find_best_im()
            gt = self.best_im
        else:
            # Check if gt is valid
            if type(gt) != list and type(gt) != np.ndarray:
                raise TypeError("The parameter 'gt' should be a list.")
        
        new_data = []
        # Transform every observation in the data
        for obs in self.data:
            new_data.append(utils.gt_state(obs, gt))
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

    def calc_bias(self, op):
        """
        Calculate the bias of operator
        """
        bias = 0
        for obs in self.data:
            # Determine if value for operator is +1 or -1 in this observation
            bias += ((-1)**(op&obs).bit_count())
        return abs(bias)
    
    def find_best_im(self):
        """
        Find the best im (independent model) for the data

        Returns
        -------
        best_im : list
            array that represents the im with the n most biased operators
        """
        # Array with the spinoperators in the best IM
        im = []
        # Complete model that can be formed with the IM 
        model_im = []

        # Generate all operators
        all_ops = utils.generate_all_ops(self.n_var)

        # Find the n most biased operator
        for _ in range(self.n_var):
            max_bias = 0
            most_bias_op = all_ops[0]

            # Loop over all available operators
            for op in all_ops:
                # Calculate bias of the operator
                current_bias = self.calc_bias(op)
                # Check if this is max biased operator so far
                if current_bias > max_bias:
                    most_bias_op = op
                    max_bias = current_bias

            # Add most biased operator to the best IM
            im.append(most_bias_op)
            model_im.append(most_bias_op)
            all_ops = np.delete(all_ops, np.where(all_ops == most_bias_op))

            # Filter out operators that are not independent from operators in IM basis
            for op in model_im[:]:
                comb_op = op^most_bias_op
                model_im.append(comb_op)
                all_ops = np.delete(all_ops, np.where(all_ops == comb_op))
        
        self.best_im = im
        return im
    
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
            # Check if this is the best one yet
            if evidence > best_ev:
                best_ev = evidence
                best_mcm = mcm
        
        self.best_mcm = best_mcm
        self.best_evidence = best_ev

        return best_mcm
        