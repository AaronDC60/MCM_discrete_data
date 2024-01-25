import numpy as np
import math
import copy

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
        self.file = file
        # Process data by writing observations in integer representation
        self.data, self.n_var = utils.process_data_int(file)

        # Generate all possible MCMs
        self.mcms = []
        utils.generate_partitions(0, self.n_var, [], self.mcms)

        # Storage for best results
        self.best_mcm = None
        self.best_evidence = 0

        # Storage for best IM
        self.best_im = [2**i for i in range(self.n_var)]
    
    def reset_data(self):
        """Transform data back to the original data."""
        self.data, _ = utils.process_data_int(self.file)
    
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
        N = len(self.data)
        evidence = 0
        for subpart in mcm:
            r = len(subpart)
            evidence += (math.lgamma(2**(r-1)) - math.lgamma(N + 2**(r-1)))
            
            # Extract the part of the dataset that represents the subpartition
            relevant_bits = np.sum([2**i for i in subpart])
            part_data = list(map(lambda x: x&relevant_bits, self.data))
            # Count the number of times the different patterns occur
            counts = np.unique(part_data, return_counts=True)[1]
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
        # Current MCM is the best one so far
        best_mcm = mcm
        best_ev = self.calc_log_evidence(mcm)
        # Print result
        if print_search:
            print(mcm, best_ev)
        # Add new subpartition
        mcm.append([])
        while True:
            # Try each one in the first subpartition as a member of the second subpartition to see if evidence increases
            for i in range(len(mcm[0])):
                # Hard copy
                new_mcm = copy.deepcopy(mcm)
                # Move ith member from first to second partition
                new_mcm[0] = mcm[0][:i] + mcm[0][i+1:]
                new_mcm[1].append(mcm[0][i])
                # Check the evidence
                ev = self.calc_log_evidence(new_mcm)
                if ev > best_ev:
                    best_mcm = new_mcm
                    best_ev = ev
                # Print result
                if print_search:
                   print(new_mcm, ev)

            # Continue with moving another one only if evidence improved
            if best_mcm != mcm:
                mcm = best_mcm
            else:
                break
            
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
    
    def find_best_im(self, max_interactions=None):
        """
        Find the best IM (independent model) for the data and transform data in that basis

        Parameters
        ----------
        max_interactions : int
            Maximal order of interaction to restrict the search

        Returns
        -------
        best_im : list
            array that represents the im with the n most biased operators in the original basis
        """
        while True:
            # Array with the spinoperators in the best IM
            im = []
            # Complete model that can be formed with the IM 
            model_im = []

            if max_interactions is None:
                # Include all operators
                all_ops = utils.generate_all_ops(self.n_var)
            else:
                # Generate all operators upto a specific order
                all_ops = utils.generate_ops_upto_order_n(self.n_var, max_interactions)

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
            
            # If all operators where generated, overall best IM is found in one iteration
            if max_interactions == None:
                self.best_im = im
                self.transform_data(im)
                break
            # If the basis operators are the best IM, algorithm is converged
            if im == [2**i for i in range(self.n_var)]:
                break
            # Update best IM
            self.best_im = utils.gt_model(self.best_im, im)
            self.transform_data(im)
        return self.best_im
    
    def find_best_mcm(self, method='exhaustive', print_search=False):
        """
        Find the best mcm for the data

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
            raise NameError('Unknown method. Options are "exhaustive", "greedy" and "divide_and_conquer.')
        
        self.best_mcm = best_mcm
        self.best_evidence = best_ev
        return best_mcm
