import numpy as np

class model_generator:

    def __init__(self, n_var):
        # Check type of the input parameters
        if type(n_var) != int:
            raise TypeError("The parameter n_var should be an integer instead of %s."%type(n_var))
        # Total number of spin_variables in the system
        self.n_var = n_var

        # List to store all mcms
        self.mcms = []
    
    def generate_all_ops(self):
        """
        Generate all possible spinoperators (complete model).
        
        Returns
        -------
        operators : array
            array containing the spinoperators
        """
        # Start with the the all zero operator
        states = [0]
        for i in range(self.n_var):
            for state in states[:]:
                states.append(state + 2**i)
        return np.array(states[1:])
    
    def generate_ops_upto_order_n(self, n_inter):
        """
        Generate all possible spinoperators with n_inter as highest order interaction terms.

        Parameter
        ---------
        n_inter : int
            Max number of spin variables in a single interaction
        
        Returns
        -------
        operators : array
            array containing the spinoperators
        """
        if type(n_inter) != int:
            raise TypeError("The parameter n_inter should be an integer instead of %s."%type(n_inter))
        # The max number of variables per interaction should be smaller than the total number of variables
        if n_inter > self.n_var:
            raise ValueError("The maximum number of variables per interaction (%i) should be smaller than the total number of variables (%i)."%(n_inter, self.n_var))
        
        # Start with the the all zero operator
        states = [0]
        for i in range(self.n_var):
            for state in states[:]:
                if state.bit_count() < n_inter:
                    states.append(state + 2**i)
        return np.array(states[1:])

    def generate_pairwise_ops(self):
        """Generate all pairwise operators (Ising model)."""
        return self.generate_ops_upto_order_n(2)
    
    def generate_all_mcms(self):
        """
        Generate all possible MCMs for the current number of spinvariables.
        All the generated MCMs are also stored in self.mcms

        Returns
        -------
        all_mcms : list
            List with all the possible MCMs
        """
        # Reset list to empty list
        self.mcms = []
        self.generate_partitions([], 0)
        return self.mcms
    
    def generate_partitions(self, part, i):
        if i == self.n_var:
            # All n variables are added to a subpartition (-> complete partition)
            # Generate the MCM that corresponds to this partition and return to create next partition
            self.generate_mcm_ops(part)
            return
        
        for j in range(len(part)):
            # Add spinvariable to a subpartition (generate child node) and continue with the next spinvariable
            part[j].append(i)
            self.generate_partitions(part, i+1)
            # Remove spin variable from the current subpartition (going back to the parent node in the tree)
            part[j].pop()
        
        # Add spinvariable as new subpartition (generate child node) and continue with the next spinvariable 
        part.append([i])
        self.generate_partitions(part, i+1)
        # Remove the current subpartition (going back to the parent node in the tree)
        part.pop()
    
    def generate_mcm_ops(self, partition):
        """
        Generate all operators for the MCM corresponding to the given partition.
        Operator representation of the MCM is added to the list of all MCMs as well.

        Parameters
        ----------
        partition : list
            2 dimensional list containing the index of each variable per subpartition
        
        Returns
        -------
        ops : array
            All operators belonging to the MCM
        """
        ops = []
        for subpart in partition:
            # Generate all operators for each subpartition
            for i in range(len(subpart)):
                ops.append(2**subpart[i])
                for j in range(i+1, len(subpart)):
                    ops.append(2**subpart[i] + 2**subpart[j])
        
        self.mcms.append(np.sort(ops))
        return np.sort(ops)
    