import numpy as np
import scipy.optimize

class ising_model:

    def __init__(self, file):
        """
        Initialize an Ising model object.

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

        # Generate all pairwise operators in integer representation
        self.spin_op = self.generate_all_operators(2)
        # List that contains all modelparameters (hi, Jij)
        self.model_param = np.random.rand(self.n_var + (self.n_var * (self.n_var - 1)) // 2)
        # List containing the model distribution
        self.model_distr = np.zeros(2**self.n_var)
        self.calc_model_distr()

        # Calculate the empirical average for the first and second moment
        self.exp_s_data = self.calc_exp_data()
        # List containing empirical distribution
        self.emp_distr = self.calc_emp_distr()
    
    def generate_all_operators(self, n_inter):
        """
        Generate all possible spinoperators with at most n_inter variables.

        Parameter
        ---------
        n_inter : int
            Max number of spin variables in a single interaction
        
        Returns
        -------
        operators : array
            array containing the spinoperators
        """
        # Start with the the all zero operator (removed at the end)
        states = [self.n_var*'0']
        for i in range(self.n_var):
            for state in states[:]:
                # Generate new state with the ith bit equal to one if the current number of bits is low enough
                if state.count('1') < n_inter:
                    states.append(state[:self.n_var-i-1] + '1' + state[self.n_var-i:])
        
        # Convert string to integer representation
        return np.array([int(i, 2) for i in states[1:]])
    
    def count_ones(self, n):
        """
        Count the number of ones in the binary representation of an integer n.

        Parameters
        ----------
        n : int
            integer for which the number of ones in the binary representation is calculated
        
        Returns
        -------
        counter : int
            number of 1s in the binary representation of n
        """
        counter = 0
        while n:
            # Check if the last bit is a one
            counter += n & 1
            # Shift all bits to the right
            n >>= 1
        return counter
    
    def set_model_param(self, param):
        """
        Set the values for the model paramaters.

        Parameters
        ----------
        param : list 
            new values for the model parameters
        """
        # Check if the length of the input is correct
        if len(param) != len(self.model_param):
            raise ValueError("Number of model parameters does not match the given input.")
        self.model_param = param
        # Recalculate the model distribution with new parameters
        self.calc_model_distr()
    
    def calc_emp_distr(self):
        """
        Calculate the empirical distribution
        """
        emp_distr = np.empty(2**self.n_var)
        for state in range(2**self.n_var):
            emp_distr[state] = np.count_nonzero(self.data == state)
        return emp_distr / len(self.data)

    def calc_exp_data(self):
        """
        Calculate the empirical averages for all the spinoperators.

        Returns
        -------
        <phi_mu> : array
            array with the empirical average for every spinoperator
        """
        exp_s = np.zeros(len(self.model_param))

        for obs in self.data:
            for i, op in enumerate(self.spin_op):
                #value = self.count_ones(obs & op)%2
                value = bin(obs & op).count('1')
                exp_s[i] += ((-1)**(value))

        return exp_s / len(self.data)

    def calculate_p(self, state):
        """
        Calculate the (non-normalized) probability for a given state.

        Parameters
        ----------
        s : int
            state represented as integer
        
        Returns
        -------
        p_s : float
            probability of state s
        """
        value = 0

        for i, op in enumerate(self.spin_op):
            #value += self.model_param[i] * (-1)**(self.count_ones(state & op)%2)
            value += self.model_param[i] * (-1)**bin(state & op).count('1')

        return np.exp(value)

    def calc_model_distr(self):
        """Calculate the model distribution for the current model parameters."""
        for state in range(2**self.n_var):
            self.model_distr[state] = self.calculate_p(state)
        self.model_distr /= np.sum(self.model_distr)
    
    def calc_exp_model(self):
        """
        Calculate the expected value for the spin operators given the current model parameters.

        Returns
        -------
        <phi_mu> : array
            expected value for every spinoperator
        """
        exp_s = np.zeros(len(self.model_param))

        for state in range(2**self.n_var):
            for i, op in enumerate(self.spin_op):
                #exp_s[i] += (self.model_distr[state] * (-1)**(self.count_ones(state & op)%2))
                exp_s[i] += (self.model_distr[state] * (-1)**bin(state & op).count('1'))
        
        return exp_s

    def calc_KL_div(self):
        """
        Calculate the Kullback-Leibler divergence between the empirical distribution and the model distribution.

        Returns
        ------
        kl_div : float
            KL divergence
        """
        kl_div = 0

        for i in range(2**self.n_var):
            if self.emp_distr[i] != 0:
                kl_div -= (self.emp_distr[i] * np.log(self.model_distr[i] / self.emp_distr[i]))

        return kl_div

    def calc_jacobian(self):
        """
        Calculate the jacobian (derivative of the negative loglikelihood with respect to the model parameters.)

        Returns
        -------
        jacobian : array
            Jacobian given the current model parameters
        """
        exp_data = self.exp_s_data
        exp_model = self.calc_exp_model()

        jacobian = np.empty(len(exp_model))

        for i in range(len(jacobian)):
            jacobian[i] = exp_model[i] - exp_data[i]
        
        return jacobian
    
    def f_x(self, model_param):
        """
        Function to minimize ((KL divergence) in the steepest descent algorithm.

        Parameters
        ----------
        param : array
            new model parameters
        
        Returns
        -------
        kl_div : float
            KL divergence
        """
        # Set new parameters (+ recalculate model distribution)
        self.set_model_param(model_param)
        # Calculate new KL divergence with the empirical distribution
        return self.calc_KL_div()
    
    def grad_f(self, model_param):
        """
        Gradient of the function to minimize in the steepest descent algorithm.

        Parameters
        ----------
        param : array
            new model parameters
        
        Returns
        -------
        jacobian : array
            Jacobian for the given model parameters
        """
        # Set new parameters (+ recalculate model distribution)
        self.set_model_param(model_param)
        # Calculate the new gradient
        return self.calc_jacobian()

    def fit_model(self, n_iter = 500):
        """
        Find model parameters using the steepest descent algorithm.

        Parameters
        ----------
        n_iter : int
            Maximum number of iterations in the algorithm
        """
        for _ in range(n_iter):

            model_param = self.model_param
            s = - self.calc_jacobian()
            # perform a linesearch
            result = scipy.optimize.line_search(self.f_x, self.grad_f, model_param, s)
            alpha = result[0]

            # Update the model parameters
            self.set_model_param(model_param + alpha * s)

            # Check convergence
            if np.allclose(self.model_param, model_param):
                break
