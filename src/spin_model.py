import numpy as np
import scipy.optimize

from . import utils

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

        # List containing the model distribution
        self.model_distr = np.zeros(2**self.n_var)
        # Generate all pairwise operators in integer representation
        self.spin_op = utils.generate_ops_upto_order_n(self.n_var, 2)
        # List that contains all modelparameters
        self.param = np.zeros(len(self.spin_op))
        self.set_param(np.random.rand(len(self.spin_op)))

        # List containing empirical distribution
        self.emp_distr = self.calc_emp_distr()
        # Calculate the empirical average for the first and second moment
        self.exp_s_data = self.calc_exp_data()
    
    def set_param(self, param):
        """
        Set the values for the model paramaters.

        Parameters
        ----------
        param : list 
            new values for the model parameters
        """
        # Check if the length of the input is correct
        if len(param) != len(self.spin_op):
            raise ValueError("Number of model parameters does not match the given input.")
        self.param = param
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
        exp_data = utils.fwht(self.emp_distr)
        return exp_data[self.spin_op]

    def calc_model_distr(self):
        """Calculate the model distribution for the current model parameters."""
        g = np.zeros(2**self.n_var)
        g[self.spin_op] = self.param

        energy = utils.fwht(g)
        model_distr = np.exp(energy)
        self.model_distr = model_distr / np.sum(model_distr)
    
    def calc_exp_model(self):
        """
        Calculate the expected value for the spin operators given the current model parameters.

        Returns
        -------
        <phi_mu> : array
            expected value for every spinoperator
        """
        exp_model = utils.fwht(self.model_distr)
        return exp_model[self.spin_op]

    def calc_KL_div(self):
        """
        Calculate the Kullback-Leibler divergence between the empirical distribution and the model distribution.

        Returns
        ------
        kl_div : float
            KL divergence
        """
        div = self.emp_distr / self.model_distr
        log = np.log(div, out=np.zeros_like(div), where=div!=0)

        kl_div = self.emp_distr @ log

        return kl_div

    def calc_jacobian(self):
        """
        Calculate the jacobian (derivative of the negative loglikelihood with respect to the model parameters.)

        Returns
        -------
        jacobian : array
            Jacobian given the current model parameters
        """
        jacobian = utils.fwht(self.model_distr - self.emp_distr)
        return jacobian[self.spin_op]
    
    def f_x(self, param):
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
        self.set_param(param)
        # Calculate new KL divergence with the empirical distribution
        return self.calc_KL_div()
    
    def grad_f(self, param):
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
        self.set_param(param)
        # Calculate the new gradient
        return self.calc_jacobian()

    def fit_param(self, n_iter = 500):
        """
        Find model parameters using the steepest descent algorithm.

        Parameters
        ----------
        n_iter : int
            Maximum number of iterations in the algorithm
        """
        for _ in range(n_iter):

            param = self.param
            s = - self.calc_jacobian()
            # perform a linesearch
            result = scipy.optimize.line_search(self.f_x, self.grad_f, param, s)
            alpha = result[0]

            # Update the model parameters
            self.set_param(param + alpha * s)

            # Check convergence
            if np.allclose(self.param, param):
                break
