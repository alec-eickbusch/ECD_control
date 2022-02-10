TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

print(
    "\nNeed tf version 2.3.0 or later. Using tensorflow version: "
    + tf.__version__
    + "\n"
)
import ECD_control.ECD_optimization.tf_quantum as tfq
from ECD_control.ECD_optimization.visualization import VisualizationMixin
import qutip as qt
import datetime
import time


class GateSet:

    """
    This class is intended to be a barebones implementation of a specific gate set. Here, we only want to define the blocks in the gate
    set and the parameters that will be optimized. This class will be passed to the GateSynthesizer class which will call the optimizer
    your choice to optimize the GateSet parameters.
    """
    
    def __init__(
        self,
        dimension=20,
        N_blocks=20,
        n_parameters=20,
        name="ECD_control",
        **kwargs
    ): # some of the above may not be necessary. i.e. dimension, N_blocks, n_parameters are implicit in some of the defs below. think about this
        self.parameters = {
            'dimension' : dimension,
            'n_parameters' : n_parameters,
            'N_blocks' : N_blocks,
            'name' : name,
            }
        self.parameters.update(kwargs)

        assert()
        

    def modify_parameters(self, **kwargs):
        # currently, does not support changing optimization type.
        # todo: update for multi-state optimization and unitary optimziation
        parameters = kwargs
        for param, value in self.parameters.items():
            if param not in parameters:
                parameters[param] = value
        self.__init__(**parameters)

    @tf.function
    def batch_construct_block_operators(self, opt_params: list, *args):
        """
        This function must take a list of tf.Variable defined in the same order as randomize_and_set_vars()
        and construct a batch of block operators. Note that the performance of the optimization depends heavily
        on your implementation of this function. For the best performance, do everything with vectorized operations
        and decorate your implementation with @tf.function.

        Parameters
        -----------
        opt_params  :   List of optimization parameters. This list must be of the same length
                        as the one defined in ``randomize_and_set_vars``. Each element in the list
                        should be of dimension (N_blocks, N_multistart).
        
        Returns
        -----------
        tf.tensor of block operators U of size (N_multistart, U.shape)
        """
        
        pass

    def randomize_and_set_vars(self):
        """
        This function creates the tf variables over which we will optimize and randomizes their initial values.

        Returns
        -----------
        list of tf.Variable of dimension (N_blocks, N_multistart) with initialized values.
        """

        pass

    def create_optimization_mask(self, *args):
        """
        Under construction.
        """
        
        self.optimization_mask = None
        pass