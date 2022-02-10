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
    
    def __init__(
        self,
        optimization_type="state transfer",
        target_unitary=None,
        P_cav=None,
        N_cav=None,
        initial_states=None,
        target_states=None,
        expectation_operators=None,
        target_expectation_values=None,
        N_multistart=10,
        N_blocks=20,
        term_fid=0.99,  # can set >1 to force run all epochs
        dfid_stop=1e-4,  # can be set= -1 to force run all epochs
        learning_rate=0.01,
        epoch_size=10,
        epochs=100,
        use_etas=False,
        use_displacements=False,
        no_CD_end=False,
        name="ECD_control",
        filename=None,
        comment="",
        use_phase=False,  # include the phase in the optimization cost function. Important for unitaries.
        timestamps=[],
        **kwargs
    ):
        self.parameters = {
            'optimization_type' : optimization_type,
            'target_unitary' : target_unitary,
            'P_cav' : P_cav,
            'N_cav' : N_cav,
            'initial_states' : initial_states,
            'target_states' : target_states,
            'expectation_operators' : expectation_operators,
            'target_expectation_values' : target_expectation_values,
            'N_multistart' : N_multistart,
            'N_blocks' : N_blocks,
            'term_fid' : term_fid,
            'dfid_stop' : dfid_stop,
            'learning_rate' : learning_rate,
            'epoch_size' : epoch_size,
            'epochs' : epochs,
            'use_etas' : use_etas,
            'use_displacements' : use_displacements,
            'no_CD_end' : no_CD_end,
            'name' : name,
            'filename' : filename,
            'comment' : comment,
            'use_phase' : use_phase,
            'timestamps' : timestamps,
            }
        self.parameters.update(kwargs)
        

    def modify_parameters(self, **kwargs):
        # currently, does not support changing optimization type.
        # todo: update for multi-state optimization and unitary optimziation
        parameters = kwargs
        for param, value in self.parameters.items():
            if param not in parameters:
                parameters[param] = value
        # handle things that are not in self.parameters:
        parameters["initial_states"] = (
            parameters["initial_states"]
            if "initial_states" in parameters
            else self.initial_states
        )
        parameters["target_states"] = (
            parameters["target_states"]
            if "target_states" in parameters
            else self.target_states
        )
        parameters["filename"] = (
            parameters["filename"] if "filename" in parameters else self.filename
        )
        parameters["timestamps"] = (
            parameters["timestamps"] if "timestamps" in parameters else self.timestamps
        )
        self.__init__(**parameters)

    @tf.function
    def batch_construct_block_operators(self, opt_params, *args):
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
        This function needs to return a list of tf.Variable of dimension (N_blocks, N_multistart) with initialized values.
        """

        pass

    def create_optimization_mask(self, *args):
        """
        Under construction.
        """
        
        self.optimization_mask = None
        pass