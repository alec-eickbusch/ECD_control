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
from ECD_control.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time


class GateSynthesizer:

    def __init__(
        self,
        gateset: GateSet,
        optimization_type="state transfer",
        target_unitary=None,
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
        self.gateset = gateset