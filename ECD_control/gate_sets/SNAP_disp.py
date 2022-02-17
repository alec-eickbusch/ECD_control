TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
END_OPT_STRING = "\n" + "=" * 60 + "\n"
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # supress warnings
import h5py

import ECD_control.ECD_optimization.tf_quantum as tfq
from ECD_control.gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time
from .operators import DisplacementOperator, SNAP


class SNAP_disp(GateSet):
    def __init__(self, N_blocks=20, name="SNAP_control", **kwargs):
        super().__init__(name)
        # combine all keyword arguments
        self.parameters = {
            **self.parameters,
            **kwargs,
        }  # python 3.9: self.parameters | kwargs
        self.disp_op = DisplacementOperator(self.parameters["N_cav"])
        self.snap = SNAP(self.parameters["N_cav"])

    @property
    def parameter_names(self):

        params = [
            "betas_rho",
            "betas_angle",
            "thetas"
        ]

        # for k in range(self.parameters["N_snap"]):
        #     params.append("theta" + str(k))

        return params

    def randomization_ranges(self):
        return {
            "betas_rho": (
                -1 * self.parameters["beta_scale"],
                1 * self.parameters["beta_scale"],
            ),
            "betas_angle": (-np.pi, np.pi),
            "thetas": (-np.pi, np.pi, self.parameters["N_snap"])
        }


    @tf.function
    def batch_construct_block_operators(self, opt_vars):
        betas_rho = opt_vars["betas_rho"]
        betas_angle = opt_vars["betas_angle"]
        thetas = opt_vars["thetas"]

        Bs = (
            tf.cast(betas_rho, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle, dtype=tf.complex64)
            )
        )

        Ds = self.disp_op(Bs)
        SNAPs = self.snap(thetas)
        
        blocks = Ds @ SNAPs
        return blocks

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["betas"] = tf.Variable(
            tf.cast(opt_params["betas_rho"], dtype=tf.complex64)
            * tf.math.exp(1j * tf.cast(opt_params["betas_angle"], dtype=tf.complex64)),
            name="betas",
            dtype=tf.complex64,
        )
        processed_params["thetas"] = (opt_params["thetas"] + np.pi) % (
            2 * np.pi
        ) - np.pi

        return processed_params