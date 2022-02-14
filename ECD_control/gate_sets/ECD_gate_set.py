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

class ECDGateSet(GateSet):

    def __init__(self, N_blocks=20, name="ECD_control", **kwargs):
        super().__init__(N_blocks, name)
        # combine all keywork arguments
        self.parameters = {**self.parameters, **kwargs} # python 3.9: self.parameters | kwargs
        self._construct_needed_matrices()

    def _construct_needed_matrices(self):
        N_cav = self.parameters["N_cav"]
        q = tfq.position(N_cav)
        p = tfq.momentum(N_cav)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)

        # if self.parameters["optimization_type"] == "unitary":
        #     P_cav = self.parameters["P_cav"]
        #     partial_I = np.array(qt.identity(N_cav))
        #     for j in range(P_cav, N_cav):
        #         partial_I[j, j] = 0
        #     partial_I = qt.Qobj(partial_I)
        #     self.P_matrix = tfq.qt2tf(qt.tensor(qt.identity(2), partial_I))

    
    def create_optimization_masks(self, length):

        masks = []
        if self.parameters['beta_mask'] is None:
            beta_mask = np.ones(
                shape=(self.parameters["N_blocks"], length),
                dtype=np.float32,
            )
            if self.parameters["no_CD_end"]:
                beta_mask[-1, :] = 0  # don't optimize final CD
            masks.append(beta_mask)
        else:
            # TODO: add mask to self.parameters for saving if it's non standard!
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if self.parameters['alpha_mask'] is None and self.parameters['use_displacements']:
            alpha_mask = np.ones(
                shape=(1, length), dtype=np.float32,
            )
            masks.append(alpha_mask)
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if self.parameters['phi_mask'] is None:
            phi_mask = np.ones(
                shape=(self.parameters["N_blocks"], length),
                dtype=np.float32,
            )
            phi_mask[0, :] = 0  # stop gradient on first phi entry
            masks.append(phi_mask)
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if self.parameters['eta_mask'] is None:
            eta_mask = np.ones(
                shape=(self.parameters["N_blocks"], length),
                dtype=np.float32,
            )
            phi_mask[0, :] = 0  # stop gradient on first phi entry
            masks.append(eta_mask)
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if self.parameters['theta_mask'] is None:
            theta_mask = np.ones(
                shape=(self.parameters["N_blocks"], length),
                dtype=np.float32,
            )
            masks.append(theta_mask)
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        
        return masks


    @tf.function
    def batch_construct_displacement_operators(self, alphas):

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        re_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.real(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )
        im_a = tf.reshape(
            sqrt2 * tf.cast(tf.math.imag(alphas), dtype=tf.complex64),
            [alphas.shape[0], alphas.shape[1], 1],
        )

        # Exponentiate diagonal matrices
        expm_q = tf.linalg.diag(tf.math.exp(1j * im_a * self._eig_q))
        expm_p = tf.linalg.diag(tf.math.exp(-1j * re_a * self._eig_p))
        expm_c = tf.linalg.diag(tf.math.exp(-0.5 * re_a * im_a * self._qp_comm))

        # Apply Baker-Campbell-Hausdorff
        return tf.cast(
            self._U_q
            @ expm_q
            @ tf.linalg.adjoint(self._U_q)
            @ self._U_p
            @ expm_p
            @ tf.linalg.adjoint(self._U_p)
            @ expm_c,
            dtype=tf.complex64,
        )

    @tf.function
    def batch_construct_block_operators(self, opt_vars):

        if self.parameters['use_displacements']:
            alphas_rho = opt_vars["alphas_rho"]
            alphas_angle = opt_vars["alphas_angle"]
        betas_rho = opt_vars["betas_rho"]
        betas_angle = opt_vars["betas_angle"]
        phis = opt_vars["phis"]
        etas = opt_vars["etas"]
        thetas = opt_vars["thetas"]
        
        # conditional displacements
        Bs = (
            tf.cast(betas_rho, dtype=tf.complex64)
            / tf.constant(2, dtype=tf.complex64)
            * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(betas_angle, dtype=tf.complex64)
            )
        )

        # final displacement
        if self.parameters['use_displacements']:
            D = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
                tf.constant(1j, dtype=tf.complex64)
                * tf.cast(alphas_angle, dtype=tf.complex64)
            )
            ds_end = self.batch_construct_displacement_operators(D)
        else:
            D = tf.zeros((1, betas_rho.shape[1]))
            ds_end = self.batch_construct_displacement_operators(D)
            # ds_end = tf.eye(self.parameters['N_cav'], batch_shape=) # figure this out; faster
        
        ds_g = self.batch_construct_displacement_operators(Bs)
        ds_e = tf.linalg.adjoint(ds_g)

        Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
            2, dtype=tf.float32
        )
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
        Phis = tf.cast(
            tf.reshape(Phis, [Phis.shape[0], Phis.shape[1], 1, 1]), dtype=tf.complex64
        )
        etas = tf.cast(
            tf.reshape(etas, [etas.shape[0], etas.shape[1], 1, 1]), dtype=tf.complex64
        )
        Thetas = tf.cast(
            tf.reshape(Thetas, [Thetas.shape[0], Thetas.shape[1], 1, 1]),
            dtype=tf.complex64,
        )

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        im = tf.constant(1j, dtype=tf.complex64)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)
        cos_e = tf.math.cos(etas)
        sin_e = tf.math.sin(etas)

        # constructing the blocks of the matrix
        ul = (cos + im * sin * cos_e) * ds_g
        ll = exp * sin * sin_e * ds_e
        ur = tf.constant(-1, dtype=tf.complex64) * exp_dag * sin * sin_e * ds_g
        lr = (cos - im * sin * cos_e) * ds_e

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        # append a final block matrix with a single displacement in each quadrant
        blocks = tf.concat(
            [
                -1j * tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2),
                tf.concat(
                    [
                        tf.concat([ds_end, tf.zeros_like(ds_end)], 3),
                        tf.concat([tf.zeros_like(ds_end), ds_end], 3),
                    ],
                    2,
                ),
            ],
            0,
        )
        return blocks

    def randomize_and_set_vars(self, parallel):

        init_vars = {}

        beta_scale = self.parameters["beta_scale"]
        alpha_scale = self.parameters["alpha_scale"]
        theta_scale = self.parameters["theta_scale"]
        betas_rho = np.random.uniform(
            0,
            beta_scale,
            size=(self.parameters["N_blocks"], parallel),
        )
        betas_angle = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], parallel),
        )
        if self.parameters["use_displacements"]:
            alphas_rho = np.random.uniform(
                0, alpha_scale, size=(1, parallel),
            )
            alphas_angle = np.random.uniform(
                -np.pi, np.pi, size=(1, parallel),
            )
        phis = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], parallel),
        )
        if self.parameters["use_etas"]:  # eta range is 0 to pi.
            etas = np.random.uniform(
                -np.pi,
                np.pi,
                size=(self.parameters["N_blocks"], parallel),
            )
        thetas = np.random.uniform(
            -1 * theta_scale,
            theta_scale,
            size=(self.parameters["N_blocks"], parallel),
        )
        phis[0] = 0  # everything is relative to first phi
        if self.parameters["no_CD_end"]:
            betas_rho[-1] = 0
            betas_angle[-1] = 0
        init_vars["betas_rho"] = tf.Variable(
            betas_rho, dtype=tf.float32, trainable=True, name="betas_rho",
        )
        init_vars["betas_angle"] = tf.Variable(
            betas_angle, dtype=tf.float32, trainable=True, name="betas_angle",
        )
        if self.parameters["use_displacements"]:
            init_vars["alphas_rho"] = tf.Variable(
                alphas_rho, dtype=tf.float32, trainable=True, name="alphas_rho",
            )
            init_vars["alphas_angle"] = tf.Variable(
                alphas_angle, dtype=tf.float32, trainable=True, name="alphas_angle",
            )
        init_vars["phis"] = tf.Variable(phis, dtype=tf.float32, trainable=True, name="phis",)
        if self.parameters["use_etas"]:
            init_vars["etas"] = tf.Variable(
                etas, dtype=tf.float32, trainable=True, name="etas",
            )
        else:
            init_vars["etas"] = (tf.constant(
                (np.pi / 2.0) * np.ones_like(phis), name="etas", dtype=tf.float32,
            ))

        init_vars["thetas"] = (tf.Variable(
            thetas, dtype=tf.float32, trainable=True, name="thetas",
        ))

        return init_vars

    def preprocess_params_before_saving(self, opt_params, *args):
        processed_params = {}
        processed_params["betas"] = tf.Variable(tf.cast(opt_params["betas_rho"], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(opt_params["betas_angle"], dtype=tf.complex64)), name='betas', dtype=tf.complex64)
        if self.parameters['use_displacements']:
            processed_params["alphas"] = tf.Variable(tf.cast(opt_params["alphas_rho"], dtype=tf.complex64) * tf.math.exp(1j * tf.cast(opt_params["alphas_angle"], dtype=tf.complex64)), name='alphas', dtype=tf.complex64)
        processed_params["phis"] = ((opt_params["phis"] + np.pi) % (2 * np.pi) - np.pi)
        processed_params["etas"] = ((opt_params["etas"] + np.pi) % (2 * np.pi) - np.pi)
        processed_params["thetas"] = ((opt_params["thetas"] + np.pi) % (2 * np.pi) - np.pi)

        return processed_params

        # can do this better if we switch to a dict, then names are also automatic

"""
order of variables:
betas_rho
betas_angle
alphas_rho
alphas_angle
phis
etas
thetas
"""