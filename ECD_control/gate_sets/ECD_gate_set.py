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
from ECD_control.ECD_gate_sets.gate_set import GateSet
import qutip as qt
import datetime
import time

class ECDGateSet(GateSet):
    
    
    def create_optimization_masks(
        self, beta_mask=None, alpha_mask=None, phi_mask=None, eta_mask=None, theta_mask=None
    ):
        if beta_mask is None:
            beta_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
            if self.parameters["no_CD_end"]:
                beta_mask[-1, :] = 0  # don't optimize final CD
        else:
            # TODO: add mask to self.parameters for saving if it's non standard!
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if alpha_mask is None:
            alpha_mask = np.ones(
                shape=(1, self.parameters["N_multistart"]), dtype=np.float32,
            )
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if phi_mask is None:
            phi_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
            phi_mask[0, :] = 0  # stop gradient on first phi entry
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if eta_mask is None:
            eta_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
            phi_mask[0, :] = 0  # stop gradient on first phi entry
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        if theta_mask is None:
            theta_mask = np.ones(
                shape=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                dtype=np.float32,
            )
        else:
            raise Exception(
                "need to implement non-standard masks for batch optimization"
            )
        self.beta_mask = beta_mask
        self.alpha_mask = alpha_mask
        self.phi_mask = phi_mask
        self.eta_mask = eta_mask
        self.theta_mask = theta_mask


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
    def batch_construct_block_operators(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, etas, thetas
    ):
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
        D = tf.cast(alphas_rho, dtype=tf.complex64) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(alphas_angle, dtype=tf.complex64)
        )

        ds_end = self.batch_construct_displacement_operators(D)
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

    def randomize_and_set_vars(self):
        beta_scale = self.parameters["beta_scale"]
        alpha_scale = self.parameters["alpha_scale"]
        theta_scale = self.parameters["theta_scale"]
        betas_rho = np.random.uniform(
            0,
            beta_scale,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        betas_angle = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        if self.parameters["use_displacements"]:
            alphas_rho = np.random.uniform(
                0, alpha_scale, size=(1, self.parameters["N_multistart"]),
            )
            alphas_angle = np.random.uniform(
                -np.pi, np.pi, size=(1, self.parameters["N_multistart"]),
            )
        phis = np.random.uniform(
            -np.pi,
            np.pi,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        if self.parameters["use_etas"]:  # eta range is 0 to pi.
            etas = np.random.uniform(
                -np.pi,
                np.pi,
                size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
            )
        thetas = np.random.uniform(
            -1 * theta_scale,
            theta_scale,
            size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
        )
        phis[0] = 0  # everything is relative to first phi
        if self.parameters["no_CD_end"]:
            betas_rho[-1] = 0
            betas_angle[-1] = 0
        self.betas_rho = tf.Variable(
            betas_rho, dtype=tf.float32, trainable=True, name="betas_rho",
        )
        self.betas_angle = tf.Variable(
            betas_angle, dtype=tf.float32, trainable=True, name="betas_angle",
        )
        if self.parameters["use_displacements"]:
            self.alphas_rho = tf.Variable(
                alphas_rho, dtype=tf.float32, trainable=True, name="alphas_rho",
            )
            self.alphas_angle = tf.Variable(
                alphas_angle, dtype=tf.float32, trainable=True, name="alphas_angle",
            )
        else:
            self.alphas_rho = tf.constant(
                np.zeros(shape=((1, self.parameters["N_multistart"]))),
                dtype=tf.float32,
            )
            self.alphas_angle = tf.constant(
                np.zeros(shape=((1, self.parameters["N_multistart"]))),
                dtype=tf.float32,
            )
        self.phis = tf.Variable(phis, dtype=tf.float32, trainable=True, name="phis",)
        if self.parameters["use_etas"]:
            self.etas = tf.Variable(
                etas, dtype=tf.float32, trainable=True, name="etas",
            )
        else:
            self.etas = tf.constant(
                (np.pi / 2.0) * np.ones_like(phis), dtype=tf.float32,
            )

        self.thetas = tf.Variable(
            thetas, dtype=tf.float32, trainable=True, name="thetas",
        )