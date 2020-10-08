#%%
# a minimial implementation of a discrete version of grape
# consisting of a sequence of conditional displacement,
# displacement, and rotation pulses, with
# tuneable parameters

import numpy as np
import tensorflow as tf

#%%
import CD_control.tf_quantum as tfq
import matplotlib.pyplot as plt
from CD_control.helper_functions import plot_wigner
import qutip as qt
from datetime import datetime

#%%
class CD_control_tf:

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        initial_state=None,
        target_state=None,
        target_unitary=None,
        N_blocks=1,
        Bs=None,
        alphas=None,
        Phis=None,
        Thetas=None,
        max_alpha=5,
        max_beta=5,
        saving_directory=None,
        name="CD_control",
        term_fid=0.999,
        beta_penalty_multiplier=0,
        use_displacements=True,
        no_CD_end=True,
    ):

        self.initial_state = tfq.qt2tf(initial_state)
        self.target_state = tfq.qt2tf(target_state)
        self.N_blocks = N_blocks
        self.Bs = (
            tf.Variable(Bs, dtype=tf.complex64, trainable=True)
            if Bs is not None
            else tf.Variable(tf.zeros(N_blocks, dtype=tf.complex64), trainable=True)
        )
        self.alphas = (
            tf.Variable(alphas, dtype=tf.complex64, trainable=True)
            if alphas is not None
            else tf.Variable(tf.zeros(N_blocks, dtype=tf.complex64), trainable=True)
        )
        self.Phis = (
            tf.Variable(Phis, dtype=tf.float32, trainable=True)
            if Phis is not None
            else tf.Variable(tf.zeros(N_blocks, dtype=tf.float32), trainable=True)
        )
        self.Thetas = (
            tf.Variable(Thetas, dtype=tf.float32, trainable=True)
            if Thetas is not None
            else tf.Variable(tf.zeros(N_blocks, dtype=tf.float32), trainable=True)
        )

        self.max_alpha = max_alpha if use_displacements else 0.0
        self.max_beta = max_beta
        self.saving_directory = saving_directory
        self.name = name
        self.term_fid = term_fid
        self.beta_penalty_multiplier = beta_penalty_multiplier
        self.use_displacements = use_displacements
        self.no_CD_end = no_CD_end

        # todo: handle case when initial state is a tf object.
        self.N_cav = initial_state.dims[0][1]
        self.a = tfq.destroy(self.N_cav)
        self.adag = tfq.create(self.N_cav)
        self.q = tfq.position(self.N_cav)
        self.p = tfq.momentum(self.N_cav)
        self.n = tfq.num(self.N_cav)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(self.q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(self.p)
        (self._eig_n, self._U_n) = tf.linalg.eigh(self.n)

        self._qp_comm = tf.linalg.diag_part(self.q @ self.p - self.p @ self.q)

    @tf.function
    def construct_displacement_operators(self, Bs):

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.complex64))
        amplitude = sqrt2 * tf.cast(
            tf.reshape(Bs, [Bs.shape[0], 1]), dtype=tf.complex64
        )

        # Take real/imag of amplitude for the commutator part of the expansion
        re_a = tf.cast(tf.math.real(amplitude), dtype=tf.complex64)
        im_a = tf.cast(tf.math.imag(amplitude), dtype=tf.complex64)

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
    def construct_block_operators(self, Bs, Phis, Thetas):

        ds = self.construct_displacement_operators(Bs)
        ds_dag = tf.linalg.adjoint(ds)
        Phis = tf.cast(tfq.matrix_flatten(Phis), dtype=tf.complex64)
        Thetas = tf.cast(tfq.matrix_flatten(Thetas), dtype=tf.complex64)

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)

        # constructing the blocks of the matrix
        ul = cos * ds
        ll = exp * sin * ds_dag
        ur = tf.constant(-1, dtype=tf.complex64) * exp_dag * sin * ds
        lr = cos * ds_dag

        blocks = tf.concat([tf.concat([ul, ur], 2), tf.concat([ll, lr], 2)], 1)

        return blocks

    # TODO: use tf.einsum to quickly do these contractions
    @tf.function
    def state_overlap(self, Bs, Phis, Thetas):
        # U = tf.eye(2, 2, dtype=tf.complex64)
        bs = self.construct_block_operators(Bs, Phis, Thetas)
        psi = self.initial_state
        for U in tf.reverse(bs, axis=[0]):
            psi = U @ psi
        psi_target_dag = tf.linalg.adjoint(self.target_state)
        overlap = psi_target_dag @ psi
        return overlap

    @tf.function
    def state_fidelity(self, Bs, Phis, Thetas):
        overlap = self.state_overlap(Bs, Phis, Thetas)
        fid = tf.cast(overlap * tf.math.conj(overlap), dtype=tf.float32)
        return fid

    def optimize(self, num_rounds):
        optimizer = tf.optimizers.Adam(0.05)
        variables = [self.Bs, self.Phis, self.Thetas]

        # todo: change to log
        @tf.function
        def loss_fun(fid):
            minus = tf.constant(-1, dtype=tf.float32)
            return minus * fid

        for i in range(num_rounds):
            with tf.GradientTape() as tape:
                fid = self.state_fidelity(self.Bs, self.Phis, self.Thetas)
                loss = loss_fun(fid)
                dloss_dvar = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(dloss_dvar, variables))
            print("Loss: {}".format(loss.numpy()))

    # TODO: update for tf
    def randomize(self, beta_scale=None, alpha_scale=None):
        beta_scale = self.max_beta if beta_scale is None else beta_scale
        alpha_scale = self.max_alpha if alpha_scale is None else alpha_scale
        ang_beta = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_beta = np.random.uniform(0, beta_scale, self.N_blocks)
        ang_alpha = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_alpha = np.random.uniform(0, alpha_scale, self.N_blocks)
        phis = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        thetas = np.random.uniform(0, np.pi, self.N_blocks)
        self.betas = np.array(np.exp(1j * ang_beta) * rho_beta, dtype=np.complex128)
        if self.no_CD_end:
            self.betas[-1] = 0.0 + 1j * 0.0
        if self.use_displacements:
            self.alphas = np.array(
                np.exp(1j * ang_alpha) * rho_alpha, dtype=np.complex128
            )
        else:
            self.alphas = np.zeros(self.N_blocks, dtype=np.complex128)
        self.phis = np.array(phis, dtype=np.float64)
        self.thetas = np.array(thetas, dtype=np.float64)

    # when parameters is specificed, normalize_angles is being used
    # during the optimization. In this case, it normalizes the parameters
    # and returns the parameters without updating self.
    # if parameters not specified, just normalize self's angles.
    # todo: make faster with numpy...
    def normalize_angles(self, phis=None, thetas=None):
        do_return = phis is not None
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        thetas_new = []
        for theta in thetas:
            while theta < -np.pi:
                theta = theta + 2 * np.pi
            while theta > np.pi:
                theta = theta - 2 * np.pi
            thetas_new.append(theta)
        thetas = np.array(thetas_new)
        phis_new = []
        for phi in phis:
            while phi < -np.pi:
                phi = phi + 2 * np.pi
            while phi > np.pi:
                phi = phi - 2 * np.pi
            phis_new.append(phi)
        phis = np.array(phis_new)
        if do_return:
            return phis, thetas
        else:
            self.thetas = thetas
            self.phis = phis

    def save(self):
        datestr = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        filestring = self.saving_directory + self.name + "_" + datestr
        filename_np = filestring + ".npz"
        filename_qt = filestring + ".qt"
        np.savez(
            filename_np,
            betas=self.betas,
            alphas=self.alphas,
            phis=self.phis,
            thetas=self.thetas,
            max_alpha=self.max_alpha,
            max_beta=self.max_beta,
            name=self.name,
            circuits=self.circuits,
        )
        print("\n\nparameters saved as: " + filename_np)
        qt.qsave([self.initial_state, self.target_state], filename_qt)
        print("states saved as: " + filename_qt)
        self.print_info()
        # print('name for loading:' + filestring)
        return filestring

    def load(self, filestring):
        filename_np = filestring + ".npz"
        filename_qt = filestring + ".qt"
        f = np.load(filename_np)
        betas, alphas, phis, thetas, max_alpha, max_beta, name = (
            f["betas"],
            f["alphas"],
            f["phis"],
            f["thetas"],
            f["max_alpha"],
            f["max_beta"],
            str(f["name"]),
        )
        circuits = f["circuits"] if ["circuits"] in f else []
        print("loaded parameters from:" + filename_np)
        f.close()
        states = qt.qload(filename_qt)
        initial_state, target_state = states[0], states[1]
        print("loaded states from:" + filename_qt)
        self.__init__(
            initial_state=initial_state,
            target_state=target_state,
            N_blocks=len(betas),
            betas=betas,
            alphas=alphas,
            phis=phis,
            thetas=thetas,
            max_alpha=max_alpha,
            max_beta=max_beta,
            name=name,
            circuits=circuits,
        )
        self.print_info()

    def print_info(self, betas=None, alphas=None, phis=None, thetas=None, human=True):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        phis, thetas = self.normalize_angles(phis, thetas)
        f = (
            self.unitary_fidelity(betas, alphas, phis, thetas)
            if self.unitary_optimization
            else self.fidelity(betas, alphas, phis, thetas)
        )
        if human:
            with np.printoptions(precision=5, suppress=True):
                print("\n\n" + str(self.name))
                print("N_blocks:     " + str(self.N_blocks))
                print("betas:        " + str(betas))
                print("alphas:       " + str(alphas))
                print("phis (deg):   " + str(phis * 180.0 / np.pi))
                print("thetas (deg): " + str(thetas * 180.0 / np.pi))
                print("Fidelity:     %.5f" % f)
                print("\n")
        else:
            print("\n\n" + str(self.name))
            print("N_blocks: " + repr(self.N_blocks))
            print("betas: " + repr(betas))
            print("alphas: " + repr(alphas))
            print("phis: " + repr(phis))
            print("thetas: " + repr(thetas))
            print("Fidelity: " + repr(f))
            print("\n")


# %%
