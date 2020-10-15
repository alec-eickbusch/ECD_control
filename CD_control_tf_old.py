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
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        max_alpha=5,
        max_beta=5,
        saving_directory=None,
        name="CD_control",
        term_fid=0.999,
        beta_penalty_multiplier=0,
        use_displacements=True,
        unitary_optimization=False,
        no_CD_end=True,
    ):

        self.initial_state = (
            tfq.qt2tf(initial_state) if initial_state is not None else None
        )
        self.target_state = (
            tfq.qt2tf(target_state) if target_state is not None else None
        )
        self.target_unitary = (
            tfq.qt2tf(target_unitary) if target_unitary is not None else None
        )
        self.unitary_optimization = unitary_optimization
        self.N_blocks = N_blocks

        self.set_tf_vars(betas=betas, phis=phis, thetas=thetas)

        self.max_alpha = max_alpha if use_displacements else 0.0
        self.max_beta = max_beta
        self.saving_directory = saving_directory
        self.name = name
        self.term_fid = term_fid
        self.beta_penalty_multiplier = beta_penalty_multiplier
        self.use_displacements = use_displacements
        self.no_CD_end = no_CD_end

        # todo: handle case when initial state is a tf object.
        if unitary_optimization:
            self.N_cav = target_unitary.dims[0][1]
        else:
            self.N_cav = initial_state.dims[0][1]
        self.a = tfq.destroy(self.N_cav)
        self.adag = tfq.create(self.N_cav)
        self.q = tfq.position(self.N_cav)
        self.p = tfq.momentum(self.N_cav)
        self.n = tfq.num(self.N_cav)
        self.I = tfq.qt2tf(qt.tensor(qt.identity(2), qt.identity(self.N_cav)))

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(self.q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(self.p)
        (self._eig_n, self._U_n) = tf.linalg.eigh(self.n)

        self._qp_comm = tf.linalg.diag_part(self.q @ self.p - self.p @ self.q)

    @tf.function
    def construct_displacement_operators(self, alphas_rho, alphas_angle):

        # Reshape amplitudes for broadcast against diagonals
        sqrt2 = tf.math.sqrt(tf.constant(2, dtype=tf.float32))
        cosines = tf.math.cos(alphas_angle)
        sines = tf.math.sin(alphas_angle)
        re_a = tf.cast(
            tf.reshape(sqrt2 * alphas_rho * cosines, [alphas_rho.shape[0], 1]),
            dtype=tf.complex64,
        )
        im_a = tf.cast(
            tf.reshape(sqrt2 * alphas_rho * sines, [alphas_rho.shape[0], 1]),
            dtype=tf.complex64,
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
    def construct_block_operators(self, betas_rho, betas_angle, phis, thetas):

        Bs_g_rho = betas_rho / tf.constant(2, dtype=tf.float32)
        Bs_g_angle = betas_angle
        ds = self.construct_displacement_operators(Bs_g_rho, Bs_g_angle)
        ds_dag = tf.linalg.adjoint(ds)
        Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
            2, dtype=tf.float32
        )
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
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
    def state_overlap(self, betas_rho, betas_angle, phis, thetas):
        # U = tf.eye(2, 2, dtype=tf.complex64)
        bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        psi = self.initial_state
        for U in tf.reverse(bs, axis=[0]):
            psi = U @ psi
        psi_target_dag = tf.linalg.adjoint(self.target_state)
        overlap = psi_target_dag @ psi
        return overlap

    @tf.function
    def state_fidelity(self, betas_rho, betas_angle, phis, thetas):
        overlap = self.state_overlap(betas_rho, betas_angle, phis, thetas)
        fid = tf.cast(overlap * tf.math.conj(overlap), dtype=tf.float32)
        return fid

    @tf.function
    def U_tot(self, betas_rho, betas_angle, phis, thetas):
        bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        U_c = tf.scan(lambda a, b: tf.matmul(a, b), bs)[-1]
        # U_c = self.I
        # for U in bs:
        #     U_c = U_c @ U  # following convention of state_fidelity..
        return U_c

    @tf.function
    def state_fidelity_unitary(self, betas_rho, betas_angle, phis, thetas):
        U_circuit = self.U_tot(betas_rho, betas_angle, phis, thetas)
        D = self.N_cav * 2
        overlap = tf.linalg.trace(tf.linalg.adjoint(self.target_unitary) @ U_circuit)
        return tf.abs((1.0 / D) * overlap) ** 2

    def optimize(self, learning_rate=0.1, epoch_size=100, epochs=100, df_stop=1e-5):
        optimizer = tf.optimizers.Adam(learning_rate)
        variables = [self.betas_rho, self.betas_angle, self.phis, self.thetas]

        # todo: change to log
        @tf.function
        def loss_fun(fid):
            # minus = tf.constant(-1, dtype=tf.float32)
            return tf.math.log(1 - fid)

        fid_func = (
            self.state_fidelity
            if not self.unitary_optimization
            else self.state_fidelity_unitary
        )
        fid = fid_func(self.betas_rho, self.betas_angle, self.phis, self.thetas)
        initial_loss = loss_fun(fid)
        print("Epoch: 0 Fidelity: {}".format(1 - np.exp(initial_loss.numpy())))
        # def callback_early_stop()

        for epoch in range(epochs + 1)[1:]:
            for _ in range(epoch_size):
                with tf.GradientTape() as tape:
                    new_fid = fid_func(
                        self.betas_rho, self.betas_angle, self.phis, self.thetas
                    )
                    loss = loss_fun(new_fid)
                    dloss_dvar = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(dloss_dvar, variables))
            df = new_fid - fid
            fid = new_fid
            print("Epoch: {} Fidelity: {} dF: {}".format(epoch, fid, df))
            if fid >= self.term_fid:
                self.print_info()
                return fid
            if np.abs(df) < df_stop:
                self.print_info()
                return new_fid
            fid = new_fid

        self.print_info()
        fid = 1 - np.exp(loss.numpy())
        return fid[0, 0]

    # TODO: update for tf
    def randomize(self, beta_scale=None, alpha_scale=None):
        beta_scale = self.max_beta if beta_scale is None else beta_scale
        alpha_scale = self.max_alpha if alpha_scale is None else alpha_scale
        ang_beta = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_beta = np.random.uniform(0, beta_scale, self.N_blocks)
        ang_alpha = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_alpha = np.random.uniform(0, alpha_scale, self.N_blocks)
        phis = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        thetas = np.random.uniform(-np.pi, np.pi, self.N_blocks)

        self.betas_rho = tf.Variable(rho_beta, dtype=tf.float32, trainable=True)
        self.betas_angle = tf.Variable(ang_beta, dtype=tf.float32, trainable=True)
        self.alphas_rho = tf.Variable(rho_alpha, dtype=tf.float32, trainable=True)
        self.alphas_angle = tf.Variable(ang_alpha, dtype=tf.float32, trainable=True)
        self.phis = tf.Variable(phis, dtype=tf.float32, trainable=True)
        self.thetas = tf.Variable(thetas, dtype=tf.float32, trainable=True)

    def get_numpy_vars(self, betas_rho=None, betas_angle=None, phis=None, thetas=None):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas

        betas = betas_rho.numpy() * np.exp(1j * betas_angle.numpy())
        phis = phis.numpy()
        thetas = thetas.numpy()

        return betas, phis, thetas

    def set_tf_vars(self, betas=None, phis=None, thetas=None):
        # if None, set to zero
        self.betas_rho = (
            tf.Variable(np.abs(np.array(betas)), dtype=tf.float32, trainable=True)
            if betas is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )
        self.betas_angle = (
            tf.Variable(np.angle(np.array(betas)), dtype=tf.float32, trainable=True)
            if betas is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )
        """
        self.alphas_rho = (
            tf.Variable(np.abs(np.array(self.alphas)), dtype=tf.float32, trainable=True)
            if alphas is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )
        self.alphas_angle = (
            tf.Variable(np.angle(np.array(self.alphas)), dtype=tf.float32, trainable=True)
            if alphas is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )
        """
        self.phis = (
            tf.Variable(phis, dtype=tf.float32, trainable=True)
            if phis is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )
        self.thetas = (
            tf.Variable(thetas, dtype=tf.float32, trainable=True)
            if phis is not None
            else tf.Variable(tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True)
        )

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

    def print_info(
        self, betas_rho=None, betas_angle=None, phis=None, thetas=None, human=True
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        f = (
            self.state_fidelity(betas_rho, betas_angle, phis, thetas)
            if not self.unitary_optimization
            else self.state_fidelity_unitary(betas_rho, betas_angle, phis, thetas)
        )
        betas, phis, thetas = self.get_numpy_vars(betas_rho, betas_angle, phis, thetas)
        if human:
            with np.printoptions(precision=5, suppress=True):
                print("\n\n" + str(self.name))
                print("N_blocks:     " + str(self.N_blocks))
                print("betas:        " + str(betas))
                # print("alphas:       " + str(alphas))
                print("phis (deg):   " + str(phis * 180.0 / np.pi))
                print("thetas (deg): " + str(thetas * 180.0 / np.pi))
                print("Fidelity:     %.5f" % f)
                print("\n")
        else:
            print("\n\n" + str(self.name))
            print("N_blocks: " + repr(self.N_blocks))
            print("betas: " + repr(betas))
            # print("alphas: " + repr(alphas))
            print("phis: " + repr(phis))
            print("thetas: " + repr(thetas))
            print("Fidelity: " + repr(f))
            print("\n")


# %%
