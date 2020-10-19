#%%
# a minimial implementation of a discrete version of grape
# consisting of a sequence of conditional displacement,
# displacement, and rotation pulses, with
# tuneable parameters

import numpy as np
import tensorflow as tf

print(
    "\nNeed tf version 2.3.0 or later. Using tensorflow version: "
    + tf.__version__
    + "\n"
)
#%%
import unitary_decomposition_optimizer.tf_quantum as tfq
import qutip as qt
from datetime import datetime

#%%
class BatchedGlobalOptimizer:

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        optimization_type="state transfer",
        target_unitary=None,
        P_cav=None,
        initial_states=None,
        target_states=None,
        expectation_operators=None,
        target_expectation_values=None,
        N_multistart=10,
        N_blocks=20,
        # desired_loss=np.log(1e-3),\
        term_fid=0.99,
        use_displacements=True,
        no_CD_end=True,
    ):
        self.optimization_type = optimization_type
        if self.optimization_type == "state transfer":
            # self.initial_states = tf.stack(
            #    [tfq.qt2tf(state) for state in initial_states]
            # )
            # todo: can instead store dag of target states
            # self.target_states = tf.stack([tfq.qt2tf(state) for state in target_states])
            self.initial_state = tfq.qt2tf(initial_states[0])
            self.target_state = tfq.qt2tf(target_states[0])
            self.N_cav = self.initial_state.numpy().shape[0] // 2
        elif self.optimization_type == "unitary":
            self.target_unitary = tfq.qt2tf(target_unitary)
            self.N_cav = self.target_unitary.numpy().shape[0] // 2
            self.P_cav = P_cav if P_cav is not None else self.N_cav
        elif self.optimization_type == "expectation":
            pass
            # todo: handle this case.
        else:
            raise ValueError(
                "optimization_type must be one of \{'state transfer', 'unitary', 'expectation'\}"
            )
        self.N_blocks = N_blocks
        self.N_multistart = N_multistart
        self.use_displacements = use_displacements
        self.betas_rho = []
        self.betas_angle = []
        self.alphas_rho = []
        self.alphas_angle = []
        self.phis = []
        self.thetas = []
        self.term_fid = term_fid
        self.no_CD_end = no_CD_end
        self.randomize_and_set_vars()
        # self.set_tf_vars(betasG=betas, alphas=alphas, phis=phis, thetas=thetas)

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

        if self.optimization_type == "unitary":
            partial_I = np.array(qt.identity(self.N_cav))
            for j in range(self.P_cav, self.N_cav):
                partial_I[j, j] = 0
            partial_I = qt.Qobj(partial_I)
            self.P_matrix = tfq.qt2tf(qt.tensor(qt.identity(2), partial_I))

    @tf.function
    def construct_displacement_operators_rho_angle(self, alphas_rho, alphas_angle):

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
    def construct_block_operators(self, betas_rho, betas_angle, phis, thetas):

        Bs_g_rho = betas_rho / tf.constant(2, dtype=tf.float32)
        Bs_g_angle = betas_angle
        ds = self.construct_displacement_operators_rho_angle(Bs_g_rho, Bs_g_angle)
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

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        blocks = -1j * tf.concat([tf.concat([ll, lr], 2), tf.concat([ul, ur], 2)], 1)
        return blocks

    @tf.function
    def batch_construct_block_operators_with_displacements(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):

        Bs_g = tf.cast(betas_rho, dtype=tf.complex64) / tf.constant(
            2, dtype=tf.complex64
        ) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(betas_angle, dtype=tf.complex64)
        ) + tf.cast(
            alphas_rho, dtype=tf.complex64
        ) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(alphas_angle, dtype=tf.complex64)
        )
        Bs_e = tf.constant(-1, dtype=tf.complex64) * tf.cast(
            betas_rho, dtype=tf.complex64
        ) / tf.constant(2, dtype=tf.complex64) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(betas_angle, dtype=tf.complex64)
        ) + tf.cast(
            alphas_rho, dtype=tf.complex64
        ) * tf.math.exp(
            tf.constant(1j, dtype=tf.complex64)
            * tf.cast(alphas_angle, dtype=tf.complex64)
        )
        ds_g = self.batch_construct_displacement_operators(Bs_g)
        ds_e = self.batch_construct_displacement_operators(Bs_e)
        Phis = phis - tf.constant(np.pi, dtype=tf.float32) / tf.constant(
            2, dtype=tf.float32
        )
        Thetas = thetas / tf.constant(2, dtype=tf.float32)
        Phis = tf.cast(
            tf.reshape(Phis, [Phis.shape[0], Phis.shape[1], 1, 1]), dtype=tf.complex64
        )
        Thetas = tf.cast(
            tf.reshape(Thetas, [Thetas.shape[0], Thetas.shape[1], 1, 1]),
            dtype=tf.complex64,
        )

        exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
        exp_dag = tf.linalg.adjoint(exp)
        cos = tf.math.cos(Thetas)
        sin = tf.math.sin(Thetas)

        # constructing the blocks of the matrix
        ul = cos * ds_g
        ll = exp * sin * ds_e
        ur = tf.constant(-1, dtype=tf.complex64) * exp_dag * sin * ds_g
        lr = cos * ds_e

        # without pi pulse, block matrix is:
        # (ul, ur)
        # (ll, lr)
        # however, with pi pulse included:
        # (ll, lr)
        # (ul, ur)
        # pi pulse also adds -i phase, however don't need to trck it unless using multiple oscillators.a
        blocks = -1j * tf.concat([tf.concat([ll, lr], 3), tf.concat([ul, ur], 3)], 2)
        return blocks

    @tf.function
    def state(
        self,
        i=0,
        betas_rho=None,
        betas_angle=None,
        alphas_rho=None,
        alphas_angle=None,
        phis=None,
        thetas=None,
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        alphas_rho = self.alphas_rho if alphas_rho is None else alphas_rho
        alphas_angle = self.alphas_angle if alphas_angle is None else alphas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        if self.use_displacements:
            bs = self.construct_block_operators_with_displacements(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
        else:
            bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        psi = self.initial_state
        for U in bs[:i]:
            psi = U @ psi
        return psi

    # TODO: How does it handle the if statements here?
    @tf.function
    def batch_final_states(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        if self.use_displacements:
            bs = self.batch_construct_block_operators_with_displacements(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
        else:
            bs = self.batch_construct_block_operators(
                betas_rho, betas_angle, phis, thetas
            )

        psis = tf.stack([self.initial_state] * self.N_multistart)
        # note: might be able to use tf.einsum or tf.scan (as done in U_tot) here.
        for U in bs:
            psis = U @ psis
        return psis

    @tf.function
    def batch_state_overlaps(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        psifs = self.batch_final_states(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        # todo: could be batched beforehand
        psits = tf.stack([self.target_state] * self.N_multistart)
        psi_ts_dag = tf.linalg.adjoint(psits)
        overlaps = psi_ts_dag @ psifs
        return overlaps

    @tf.function
    def batch_state_fidelities(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        overlaps = self.batch_state_overlaps(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        fids = tf.cast(overlaps * tf.math.conj(overlaps), dtype=tf.float32)
        return fids

    """
    @tf.function
    def state_fidelity(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        overlap = self.state_overlap(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        fid = tf.cast(overlap * tf.math.conj(overlap), dtype=tf.float32)
        return fid
    """

    @tf.function
    def mult_bin_tf(self, a):
        while a.shape[0] > 1:
            if a.shape[0] % 2 == 1:
                a = tf.concat(
                    [a[:-2], [tf.matmul(a[-2], a[-1])]], 0
                )  # maybe there's a faster way to deal with immutable constants
            a = tf.matmul(a[::2, ...], a[1::2, ...])
        return a[0]

    @tf.function
    def U_tot(self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas):
        if self.use_displacements:
            bs = self.construct_block_operators_with_displacements(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
        else:
            bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        # U_c = tf.scan(lambda a, b: tf.matmul(b, a), bs)[-1]
        U_c = self.mult_bin_tf(
            tf.reverse(bs, axis=[0])
        )  # [U_1,U_2,..] -> [U_N,U_{N-1},..]-> U_N @ U_{N-1} @ .. @ U_1
        # U_c = self.I
        # for U in bs:
        #     U_c = U @ U_c
        return U_c

    @tf.function
    def unitary_fidelity(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        U_circuit = self.U_tot(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        D = tf.constant(self.P_cav * 2, dtype=tf.complex64)
        overlap = tf.linalg.trace(
            tf.linalg.adjoint(self.target_unitary) @ self.P_matrix @ U_circuit
        )
        return tf.cast(
            (1.0 / D) ** 2 * overlap * tf.math.conj(overlap), dtype=tf.float32
        )

    def set_unitary_fidelity_state_basis(self, states):
        self.initial_unitary_states = states
        self.target_unitary_states = self.target_unitary @ states  # using broadcasting

    # returns <psi_f | O | psi_f>
    @tf.function
    def expectation_value(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas, O
    ):
        psif = self.final_state(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        psif_dag = tf.linalg.adjoint(psif)
        expect = psif_dag @ O @ psif
        return expect

    def optimize(
        self,
        learning_rate=0.01,
        epoch_size=100,
        epochs=100,
        dloss_stop=1e-6,
        beta_mask=None,
        phi_mask=None,
        theta_mask=None,
        alpha_mask=None,
        callback_fun=None,
    ):
        optimizer = tf.optimizers.Adam(learning_rate)
        if self.use_displacements:
            variables = [
                self.betas_rho,
                self.betas_angle,
                self.alphas_rho,
                self.alphas_angle,
                self.phis,
                self.thetas,
            ]
        else:
            variables = [self.betas_rho, self.betas_angle, self.phis, self.thetas]

        """
        if beta_mask is None:
            beta_mask = np.ones(self.N_blocks)
            if self.no_CD_end:
                beta_mask[-1] = 0  # don't optimize final CD

        if phi_mask is None:
            phi_mask = np.ones(self.N_blocks)
            phi_mask[0] = 0  # stop gradient on first phi entry

        if theta_mask is None:
            theta_mask = np.ones(self.N_blocks)

        if alpha_mask is None:
            alpha_mask = np.ones(self.N_blocks)

        beta_mask = tf.constant(beta_mask, dtype=tf.float32)
        phi_mask = tf.constant(phi_mask, dtype=tf.float32)
        theta_mask = tf.constant(theta_mask, dtype=tf.float32)
        alpha_mask = tf.constant(alpha_mask, dtype=tf.float32)

        @tf.function
        def entry_stop_gradients(target, mask):
            mask_h = tf.abs(mask - 1)
            return tf.stop_gradient(mask_h * target) + mask * target

        if self.optimize_expectation:

            @tf.function
            def loss_fun(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            ):
                expect = self.expectation_value(
                    betas_rho,
                    betas_angle,
                    alphas_rho,
                    alphas_angle,
                    phis,
                    thetas,
                    self.O,
                )
                return tf.math.log(1 - tf.math.real(expect))

        if self.unitary_optimization:
            if self.unitary_optimization == "states":

                @tf.function
                def loss_fun(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                ):
                    fid = self.unitary_fidelity_state_decomp(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                    )
                    return tf.math.log(1 - fid)

            else:

                @tf.function
                def loss_fun(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                ):
                    fid = self.unitary_fidelity(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                    )
                    return tf.math.log(1 - fid)

        else:

            @tf.function
            def loss_fun(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            ):
                fid = self.state_fidelity(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                )e
                return tf.math.log(1 - fid)
        """

        @tf.function
        def loss_fun(fids):
            # I think it's important that the log is taken before the avg
            losses = tf.math.log(1 - fids)
            avg_loss = tf.reduce_sum(losses) / self.N_multistart
            return avg_loss

        term_loss = np.log(1 - self.term_fid)

        # format of callback will always be
        # callback_fun(self, loss, dloss, epoch)
        # passing self to callback will allow one to print any values of the variables
        if callback_fun is None:

            def callback_fun(obj, fids, epoch):
                avg_fid = tf.reduce_sum(fids)/self.N_multistart
                max_fid = tf.reduce_max(fids)
                print(
                    "Epoch: %d Avg Fid: %.6f Max fid: %.6f"
                    % (epoch, avg_fid, max_fid)
                )

        initial_fids = self.batch_state_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        initial_loss = loss_fun(initial_fids)
        # callback_fun(self, initial_loss.numpy(), 0, 0)

        losses = []
        losses.append(initial_loss.numpy())
        loss = initial_loss
        all_fids = []
        for epoch in range(epochs + 1)[1:]:
            for _ in range(epoch_size):
                with tf.GradientTape() as tape:
                    """
                    betas_rho = entry_stop_gradients(self.betas_rho, beta_mask)
                    betas_angle = entry_stop_gradients(self.betas_angle, beta_mask)
                    if self.use_displacements:
                        alphas_rho = entry_stop_gradients(self.alphas_rho, alpha_mask)
                        alphas_angle = entry_stop_gradients(
                            self.alphas_angle, alpha_mask
                        )
                    else:
                        alphas_rho = self.alphas_rho
                        alphas_angle = self.alphas_angle
                    phis = entry_stop_gradients(self.phis, phi_mask)
                    thetas = entry_stop_gradients(self.thetas, theta_mask)
                    """
                    fids = self.batch_state_fidelities(
                        self.betas_rho,
                        self.betas_angle,
                        self.alphas_rho,
                        self.alphas_angle,
                        self.phis,
                        self.thetas,
                    )
                    new_loss = loss_fun(fids)
                    dloss_dvar = tape.gradient(new_loss, variables)
                optimizer.apply_gradients(zip(dloss_dvar, variables))
            dloss = new_loss - loss
            loss = new_loss
            losses.append(loss.numpy())
            all_fids.append(fids)
            # print(fids)
            callback_fun(self,fids, epoch)
            condition = tf.greater(fids, self.term_fid)
            if tf.reduce_any(condition):
                # self.normalize_angles()
                #self.print_info()
                return np.squeeze(np.array(all_fids)).T
            # if np.abs(dloss) < dloss_stop:
            # self.normalize_angles()
            # self.print_info()
            # return np.squeeze(np.array(losses))
        # self.normalize_angles()
        #self.print_info()
        return np.squeeze(np.array(all_fids)).T

    # TODO: update for tf
    def randomize_and_set_vars(self, beta_scale=None, alpha_scale=None):
        beta_scale = 1.0 if beta_scale is None else beta_scale
        alpha_scale = 1.0 if alpha_scale is None else alpha_scale
        self.betas_rho = tf.Variable(
            np.random.uniform(0, beta_scale, size=(self.N_blocks, self.N_multistart)),
            dtype=tf.float32,
            trainable=True,
            name="betas_rho",
        )
        self.betas_angle = tf.Variable(
            np.random.uniform(-np.pi, np.pi, size=(self.N_blocks, self.N_multistart)),
            dtype=tf.float32,
            trainable=True,
            name="betas_angle",
        )
        if self.use_displacements:
            self.alphas_rho = tf.Variable(
                np.random.uniform(
                    0, beta_scale, size=(self.N_blocks, self.N_multistart)
                ),
                dtype=tf.float32,
                trainable=True,
                name="alphas_rho",
            )
            self.alphas_angle = tf.Variable(
                np.random.uniform(
                    -np.pi, np.pi, size=(self.N_blocks, self.N_multistart)
                ),
                dtype=tf.float32,
                trainable=True,
                name="alphas_angle",
            )
        self.phis = tf.Variable(
            np.random.uniform(-np.pi, np.pi, size=(self.N_blocks, self.N_multistart)),
            dtype=tf.float32,
            trainable=True,
            name="betas_rho",
        )
        self.thetas = tf.Variable(
            np.random.uniform(-np.pi, np.pi, size=(self.N_blocks, self.N_multistart)),
            dtype=tf.float32,
            trainable=True,
            name="betas_angle",
        )

    def get_numpy_vars(
        self,
        betas_rho=None,
        betas_angle=None,
        alphas_rho=None,
        alphas_angle=None,
        phis=None,
        thetas=None,
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        alphas_rho = self.alphas_rho if alphas_rho is None else alphas_rho
        alphas_angle = self.alphas_angle if alphas_angle is None else alphas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas

        betas = betas_rho.numpy() * np.exp(1j * betas_angle.numpy())
        alphas = alphas_rho.numpy() * np.exp(1j * alphas_angle.numpy())
        phis = phis.numpy()
        thetas = thetas.numpy()

        return betas, alphas, phis, thetas

    def set_tf_vars(self, betas=None, alphas=None, phis=None, thetas=None):
        # if None, sets to zero
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
        if self.use_displacements:
            self.alphas_rho = (
                tf.Variable(np.abs(np.array(alphas)), dtype=tf.float32, trainable=True)
                if alphas is not None
                else tf.Variable(
                    tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True
                )
            )
            self.alphas_angle = (
                tf.Variable(
                    np.angle(np.array(alphas)), dtype=tf.float32, trainable=True
                )
                if alphas is not None
                else tf.Variable(
                    tf.zeros(self.N_blocks, dtype=tf.float32), trainable=True
                )
            )
        else:
            self.alphas_rho = tf.constant(tf.zeros(self.N_blocks, dtype=tf.float32))
            self.alphas_angle = tf.constant(tf.zeros(self.N_blocks, dtype=tf.float32))

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
    def normalize_angles(self):
        betas, alphas, phis, thetas = self.get_numpy_vars()
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
        self.set_tf_vars(betas, alphas, phis, thetas)

    """
    def save(self):
        datestr = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        filestring = self.saving_directory + self.name + "_" + datestr
        filename_np = filestring + ".npz"
        filename_qt = filestring + ".qt"
        betas, phis, thetas = self.get_numpy_vars(betas_rho, betas_angle, phis, thetas)
        np.savez(
            filename_np,
            betas=betas,
            phis=phis,
            thetas=thetas,
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
    """

    def print_info(
        self,
        betas_rho=None,
        betas_angle=None,
        alphas_rho=None,
        alphas_angle=None,
        phis=None,
        thetas=None,
        human=True,
    ):
        betas_rho = self.betas_rho if betas_rho is None else betas_rho
        betas_angle = self.betas_angle if betas_angle is None else betas_angle
        alphas_rho = self.alphas_rho if alphas_rho is None else alphas_rho
        alphas_angle = self.alphas_angle if alphas_angle is None else alphas_angle
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        f = (
            self.state_fidelity(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
            if not self.unitary_optimization
            else (
                self.unitary_fidelity_state_decomp(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                )
                if self.unitary_optimization == "states"
                else self.unitary_fidelity(
                    betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
                )
            )
        )
        betas, alphas, phis, thetas = self.get_numpy_vars(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        if human:
            with np.printoptions(precision=5, suppress=True):
                print("\n\n" + str(self.name))
                print("N_blocks:      " + str(self.N_blocks))
                print("Unitary opt :  " + str(self.unitary_optimization))
                # todo: add expectation optimization info
                print("N_cav:         " + str(self.N_cav))
                print("P_cav:         " + str(self.P_cav))
                print("Term fid:      %.6f" % self.term_fid)
                print("displacements: " + str(self.use_displacements))
                print("betas:         " + str(betas))
                print("alphas:        " + str(alphas))
                print("phis (deg):    " + str(phis * 180.0 / np.pi))
                print("thetas (deg):  " + str(thetas * 180.0 / np.pi))
                print("Fidelity:      %.6f" % f)
                print("\n")
        else:
            print("\n\n" + str(self.name))
            print("N_blocks: " + repr(self.N_blocks))
            print("Unitary optimization: " + str(self.unitary_optimization))
            # todo: add expectation optimization info
            print("N_cav:         " + str(self.N_cav))
            print("P_cav: " + str(self.P_cav))
            print("Term fid: %.6f" % self.term_fid)
            print("Use displacements: " + str(self.use_displacements))
            print("betas: " + repr(betas))
            print("alphas: " + repr(alphas))
            print("phis: " + repr(phis))
            print("thetas: " + repr(thetas))
            print("Fidelity: " + repr(f))
            print("\n")


# %%
