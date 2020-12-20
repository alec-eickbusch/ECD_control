#%%
# note: timestamp can't use "/" character for h5 saving.
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
import CD_control_optimization.tf_quantum as tfq
from CD_control_optimization.visualization import VisualizationMixin
import qutip as qt
import datetime
import time


class BatchOptimizer(VisualizationMixin):

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
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
        beta_scale=1.0,
        alpha_scale=1.0,
        use_displacements=True,
        no_CD_end=True,
        beta_mask=None,
        phi_mask=None,
        theta_mask=None,
        alpha_mask=None,
        name="CD_control",
        filename=None,
        comment="",
        timestamps=[],
        **kwargs
    ):
        self.parameters = {
            "optimization_type": optimization_type,
            "N_multistart": N_multistart,
            "N_blocks": N_blocks,
            "term_fid": term_fid,
            "dfid_stop": dfid_stop,
            "no_CD_end": no_CD_end,
            "learning_rate": learning_rate,
            "epoch_size": epoch_size,
            "epochs": epochs,
            "beta_scale": beta_scale,
            "alpha_scale": alpha_scale,
            "use_displacements": use_displacements,
            "name": name,
            "comment": comment,
        }
        self.parameters.update(kwargs)
        if (
            self.parameters["optimization_type"] == "state transfer"
            or self.parameters["optimization_type"] == "analysis"
        ):
            self.batch_fidelities = (
                self.batch_state_transfer_fidelities
            )  # set fidelity function

            self.initial_states = tf.stack(
                [tfq.qt2tf(state) for state in initial_states]
            )

            self.target_unitary = tfq.qt2tf(target_unitary)

            # if self.target_unitary is not None: TODO
            #     raise Exception("Need to fix target_unitary multi-state transfer generation!")

            self.target_states = (  # store dag
                tf.stack([tfq.qt2tf(state) for state in target_states])
                if self.target_unitary is None
                else self.target_unitary @ self.initial_states
            )

            self.target_states_dag = tf.linalg.adjoint(
                self.target_states
            )  # store dag to avoid having to take adjoint

            N_cav = self.initial_states[0].numpy().shape[0] // 2
        elif self.parameters["optimization_type"] == "unitary":
            self.target_unitary = tfq.qt2tf(target_unitary)
            N_cav = self.target_unitary.numpy().shape[0] // 2
            P_cav = P_cav if P_cav is not None else N_cav
            raise Exception("Need to implement unitary optimization")

        elif self.parameters["optimization_type"] == "expectation":
            raise Exception("Need to implement expectation optimization")
        else:
            raise ValueError(
                "optimization_type must be one of {'state transfer', 'unitary', 'expectation', 'analysis'}"
            )

        self.parameters["N_cav"] = N_cav
        if P_cav is not None:
            self.parameters["P_cav"] = P_cav

        # TODO: handle case when you pass initial params. In that case, don't randomize, but use "set_tf_vars()"
        self.randomize_and_set_vars()
        # self.set_tf_vars(betas=betas, alphas=alphas, phis=phis, thetas=thetas)

        self._construct_needed_matrices()
        self._construct_optimization_masks(beta_mask, alpha_mask, phi_mask, theta_mask)

        # opt data will be a dictionary of dictonaries used to store optimization data
        # the dictionary will be addressed by timestamps of optmization.
        # each opt will append to opt_data a dictionary
        # this dictionary will contain optimization parameters and results

        self.timestamps = timestamps
        self.filename = (
            filename
            if (filename is not None and filename is not "")
            else self.parameters["name"]
        )
        path = self.filename.split(".")
        if len(path) < 2 or (len(path) == 2 and path[-1] != ".h5"):
            self.filename = path[0] + ".h5"

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

    def _construct_needed_matrices(self):
        N_cav = self.parameters["N_cav"]
        q = tfq.position(N_cav)
        p = tfq.momentum(N_cav)

        # Pre-diagonalize
        (self._eig_q, self._U_q) = tf.linalg.eigh(q)
        (self._eig_p, self._U_p) = tf.linalg.eigh(p)

        self._qp_comm = tf.linalg.diag_part(q @ p - p @ q)

        if self.parameters["optimization_type"] == "unitary":
            P_cav = self.parameters["P_cav"]
            partial_I = np.array(qt.identity(N_cav))
            for j in range(P_cav, N_cav):
                partial_I[j, j] = 0
            partial_I = qt.Qobj(partial_I)
            self.P_matrix = tfq.qt2tf(qt.tensor(qt.identity(2), partial_I))

    def _construct_optimization_masks(
        self, beta_mask=None, alpha_mask=None, phi_mask=None, theta_mask=None
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
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
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

    """
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
        if self.parameters["use_displacements"]:
            bs = self.construct_block_operators_with_displacements(
                betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
            )
        else:
            bs = self.construct_block_operators(betas_rho, betas_angle, phis, thetas)
        psi = self.initial_states[0]
        for U in bs[:i]:
            psi = U @ psi
        return psi
    """

    @tf.function
    def batch_state_transfer_fidelities(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        bs = self.batch_construct_block_operators(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        psis = tf.stack([self.initial_states] * self.parameters["N_multistart"])
        for U in bs:
            psis = tf.einsum(
                "mij,msjk->msik", U, psis
            )  # m: multistart, s:multiple states
        overlaps = self.target_states_dag @ psis  # broadcasting
        overlaps = tf.reduce_mean(overlaps, axis=1)
        overlaps = tf.squeeze(overlaps)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer
        fids = tf.cast(overlaps * tf.math.conj(overlaps), dtype=tf.float32)
        return fids

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
    def U_tot(self,):
        bs = self.batch_construct_block_operators(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        # U_c = tf.scan(lambda a, b: tf.matmul(b, a), bs)[-1]
        U_c = self.mult_bin_tf(
            tf.reverse(bs, axis=[0])
        )  # [U_1,U_2,..] -> [U_N,U_{N-1},..]-> U_N @ U_{N-1} @ .. @ U_1
        # U_c = self.I
        # for U in bs:
        #     U_c = U @ U_c
        return U_c

    """
    @tf.function
    def unitary_fidelity(
        self, betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
    ):
        U_circuit = self.U_tot(
            betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas
        )
        D = tf.constant(self.parameters["P_cav"] * 2, dtype=tf.complex64)
        overlap = tf.linalg.trace(
            tf.linalg.adjoint(self.target_unitary) @ self.P_matrix @ U_circuit
        )
        return tf.cast(
            (1.0 / D) ** 2 * overlap * tf.math.conj(overlap), dtype=tf.float32
        )
    """

    def optimize(self, do_prints=True):

        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.timestamps.append(timestamp)
        print("Start time: " + timestamp)
        # start time
        start_time = time.time()
        optimizer = tf.optimizers.Adam(self.parameters["learning_rate"])
        if self.parameters["use_displacements"]:
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

        @tf.function
        def entry_stop_gradients(target, mask):
            mask_h = tf.abs(mask - 1)
            return tf.stop_gradient(mask_h * target) + mask * target

        """
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
            avg_loss = tf.reduce_sum(losses) / self.parameters["N_multistart"]
            return avg_loss

        def callback_fun(obj, fids, dfids, epoch):
            elapsed_time_s = time.time() - start_time
            time_per_epoch = elapsed_time_s / epoch if epoch is not 0 else 0.0
            epochs_left = self.parameters["epochs"] - epoch
            expected_time_remaining = epochs_left * time_per_epoch
            fidelities_np = np.squeeze(np.array(fids))
            betas_np, alphas_np, phis_np, thetas_np = self.get_numpy_vars()
            if epoch == 0:
                self._save_optimization_data(
                    timestamp,
                    fidelities_np,
                    betas_np,
                    alphas_np,
                    phis_np,
                    thetas_np,
                    elapsed_time_s,
                    append=False,
                )
            else:
                self._save_optimization_data(
                    timestamp,
                    fidelities_np,
                    betas_np,
                    alphas_np,
                    phis_np,
                    thetas_np,
                    elapsed_time_s,
                    append=True,
                )
            avg_fid = tf.reduce_sum(fids) / self.parameters["N_multistart"]
            max_fid = tf.reduce_max(fids)
            avg_dfid = tf.reduce_sum(dfids) / self.parameters["N_multistart"]
            max_dfid = tf.reduce_max(dfids)
            if do_prints:
                print(
                    "\r Epoch: %d / %d Max Fid: %.6f Avg Fid: %.6f Max dFid: %.6f Avg dFid: %.6f"
                    % (
                        epoch,
                        self.parameters["epochs"],
                        max_fid,
                        avg_fid,
                        max_dfid,
                        avg_dfid,
                    )
                    + " Elapsed time: "
                    + str(datetime.timedelta(seconds=elapsed_time_s))
                    + " Remaing time: "
                    + str(datetime.timedelta(seconds=expected_time_remaining)),
                    end="",
                )

        initial_fids = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        fids = initial_fids
        callback_fun(self, fids, 0, 0)
        for epoch in range(self.parameters["epochs"] + 1)[1:]:
            for _ in range(self.parameters["epoch_size"]):
                with tf.GradientTape() as tape:
                    betas_rho = entry_stop_gradients(self.betas_rho, self.beta_mask)
                    betas_angle = entry_stop_gradients(self.betas_angle, self.beta_mask)
                    if self.parameters["use_displacements"]:
                        alphas_rho = entry_stop_gradients(
                            self.alphas_rho, self.alpha_mask
                        )
                        alphas_angle = entry_stop_gradients(
                            self.alphas_angle, self.alpha_mask
                        )
                    else:
                        alphas_rho = self.alphas_rho
                        alphas_angle = self.alphas_angle
                    phis = entry_stop_gradients(self.phis, self.phi_mask)
                    thetas = entry_stop_gradients(self.thetas, self.theta_mask)
                    new_fids = self.batch_fidelities(
                        betas_rho, betas_angle, alphas_rho, alphas_angle, phis, thetas,
                    )
                    new_loss = loss_fun(new_fids)
                    dloss_dvar = tape.gradient(new_loss, variables)
                optimizer.apply_gradients(zip(dloss_dvar, variables))
            dfids = new_fids - fids
            fids = new_fids
            callback_fun(self, fids, dfids, epoch)
            condition_fid = tf.greater(fids, self.parameters["term_fid"])
            condition_dfid = tf.greater(dfids, self.parameters["dfid_stop"])
            if tf.reduce_any(condition_fid):
                print("\n\n Optimization stopped. Term fidelity reached.\n")
                termination_reason = "term_fid"
                break
            if not tf.reduce_any(condition_dfid):
                print("\n max dFid: %6f" % tf.reduce_max(dfids).numpy())
                print("dFid stop: %6f" % self.parameters["dfid_stop"])
                print("\n\n Optimization stopped.  No dfid is greater than dfid_stop\n")
                termination_reason = "dfid"
                break

        if epoch == self.parameters["epochs"]:
            termination_reason = "epochs"
            print(
                "\n\nOptimization stopped.  Reached maximum number of epochs. Terminal fidelity not reached.\n"
            )
        self._save_termination_reason(timestamp, termination_reason)
        timestamp_end = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        elapsed_time_s = time.time() - start_time
        epoch_time_s = elapsed_time_s / epoch
        step_time_s = epoch_time_s / self.parameters["epochs"]
        self.print_info()
        print("all data saved as: " + self.filename)
        print("termination reason: " + termination_reason)
        print("optimization timestamp (start time): " + timestamp)
        print("timestamp (end time): " + timestamp_end)
        print("elapsed time: " + str(datetime.timedelta(seconds=elapsed_time_s)))
        print(
            "Time per epoch (epoch size = %d): " % self.parameters["epoch_size"]
            + str(datetime.timedelta(seconds=epoch_time_s))
        )
        print(
            "Time per Adam step (N_multistart = %d, N_cav = %d): "
            % (self.parameters["N_multistart"], self.parameters["N_cav"])
            + str(datetime.timedelta(seconds=step_time_s))
        )
        print(END_OPT_STRING)
        return timestamp

    # if append is True, it will assume the dataset is already created and append only the
    # last aquired values to it.
    # TODO: if needed, could use compression when saving data.
    def _save_optimization_data(
        self,
        timestamp,
        fidelities_np,
        betas_np,
        alphas_np,
        phis_np,
        thetas_np,
        elapsed_time_s,
        append,
    ):
        if not append:
            with h5py.File(self.filename, "a") as f:
                grp = f.create_group(timestamp)
                for parameter, value in self.parameters.items():
                    grp.attrs[parameter] = value
                grp.attrs["termination_reason"] = "outside termination"
                grp.attrs["elapsed_time_s"] = elapsed_time_s
                if self.target_unitary is not None:
                    grp.create_dataset(
                        "target_unitary", data=self.target_unitary.numpy()
                    )
                grp.create_dataset("initial_states", data=self.initial_states.numpy())
                grp.create_dataset("target_states", data=self.target_states.numpy())
                # dims = [[2, int(self.initial_states[0].numpy().shape[0] / 2)], [1, 1]]
                grp.create_dataset(
                    "fidelities",
                    chunks=True,
                    data=[fidelities_np],
                    maxshape=(None, self.parameters["N_multistart"]),
                )
                grp.create_dataset(
                    "betas",
                    data=[betas_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
                grp.create_dataset(
                    "alphas",
                    data=[alphas_np],
                    chunks=True,
                    maxshape=(None, self.parameters["N_multistart"], 1,),
                )
                grp.create_dataset(
                    "phis",
                    data=[phis_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
                grp.create_dataset(
                    "thetas",
                    data=[thetas_np],
                    chunks=True,
                    maxshape=(
                        None,
                        self.parameters["N_multistart"],
                        self.parameters["N_blocks"],
                    ),
                )
        else:  # just append the data
            with h5py.File(self.filename, "a") as f:
                f[timestamp]["fidelities"].resize(
                    f[timestamp]["fidelities"].shape[0] + 1, axis=0
                )
                f[timestamp]["betas"].resize(f[timestamp]["betas"].shape[0] + 1, axis=0)
                f[timestamp]["alphas"].resize(
                    f[timestamp]["alphas"].shape[0] + 1, axis=0
                )
                f[timestamp]["phis"].resize(f[timestamp]["phis"].shape[0] + 1, axis=0)
                f[timestamp]["thetas"].resize(
                    f[timestamp]["thetas"].shape[0] + 1, axis=0
                )

                f[timestamp]["fidelities"][-1] = fidelities_np
                f[timestamp]["betas"][-1] = betas_np
                f[timestamp]["alphas"][-1] = alphas_np
                f[timestamp]["phis"][-1] = phis_np
                f[timestamp]["thetas"][-1] = thetas_np
                f[timestamp].attrs["elapsed_time_s"] = elapsed_time_s

    def _save_termination_reason(self, timestamp, termination_reason):
        with h5py.File(self.filename, "a") as f:
            f[timestamp].attrs["termination_reason"] = termination_reason

    def randomize_and_set_vars(self):
        beta_scale = self.parameters["beta_scale"]
        alpha_scale = self.parameters["alpha_scale"]
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
        thetas = np.random.uniform(
            -np.pi,
            np.pi,
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
        self.phis = tf.Variable(
            phis, dtype=tf.float32, trainable=True, name="betas_rho",
        )
        self.thetas = tf.Variable(
            thetas, dtype=tf.float32, trainable=True, name="betas_angle",
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

        # these will have shape N_multistart x N_blocks
        return betas.T, alphas.T, phis.T, thetas.T

    def set_tf_vars(self, betas=None, alphas=None, phis=None, thetas=None):
        # reshaping for N_multistart = 1
        if betas is not None:
            if len(betas.shape) < 2:
                betas = betas.reshape(betas.shape + (1,))
                self.parameters["N_multistart"] = 1
            betas_rho = np.abs(betas)
            betas_angle = np.angle(betas)
            self.betas_rho = tf.Variable(
                betas_rho, dtype=tf.float32, trainable=True, name="betas_rho"
            )
            self.betas_angle = tf.Variable(
                betas_angle, dtype=tf.float32, trainable=True, name="betas_angle",
            )
        if alphas is not None:
            if len(alphas.shape) < 2:
                alphas = alphas.reshape(alphas.shape + (1,))
                self.parameters["N_multistart"] = 1
            alphas_rho = np.abs(alphas)
            alphas_angle = np.angle(alphas)
            if self.parameters["use_displacements"]:
                self.alphas_rho = tf.Variable(
                    alphas_rho, dtype=tf.float32, trainable=True, name="alphas_rho",
                )
                self.alphas_angle = tf.Variable(
                    alphas_angle, dtype=tf.float32, trainable=True, name="alphas_angle",
                )
            else:
                self.alphas_rho = tf.constant(
                    np.zeros(shape=((1, self.parameters["N_multistart"],))),
                    dtype=tf.float32,
                )
                self.alphas_angle = tf.constant(
                    np.zeros(shape=((1, self.parameters["N_multistart"],))),
                    dtype=tf.float32,
                )

        if phis is not None:
            if len(phis.shape) < 2:
                phis = phis.reshape(phis.shape + (1,))
                self.parameters["N_multistart"] = 1
            self.phis = tf.Variable(
                phis, dtype=tf.float32, trainable=True, name="betas_rho",
            )
        if thetas is not None:
            if len(thetas.shape) < 2:
                thetas = thetas.reshape(thetas.shape + (1,))
                self.parameters["N_multistart"] = 1
            self.thetas = tf.Variable(
                thetas, dtype=tf.float32, trainable=True, name="betas_angle",
            )

    def best_circuit(self):
        fids = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        max_idx = tf.argmax(fids).numpy()
        all_betas, all_alphas, all_phis, all_thetas = self.get_numpy_vars(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        max_fid = fids[max_idx].numpy()
        betas = all_betas[max_idx]
        alphas = all_alphas[max_idx]
        phis = all_phis[max_idx]
        thetas = all_thetas[max_idx]
        return {
            "fidelity": max_fid,
            "betas": betas,
            "alphas": alphas,
            "phis": phis,
            "thetas": thetas,
        }

    def all_fidelities(self):
        fids = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        return fids.numpy()

    def best_fidelity(self):
        fids = self.batch_fidelities(
            self.betas_rho,
            self.betas_angle,
            self.alphas_rho,
            self.alphas_angle,
            self.phis,
            self.thetas,
        )
        max_idx = tf.argmax(fids).numpy()
        max_fid = fids[max_idx].numpy()
        return max_fid

    def print_info(self):
        best_circuit = self.best_circuit()
        with np.printoptions(precision=5, suppress=True):
            for parameter, value in self.parameters.items():
                print(parameter + ": " + str(value))
            print("filename: " + self.filename)
            print("\nBest circuit parameters found:")
            print("betas:         " + str(best_circuit["betas"]))
            print("alphas:        " + str(best_circuit["alphas"]))
            print("phis (deg):    " + str(best_circuit["phis"] * 180.0 / np.pi))
            print("thetas (deg):  " + str(best_circuit["thetas"] * 180.0 / np.pi))
            print("Max Fidelity:  %.6f" % best_circuit["fidelity"])
            print("\n")
