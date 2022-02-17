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
from typing import List, Dict

# might want to have a separate set of optmizer parameters that are specific to the optimizer.
# these can be passed as another dictionary


class GateSynthesizer:
    def __init__(
        self,
        gateset: GateSet,
        N_blocks: 10,
        optimization_type="state transfer",
        optimization_masks=None,  # optional dictionary of masks
        target_unitary=None,
        initial_states=None,
        target_states=None,
        expectation_operators=None,
        target_expectation_values=None,
        N_multistart=10,
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
        do_prints=True,
        **kwargs
    ):
        self.parameters = {
            "N_blocks": N_blocks,
            "optimization_type": optimization_type,
            "optimization_masks": optimization_masks,
            "target_unitary": target_unitary,
            "initial_states": initial_states,
            "target_states": target_states,
            "expectation_operators": expectation_operators,
            "target_expectation_values": target_expectation_values,
            "N_multistart": N_multistart,
            "term_fid": term_fid,
            "dfid_stop": dfid_stop,
            "learning_rate": learning_rate,
            "epoch_size": epoch_size,
            "epochs": epochs,
            "name": name,
            "filename": filename,
            "comment": comment,
            "use_phase": use_phase,
            "timestamps": timestamps,
            "do_prints": do_prints,
        }
        self.parameters.update(kwargs)
        self.gateset = gateset

        # self.GateSet = GateSet
        # self.parameters = self.GateSet.parameters

        # TODO: handle case when you pass initial params. In that case, don't randomize, but use "set_tf_vars()"
        self.opt_vars = self.randomize_and_set_vars()

        # self._construct_optimization_masks(beta_mask, alpha_mask, phi_mask,eta_mask, theta_mask)

        self.optimization_mask = self.create_optimization_mask()

        # opt data will be a dictionary of dictonaries used to store optimization data
        # the dictionary will be addressed by timestamps of optmization.
        # each opt will append to opt_data a dictionary
        # this dictionary will contain optimization parameters and results

        self.timestamps = self.parameters["timestamps"]
        self.filename = (
            self.parameters["filename"]
            if (
                self.parameters["filename"] is not None
                and self.parameters["filename"] != ""
            )
            else self.parameters["name"]
        )
        path = self.filename.split(".")
        if len(path) < 2 or (len(path) == 2 and path[-1] != ".h5"):
            self.filename = path[0] + ".h5"
        self.batch_fidelities = None
        if (
            self.parameters["optimization_type"] == "state transfer"
            or self.parameters["optimization_type"] == "analysis"
        ):
            self.batch_fidelities = (
                self.batch_state_transfer_fidelities
                if self.parameters["use_phase"]
                else self.batch_state_transfer_fidelities_real_part
            )
            # set fidelity function

            self.initial_states = tf.stack(
                [tfq.qt2tf(state) for state in self.parameters["initial_states"]]
            )

            self.target_unitary = tfq.qt2tf(self.parameters["target_unitary"])

            # if self.target_unitary is not None: TODO
            #     raise Exception("Need to fix target_unitary multi-state transfer generation!")

            self.target_states = (  # store dag
                tf.stack(
                    [tfq.qt2tf(state) for state in self.parameters["target_states"]]
                )
                if self.target_unitary is None
                else self.target_unitary @ self.initial_states
            )

            self.target_states_dag = tf.linalg.adjoint(
                self.target_states
            )  # store dag to avoid having to take adjoint

            N_cav = self.initial_states[0].numpy().shape[0] // 2
        elif self.parameters["optimization_type"] == "unitary":
            self.target_unitary = tfq.qt2tf(self.parameters["target_unitary"])
            N_cav = self.target_unitary.numpy().shape[0] // 2
            P_cav = (
                self.parameters["P_cav"]
                if self.parameters["P_cav"] is not None
                else N_cav
            )
            raise Exception("Need to implement unitary optimization")

        elif self.parameters["optimization_type"] == "expectation":
            raise Exception("Need to implement expectation optimization")
        elif (
            self.parameters["optimization_type"] == "calculation"
        ):  # using functions but not doing opt
            pass
        else:
            raise ValueError(
                "optimization_type must be one of {'state transfer', 'unitary', 'expectation', 'analysis', 'calculation'}"
            )

    @tf.function
    def batch_state_transfer_fidelities(self, opt_params: Dict[str, tf.Variable]):
        bs = self.gateset.batch_construct_block_operators(opt_params)
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

    # here, including the relative phase in the cost function by taking the real part of the overlap then squaring it.
    # need to think about how this is related to the fidelity.
    @tf.function
    def batch_state_transfer_fidelities_real_part(self, opt_params: Dict[str, tf.Variable]):
        bs = self.gateset.batch_construct_block_operators(opt_params)
        psis = tf.stack([self.initial_states] * self.parameters["N_multistart"])
        for U in bs:
            psis = tf.einsum(
                "mij,msjk->msik", U, psis
            )  # m: multistart, s:multiple states
        overlaps = self.target_states_dag @ psis  # broadcasting
        overlaps = tf.reduce_mean(tf.math.real(overlaps), axis=1)
        overlaps = tf.squeeze(overlaps)
        # squeeze after reduce_mean which uses axis=1,
        # which will not exist if squeezed before for single state transfer
        # don't need to take the conjugate anymore
        fids = tf.cast(overlaps * overlaps, dtype=tf.float32)
        return fids

    @tf.function
    def loss_fun(self, fids):
        # I think it's important that the log is taken before the avg
        losses = tf.math.log(1 - fids)
        avg_loss = tf.reduce_sum(losses) / self.parameters["N_multistart"]
        return avg_loss

    def callback_fun(self, fids, dfids, epoch, timestamp, start_time):
        elapsed_time_s = time.time() - start_time
        time_per_epoch = elapsed_time_s / epoch if epoch != 0 else 0.0
        epochs_left = self.parameters["epochs"] - epoch
        expected_time_remaining = epochs_left * time_per_epoch
        fidelities_np = np.squeeze(np.array(fids))

        if epoch == 0:
            self._save_optimization_data(
                timestamp, fidelities_np, elapsed_time_s, append=False,
            )
        else:
            self._save_optimization_data(
                timestamp, fidelities_np, elapsed_time_s, append=True,
            )
        avg_fid = tf.reduce_sum(fids) / self.parameters["N_multistart"]
        max_fid = tf.reduce_max(fids)
        avg_dfid = tf.reduce_sum(dfids) / self.parameters["N_multistart"]
        max_dfid = tf.reduce_max(dfids)
        extra_string = " (real part)" if self.parameters["use_phase"] else ""
        if self.parameters["do_prints"]:
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
                + str(datetime.timedelta(seconds=expected_time_remaining))
                + extra_string,
                end="",
            )
        return

    def get_numpy_vars(self, opt_vars: List[tf.Variable]):

        return [
            (k.numpy().T)
            for k in self.gateset.preprocess_params_before_saving(opt_vars)
        ]

    def best_circuit(self):
        fids = self.batch_fidelities(self.opt_vars)
        fids = np.atleast_1d(fids.numpy())
        max_idx = np.argmax(fids)
        tf_vars = self.gateset.preprocess_params_before_saving(self.opt_vars)
        best_params = {}
        for key, value in tf_vars.items():
            best_params[key] = value[:, max_idx]  # first index is always N_multistart

        return best_params

    def all_fidelities(self):
        fids = self.batch_fidelities(self.opt_vars)
        return fids.numpy()

    def best_fidelity(self):
        fids = self.batch_fidelities(self.opt_vars)
        max_idx = tf.argmax(fids).numpy()
        max_fid = fids[max_idx].numpy()
        return max_fid

    def print_info(self):
        best_circuit = self.best_circuit()
        with np.printoptions(precision=5, suppress=True):
            for parameter, value in self.parameters.items():
                if parameter is "initial_states" or "final_states":
                    continue
                print(parameter + ": " + str(value))
            print("filename: " + self.filename)
            print("\nBest circuit parameters found:")

            for k in best_circuit.keys():
                print(k + str(best_circuit[k]))
            print("\n")

    def _save_optimization_data(
        self, timestamp, fidelities_np, elapsed_time_s, append,
    ):
        if not append:
            with h5py.File(self.filename, "a") as f:
                grp = f.create_group(timestamp)
                for parameter, value in self.parameters.items():
                    if value is None:
                        continue
                    if type(value) is dict:
                        for key,item in value.items():
                            if item is None:
                                continue
                            grp.attrs[key] = item
                grp.attrs["termination_reason"] = "outside termination"
                grp.attrs["elapsed_time_s"] = elapsed_time_s
                if self.target_unitary is not None:
                    grp.create_dataset(
                        "target_unitary", data=self.target_unitary.numpy()
                    )
                grp.create_dataset("initial_states", data=self.initial_states.numpy())
                grp.create_dataset("target_states", data=self.target_states.numpy())

                grp.create_dataset(
                    "fidelities",
                    chunks=True,
                    data=[fidelities_np],
                    maxshape=(None, self.parameters["N_multistart"]),
                )

                for key, value in self.gateset.preprocess_params_before_saving(
                    self.opt_vars
                ).items():
                    if len(value.shape) == 2:
                        grp.create_dataset(
                            key,
                            data=[np.swapaxes(value.numpy(), 0, 1)],
                            chunks=True,
                            maxshape=(
                                None,
                                self.parameters["N_multistart"],
                                self.parameters["N_blocks"],
                            ),
                        )
                    elif len(value.shape) == 3:
                        grp.create_dataset(
                            key,
                            data=[np.swapaxes(value.numpy(), 0, 1)],
                            chunks=True,
                            maxshape=(
                                None,
                                self.parameters["N_multistart"],
                                self.parameters["N_blocks"],
                                value.shape[-1]
                            ),
                        )
                    else:
                        raise ValueError(key + " has more than three indices. This is not currently supported.")

        else:  # just append the data
            with h5py.File(self.filename, "a") as f:

                f[timestamp]["fidelities"].resize(
                    f[timestamp]["fidelities"].shape[0] + 1, axis=0
                )
                f[timestamp]["fidelities"][-1] = fidelities_np

                for key, value in self.gateset.preprocess_params_before_saving(
                    self.opt_vars
                ).items():
                    f[timestamp][key].resize(f[timestamp][key].shape[0] + 1, axis=0)
                    f[timestamp][key][-1] = np.swapaxes(value.numpy(), 0, 1)

                f[timestamp].attrs["elapsed_time_s"] = elapsed_time_s

    def _save_termination_reason(self, timestamp, termination_reason):
        with h5py.File(self.filename, "a") as f:
            f[timestamp].attrs["termination_reason"] = termination_reason

    def randomize_and_set_vars(self):

        init_vars = {}

        init_scales = self.gateset.randomization_ranges()
        init_vars = {}
        for var_name in self.gateset.parameter_names:
            scale = init_scales[var_name]
            if len(scale) == 2:
                var_np = np.random.uniform(
                    scale[0],
                    scale[1],
                    size=(self.parameters["N_blocks"], self.parameters["N_multistart"]),
                )
            elif len(scale) == 3:
                var_np = np.random.uniform(
                    scale[0],
                    scale[1],
                    size=(self.parameters["N_blocks"], self.parameters["N_multistart"], scale[2]),
                )
            else:
                raise ValueError("You have not specified the correct number of parameters to initialize " + var_name)
            var_tf = tf.Variable(
                var_np, dtype=tf.float32, trainable=True, name=var_name
            )
            init_vars[var_name] = var_tf
        return init_vars

    def create_optimization_mask(self):
        masks = {}
        if self.parameters['optimization_masks'] is None:
            self.parameters['optimization_masks'] = {
                var_name: None for var_name in self.gateset.parameter_names
            }
        for var_name in self.gateset.parameter_names:
            if self.parameters['optimization_masks'][var_name] is None:
                var_mask = np.ones(
                    shape=(
                        self.parameters["N_blocks"],
                        self.parameters["N_multistart"],
                    ),
                    dtype=np.float32,
                )
            else:
                var_mask = np.array(
                    np.tile(
                        self.parameters['optimization_masks'][var_name],
                        self.parameters["N_multistart"],
                    ).reshape(
                        self.parameters["N_blocks"], self.parameters["N_multistart"]
                    ),
                    dtype=np.float32,
                )

            if var_mask.shape == self.opt_vars[var_name].shape:
                masks[var_name] = var_mask

            elif abs(len(var_mask.shape) - len(self.opt_vars[var_name].shape)) == 1:
                # in this case there may be a third dimension to one of the variables
                var_mask_dim = np.repeat(var_mask[:, :, None], self.opt_vars[var_name].shape[2], axis=2)
                masks[var_name] = var_mask_dim
            else:
                raise ValueError("Cannot create mask for variable " + var_name)

        return masks

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
            self.etas,
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
