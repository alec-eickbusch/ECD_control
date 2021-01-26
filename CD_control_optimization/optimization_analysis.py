import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
from matplotlib.ticker import MaxNLocator
from CD_control_optimization.batch_optimizer import BatchOptimizer
from scipy.interpolate import interp2d
from tqdm import tqdm
from sklearn.manifold import TSNE

plt.rcParams.update(
    {
        "figure.figsize": (11.7, 8.27),
        "font.size": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


N_BLOCKS = "N_blocks"
TIMESTAMP_SEP = ","


class OptimizationAnalysis:
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(self.filename, "a") as f:
            self.timestamps = list(f.keys())
        # remove the "sweep" data sets.
        self.timestamps = [
            timestamp
            for timestamp in self.timestamps
            if timestamp.split(" ")[0] != "sweep"
        ]
        # TODO: is it gaurenteed that the most recent timestamp will be [-1]?
        self.data = {}

    def get_data(self, timestamp):
        if timestamp not in self.data:
            self._load_data(timestamps=[timestamp])
        return self.data[timestamp]

    def _load_data(self, timestamps=None):
        if timestamps is None:
            timestamps = [self.timestamps[-1]]
        if not isinstance(timestamps, list):
            timestamps = [timestamps]
        for timestamp in timestamps:
            if timestamp in self.data:
                continue
            self.data[timestamp] = {}
            with h5py.File(self.filename, "a") as f:
                self.data[timestamp]["parameters"] = dict(f[timestamp].attrs.items())
                self.data[timestamp]["betas"] = f[timestamp]["betas"][()]
                self.data[timestamp]["alphas"] = f[timestamp]["alphas"][()]
                self.data[timestamp]["phis"] = f[timestamp]["phis"][()]
                self.data[timestamp]["thetas"] = f[timestamp]["thetas"][()]
                self.data[timestamp]["fidelities"] = f[timestamp]["fidelities"][()]
                if "initial_states" in f[timestamp]:
                    initial_states = f[timestamp]["initial_states"][()]
                    target_states = f[timestamp]["target_states"][()]
                    N = initial_states.shape[1] // 2
                    dims = [[2, N], [1, 1]]
                    self.data[timestamp]["N"] = N
                    self.data[timestamp]["initial_states"] = [
                        qt.Qobj(initial_state, dims=dims)
                        for initial_state in initial_states
                    ]
                    self.data[timestamp]["target_states"] = [
                        qt.Qobj(target_state, dims=dims)
                        for target_state in target_states
                    ]
                    if len(self.data[timestamp]["initial_states"]) == 1:
                        self.data[timestamp]["initial_state"] = self.data[timestamp][
                            "initial_states"
                        ][0]
                        self.data[timestamp]["target_state"] = self.data[timestamp][
                            "target_states"
                        ][0]

    def get_opt_object(self, timestamp=None, N_multistart=1):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        parameters = self.get_data(timestamp)["parameters"]
        parameters["optimization_type"] = "analysis"
        parameters["N_multistart"] = N_multistart
        return BatchOptimizer(
            initial_states=self.data[timestamp]["initial_states"],
            target_states=self.data[timestamp]["target_states"],
            **parameters,
        )

    def fidelities(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["fidelities"]

    def best_fidelities(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        indx = self.idx_of_best_circuit(timestamp)
        fids = self.fidelities(timestamp).T[indx]
        return fids

    def parameters(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["parameters"]

    # given a timestamp and epoch, return a list of epochs which have a fidelity larger than success_fid
    def successful_idxs(self, timestamp=None, epoch=-1, success_fid=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        fidelities = self.fidelities(timestamp)[epoch]
        success_fid = (
            self.parameters(timestamp)["term_fid"]
            if success_fid is None
            else success_fid
        )
        return np.where(fidelities > success_fid)[0]

    def success_fraction(self, timestamp=None, epoch=-1, success_fid=None):
        num_success = len(self.successful_idxs(timestamp, epoch, success_fid))
        return num_success / self.parameters(timestamp)["N_multistart"]

    def idx_of_best_circuit(self, timestamp=None, epoch=-1):
        fidelities = self.fidelities(timestamp)[epoch]
        idx = np.argmax(fidelities)
        return idx

    def initial_state(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["initial_state"]

    def target_state(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["target_state"]

    def best_circuit(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        idx = self.idx_of_best_circuit(timestamp)
        betas = self.get_data(timestamp)["betas"][-1][idx]
        alphas = self.get_data(timestamp)["alphas"][-1][idx]
        phis = self.get_data(timestamp)["phis"][-1][idx]
        thetas = self.get_data(timestamp)["thetas"][-1][idx]
        max_fid = self.get_data(timestamp)["fidelities"][-1][idx]
        return {
            "fidelity": max_fid,
            "betas": betas,
            "alphas": alphas,
            "phis": phis,
            "thetas": thetas,
        }

    def best_U_tot(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        idx = self.idx_of_best_circuit(timestamp)
        betas = self.get_data(timestamp)["betas"][-1][idx]
        alphas = self.get_data(timestamp)["alphas"][-1][idx]
        phis = self.get_data(timestamp)["phis"][-1][idx]
        thetas = self.get_data(timestamp)["thetas"][-1][idx]
        opt = self.get_opt_object()
        opt.set_tf_vars(betas, alphas, phis, thetas)
        return opt.U_tot()[0]  # since multistart is 1, we must take the first element

    def U_benchmark(self, initial_state, target_state, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        U_tot = self.best_U_tot(timestamp=timestamp)
        fid = np.abs(target_state.dag() @ U_tot @ initial_state) ** 2
        return fid

    def print_info(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        best_circuit = self.best_circuit(timestamp)
        with np.printoptions(precision=5, suppress=True):
            for parameter, value in self.get_data(timestamp)["parameters"].items():
                print(parameter + ": " + str(value))
            print("filename: " + self.filename)
            print("\nBest circuit parameters found:")
            print("betas:         " + str(best_circuit["betas"]))
            print("alphas:        " + str(best_circuit["alphas"]))
            print("phis (deg):    " + str(best_circuit["phis"] * 180.0 / np.pi))
            print("thetas (deg):  " + str(best_circuit["thetas"] * 180.0 / np.pi))
            print("Max Fidelity:  %.6f" % best_circuit["fidelity"])
            print("\n")

    def betas(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["betas"]

    def alphas(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        return self.get_data(timestamp)["alphas"]

    # average over BLOCKS in the circuit
    def average_magnitude_betas(self, timestamp=None):
        betas = self.betas(timestamp)
        return np.mean(np.abs(betas), axis=-1)

    def average_magnitude_alphas(self, timestamp=None):
        alphas = self.alphas(timestamp)
        return np.mean(np.abs(alphas), axis=-1)

    def plot_fidelities(self, timestamp=None, fig=None, ax=None, log=True, **kwargs):
        fidelities = self.fidelities(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
        ax = ax if ax is not None else fig.subplots()
        end_fids = fidelities[:, -1]
        indxs = np.argsort(end_fids)
        fidelities = fidelities[indxs]
        for fids in fidelities[:-1]:
            if log:
                ax.semilogy(1 - fids, ":", linewidth=0.5, **kwargs)
            else:
                ax.plot(fids, ":", linewidth=0.5, **kwargs)
        if log:
            ax.semilogy(1 - fidelities[-1], **kwargs)
        else:
            ax.plot(fidelities[-1], **kwargs)
        ax.set_xlabel("epoch")
        if log:
            ax.set_ylabel("infidelity")
        else:
            ax.set_ylabel("fidelity")
        fig.tight_layout()

    def plot_best_fidelity(self, timestamp=None, fig=None, ax=None, log=True, **kwargs):
        fids = self.best_fidelities(timestamp)
        fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
        ax = ax if ax is not None else fig.subplots()
        if log:
            ax.semilogy(1 - fids, **kwargs)
        else:
            ax.plot(fids, **kwargs)
        ax.set_xlabel("epoch")
        if log:
            ax.set_ylabel("infidelity")
        else:
            ax.set_ylabel("fidelity")
        fig.tight_layout()

    def plot_tSNE_betas(
        self, timestamp=None, fig=None, ax=None, log=True, min_fid=0, **kwargs
    ):
        fids = self.fidelities(timestamp)
        # num_epochs = fids.shape[0]
        fids = fids[-1]
        if not np.any(fids >= min_fid):
            print("No fidelities above: " + str(min_fid))
            return

        fig = fig if fig is not None else plt.figure()
        ax = ax if ax is not None else fig.subplots()
        betas = np.abs(self.betas(timestamp)[-1])
        fids = fids if log is False else np.log10(1 - fids)
        self.plot_tSNE(betas, fids, fig=fig, ax=ax, log=log, **kwargs)
        plt.show()

    def plot_tSNE_alphas(
        self, timestamp=None, fig=None, ax=None, log=True, min_fid=0, **kwargs
    ):
        fids = self.fidelities(timestamp)
        # num_epochs = fids.shape[0]
        fids = fids[-1]
        if not np.any(fids >= min_fid):
            return

        fig = fig if fig is not None else plt.figure()
        ax = ax if ax is not None else fig.subplots()
        alphas = np.abs(self.alphas(timestamp)[-1])
        fids = fids if log is False else np.log10(1 - fids)
        self.plot_tSNE(alphas, fids, fig=fig, ax=ax, log=log, **kwargs)
        plt.show()

    def plot_tSNE(self, X, y, fig=None, ax=None, log=True, **kwargs):
        tsne = TSNE()
        X_embedded = tsne.fit_transform(X)
        import seaborn as sns

        palette = sns.color_palette("magma", as_cmap=True)
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=y,
            legend="brief",
            palette=palette,
        )

    def plot_average_magnitude_beta(self, timestamp=None, fig=None, ax=None):
        average_mag_betas = self.average_magnitude_betas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()
        for mag_betas in average_mag_betas:
            ax.plot(mag_betas)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\beta|$")
        fig.tight_layout()

    def plot_average_magnitude_alpha(self, timestamp=None, fig=None, ax=None):
        average_mag_alphas = self.average_magnitude_alphas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()
        for mag_alphas in average_mag_alphas:
            ax.plot(mag_alphas)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\alpha|$")
        fig.tight_layout()

    def plot_mag_betas(self, timestamp=None, fig=None):
        # will have shape num blocks, # multistart, #epochs with transpose
        betas = self.betas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(14, 3), dpi=200)
        axs = fig.subplots(1, betas.shape[0])
        for circuit_block, bs in enumerate(betas):
            for beta_per_epoch in bs:
                axs[circuit_block].plot(np.abs(beta_per_epoch))
            axs[circuit_block].set_xlabel("epoch")
            axs[circuit_block].set_ylabel(r"$|\beta|$")
            axs[circuit_block].set_title(r"$CD_{%d}(\beta)$" % circuit_block)
        fig.tight_layout()

    def plot_phase_betas(self, timestamp=None, fig=None):
        # will have shape num blocks, # multistart, #epochs with transpose
        betas = self.betas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(14, 3), dpi=200)
        axs = fig.subplots(1, betas.shape[0])
        for circuit_block, bs in enumerate(betas):
            for beta_per_epoch in bs:
                axs[circuit_block].plot(180 * np.angle(beta_per_epoch) / np.pi)
            axs[circuit_block].set_xlabel("epoch")
            axs[circuit_block].set_ylabel(r"arg$(\beta)$ (deg)")
            axs[circuit_block].set_title(r"$CD_{%d}(\beta)$" % circuit_block)
        fig.tight_layout()

    def plot_mag_alphas(self, timestamp=None, fig=None):
        # will have shape num blocks, # multistart, #epochs with transpose
        alphas = self.alphas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(14, 3), dpi=200)
        axs = fig.subplots(1, alphas.shape[0])
        for circuit_block, bs in enumerate(alphas):
            for alpha_per_epoch in bs:
                axs[circuit_block].plot(np.abs(alpha_per_epoch))
            axs[circuit_block].set_xlabel("epoch")
            axs[circuit_block].set_ylabel(r"$|\alpha|$")
            axs[circuit_block].set_title(r"$D_{%d}(\alpha)$" % circuit_block)
        fig.tight_layout()

    def plot_phase_alphas(self, timestamp=None, fig=None):
        # will have shape num blocks, # multistart, #epochs with transpose
        alphas = self.alphas(timestamp).T
        fig = fig if fig is not None else plt.figure(figsize=(14, 3), dpi=200)
        axs = fig.subplots(1, alphas.shape[0])
        for circuit_block, bs in enumerate(alphas):
            for alpha_per_epoch in bs:
                axs[circuit_block].plot(180 * np.angle(alpha_per_epoch) / np.pi)
            axs[circuit_block].set_xlabel("epoch")
            axs[circuit_block].set_ylabel(r"arg$(\alpha)$ (deg)")
            axs[circuit_block].set_title(r"$D_{%d}(\alpha)$" % circuit_block)
        fig.tight_layout()


class OptimizationSweepsAnalysis:
    def __init__(self, filename, averaging_over=None):
        self.averaging_over = averaging_over
        self._load_data = (
            self._load_data_averaged
            if averaging_over is not None
            else self._load_data_single
        )

        self.filename = filename
        with h5py.File(self.filename, "a") as f:
            group_names = list(f.keys())
        self.sweep_names = []
        for group_name in group_names:
            if group_name.split(" ")[0] == "sweep":
                self.sweep_names.append(group_name)
            # TODO: is it gaurenteed that the most recent sweep will be [-1]?
        self.data = {}
        self.opt_analysis_obj = OptimizationAnalysis(self.filename)

    def get_data(self, sweep_name):
        if sweep_name not in self.data:
            self._load_data(sweep_names=[sweep_name])
        return self.data[sweep_name]

    def _load_data_single(self, sweep_names=None):
        if sweep_names is None:
            sweep_names = [self.sweep_names[-1]]
        if not isinstance(sweep_names, list):
            sweep_names = [sweep_names]
        for sweep_name in sweep_names:
            if sweep_name in self.data:
                continue
            self.data[sweep_name] = {}
            with h5py.File(self.filename, "a") as f:
                self.data[sweep_name]["sweep_param_names"] = f[sweep_name].attrs[
                    "sweep_param_names"
                ]
                self.data[sweep_name]["timestamps"] = f[sweep_name].attrs["timestamps"]
                self.data[sweep_name]["fidelities"] = f[sweep_name]["fidelities"][()]
                self.data[sweep_name]["sweep_param_values"] = f[sweep_name][
                    "sweep_param_values"
                ][()]

                self.data[sweep_name]["best_circuits"] = []
                for timestamp in self.data[sweep_name]["timestamps"]:
                    self.data[sweep_name]["best_circuits"].append(
                        self.opt_analysis_obj.best_circuit(timestamp)
                    )

    def _load_data_averaged(self, sweep_names=None):
        if sweep_names is None:
            sweep_names = [self.sweep_names[-1]]
        if not isinstance(sweep_names, list):
            sweep_names = [sweep_names]
        for sweep_name in sweep_names:
            if sweep_name in self.data:
                continue
            self.data[sweep_name] = {}
            with h5py.File(self.filename, "a") as f:
                sweep_param_names = list(f[sweep_name].attrs["sweep_param_names"])
                # copy list
                self.data[sweep_name]["sweep_param_names"] = sweep_param_names[:]

                sweep_param_values = f[sweep_name]["sweep_param_values"][()]
                self.data[sweep_name]["sweep_param_values"] = sweep_param_values

                timestamps = f[sweep_name].attrs["timestamps"]
                self.data[sweep_name]["timestamps"] = []

                avg_indx = sweep_param_names.index(self.averaging_over)
                remaining_param_names = list(sweep_param_names)
                remaining_param_names.remove(self.averaging_over)

                fidelities = f[sweep_name]["fidelities"][()]
                self.data[sweep_name]["fidelities"] = []
                remaining_param_values = np.delete(sweep_param_values, avg_indx, axis=1)
                remaining_param_unique_values = np.unique(
                    remaining_param_values, axis=0
                )

                # "fidelity": max_fid,
                # "betas": betas,
                # "alphas": alphas,
                # "phis": phis,
                # "thetas": thetas,
                best_circuits = np.array(
                    [
                        self.opt_analysis_obj.best_circuit(timestamp)
                        for timestamp in timestamps
                    ]
                )
                self.data[sweep_name]["best_circuits"] = []

                remaining_sweep_param_values = []

                for remaining_param_unique_val in tqdm(remaining_param_unique_values):
                    indxs = self.get_fixed_indx(
                        sweep_name=sweep_name,
                        fixed_param_names=remaining_param_names,
                        fixed_param_values=list(remaining_param_unique_val),
                    )
                    self.data[sweep_name]["timestamps"].append(
                        TIMESTAMP_SEP.join(timestamps[indxs])
                    )
                    remaining_sweep_param_values.append(remaining_param_unique_val)
                    fids = np.array(0.0)
                    for fid_vals in fidelities[indxs]:
                        fids = fids + np.sort(fid_vals)  # add max to max

                    best_circuit = {
                        "fidelity": 0,
                        "betas": np.array(0.0j),
                        "alphas": np.array(0.0j),
                        "phis": np.array(0.0j),
                        "thetas": np.array(0.0j),
                    }
                    num_circuits = len(best_circuits[indxs])
                    for circuit in best_circuits[indxs]:
                        best_circuit["fidelity"] += circuit["fidelity"]
                        best_circuit["betas"] = best_circuit["betas"] + np.array(
                            circuit["betas"]
                        )
                        best_circuit["alphas"] = best_circuit["alphas"] + np.array(
                            circuit["alphas"]
                        )
                        best_circuit["phis"] = best_circuit["phis"] + np.array(
                            circuit["phis"]
                        )
                        best_circuit["thetas"] = best_circuit["thetas"] + np.array(
                            circuit["thetas"]
                        )
                    # averaging
                    best_circuit["fidelity"] = best_circuit["fidelity"] / num_circuits
                    best_circuit["betas"] = best_circuit["betas"] / num_circuits
                    best_circuit["alphas"] = best_circuit["alphas"] / num_circuits
                    best_circuit["phis"] = best_circuit["phis"] / num_circuits
                    best_circuit["thetas"] = best_circuit["thetas"] / num_circuits
                    self.data[sweep_name]["best_circuits"].append(best_circuit)
                    self.data[sweep_name]["fidelities"].append(fids / num_circuits)
                # do this after because get_fixed_indx needs full param_values and param_names
                self.data[sweep_name]["fidelities"] = np.array(
                    self.data[sweep_name]["fidelities"]
                )
                self.data[sweep_name]["sweep_param_values"] = np.array(
                    remaining_sweep_param_values
                )
                self.data[sweep_name]["sweep_param_names"] = remaining_param_names

    def timestamps(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return self.get_data(sweep_name)["timestamps"]

    def fidelities(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return self.get_data(sweep_name)["fidelities"]

    def abs_mean_betas(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        best_circuits = self.best_circuits(sweep_name)
        betas = [np.mean(np.abs(circuit["betas"])) for circuit in best_circuits]
        return np.array(betas)

    def abs_sum_betas(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        best_circuits = self.best_circuits(sweep_name)
        betas = [np.sum(np.abs(circuit["betas"])) for circuit in best_circuits]
        return np.array(betas)

    def abs_mean_alphas(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        best_circuits = self.best_circuits(sweep_name)
        alphas = [np.mean(np.abs(circuit["alphas"])) for circuit in best_circuits]
        return np.array(alphas)

    def abs_sum_alphas(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        best_circuits = self.best_circuits(sweep_name)
        alphas = [np.sum(np.abs(circuit["alphas"])) for circuit in best_circuits]
        return np.array(alphas)

    def best_fidelities(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return np.amax(self.fidelities(sweep_name), 1)

    def success_fracs(self, success_fid=None, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fids = self.fidelities(sweep_name)
        return np.mean(fids >= success_fid, axis=1)

    def sweep_param_values(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return np.array(self.get_data(sweep_name)["sweep_param_values"])

    def sweep_param_names(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return list(self.get_data(sweep_name)["sweep_param_names"])

    # for each optimization in the sweep, find the best beta
    def best_circuits(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        return self.get_data(sweep_name)["best_circuits"]

    def best_U_tots(self, sweep_name=None):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]

        timestamps = self.timestamps(sweep_name)

        # e.g. [[1,1],[2,1],[3,1],...]
        sweep_param_values = self.sweep_param_values(sweep_name)

        # e.g. ["N_blocks", "max_fock"]
        sweep_param_names = self.sweep_param_names(sweep_name)

        U_tots = []
        for timestamp in timestamps:
            U_tots.append(self.opt_analysis_obj.best_U_tot(timestamp))
        return U_tots, sweep_param_values, sweep_param_names

    def get_fixed_indx(
        self, sweep_name=None, fixed_param_names=[], fixed_param_values=[]
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        sweep_param_names = self.sweep_param_names(sweep_name)
        sweep_param_values = self.sweep_param_values(sweep_name)
        if len(fixed_param_names) == 0:
            return list(range(len(sweep_param_values)))
        indxs = []
        ordered_fixed_param_values = []
        for param_name in sweep_param_names:
            if param_name not in fixed_param_names:
                indxs.append(sweep_param_names.index(param_name))
            else:
                ordered_fixed_param_values.append(
                    fixed_param_values[fixed_param_names.index(param_name)]
                )
        all_fixed_param_values = np.delete(sweep_param_values, indxs, axis=1)
        indxs = np.where(
            (all_fixed_param_values == tuple(ordered_fixed_param_values)).all(axis=1)
        )[0]
        return indxs

    def plot_best_fidelities_vs_epoch(
        self,
        sweep_name=None,
        fixed_param_names=[],
        fixed_param_values=[],
        fig=None,
        ax=None,
        log=True,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(4, 3), dpi=200)
        ax = ax if ax is not None else fig.subplots()
        timestamps = self.timestamps(sweep_name=sweep_name)
        sweep_param_values = self.sweep_param_values(sweep_name=sweep_name)

        indxs = self.get_fixed_indx(sweep_name, fixed_param_names, fixed_param_values)

        timestamps = timestamps[indxs]
        sweep_param_values = sweep_param_values[indxs]

        for i in tqdm(range(0, len(timestamps))):
            timestamp = timestamps[i].split(TIMESTAMP_SEP)[0]
            sweep_param_value = sweep_param_values[i]
            self.opt_analysis_obj.plot_best_fidelity(
                timestamp=timestamp, fig=fig, ax=ax, label=str(sweep_param_value)
            )

    def plot_sweep_metric(
        self,
        metric,
        sweep_name=None,
        fixed_param_names=[],
        fixed_param_values=[],
        fig=None,
        ax=None,
        **kwargs,
    ):
        """
        metric:             metric to plot (e.g. fidelities)
        sweep_name:         name of sweep description group
        fixed_param_names:   fidelity plotted against this sweep parameter 
                            (e.g. 'N_blocks' from['N_blocks', 'target_fock'])
        fixed_param_values: list of values specifying other fixed parameters
                            (e.g. [3] for ['target_fock'] = [3])
        fig:                figure            
        ax:                 subplots of figure
        log:                boolean, plot log or linear
        **kwargs:           other plotting attributes
        """
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]

        sweep_param_names = self.sweep_param_names(sweep_name)

        if len(fixed_param_names) + 1 != len(sweep_param_names) or len(
            fixed_param_names
        ) != len(fixed_param_values):
            raise Exception(
                "Please properly fix "
                + str(len(sweep_param_names) - 1)
                + " parameters out of: "
                + str(sweep_param_names)
            )

        find_sweep_param_name = sweep_param_names[:]
        for fixed_param_name in fixed_param_names:
            if fixed_param_name not in sweep_param_names:
                raise Exception(
                    "Please properly fix "
                    + str(len(sweep_param_names) - 1)
                    + " parameters out of: "
                    + str(sweep_param_names)
                )
            find_sweep_param_name.remove(fixed_param_name)
        sweep_param_name = find_sweep_param_name[0]
        sweep_param_name_indx = sweep_param_names.index(sweep_param_name)

        indxs = self.get_fixed_indx(sweep_name, fixed_param_names, fixed_param_values)
        sweep_param_values = self.sweep_param_values(sweep_name)[indxs][
            :, sweep_param_name_indx
        ]  # y

        metric = metric[indxs] if indxs is not None else metric  # y

        sort_indx = np.argsort(sweep_param_values)
        sweep_param_values = sweep_param_values[sort_indx]
        metric = metric[sort_indx]

        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)

        ax = ax if ax is not None else fig.subplots()

        label = None
        if len(fixed_param_names) > 0:
            label = ", ".join(fixed_param_names) + ": " + str(fixed_param_values)
        if "label" in kwargs:
            label = kwargs.pop("label")

        ax.plot(sweep_param_values, metric, ":.", label=label, **kwargs)
        ax.set_xlabel(sweep_param_name)
        if sweep_param_name == N_BLOCKS:
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # uses integers as ticks
        fig.tight_layout()

    def plot_multi_sweep_metric(
        self,
        sweep_name=None,
        fig=None,
        ax=None,
        sweep_param_name=None,
        plot_sweep_metric_func=None,
        color_gradient=True,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        sweep_param_names = self.sweep_param_names(sweep_name)
        indx = sweep_param_names.index(sweep_param_name)
        fixed_param_names = sweep_param_names
        fixed_param_names.remove(sweep_param_name)

        sweep_param_values = self.sweep_param_values(sweep_name)
        all_fixed_param_values = np.delete(sweep_param_values, indx, axis=1)
        all_fixed_param_values = sorted(set([tuple(x) for x in all_fixed_param_values]))

        plot_sweep_metric_func = (
            self.plot_sweep_fidelities
            if not plot_sweep_metric_func
            else plot_sweep_metric_func
        )

        if color_gradient:
            parameters = np.array(all_fixed_param_values)[:,0]
            s_m = self._gradient_multiline_colorbar(parameters)

        for fixed_param_values in all_fixed_param_values:
            if color_gradient:
                kwargs['color'] = s_m.to_rgba(fixed_param_values[0])
            plot_sweep_metric_func(
                sweep_name,
                fixed_param_names=fixed_param_names,
                fixed_param_values=list(fixed_param_values),
                fig=fig,
                ax=ax,
                **kwargs,
            )
        if color_gradient:
            cbar  = plt.colorbar(s_m, ticks=parameters)
            cbar.ax.tick_params(labelsize=10) 
            cbar.ax.set_title(fixed_param_names[0], rotation=0, size=10)
            ax.get_legend().remove()

        

    def _gradient_multiline_colorbar(self, parameters):
        # taken from https://stackoverflow.com/questions/26545897/drawing-a-colorbar-aside-a-line-plot-using-matplotlib/26562639#26562639
        norm = mpl.colors.Normalize(
            vmin=np.min(parameters), vmax=np.max(parameters)
        )
        c_m = mpl.cm.plasma_r
        s_m = mpl.cm.ScalarMappable(cmap=c_m, norm=norm)
        s_m.set_array([])
        return s_m

    def _run_U_benchmark(
        self,
        sweep_name=None,
        benchmark_name="benchmark",
        initial_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
        target_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        timestamps = self.timestamps(sweep_name)

        self.get_data(sweep_name)[benchmark_name] = np.zeros(len(timestamps))
        for i in tqdm(range(len(timestamps))):
            N = self.opt_analysis_obj.get_data(timestamps[i])["N"]
            initial_state = initial_state_func(N)
            target_state = target_state_func(N)
            self.get_data(sweep_name)[benchmark_name][
                i
            ] = self.opt_analysis_obj.U_benchmark(
                initial_state, target_state, timestamp=timestamps[i]
            )
        return

    def plot_sweep_U_benchmark(
        self,
        sweep_name=None,
        fixed_param_names=[],
        fixed_param_values=[],
        fig=None,
        ax=None,
        log=True,
        benchmark_name="benchmark",
        initial_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
        target_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        if benchmark_name not in self.get_data(sweep_name):
            self._run_U_benchmark(
                sweep_name, benchmark_name, initial_state_func, target_state_func
            )

        metric = self.get_data(sweep_name)[benchmark_name]
        if log:
            metric = np.log10(1 - metric)

        self.plot_sweep_metric(
            metric,
            sweep_name=sweep_name,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            fig=fig,
            ax=ax,
            **kwargs,
        )

        if log:
            ax.set_ylabel("best $\log_{10}$(infidelity)")
            plt.legend(loc="upper right", prop={"size": 3})
        else:
            ax.set_ylabel("best fidelity")
            plt.legend(loc="lower right", prop={"size": 3})

    def plot_multi_sweep_U_benchmark(
        self,
        initial_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
        target_state_func=lambda N: qt.tensor(qt.basis(2, 0), qt.basis(N, 0)),
        benchmark_name="benchmark",
        sweep_name=None,
        fig=None,
        ax=None,
        sweep_param_name=None,
        log=True,
        **kwargs,
    ):
        self.plot_multi_sweep_metric(
            initial_state_func=initial_state_func,
            target_state_func=target_state_func,
            benchmark_name=benchmark_name,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            sweep_param_name=sweep_param_name,
            plot_sweep_metric_func=self.plot_sweep_U_benchmark,
            log=log,
            **kwargs,
        )

    def plot_sweep_fidelities(
        self,
        sweep_name=None,
        fixed_param_names=[],
        fixed_param_values=[],
        fig=None,
        ax=None,
        log=True,
        **kwargs,
    ):
        """
        sweep_name:         name of sweep description group
        fixed_param_names:   fidelity plotted against this sweep parameter 
                            (e.g. 'N_blocks' from['N_blocks', 'target_fock'])
        fixed_param_values: list of values specifying other fixed parameters
                            (e.g. [3] for ['target_fock'] = [3])
        fig:                figure            
        ax:                 subplots of figure
        log:                boolean, plot log or linear
        **kwargs:           other plotting attributes
        """
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        all_fids = self.best_fidelities(sweep_name)
        if log:
            all_fids = np.log10(1 - all_fids)

        self.plot_sweep_metric(
            all_fids,
            sweep_name=sweep_name,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            fig=fig,
            ax=ax,
            **kwargs,
        )

        if log:
            ax.set_ylabel("best $\log_{10}$(infidelity)")
            plt.legend(loc="upper right", prop={"size": 3})
        else:
            ax.set_ylabel("best fidelity")
            plt.legend(loc="lower right", prop={"size": 3})

    def plot_multi_sweep_fidelities(
        self,
        sweep_name=None,
        fig=None,
        ax=None,
        sweep_param_name=None,
        log=True,
        **kwargs,
    ):
        self.plot_multi_sweep_metric(
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            sweep_param_name=sweep_param_name,
            plot_sweep_metric_func=self.plot_sweep_fidelities,
            log=log,
            **kwargs,
        )

    def min_N_blocks_to_reach_fid(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        plot=True,
        log=False,
        fixed_param_names=[],
        fixed_param_values=[],
        fit=None,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]

        try:
            remaining_param_names = self.sweep_param_names(sweep_name)
            remaining_param_names.remove(N_BLOCKS)
            for fixed_param_name in fixed_param_names:
                remaining_param_names.remove(fixed_param_name)
            sweep_param_name = remaining_param_names[0]
            remaining_param_names.remove(sweep_param_name)
            assert len(remaining_param_names) == 0
        except:
            temp = self.sweep_param_names(sweep_name)
            temp.remove(N_BLOCKS)
            raise Exception(
                "Please properly fix "
                + str(len(temp) - 1)
                + " parameters out of: "
                + str(temp)
            )

        sweep_param_names = self.sweep_param_names(sweep_name)
        N_blocks_indx = sweep_param_names.index(N_BLOCKS)
        sweep_param_indx = sweep_param_names.index(sweep_param_name)

        sweep_param_values = self.sweep_param_values(sweep_name)
        unique_sweep_param_values = sorted(set(sweep_param_values[:, sweep_param_indx]))

        all_fids = self.best_fidelities(sweep_name)
        data = {"min_N_blocks": [], "sweep_param_values": []}
        for sweep_param_value in unique_sweep_param_values:
            indxs = self.get_fixed_indx(
                sweep_name=sweep_name,
                fixed_param_names=fixed_param_names + [sweep_param_name],
                fixed_param_values=fixed_param_values + [sweep_param_value],
            )
            N_blocks_swept = sweep_param_values[indxs][:, N_blocks_indx]
            sort_indxs = np.argsort(N_blocks_swept)
            N_blocks_swept = N_blocks_swept[sort_indxs]
            fids = all_fids[indxs][sort_indxs]
            satisfying_indxs = np.where(fids >= success_fid)[0]
            if len(satisfying_indxs) > 0:
                min_indx = np.min(satisfying_indxs)
                min_N_block = N_blocks_swept[min_indx]
                data["min_N_blocks"].append(min_N_block)
                data["sweep_param_values"].append(sweep_param_value)
        x = np.array(data["sweep_param_values"])
        y = np.array(data["min_N_blocks"])
        y_fit = []
        if fit is not None and len(x) > 0:
            p = np.poly1d(np.polyfit(x, y, fit))
            y_fit = p(x)

        if not plot:
            return x, sweep_param_name, y, y_fit

        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        label = "" if "label" not in kwargs else kwargs.pop("label")
        if fit is not None:
            ax.plot(x, y_fit, "-", label="Poly Fit " + label)

        ax.plot(x, y, ":o", **kwargs, label="Simulation " + label)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(sweep_param_name, size=8)
        ax.set_ylabel(
            "Minimum N Blocks to reach " + str(100 * success_fid) + "% Fidelity", size=8
        )
        if fit is not None:
            plt.legend(loc="lower right", prop={"size": 6})
        fig.tight_layout()

    def min_abs_sum_betas_to_reach_fid(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        plot=True,
        log=False,
        fixed_param_names=[],
        fixed_param_values=[],
        fit=None,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]

        try:
            remaining_param_names = self.sweep_param_names(sweep_name)
            remaining_param_names.remove(N_BLOCKS)
            for fixed_param_name in fixed_param_names:
                remaining_param_names.remove(fixed_param_name)
            sweep_param_name = remaining_param_names[0]
            remaining_param_names.remove(sweep_param_name)
            assert len(remaining_param_names) == 0
        except:
            temp = self.sweep_param_names(sweep_name)
            temp.remove(N_BLOCKS)
            raise Exception(
                "Please properly fix "
                + str(len(temp) - 1)
                + " parameters out of: "
                + str(temp)
            )

        sweep_param_names = self.sweep_param_names(sweep_name)
        sweep_param_indx = sweep_param_names.index(sweep_param_name)

        sweep_param_values = self.sweep_param_values(sweep_name)
        unique_sweep_param_values = sorted(set(sweep_param_values[:, sweep_param_indx]))

        all_fids = self.best_fidelities(sweep_name)
        abs_sum_betas = self.abs_sum_betas(sweep_name)

        data = {"min_abs_sum_beta": [], "sweep_param_values": []}
        for sweep_param_value in unique_sweep_param_values:
            indxs = self.get_fixed_indx(
                sweep_name=sweep_name,
                fixed_param_names=fixed_param_names + [sweep_param_name],
                fixed_param_values=fixed_param_values + [sweep_param_value],
            )
            abs_sum_betas_achieved = abs_sum_betas[indxs]
            sort_indxs = np.argsort(abs_sum_betas_achieved)
            abs_sum_betas_achieved = abs_sum_betas_achieved[sort_indxs]
            fids = all_fids[indxs][sort_indxs]
            satisfying_indxs = np.where(fids >= success_fid)[0]
            if len(satisfying_indxs) > 0:
                min_indx = np.min(satisfying_indxs)
                min_abs_sum_betas = abs_sum_betas_achieved[min_indx]
                data["min_abs_sum_beta"].append(min_abs_sum_betas)
                data["sweep_param_values"].append(sweep_param_value)
        x = np.array(data["sweep_param_values"])
        y = np.array(data["min_abs_sum_beta"])
        y_fit = []
        if fit is not None and len(x) > 0:
            p = np.poly1d(np.polyfit(x, y, fit))
            y_fit = p(x)

        if not plot:
            return x, sweep_param_name, y, y_fit

        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        if fit is not None:
            ax.plot(x, y_fit, "-", label="Poly Fit")

        ax.plot(x, y, ":o", **kwargs, label="Simulation")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(sweep_param_name, size=8)
        ax.set_ylabel(
            "Minimum $\\Sigma_i |\\beta_i|$ to reach "
            + str(100 * success_fid)
            + "% Fidelity",
            size=8,
        )
        if fit is not None:
            plt.legend(loc="lower right", prop={"size": 6})
        fig.tight_layout()

    def min_abs_sum_alphas_to_reach_fid(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        plot=True,
        log=False,
        fixed_param_names=[],
        fixed_param_values=[],
        fit=None,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]

        try:
            remaining_param_names = self.sweep_param_names(sweep_name)
            remaining_param_names.remove(N_BLOCKS)
            for fixed_param_name in fixed_param_names:
                remaining_param_names.remove(fixed_param_name)
            sweep_param_name = remaining_param_names[0]
            remaining_param_names.remove(sweep_param_name)
            assert len(remaining_param_names) == 0
        except:
            temp = self.sweep_param_names(sweep_name)
            temp.remove(N_BLOCKS)
            raise Exception(
                "Please properly fix "
                + str(len(temp) - 1)
                + " parameters out of: "
                + str(temp)
            )

        sweep_param_names = self.sweep_param_names(sweep_name)
        sweep_param_indx = sweep_param_names.index(sweep_param_name)

        sweep_param_values = self.sweep_param_values(sweep_name)
        unique_sweep_param_values = sorted(set(sweep_param_values[:, sweep_param_indx]))

        all_fids = self.best_fidelities(sweep_name)
        abs_sum_alphas = self.abs_sum_alphas(sweep_name)

        data = {"min_abs_sum_alpha": [], "sweep_param_values": []}
        for sweep_param_value in unique_sweep_param_values:
            indxs = self.get_fixed_indx(
                sweep_name=sweep_name,
                fixed_param_names=fixed_param_names + [sweep_param_name],
                fixed_param_values=fixed_param_values + [sweep_param_value],
            )
            abs_sum_alphas_achieved = abs_sum_alphas[indxs]
            sort_indxs = np.argsort(abs_sum_alphas_achieved)
            abs_sum_alphas_achieved = abs_sum_alphas_achieved[sort_indxs]
            fids = all_fids[indxs][sort_indxs]
            satisfying_indxs = np.where(fids >= success_fid)[0]
            if len(satisfying_indxs) > 0:
                min_indx = np.min(satisfying_indxs)
                min_abs_sum_alphas = abs_sum_alphas_achieved[min_indx]
                data["min_abs_sum_alpha"].append(min_abs_sum_alphas)
                data["sweep_param_values"].append(sweep_param_value)
        x = np.array(data["sweep_param_values"])
        y = np.array(data["min_abs_sum_alpha"])
        y_fit = []
        if fit is not None and len(x) > 0:
            p = np.poly1d(np.polyfit(x, y, fit))
            y_fit = p(x)

        if not plot:
            return x, sweep_param_name, y, y_fit

        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        if fit is not None:
            ax.plot(x, y_fit, "-", label="Poly Fit")

        ax.plot(x, y, ":o", **kwargs, label="Simulation")
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(sweep_param_name, size=8)
        ax.set_ylabel(
            "Minimum $|\\alpha_N|$ to reach " + str(100 * success_fid) + "% Fidelity",
            size=8,
        )
        if fit is not None:
            plt.legend(loc="lower right", prop={"size": 6})
        fig.tight_layout()

    def plot_2d(
        self,
        x_list,
        y_list,
        z_list,
        fig=None,
        ax=None,
        outlier_val=2,
        types=["scatter"],
        **kwargs,
    ):
        """
        'scatter', 'interpolate', 'extrapolate', None
        """
        default_types = ["interpolate", "extrapolate", "scatter"]
        if not isinstance(types, list):
            types = [types]
        for type in types:
            if type not in default_types:
                raise Exception("Please choose type(s) from: " + str(default_types))
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        x_list = list(x_list)
        y_list = list(y_list)
        z_list = list(z_list)

        x_coords = sorted(set(x_list))
        y_coords = sorted(set(y_list))
        xy_list = [(x_list[i], y_list[i]) for i in range(len(x_list))]
        Z = np.ones((len(y_coords), len(x_coords)))
        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                if (x_coords[i], y_coords[j]) in xy_list:
                    Z[j][i] = z_list[xy_list.index((x_coords[i], y_coords[j]))]
                else:
                    Z[j][i] = None

        if "extrapolate" in types:
            for j in range(Z.shape[0]):
                for i in range(1, Z.shape[1]):
                    if np.isnan(Z[j][i]):
                        Z[j][i] = Z[j][i - 1]
                        xy_list.append((x_coords[i], y_coords[j]))
                        x_list.append(x_coords[i])
                        y_list.append(y_coords[j])
                        z_list.append(Z[j][i])

        if "scatter" in types:
            plt.scatter(x_list, y_list, s=40, c=z_list, marker="o", cmap="cool")
            plt.colorbar(extend="both")
            return

        # not sure if necessary, because contourf interpolates naturally
        if "interpolate" in types:
            f = interp2d(x_list, y_list, z_list, kind="linear")
            Z = f(x_coords, y_coords)

        for i in range(len(x_coords)):
            for j in range(len(y_coords)):
                if (x_coords[i], y_coords[j]) not in xy_list:
                    Z[j][i] = outlier_val

        cmap = plt.get_cmap("cool")
        plt.contourf(
            x_coords,
            y_coords,
            Z,
            cmap=cmap,
            vmin=np.min(z_list),
            vmax=np.max(z_list),
            **kwargs,
        )
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(Z)
        m.set_clim(np.min(z_list), np.max(z_list))
        plt.colorbar(m)

    def plot_2D_min_vals(
        self,
        success_infids=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        fit=None,
        sweep_name=None,
        fig=None,
        ax=None,
        log=True,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        min_val_method=None,
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        min_val_method = (
            min_val_method
            if min_val_method is not None
            else self.min_abs_sum_betas_to_reach_fid
        )

        x_vals = []
        y_vals = []
        z_vals = []

        for success_infid in success_infids:
            (param_vals, param_name, min_vals, min_vals_fit,) = min_val_method(
                success_fid=1 - success_infid,
                sweep_name=sweep_name,
                plot=False,
                fixed_param_names=fixed_param_names,
                fixed_param_values=fixed_param_values,
                fit=fit,
            )
            param_vals = list(param_vals)
            min_vals = list(min_vals if fit is None else min_vals_fit)

            x_vals += param_vals
            z_vals += min_vals
            y_vals += [success_infid] * len(param_vals)

        outlier_val = np.max(z_vals) + 0.1
        self.plot_2d(
            x_vals,
            y_vals,
            z_vals,
            fig,
            ax,
            outlier_val=outlier_val,
            types=types,
            **kwargs,
        )
        ax.set_xlabel(param_name, size=8)
        ax.set_ylabel("Success Infidelity % Threshold", size=8)
        if log:
            plt.yscale("log")
        fig.tight_layout()

    def plot_2D_min_N_blocks(
        self,
        success_infids=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        fit=None,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        self.plot_2D_min_vals(
            success_infids=success_infids,
            fit=fit,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            min_val_method=self.min_N_blocks_to_reach_fid,
            **kwargs,
        )

        ax.set_title("Minimum N Blocks", size=8)
        fig.tight_layout()

    def plot_2D_min_abs_sum_betas(
        self,
        success_infids=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        fit=None,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        self.plot_2D_min_vals(
            success_infids=success_infids,
            fit=fit,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            min_val_method=self.min_abs_sum_betas_to_reach_fid,
            **kwargs,
        )

        ax.set_title("Minimum $\\Sigma_{i} |\\betas_i|", size=8)
        fig.tight_layout()

    def plot_2D_min_abs_sum_alphas(
        self,
        success_infids=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1],
        fit=None,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        self.plot_2D_min_vals(
            success_infids=success_infids,
            fit=fit,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            min_val_method=self.min_abs_sum_alphas_to_reach_fid,
            **kwargs,
        )

        ax.set_title("Minimum $\\Sigma_{i} |\\alphas_i|", size=8)
        fig.tight_layout()

    def plot_2D_metric(
        self,
        metric,
        outlier_val=-1,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        try:
            remaining_param_names = self.sweep_param_names(sweep_name)
            for fixed_param_name in fixed_param_names:
                remaining_param_names.remove(fixed_param_name)
            assert len(remaining_param_names) == 2
        except:
            raise Exception(
                "Please properly fix "
                + str(len(self.sweep_param_names(sweep_name)) - 2)
                + " parameters out of: "
                + str(self.sweep_param_names(sweep_name))
            )
        indxs = self.get_fixed_indx(
            sweep_name=sweep_name,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
        )
        sweep_param_names = self.sweep_param_names(sweep_name)
        sweep_param_values = self.sweep_param_values(sweep_name)[indxs]
        z_vals = metric[indxs]

        param_x_name = remaining_param_names[0]
        param_x_indx = sweep_param_names.index(param_x_name)
        param_y_name = remaining_param_names[1]
        param_y_indx = sweep_param_names.index(param_y_name)
        x_vals = sweep_param_values[:, param_x_indx]
        y_vals = sweep_param_values[:, param_y_indx]
        self.plot_2d(
            x_vals,
            y_vals,
            z_vals,
            fig,
            ax,
            outlier_val=outlier_val,
            types=types,
            **kwargs,
        )
        ax.set_xlabel(param_x_name, size=8)
        ax.set_ylabel(param_y_name, size=8)
        fig.tight_layout()

    def plot_2D_abs_mean_betas(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        metric = self.abs_mean_betas(sweep_name)
        outlier_val = np.max(metric) + 0.1
        self.plot_2D_metric(
            metric,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title("Mean |Betas|", size=8)

    def plot_2D_abs_sum_betas(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        metric = self.abs_sum_betas(sweep_name)
        outlier_val = np.max(metric) + 0.1
        self.plot_2D_metric(
            metric,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title("Sum |Betas|", size=8)

    def plot_2D_abs_mean_alphas(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        log=True,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        metric = self.abs_mean_alphas(sweep_name)
        metric = np.log10(metric) if log else metric
        outlier_val = np.max(metric) + 0.1
        self.plot_2D_metric(
            metric,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title("Mean |Alphas|" + ("(log)" if log else ""), size=8)

    def plot_2D_abs_sum_alphas(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        metric = self.abs_sum_alphas(sweep_name)
        outlier_val = np.max(metric) + 0.1
        self.plot_2D_metric(
            metric,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title("Sum |Alphas|", size=8)

    def plot_2D_success_fraction(
        self,
        success_fid=0.999,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        metric = self.success_fracs(success_fid, sweep_name)
        outlier_val = np.max(metric) + 0.1
        self.plot_2D_metric(
            metric,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title(
            "Success Fraction for Fidelity: " + str(success_fid * 100) + "%", size=8
        )

    def plot_2D_fidelity(
        self,
        sweep_name=None,
        fig=None,
        ax=None,
        fixed_param_names=[],
        fixed_param_values=[],
        log=True,
        types=["scatter"],
        **kwargs,
    ):
        sweep_name = sweep_name if sweep_name is not None else self.sweep_names[-1]
        fig = fig if fig is not None else plt.figure(figsize=(3.5, 2.5), dpi=200)
        ax = ax if ax is not None else fig.subplots()

        all_fids = self.best_fidelities(sweep_name)
        if log:
            all_fids = np.log10(1 - all_fids)
        outlier_val = np.min(all_fids) - 0.5 if log else np.max(all_fids) + 0.1
        self.plot_2D_metric(
            all_fids,
            outlier_val=outlier_val,
            sweep_name=sweep_name,
            fig=fig,
            ax=ax,
            fixed_param_names=fixed_param_names,
            fixed_param_values=fixed_param_values,
            types=types,
            **kwargs,
        )
        ax.set_title("Log Infidelity" if log else "Fidelity")
