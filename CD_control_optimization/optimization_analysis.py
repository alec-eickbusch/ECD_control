import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import qutip as qt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42})


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

    def _load_data(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        if timestamp in self.data:
            return
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
                dims = f[timestamp]["state_dims"][()]
                self.data[timestamp]["initial_states"] = [
                    qt.Qobj(initial_state, dims=dims.tolist())
                    for initial_state in initial_states
                ]
                self.data[timestamp]["target_states"] = [
                    qt.Qobj(target_state, dims=dims.tolist())
                    for target_state in target_states
                ]
                if len(self.data[timestamp]["initial_states"]) == 1:
                    self.data[timestamp]["initial_state"] = self.data[timestamp][
                        "initial_states"
                    ][0]
                    self.data[timestamp]["target_state"] = self.data[timestamp][
                        "target_states"
                    ][0]

    def fidelities(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["fidelities"]

    def parameters(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["parameters"]

    # given a timestamp and epoch, return a list of epochs which have a fidelity larger than success_fid
    def successful_idxs(self, timestamp=None, epoch=-1, success_fid=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
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
        self._load_data(timestamp)
        return self.data[timestamp]["initial_state"]

    def target_state(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["target_state"]

    def best_circuit(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        idx = self.idx_of_best_circuit(timestamp)
        betas = self.data[timestamp]["betas"][-1][idx]
        alphas = self.data[timestamp]["betas"][-1][idx]
        phis = self.data[timestamp]["betas"][-1][idx]
        thetas = self.data[timestamp]["betas"][-1][idx]
        max_fid = self.data[timestamp]["fidelities"][-1][idx]
        return {
            "fidelity": max_fid,
            "betas": betas,
            "alphas": alphas,
            "phis": phis,
            "thetas": thetas,
        }

    def print_info(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        best_circuit = self.best_circuit(timestamp)
        with np.printoptions(precision=5, suppress=True):
            for parameter, value in self.data[timestamp]["parameters"].items():
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
        self._load_data(timestamp)
        return self.data[timestamp]["betas"]

    def alphas(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["alphas"]

    # average over BLOCKS in the circuit
    def average_magnitude_betas(self, timestamp=None):
        betas = self.betas(timestamp)
        return np.mean(np.abs(betas), axis=-1)

    def average_magnitude_alphas(self, timestamp=None):
        alphas = self.alphas(timestamp)
        return np.mean(np.abs(alphas), axis=-1)

    def plot_fidelities(self, timestamp=None, fig=None, ax=None, log=True):
        fidelities = self.fidelities(timestamp).T
        if fig is None:
            fig = plt.figure(figsize=(4, 3), dpi=200)
        if ax is None:
            ax = fig.subplots()
        for fids in fidelities:
            if log:
                ax.semilogy(1 - fids)
            else:
                ax.plot(fids)
        ax.set_xlabel("epoch")
        if log:
            ax.set_ylabel("infidelity")
        else:
            ax.set_ylabel("fidelity")
        fig.tight_layout()

    def plot_average_magnitude_beta(self, timestamp=None, fig=None, ax=None):
        average_mag_betas = self.average_magnitude_betas(timestamp).T
        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
        if ax is None:
            ax = fig.subplots()
        for mag_betas in average_mag_betas:
            ax.plot(mag_betas)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\beta|$")
        fig.tight_layout()

    def plot_average_magnitude_alpha(self, timestamp=None, fig=None, ax=None):
        average_mag_alphas = self.average_magnitude_alphas(timestamp).T
        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
        if ax is None:
            ax = fig.subplots()
        for mag_alphas in average_mag_alphas:
            ax.plot(mag_alphas)
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$|\alpha|$")
        fig.tight_layout()

    def plot_mag_betas(self, timestamp=None, fig=None):
        # will have shape num blocks, # multistart, #epochs with transpose
        betas = self.betas(timestamp).T
        if fig is None:
            fig = plt.figure(figsize=(14, 3), dpi=200)
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
        if fig is None:
            fig = plt.figure(figsize=(14, 3), dpi=200)
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
        if fig is None:
            fig = plt.figure(figsize=(14, 3), dpi=200)
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
        if fig is None:
            fig = plt.figure(figsize=(14, 3), dpi=200)
        axs = fig.subplots(1, alphas.shape[0])
        for circuit_block, bs in enumerate(alphas):
            for alpha_per_epoch in bs:
                axs[circuit_block].plot(180 * np.angle(alpha_per_epoch) / np.pi)
            axs[circuit_block].set_xlabel("epoch")
            axs[circuit_block].set_ylabel(r"arg$(\alpha)$ (deg)")
            axs[circuit_block].set_title(r"$D_{%d}(\alpha)$" % circuit_block)
        fig.tight_layout()


class OptimizationSweepsAnalysis:
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(self.filename, "a") as f:
            group_names = list(f.keys())
        self.sweep_names = []
        for group_name in group_names:
            if group_name.split(" ")[0] == "sweep":
                self.sweep_names.append(group_name)
            # TODO: is it gaurenteed that the most recent sweep will be [-1]?
        self.data = {}

    def _load_data(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        if sweep_name in self.data:
            return
        self.data[sweep_name] = {}
        with h5py.File(self.filename, "a") as f:
            self.data[sweep_name]["sweep_param_name"] = f[sweep_name].attrs[
                "sweep_param_name"
            ]
            self.data[sweep_name]["timestamps"] = f[sweep_name].attrs["timestamps"]
            self.data[sweep_name]["fidelities"] = f[sweep_name]["fidelities"][()]
            self.data[sweep_name]["sweep_param_values"] = f[sweep_name][
                "sweep_param_values"
            ][()]

    def timestamps(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        return self.data[sweep_name]["timestamps"]

    def fidelities(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        return self.data[sweep_name]["fidelities"]

    def get_opt_object(self,):
        return OptimizationAnalysis(self.filename)

    def success_fracs(self, success_fid=None, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        fracs = []
        for timestamp in self.timestamps(sweep_name):
            fracs.append(
                self.get_opt_object().success_fraction(
                    timestamp=timestamp, success_fid=success_fid
                )
            )
        return np.array(fracs)

    def sweep_param_values(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        return self.data[sweep_name]["sweep_param_values"]

    # for each optimization in the sweep, find the best beta
    def best_circuits(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        circuits = []
        for timestamp in self.timestamps(sweep_name):
            circuits.append(self.get_opt_object().best_circuit(timestamp))
        return circuits

    def plot_best_mag_betas(self, sweep_name=None, fig=None, ax=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        sweep_param_values = self.data[sweep_name]["sweep_param_values"]
        sweep_param_name = self.data[sweep_name]["sweep_param_name"]
        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
        if ax is None:
            ax = fig.subplots()
        circuits = self.best_circuits(sweep_name)
        for i, value in enumerate(sweep_param_values):
            betas = circuits[i]["betas"]
            num = len(betas)
            plt.scatter(value * np.ones(num), np.abs(betas))
        sweep_param_name = self.data[sweep_name]["sweep_param_name"]
        ax.set_xlabel(sweep_param_name)
        ax.set_ylabel(r"$|\beta|$")
        plt.tight_layout()

    def plot_sweep_fidelities(
        self, sweep_name=None, labels=None, fig=None, ax=None, log=True
    ):
        if sweep_name is None:
            sweep_name = [self.sweep_names[-1]]
        if not isinstance(sweep_name, list):
            sweep_name = [sweep_name]

        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)

        if ax is None:
            ax = fig.subplots()

        if log:
            ax.set_ylabel("best infidelity")
        else:
            ax.set_ylabel("best fidelity")

        for i in range(len(sweep_name)):
            sweep = sweep_name[i]
            label = labels[i] if labels is not None else None
            fids = self.fidelities(sweep)
            sweep_param_values = self.data[sweep]["sweep_param_values"]
            if log:
                ax.semilogy(sweep_param_values, 1 - fids, ":.", label=label)
            else:
                ax.plot(sweep_param_values, fids, ":.", label=label)
        if labels is not None:
            if log:
                plt.legend(loc="upper right", prop={"size": 6})
            else:
                plt.legend(loc="lower right", prop={"size": 6})
        # sweep_param_name should be the same across sweeps
        sweep_param_name = self.data[sweep]["sweep_param_name"]
        ax.set_xlabel(sweep_param_name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # uses integers as ticks
        fig.tight_layout()

    def plot_sweep_success_fraction(
        self, success_fid=0.999, sweep_name=None, fig=None, ax=None
    ):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        fracs = self.success_fracs(sweep_name=sweep_name, success_fid=success_fid)
        sweep_param_values = self.data[sweep_name]["sweep_param_values"]
        sweep_param_name = self.data[sweep_name]["sweep_param_name"]
        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
        if ax is None:
            ax = fig.subplots()
        ax.plot(sweep_param_values, fracs, ":.", color="black")
        ax.set_xlabel(sweep_param_name)
        ax.set_ylabel("Fraction with F > %.3f" % success_fid)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # uses integers as ticks
        fig.tight_layout()

