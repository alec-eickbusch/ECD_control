import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import qutip as qt

plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42})


class OptimizationAnalysis:
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(self.filename, "a") as f:
            self.timestamps = list(f.keys())
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
            self.data[timestamp]["betas"] = f[timestamp]["betas"].value
            self.data[timestamp]["alphas"] = f[timestamp]["alphas"].value
            self.data[timestamp]["phis"] = f[timestamp]["phis"].value
            self.data[timestamp]["thetas"] = f[timestamp]["thetas"].value
            self.data[timestamp]["fidelities"] = f[timestamp]["fidelities"].value
            if "initial_state" in f[timestamp]:
                initial_state = f[timestamp]["initial_state"].value
                initial_state_dims = f[timestamp]["initial_state_dims"].value
                target_state = f[timestamp]["target_state"].value
                target_state_dims = f[timestamp]["target_state_dims"].value
            self.data[timestamp]["initial_state"] = qt.Qobj(
                initial_state, dims=initial_state_dims.tolist()
            )
            self.data[timestamp]["target_state"] = qt.Qobj(
                target_state, dims=target_state_dims.tolist()
            )

    def fidelities(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["fidelities"]

    def idx_of_best_circuit(self, timestamp=None):
        fidelities = self.fidelities(timestamp)[-1]
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
            self.data[sweep_name]["timestamps"] = str(
                f[sweep_name].attrs["timestamps"]
            ).split(",")
            self.data[sweep_name]["fidelities"] = f[sweep_name]["fidelities"].value
            self.data[sweep_name]["sweep_param_values"] = f[sweep_name][
                "sweep_param_values"
            ].value

    def fidelities(self, sweep_name=None):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        self._load_data(sweep_name)
        return self.data[sweep_name]["fidelities"]

    def plot_sweep_fidelities(self, sweep_name=None, fig=None, ax=None, log=False):
        if sweep_name is None:
            sweep_name = self.sweep_names[-1]
        fids = self.fidelities(sweep_name)
        sweep_param_values = self.data[sweep_name]["sweep_param_values"]
        sweep_param_name = self.data[sweep_name]["sweep_param_name"]
        if fig is None:
            fig = plt.figure(figsize=(3.5, 2.5), dpi=200)
        if ax is None:
            ax = fig.subplots()
        if log:
            ax.semilogy(sweep_param_values, 1 - fids, "--.", color="black")
        else:
            ax.plot(sweep_param_values, fids, ":.", color="black")
        ax.set_xlabel(sweep_param_name)
        if log:
            ax.set_ylabel("best infidelity")
        else:
            ax.set_ylabel("best fidelity")
        fig.tight_layout()
