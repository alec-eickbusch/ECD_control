import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42})


class OptimizationAnalysis:
    def __init__(self, filename):
        self.filename = filename
        with h5py.File(self.filename, "a") as f:
            self.timestamps = list(f.keys())
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

    def fidelities(self, timestamp=None):
        if timestamp is None:
            timestamp = self.timestamps[-1]
        self._load_data(timestamp)
        return self.data[timestamp]["fidelities"]

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
