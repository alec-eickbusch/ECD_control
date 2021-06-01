import numpy as np
import matplotlib.pyplot as plt
import ECD_control.ECD_optimization.tf_quantum as tfq
import qutip as qt


plt.rcParams.update(
    {
        "figure.figsize": (11.7, 8.27),
        "font.size": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def plot_wigner(
    state,
    alpha=0.0,
    tensor_state=True,
    contour=False,
    fig=None,
    ax=None,
    max_alpha=6,
    plot_max_alpha=None,
    cbar=False,
    npts=81,
):
    xvec = np.linspace(-max_alpha, max_alpha, npts)
    dx = xvec[1] - xvec[0]
    if plot_max_alpha is not None and plot_max_alpha > max_alpha:
        upper = np.arange(xvec[-1] + dx, plot_max_alpha + dx, dx)
        lower = np.arange(-plot_max_alpha, xvec[0], dx)
        num_pad = len(lower)
        num_pad_upper = len(upper)
        # print('num pad: %d num_pad_upper: %d' % (num_pad, num_pad_upper))
        xvec_plot = np.concatenate([lower, xvec, upper])
    else:
        xvec_plot = xvec
    roll_x = int(np.real(alpha) / dx)
    roll_y = int(np.imag(alpha) / dx)
    xvec2 = np.roll(xvec_plot, -roll_x)
    yvec2 = np.roll(xvec_plot, -roll_y)
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    if ax is None:
        ax = fig.subplots()
    if tensor_state:
        state = qt.ptrace(state, 1)
    if len(xvec_plot) > len(xvec):
        W = np.pad(W, (num_pad, num_pad), mode="constant")
    if contour:
        levels = np.linspace(-1.1, 1.1, 102)
        im = ax.contourf(
            xvec2, yvec2, W, cmap="seismic", vmin=-1, vmax=+1, levels=levels,
        )
    else:
        im = ax.pcolormesh(
            yvec2 - dx / 2.0, xvec2 - dx / 2.0, W, cmap="seismic", vmin=-1, vmax=+1
        )
    # ax.axhline(0, linestyle='-', color='black', alpha=0.2)
    # ax.axvline(0, linestyle='-', color='black', alpha=0.2)
    ax.set_xlabel(r"Re$(\alpha)$")
    ax.set_ylabel(r"Im$(\alpha)$")
    ax.grid()
    # ax.set_title(title)
    # TODO: Gaurentee that the wigners are square!
    fig.tight_layout()
    if cbar:
        fig.subplots_adjust(right=0.8, hspace=0.25, wspace=0.25)
        # todo: ensure colorbar even with plot...
        # todo: fix this colorbar

        cbar_ax = fig.add_axes([0.8, 0.225, 0.025, 0.65])
        ticks = np.linspace(-1, 1, 5)
        fig.colorbar(im, cax=cbar_ax, ticks=ticks)
        cbar_ax.set_title(r"$\frac{\pi}{2} W(\alpha)$", pad=10)
    ax.set_aspect("equal", adjustable="box")


# visulaization mixin for control optimization object
class VisualizationMixin:
    def plot_initial_states(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        for tf_state in self.initial_states:
            state = tfq.tf2qt(tf_state)
            plot_wigner(
                state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar,
            )

    def plot_final_states(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        raise Exception("Not implemented!")
        state = tfq.tf2qt(
            self.final_state(
                self.betas_rho,
                self.betas_angle,
                self.alphas_rho,
                self.alphas_angle,
                self.phis,
                self.thetas,
            )
        )
        plot_wigner(
            state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar,
        )

    def plot_target_states(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=False
    ):
        for tf_state in self.target_states:
            state = tfq.tf2qt(tf_state)
            plot_wigner(
                state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar,
            )

    def plot_state(self, i=0, contour=True, fig=None, ax=None, max_alpha=6, cbar=False):
        state = tfq.tf2qt(self.state(i=i))
        plot_wigner(
            state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar,
        )
