#%%
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

plt.rcParams.update({"font.size": 16, "pdf.fonttype": 42, "ps.fonttype": 42})
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

#%%


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
    W = (np.pi / 2.0) * qt.wigner(state, xvec, xvec, g=2)
    if len(xvec_plot) > len(xvec):
        W = np.pad(W, (num_pad, num_pad), mode="constant")
    if contour:
        levels = np.linspace(-1.1, 1.1, 102)
        im = ax.contourf(
            xvec2 - dx / 2.0,
            yvec2 - dx / 2.0,
            W,
            cmap="seismic",
            vmin=-1,
            vmax=+1,
            levels=levels,
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

        cbar_ax = fig.add_axes([0.6, 0.225, 0.025, 0.65])
        ticks = np.linspace(-1, 1, 5)
        fig.colorbar(im, cax=cbar_ax, ticks=ticks)
        cbar_ax.set_title(r"$\frac{\pi}{2} W(\alpha)$", pad=10)
    ax.set_aspect("equal", adjustable="box")


def plot_pulse(epsilon, Omega, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
    axs = fig.subplots(2, sharex=True)
    ts = np.arange(len(epsilon))
    axs[0].plot(ts, 1e3 * np.real(epsilon) / 2 / np.pi, label="I", color="firebrick")
    axs[0].plot(
        ts, 1e3 * np.imag(epsilon) / 2 / np.pi, ":", label="Q", color="firebrick"
    )
    axs[1].plot(ts, 1e3 * np.real(Omega) / 2 / np.pi, label="I", color="darkblue")
    axs[1].plot(ts, 1e3 * np.imag(Omega) / 2 / np.pi, ":", label="Q", color="darkblue")
    axs[0].set_ylabel(r"$\varepsilon(t)$ (MHz)")
    axs[1].set_ylabel(r"$\Omega(t)$ (MHz)")
    axs[1].set_xlabel("t (ns)")
    axs[0].yaxis.tick_right()
    axs[1].yaxis.tick_right()
    axs[0].yaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")
    axs[0].tick_params(direction="in")
    axs[1].tick_params(direction="in")
    y_max_e = (
        1.1
        * 1e3
        * np.max([np.abs(np.real(epsilon)), np.abs(np.imag(epsilon))])
        / 2
        / np.pi
    )
    axs[0].set_ylim([-y_max_e, +y_max_e])
    y_max_O = (
        1.1 * 1e3 * np.max([np.abs(np.real(Omega)), np.abs(np.imag(Omega))]) / 2 / np.pi
    )
    axs[1].set_ylim([-y_max_O, +y_max_O])


def plot_pulse_with_alpha(epsilon, Omega, alpha, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
    axs = fig.subplots(3, sharex=True)
    ts = np.arange(len(epsilon))
    axs[0].plot(ts, np.real(alpha), color="black", label="I")
    axs[0].plot(ts, np.imag(alpha), color="grey", label="Q")
    axs[1].plot(ts, 1e3 * np.real(epsilon) / 2 / np.pi, label="I", color="maroon")
    axs[1].plot(ts, 1e3 * np.imag(epsilon) / 2 / np.pi, label="Q", color="indianred")
    axs[2].plot(ts, 1e3 * np.real(Omega) / 2 / np.pi, label="I", color="darkblue")
    axs[2].plot(ts, 1e3 * np.imag(Omega) / 2 / np.pi, label="Q", color="steelblue")
    axs[0].set_ylabel(r"$\alpha(t)$")
    axs[1].set_ylabel(r"$\varepsilon(t)$ (MHz)")
    axs[2].set_ylabel(r"$\Omega(t)$ (MHz)")
    axs[2].set_xlabel("t (ns)")
    axs[0].yaxis.tick_right()
    axs[1].yaxis.tick_right()
    axs[2].yaxis.tick_right()
    axs[0].yaxis.set_ticks_position("both")
    axs[1].yaxis.set_ticks_position("both")
    axs[2].yaxis.set_ticks_position("both")
    axs[0].tick_params(direction="in")
    axs[1].tick_params(direction="in")
    axs[2].tick_params(direction="in")
    y_max_e = (
        1.1
        * 1e3
        * np.max([np.abs(np.real(epsilon)), np.abs(np.imag(epsilon))])
        / 2
        / np.pi
    )
    axs[1].set_ylim([-y_max_e, +y_max_e])
    y_max_O = (
        1.1 * 1e3 * np.max([np.abs(np.real(Omega)), np.abs(np.imag(Omega))]) / 2 / np.pi
    )
    axs[2].set_ylim([-y_max_O, +y_max_O])
    axs[0].legend(frameon=False, loc="upper left")
    axs[1].legend(frameon=False, loc="upper left")
    axs[2].legend(frameon=False, loc="upper left")


def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(
        ts, data_array, kind="cubic", bounds_error=False
    )  # can test different kinds


# still having problems with this version
"""
def alpha_from_epsilon(epsilon_array, dt=1, K = 0, alpha_init=0+0j, kappa_cavity=0):
    t_eval = np.linspace(0,len(epsilon_array)*dt-dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    dalpha_dt = lambda t, alpha : -1j*epsilon(t) - (kappa_cavity/2)*alpha# - 1j*4*K*np.abs(alpha)**2 * alpha
    alpha = solve_ivp(dalpha_dt,(0,len(epsilon_array)*dt-dt),y0=[alpha_init],\
                      method='RK23',t_eval=t_eval).y[0]
    return alpha
"""


def alpha_from_epsilon(epsilon):
    alpha = -1j * np.cumsum(epsilon)
    return alpha


# %%
