#%%
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

plt.rcParams.update({"font.size": 16, "pdf.fonttype": 42, "ps.fonttype": 42})
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

#%%
# todo: nice plotting without Omega
def plot_pulse(epsilon, Omega=None, fig=None):
    if Omega is None:
        Omega = np.zeros_like(epsilon)
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


# can set nbar=True to instead plot with nbar
def plot_pulse_with_alpha(
    epsilon, Omega, alpha, fig=None, nbar=False, labels=True, us=True
):
    if fig is None:
        fig = plt.figure(figsize=(8, 4), dpi=200)
    axs = fig.subplots(3, sharex=True)
    ts = np.arange(len(epsilon))
    if us:
        ts = ts / 1e3
    if nbar:
        axs[0].plot(ts, np.abs(alpha) ** 2, color="black")
    else:
        axs[0].plot(ts, np.real(alpha), color="black", label="I")
        axs[0].plot(ts, np.imag(alpha), color="grey", label="Q")
    axs[1].plot(ts, 1e3 * np.real(epsilon) / 2 / np.pi, label="I", color="maroon")
    axs[1].plot(ts, 1e3 * np.imag(epsilon) / 2 / np.pi, label="Q", color="indianred")
    axs[2].plot(ts, 1e3 * np.real(Omega) / 2 / np.pi, label="I", color="darkblue")
    axs[2].plot(ts, 1e3 * np.imag(Omega) / 2 / np.pi, label="Q", color="steelblue")
    if nbar:
        axs[0].set_ylabel(r"$\langle n(t) \rangle$")
    else:
        axs[0].set_ylabel(r"$\alpha(t)$")
    axs[1].set_ylabel(r"$\varepsilon(t)$ (MHz)")
    axs[2].set_ylabel(r"$\Omega(t)$ (MHz)")
    if us:
        axs[2].set_xlabel("t (μs)")
    else:
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
    if labels:
        axs[0].legend(frameon=False, loc="upper left")
        axs[1].legend(frameon=False, loc="upper left")
        axs[2].legend(frameon=False, loc="upper left")


def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(
        ts, data_array, kind="cubic", bounds_error=False
    )  # can test different kinds


# still having problems with this version


def alpha_from_epsilon2(
    epsilon_array, dt=1, delta=0, alpha_init=0 + 0j, kappa_cavity=0
):
    t_eval = np.linspace(0, len(epsilon_array) * dt - dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    dalpha_dt = (
        lambda t, alpha: -1j * epsilon(t) + delta * alpha - (kappa_cavity / 2) * alpha
    )  # - 1j*4*K*np.abs(alpha)**2 * alpha
    alpha = solve_ivp(
        dalpha_dt,
        (0, len(epsilon_array) * dt - dt),
        y0=[alpha_init],
        method="RK23",
        t_eval=t_eval,
    ).y[0]
    return alpha

# todo: get direction of rotation correct.
def alpha_from_epsilon(epsilon, delta=0, dt=1, alpha_init=0 + 0j):
    ts = np.arange(0, len(epsilon)) * dt
    integrand = np.exp(1j * delta * ts) * epsilon
    return np.exp(-1j * delta * ts) * (alpha_init - 1j * np.cumsum(integrand))


# %%
