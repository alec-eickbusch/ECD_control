import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# note that some pulse functions also in fpga_lib are repeated here so this file can be somewhat standalone.


def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def ring_up_smootherstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 6 * ts ** 5 - 15 * ts ** 4 + 10 * ts ** 3


def ring_up_smoothstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 3 * ts ** 2 - 2 * ts ** 3


def rotate(theta, phi=0, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    amp = 1 / energy
    wave = (1 + 0j) * wave
    return (theta / (2.0)) * amp * np.exp(1j * phi) * wave


# displace cavity by an amount alpha
def disp_gaussian(alpha, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    wave = (1 + 0j) * wave
    return (
        (np.abs(alpha) / energy) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave
    )


class FakePulse:
    def __init__(self, unit_amp, sigma, chop, detune=0):
        self.unit_amp = unit_amp
        self.sigma = sigma
        self.chop = chop
        self.detune = detune

    def make_wave(self, pad=False):
        wave = gaussian_wave(sigma=self.sigma, chop=self.chop)
        return np.real(wave), np.imag(wave)


class FakeStorage:
    def __init__(
        self,
        chi_kHz=-30.0,
        chi_prime_Hz=1.0,
        Ks_Hz=-2.0,
        epsilon_m_MHz=400.0,
        T1_us=340.0,
        unit_amp=0.05,
        sigma=15,
        chop=4,
        max_dac=0.6,
    ):
        self.chi_kHz = chi_kHz
        self.chi_prime_Hz = chi_prime_Hz
        self.Ks_Hz = Ks_Hz
        self.epsilon_m_MHz = epsilon_m_MHz
        self.max_dac = max_dac
        self.T1_us = T1_us

        self.displace = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop)


class FakeQubit:
    def __init__(self, unit_amp, sigma, chop, detune=0):
        self.pulse = FakePulse(unit_amp=unit_amp, sigma=sigma, chop=chop, detune=detune)


# Solution to linear differential equation
def alpha_from_epsilon_linear(epsilon, delta=0, kappa=0, dt=1, alpha_init=0 + 0j):
    ts = np.arange(0, len(epsilon)) * dt
    integrand = np.exp((1j * delta + kappa / 2.0) * ts) * epsilon
    return np.exp(-1 * (1j * delta + kappa / 2.0) * ts) * (
        alpha_init - 1j * np.cumsum(integrand)
    )


# Later: include chi prime for deviation during the displacements?
def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(
        ts, data_array, kind="cubic", bounds_error=False
    )  # can test different kinds


def get_flip_idxs(qubit_dac_pulse):
    return find_peaks(qubit_dac_pulse, height=np.max(qubit_dac_pulse) * 0.975)[0]


# solution to nonlinear differential equation
def alpha_from_epsilon_nonlinear(
    epsilon_array, delta=0, Ks=0, kappa=0, alpha_init=0 + 0j
):
    dt = 1
    t_eval = np.linspace(0, len(epsilon_array) * dt - dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    # todo: find correct rotation...
    dalpha_dt = lambda t, alpha: (
        -1j * delta * alpha
        - 2j * Ks * np.abs(alpha) ** 2 * alpha
        - (kappa / 2.0) * alpha
        - 1j * epsilon(t)
    )
    alpha = solve_ivp(
        dalpha_dt,
        (0, len(epsilon_array) * dt - dt),
        y0=[alpha_init],
        method="RK45",
        t_eval=t_eval,
        rtol=1e-16,  # 1e-13,
        atol=1e-16,  # 1e-13
    ).y[0]
    return alpha


# solution to nonlinear differential equation
def alpha_from_epsilon_nonlinear_finite_difference(
    epsilon_array, delta=0, Ks=0, kappa=0, alpha_init=0 + 0j
):
    dt = 1
    alpha = np.zeros_like(epsilon_array)
    alpha[0] = alpha_init
    for j in range(1, len(epsilon_array) - 1):
        alpha[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha[j]
                - 2j * Ks * np.abs(alpha[j]) ** 2 * alpha[j]
                - (kappa / 2.0) * alpha[j]
                - 1j * epsilon_array[j]
            )
            + alpha[j - 1]
        )
    return alpha


# refer to "Full Hamiltonian in the displaced frame" notes
def alpha_from_epsilon_ge(
    epsilon_array,
    delta=0,
    chi=0,
    chi_prime=0,
    Ks=0,
    kappa=0,
    alpha_g_init=0 + 0j,
    alpha_e_init=0 + 0j,
):
    dt = 1
    t_eval = np.linspace(0, len(epsilon_array) * dt - dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    rtol = 1e-15
    atol = rtol
    method = "RK45"
    # todo: find correct rotation...
    dalpha_dt_g = lambda t, alpha: (
        -1j * delta * alpha
        - 2j * Ks * np.abs(alpha) ** 2 * alpha
        - (kappa / 2.0) * alpha
        - 1j * epsilon(t)
    )
    alpha_g = solve_ivp(
        dalpha_dt_g,
        (0, len(epsilon_array) * dt - dt),
        y0=[alpha_g_init],
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    ).y[0]
    if chi == 0 and chi_prime == 0 and alpha_g_init == alpha_e_init:
        alpha_e = alpha_g
    else:
        dalpha_dt_e = lambda t, alpha: (
            -1j * delta * alpha
            - 2j * Ks * np.abs(alpha) ** 2 * alpha
            - (kappa / 2.0) * alpha
            - 1j * epsilon(t)
            - 1j * (chi + 2 * chi_prime * np.abs(alpha) ** 2) * alpha
        )
        alpha_e = solve_ivp(
            dalpha_dt_e,
            (0, len(epsilon_array) * dt - dt),
            y0=[alpha_e_init],
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        ).y[0]
    return alpha_g, alpha_e


def alpha_from_epsilon_ge_finite_difference(
    epsilon_array,
    delta=0,
    chi=0,
    chi_prime=0,
    Ks=0,
    kappa=0,
    alpha_g_init=0 + 0j,
    alpha_e_init=0 + 0j,
):
    dt = 1
    alpha_g = np.zeros_like(epsilon_array)
    alpha_e = np.zeros_like(epsilon_array)
    # todo: can handle initial condition with finite difference better...
    alpha_g[0], alpha_g[1] = alpha_g_init, alpha_g_init
    alpha_e[0], alpha_e[1] = alpha_e_init, alpha_e_init
    for j in range(1, len(epsilon_array) - 1):
        alpha_g[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha_g[j]
                - 2j * Ks * np.abs(alpha_g[j]) ** 2 * alpha_g[j]
                - (kappa / 2.0) * alpha_g[j]
                - 1j * epsilon_array[j]
            )
            + alpha_g[j - 1]
        )
        alpha_e[j + 1] = (
            2
            * dt
            * (
                -1j * delta * alpha_e[j]
                - 2j * Ks * np.abs(alpha_e[j]) ** 2 * alpha_e[j]
                - (kappa / 2.0) * alpha_e[j]
                - 1j * epsilon_array[j]
                - 1j * (chi + 2 * chi_prime * np.abs(alpha_e[j]) ** 2) * alpha_e[j]
            )
            + alpha_e[j - 1]
        )
    return alpha_g, alpha_e


def get_ge_trajectories(
    epsilon,
    delta=0,
    chi=0,
    chi_prime=0,
    Ks=0,
    kappa=0,
    flip_idxs=[],
    finite_difference=False,
):
    func = (
        alpha_from_epsilon_ge_finite_difference
        if finite_difference
        else alpha_from_epsilon_ge
    )
    f = lambda epsilon, alpha_g_init, alpha_e_init: func(
        epsilon,
        delta=delta,
        chi=chi,
        chi_prime=chi_prime,
        Ks=Ks,
        kappa=kappa,
        alpha_g_init=alpha_g_init,
        alpha_e_init=alpha_e_init,
    )
    epsilons = np.split(epsilon, flip_idxs)
    alpha_g = []  # alpha_g defined as the trajectory that starts in g
    alpha_e = []
    g_state = 0  # this bit will track if alpha_g is in g (0) or e (1)
    alpha_g_init = 0 + 0j
    alpha_e_init = 0 + 0j
    for epsilon in epsilons:
        alpha_g_current, alpha_e_current = f(epsilon, alpha_g_init, alpha_e_init)
        if g_state == 0:
            alpha_g.append(alpha_g_current)
            alpha_e.append(alpha_e_current)
        else:
            alpha_g.append(alpha_e_current)
            alpha_e.append(alpha_g_current)
        # because we will flip the qubit, the initial state for the next g trajectory will be the final from e of the current trajectory.
        alpha_g_init = alpha_e_current[-1]
        alpha_e_init = alpha_g_current[-1]
        g_state = 1 - g_state  # flip the bit
    alpha_g = np.concatenate(alpha_g)
    alpha_e = np.concatenate(alpha_e)
    return alpha_g, alpha_e


def plot_trajs_complex(alpha_g, alpha_e=None, bound=None):
    fig, ax = plt.subplots()
    ax.plot(np.real(alpha_g), np.imag(alpha_g), label="g")
    ax.fill_between(np.real(alpha_g), np.imag(alpha_g), alpha=0.2)
    if alpha_e is not None:
        ax.plot(np.real(alpha_e), np.imag(alpha_e), label="e")
        ax.fill_between(np.real(alpha_e), np.imag(alpha_e), alpha=0.2)
    if bound is not None:
        ax.set_xlim([-bound, bound])
        ax.set_ylim([-bound, bound])
    ax.grid()
    ax.legend(frameon=False)


def plot_trajs_linear(alpha_g, alpha_e=None):
    fig, ax = plt.subplots()
    ts = np.arange(len(alpha_g))
    ax.plot(ts, np.real(alpha_g), label="re(g)")
    ax.plot(ts, np.imag(alpha_g), label="im(g)")
    if alpha_e is not None:
        ax.plot(ts, np.real(alpha_e), label="re(e)")
        ax.plot(ts, np.imag(alpha_e), label="im(e)")
    ax.grid()
    ax.legend(frameon=False)


# this will use the pre-calibrated pulses.
# note that this will return the DAC pulses, not the values of epsilon and Omega.
# Buffer time can be a negative number if you wish to perform the pi pulse while the cavity is being displaced
# conditional displacement is defined as:
# D(beta/2)|eXg| + D(-beta/2) |gXe|
def conditional_displacement(
    beta,
    alpha,
    storage,
    qubit,
    buffer_time=4,
    curvature_correction=True,
    chi_prime_correction=True,
    kerr_correction=True,
    pad=True,
    finite_difference=True,
    output=False,
):
    beta = float(beta) if isinstance(beta, int) else beta
    alpha = float(alpha) if isinstance(alpha, int) else alpha
    chi = 2 * np.pi * 1e-6 * storage.chi_kHz
    chi_prime = 2 * np.pi * 1e-9 * storage.chi_prime_Hz if chi_prime_correction else 0.0
    Ks = 2 * np.pi * 1e-9 * storage.Ks_Hz

    # delta is the Hamiltonian parameter, so if it is positive, that means the cavity
    # is detuned positivly relative to the current frame, so the drive is
    # below the cavity.

    # We expect chi to be negative, so we want to drive below the cavity, hence delta should
    # be positive.

    delta = -chi / 2.0
    epsilon_m = 2 * np.pi * 1e-3 * storage.epsilon_m_MHz
    alpha = np.abs(alpha)
    beta_abs = np.abs(beta)
    beta_phase = np.angle(beta)

    def ratios(alpha, tw):
        n = np.abs(alpha) ** 2
        chi_effective = chi + 2 * chi_prime * n
        r = np.cos((chi_effective / 2.0) * tw)
        r2 = np.cos(chi_effective * tw)
        # r = np.cos((chi/2.0)*(tw + 2*tp))/np.cos((chi/2.0)*tp)
        # r2 = np.cos((chi/2.0)*(tw + 2*tp)) - np.cos((chi/2.0)*(tw + tp))
        return r, r2

    # the initial guesses
    # phase of the displacements.
    phase = beta_phase + np.pi / 2.0
    n = np.abs(alpha) ** 2
    chi_effective = chi + 2 * chi_prime * n
    # initial tw
    tw = int(np.abs(np.arcsin(beta_abs / (2 * alpha)) / chi_effective))

    # note, even with pad =False, there is a leading and trailing 0 because
    # every gaussian pulse will start / end at 0. Could maybe remove some of these
    # later to save a few ns.
    dr, di = storage.displace.make_wave(pad=False)
    d = storage.displace.unit_amp * (dr + 1j * di)
    pr, pi = qubit.pulse.make_wave(pad=False)
    # doing the same thing the FPGA does
    detune = qubit.pulse.detune
    if np.abs(detune) > 0:
        ts = np.arange(len(pr)) * 1e-9
        c_wave = (pr + 1j * pi) * np.exp(-2j * np.pi * ts * detune)
        pr, pi = np.real(c_wave), np.imag(c_wave)
    p = qubit.pulse.unit_amp * (pr + 1j * pi)

    # ratios of the displacements
    r, r2 = ratios(alpha, tw)

    # only add buffer time at the final setp
    def construct_CD(alpha, tw, r, r2, buf=0):

        cavity_dac_pulse = np.concatenate(
            [
                alpha * d * np.exp(1j * phase),
                np.zeros(tw),
                r * alpha * d * np.exp(1j * (phase + np.pi)),
                np.zeros(len(p) + 2 * buf),
                r * alpha * d * np.exp(1j * (phase + np.pi)),
                np.zeros(tw),
                r2 * alpha * d * np.exp(1j * phase),
            ]
        )
        qubit_dac_pulse = np.concatenate(
            [
                np.zeros(tw + 2 * len(d) + buf),
                p,
                np.zeros(tw + 2 * len(d) + buf),
            ]
        )
        # need to detune the pulse for chi prime
        if chi_prime_correction:
            ts = np.arange(len(cavity_dac_pulse))
            cavity_dac_pulse = cavity_dac_pulse * np.exp(1j * ts * chi_prime)
        return cavity_dac_pulse, qubit_dac_pulse

    cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r2)

    if curvature_correction:

        def integrated_beta(epsilon):
            # note that the trajectories are first solved without kerr.
            flip_idx = int(len(epsilon) / 2)
            alpha_g, alpha_e = get_ge_trajectories(
                epsilon,
                delta=delta,
                chi=chi,
                chi_prime=chi_prime,
                Ks=Ks,
                flip_idxs=[flip_idx],
                finite_difference=finite_difference,
            )
            return np.abs(alpha_g[-1] - alpha_e[-1])

        epsilon = cavity_dac_pulse * epsilon_m
        current_beta = integrated_beta(epsilon)
        diff = np.abs(current_beta) - np.abs(beta)
        ratio = np.abs(current_beta) / np.abs(beta)
        if output:
            print("tw: " + str(tw))
            print("alpha: " + str(alpha))
            print("beta: " + str(current_beta))
            print("diff: " + str(diff))
        #  could look at real/imag part...
        # for now, will only consider absolute value
        # first step: lower tw
        #
        if diff < 0:
            tw = int(tw * 1.5)
            ratio = 1.01
        tw_flag = True
        while np.abs(diff) / np.abs(beta) > 1e-3:
            if ratio > 1.0 and tw > 0 and tw_flag:
                tw = int(tw / ratio)
            else:
                tw_flag = False
                # if ratio > 1.02:
                #    ratio = 1.02
                # if ratio < 0.98:
                #    ratio = 0.98
                alpha = alpha / ratio

            # update the ratios for the new tw and alpha given chi_prime
            r, r2 = ratios(alpha, tw)
            cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r2)
            epsilon = cavity_dac_pulse * epsilon_m
            current_beta = integrated_beta(epsilon)
            diff = np.abs(current_beta) - np.abs(beta)
            ratio = np.abs(current_beta) / np.abs(beta)
            if output:
                print("tw: " + str(tw))
                print("alpha: " + str(alpha))
                print("beta: " + str(current_beta))
                print("diff: " + str(diff))
    # need to add back in the buffer time to the pulse
    cavity_dac_pulse, qubit_dac_pulse = construct_CD(alpha, tw, r, r2, buf=buffer_time)

    # the final step is kerr correction. Now, the trajectories are solved with kerr, and there is a frame update.
    # This is not yet implemented/tested fully because Kerr correction is not important with Alec's parameters.
    # Can include it when using larger Kerr.
    # Don't trust the below code, it needs to be looked at in more detail. In particular, the rate of local rotation
    # And the rate of center of mass rotation differs by a factor of 2.
    """
    if kerr_correction:
        #here, we want to get the trajectory without kerr!
        alpha_g, alpha_e = get_ge_trajectories(epsilon, chi=chi, chi_prime=chi_prime, kerr=0.0, flip_half_way=True)
        nbar_g = np.abs(alpha_g)**2
        nbar_e = np.abs(alpha_e)**2
        det_g = kerr*nbar_g
        det_e = kerr*nbar_e
        avg_det = (det_g + det_e)/2.0 #note, that the dets should be the same
        accumulated_phase = np.cumsum(avg_det)
        cavity_dac_pulse = cavity_dac_pulse*np.exp(-1j*accumulated_phase)
    else:
        accumulated_phase = np.zeros_like(epsilon)
    """
    if pad:
        while len(cavity_dac_pulse) % 4 != 0:
            cavity_dac_pulse = np.pad(cavity_dac_pulse, (0, 1), mode="constant")
            qubit_dac_pulse = np.pad(qubit_dac_pulse, (0, 1), mode="constant")
            # accumulated_phase = np.pad(accumulated_phase, (0,1), mode='edge')

    return cavity_dac_pulse, qubit_dac_pulse, alpha, tw


def double_circuit(betas, phis, thetas, final_disp=True):
    phis = [phis] if type(phis) is not list else phis
    thetas = [thetas] if type(thetas) is not list else thetas
    betas2 = []
    phis2 = [[] for _ in phis]
    thetas2 = [[] for _ in thetas]
    for i, beta in enumerate(betas):
        if np.abs(beta) > 0 and not (i == len(betas) - 1 and final_disp):
            betas2.extend([beta / 2.0, beta / 2.0])
            for j in range(len(thetas)):
                phis2[j].extend([phis[j][i], 0])
                thetas2[j].extend([thetas[j][i], np.pi])
        else:
            betas2.extend([beta])
            for j in range(len(thetas)):
                phis2[j].extend([phis[j][i]])
                thetas2[j].extend([thetas[j][i]])
    return betas2, phis2, thetas2


# if final disp is true, the final CD will be treated as a displacement of beta/2. NOTE THE IMPORTANT FACTOR OF 2
# Optionally, thetas and phis can be an array of arrays of thetas and phis, to specify circuits with the same cavity
# drives but with different qubit circuits. Useful for tomography!

# Note: thetas and phis can a list of lists, and it will return multiple version of the circuit. This
# is useful if the betas are the same but the thetas/phis are changing.
def conditional_displacement_circuit(
    betas,
    phis,
    thetas,
    storage,
    qubit,
    alpha_CD,
    final_disp=True,
    buffer_time=4,
    curvature_correction=True,
    qubit_phase_correction=True,
    chi_prime_correction=True,
    kerr_correction=False,
    pad=True,
    double_CD=False,
    finite_difference=True,
    output=False,
):
    cavity_dac_pulse = []
    if type(thetas) is not list:
        thetas = [thetas]
    if type(phis) is not list:
        phis = [phis]
    qubit_dac_pulse = [[] for _ in thetas]
    alphas = []
    tws = []
    cd_qubit_phases = []
    analytic_betas = []
    last_beta = 0

    if double_CD:
        betas, phis, thetas = double_circuit(betas, phis, thetas, final_disp=final_disp)

    for i, beta in enumerate(betas):
        if output:
            print(i)
        if (
            np.abs(beta) > 1e-3
        ):  # if it's a disp at the end less than 1e-3, it won't matter anyway. Need to handle the pi pulse in this case...
            if (
                i == len(betas) - 1 and final_disp
            ):  # todo: could put this final displacement at the g frequency...
                dr, di = storage.displace.make_wave(pad=False)
                e_cd = (
                    np.abs(beta / 2.0)
                    * storage.displace.unit_amp
                    * (dr + 1j * di)
                    * np.exp(1j * np.angle(beta))
                )
                o_cd = np.zeros_like(e_cd)
                ap = np.zeros_like(
                    e_cd
                )  # todo: update this... Can accumulate phase on this displacement also...
            elif beta == -1 * last_beta:
                e_cd = -1 * e_cd
            else:
                if (
                    beta != last_beta
                ):  # don't construct the next one if it's the same beta...no need...
                    e_cd, o_cd, alpha, tw = conditional_displacement(
                        beta,
                        alpha=alpha_CD,
                        storage=storage,
                        qubit=qubit,
                        buffer_time=buffer_time,
                        curvature_correction=curvature_correction,
                        chi_prime_correction=chi_prime_correction,
                        kerr_correction=kerr_correction,
                        finite_difference=finite_difference,
                        output=output,
                    )
                alphas.append(alpha)
                tws.append(tw)

                # getting the phase for the phase correction
                analytic_dict = analytic_CD(
                    -1j * 2 * np.pi * 1e-3 * storage.epsilon_m_MHz * e_cd,
                    o_cd,
                    2 * np.pi * 1e-6 * storage.chi_kHz,
                )
                cd_qubit_phases.append(analytic_dict["qubit_phase"])
                analytic_betas.append(analytic_dict["beta"])

        else:
            e_cd, o_cd = np.array([]), np.array([])
        last_beta = beta

        # constructing qubit part
        pr, pi = qubit.pulse.make_wave(pad=False)
        # doing the same thing the FPGA does
        detune = qubit.pulse.detune
        if np.abs(detune) > 0:
            ts = np.arange(len(pr)) * 1e-9
            c_wave = (pr + 1j * pi) * np.exp(-2j * np.pi * ts * detune)
            pr, pi = np.real(c_wave), np.imag(c_wave)
        for j in range(len(thetas)):
            theta = thetas[j][i]
            phi = phis[j][i] - qubit_phase_correction * cd_qubit_phases[-1]
            o_r = (
                qubit.pulse.unit_amp
                * (theta / np.pi)
                * (pr + 1j * pi)
                * np.exp(1j * phi)
            )  # todo: check phase convention. I believe by default it uses the built in mixer.
            qubit_dac_pulse[j].append(o_r)
            if buffer_time > 0 and len(qubit_dac_pulse[0]) > 0:
                qubit_dac_pulse[j].append(np.zeros(buffer_time))
            qubit_dac_pulse[j].append(o_cd)
            if buffer_time > 0 and len(qubit_dac_pulse[0]) > 0:
                qubit_dac_pulse[j].append(np.zeros(buffer_time))

        # constructing cavity part
        cavity_dac_pulse.append(np.zeros(len(o_r)))
        if buffer_time > 0 and len(qubit_dac_pulse[0]) > 0:
            cavity_dac_pulse.append(np.zeros(buffer_time))
        cavity_dac_pulse.append(e_cd)
        if buffer_time > 0 and len(qubit_dac_pulse[0]) > 0:
            cavity_dac_pulse.append(np.zeros(buffer_time))

    cavity_dac_pulse = np.concatenate(cavity_dac_pulse)
    qubit_dac_pulse = [np.concatenate(qp) for qp in qubit_dac_pulse]

    flip_idxs = [
        find_peaks(np.abs(qp), height=np.max(np.abs(qp)) * 0.975)[0]
        for qp in qubit_dac_pulse
    ]

    if kerr_correction:
        print("Kerr correction not implemented yet!")
    accumulated_phase = np.zeros_like(cavity_dac_pulse)

    if pad:
        while len(cavity_dac_pulse) % 4 != 0 and len(cavity_dac_pulse) < 24:
            cavity_dac_pulse = np.pad(cavity_dac_pulse, (0, 1), mode="constant")
            qubit_dac_pulse = [
                np.pad(qp, (0, 1), mode="constant") for qp in qubit_dac_pulse
            ]

    # backwards compatibility:
    qubit_dac_pulse = (
        qubit_dac_pulse[0] if len(qubit_dac_pulse) == 1 else qubit_dac_pulse
    )
    flip_idxs = flip_idxs[0] if len(flip_idxs) == 1 else flip_idxs

    return_dict = {
        "cavity_dac_pulse": cavity_dac_pulse,
        "qubit_dac_pulse": qubit_dac_pulse,
        "accumulated_phase": accumulated_phase,
        "flip_idxs": flip_idxs,
        "alphas": alphas,
        "tws": tws,
        "cd_qubit_phases": cd_qubit_phases,
        "analytic_betas": analytic_betas,
    }
    return return_dict


# uses baptiste's formulas to find the CD and phase
def analytic_CD(epsilon, Omega, chi):
    flip_idxs = get_flip_idxs(Omega)
    pm = +1
    z = []
    for i in range(len(flip_idxs) + 1):
        l_idx = 0 if i == 0 else flip_idxs[i - 1]
        r_idx = len(Omega) if i == len(flip_idxs) else flip_idxs[i]
        z.append(pm * np.ones(r_idx - l_idx))
        pm = -1 * pm
    z = np.concatenate(z)
    phi = -(chi / 2.0) * np.cumsum(z)
    gamma = np.zeros_like(phi, dtype=np.complex64)
    delta = np.zeros_like(gamma)
    for i in range(len(phi)):
        delta[i] = -1 * np.sum(np.sin(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
        gamma[i] = -1j * np.sum(np.cos(phi[: i + 1] - phi[i]) * epsilon[: i + 1])
    theta = -2 * np.cumsum(np.real(np.conj(epsilon) * delta))
    correction = 2 * np.imag(gamma[-1] * delta[-1])
    theta_prime = theta[-1] + correction
    beta = 2 * delta[-1]
    return {
        "z": z,
        "phi": phi,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "correction": correction,
        "theta_prime": theta_prime,
        "qubit_phase": theta_prime,
        "beta": beta,
    }
