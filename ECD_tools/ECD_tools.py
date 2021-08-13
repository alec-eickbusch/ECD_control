import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import eval_laguerre
from ECD_control.ECD_pulse_construction.ECD_pulse_construction import *
import qutip as qt

# pad the pulse then apply a relative delay.
def relative_delay_and_pad(
    cavity_dac_pulse, qubit_dac_pulse, relative_delay=0, pad=None
):
    if pad == None:
        pad = int(np.abs(relative_delay))
    qubit_dac_pulse = np.pad(qubit_dac_pulse, (pad, pad), mode="constant")
    cavity_dac_pulse = np.pad(cavity_dac_pulse, (pad, pad), mode="constant")
    qubit_dac_pulse = np.roll(qubit_dac_pulse, relative_delay)
    while len(cavity_dac_pulse) % 4 != 0:
        cavity_dac_pulse = np.pad(cavity_dac_pulse, (0, 1), mode="constant")
        qubit_dac_pulse = np.pad(qubit_dac_pulse, (0, 1), mode="constant")
    return cavity_dac_pulse, qubit_dac_pulse


# todo: update displacement calibration experiment
def refocus_pulse(storage, tau):

    chi = 2 * np.pi * 1e-6 * storage.chi_kHz
    chop = storage.displace.chop
    sigma = storage.displace.sigma
    t_gate = chop * sigma
    ratio = np.cos((chi / 4.0) * (tau + 3 * t_gate)) / np.cos((chi / 4.0) * t_gate)
    e = np.concatenate(
        [
            disp_gaussian(1.0, sigma=sigma, chop=chop),
            np.zeros(int(tau / 2)),
            disp_gaussian(-1 * ratio, sigma=sigma, chop=chop),
            disp_gaussian(-1 * ratio, sigma=sigma, chop=chop),
            np.zeros(int(tau / 2)),
            disp_gaussian(1.0, sigma=sigma, chop=chop),
        ]
    ).astype(np.complex64)
    return e


# Formula for area around complex curve (done simply here...)
def complex_area(complex_alpha, dt=1):
    ts = np.arange(0, len(complex_alpha)) * dt
    x = np.real(complex_alpha)
    y = np.imag(complex_alpha)
    dx_dt = np.gradient(x, dt)
    integrand = y * dx_dt
    area = np.trapz(integrand, x=ts)
    return area


# returns the phase difference between the e and g trajectories
def geometric_phase(
    epsilon, storage, use_chi_prime=True, use_kerr=True, use_kappa=True, flip_idxs=[]
):
    chi = 2 * np.pi * 1e-6 * storage.chi_kHz
    chi_prime = 2 * np.pi * 1e-9 * storage.chi_prime_Hz if use_chi_prime else 0.0
    Ks = 2 * np.pi * 1e-9 * storage.Ks_Hz if use_kerr else 0.0
    delta = -chi / 2.0
    kappa = 1 / (1e3 * storage.T1_us) if use_kappa else 0.0
    alpha_g, alpha_e = get_ge_trajectories(
        epsilon,
        delta=delta,
        chi=chi,
        chi_prime=chi_prime,
        Ks=Ks,
        kappa=kappa,
        flip_idxs=flip_idxs,
    )
    Ag = complex_area(alpha_e)
    Ae = complex_area(alpha_g)
    phase = Ae - Ag  # is it e - g or g - e?
    return phase


def ideal_geometric_phase_exp(alphas, tau, chi):
    phases = (
        -4 * np.abs(alphas) ** 2 * np.cos(chi * tau / 4.0) * np.sin(chi * tau / 4.0)
    )
    phases = (phases + np.pi) % (2 * np.pi) - np.pi
    return phases


#%% Complex CF to Wigner. For now, assuming square with same # of sample points in x and y, symmetric around zero
def CF2W(CF, betas_I, zero_pad=True, padding=20, betas_Q=None):
    betas_Q = betas_I if betas_Q is None else betas_Q
    if zero_pad:
        dbeta_I = betas_I[1] - betas_I[0]
        new_min_I = betas_I[0] - padding * dbeta_I
        new_max_I = betas_I[0] + padding * dbeta_I
        betas_I = np.pad(
            betas_I,
            (padding, padding),
            mode="linear_ramp",
            end_values=(new_min_I, new_max_I),
        )
        dbeta_Q = betas_Q[1] - betas_Q[0]
        new_min_Q = betas_Q[0] - padding * dbeta_Q
        new_max_Q = betas_Q[0] + padding * dbeta_Q
        betas_Q = np.pad(
            betas_Q,
            (padding, padding),
            mode="linear_ramp",
            end_values=(new_min_Q, new_max_Q),
        )
        CF = np.pad(CF, (padding, padding), mode="constant")
    N_I = len(betas_I)
    dbeta_I = betas_I[1] - betas_I[0]
    N_Q = len(betas_Q)
    dbeta_Q = betas_Q[1] - betas_Q[0]
    W = (dbeta_I * dbeta_Q / np.pi ** 2) * (np.fft.fft2(a=CF)).T
    # recenter
    W = np.fft.fftshift(W)
    # todo...check this N_I vs N_Q
    for j in range(N_I):
        for k in range(N_Q):
            W[j, k] = (
                np.exp(1j * np.pi * (j + k)) * W[j, k]
            )  # todo: single matrix multiply
    alpha0_Q = np.pi / (2 * dbeta_I)

    alpha0_I = np.pi / (2 * dbeta_Q)
    alphas_Q = np.linspace(-alpha0_Q, alpha0_Q, N_Q)
    alphas_I = np.linspace(
        -alpha0_I, alpha0_I, N_I
    )  # is it true that it ends at alpha0? or alpha0-dalpha?
    return W, alphas_I, alphas_Q


# here assuming dt = 1
# Will assume here that you are driving halfway between g and e
# Can change that by including some detuning, delta.
def simulate_master_equation(
    epsilon,
    rho0,
    delta=0,
    chi=2 * np.pi * 1e-6 * 28,
    chi_prime=2 * np.pi * 1e-9 * 0,
    Ks=2 * np.pi * 1e-9 * 0,
    kappa=0,
    Kq=-2 * np.pi * 1e-3 * 2 * 192,
    Omega=None,
    expect=None,
    stark_shift=True,
    dispersive_term=True,
    gamma_down_qubit=0,
    gamma_up_qubit=0,
    gamma_phi_qubit=0,
    kappa_up=0,
    output=True,
    alpha=None,
    qubit_detune=0,
    finite_difference=True,
    kappa_cop=True,
    epsilon_amplitude_multiplier=1.0,
    Omega_amplitude_multiplier=1.0,
):
    epsilon = epsilon_amplitude_multiplier * epsilon
    Omega = Omega_amplitude_multiplier * Omega
    # alpha will include classical loss.
    if (
        alpha is None
    ):  # optinally can supply alpha, for instance, if you want to reuse it.
        if output:
            print("Solving for alpha(t)")
        if np.max(np.abs(epsilon)) < 1e-9:
            alpha = np.zeros_like(epsilon)
        else:
            if finite_difference:
                alpha = alpha_from_epsilon_nonlinear_finite_difference(
                    epsilon, delta=delta, Ks=Ks, kappa=kappa, alpha_init=0 + 0j
                )
            else:
                alpha = alpha_from_epsilon_nonlinear(
                    epsilon, delta=delta, Ks=Ks, kappa=kappa, alpha_init=0 + 0j
                )

    if output:
        print("Constructing Hamiltonian")
    N = rho0.dims[0][1]
    N2 = rho0.dims[0][0]

    a = qt.tensor(qt.identity(N2), qt.destroy(N))
    q = qt.tensor(qt.destroy(N2), qt.identity(N))

    d = +1 if dispersive_term else 0
    H0 = (
        d * delta * a.dag() * a
        + d * chi * a.dag() * a * q.dag() * q
        + chi_prime * a.dag() ** 2 * a ** 2 * q.dag() * q
        + Ks * a.dag() ** 2 * a ** 2
        + Kq * q.dag() ** 2 * q ** 2
        + qubit_detune * q.dag() * q
    )
    # alpha and alpha* control
    H_alpha = (
        chi * a.dag() * q.dag() * q
        + 2 * Ks * a.dag() ** 2 * a
        + 2 * chi_prime * a.dag() ** 2 * a * q.dag() * q
    )
    # alpha^2 and alpha*^2 control
    H_alpha_sq = Ks * a.dag() ** 2 + chi_prime * a.dag() ** 2 * q.dag() * q
    # |alpha|^2 control
    s = +1 if stark_shift else 0
    H_abs_alpha_sq = (
        4 * chi_prime * a.dag() * a * q.dag() * q
        + s * chi * q.dag() * q
        + 4 * Ks * a.dag() * a
    )
    # |alpha|^2*alpha control
    H_abs_alpha_sq_alpha = 2 * chi_prime * a.dag() * q.dag() * q
    # |alpha|^4 control
    H_abs_alpha_4 = chi_prime * q.dag() * q
    # Omega control
    H_Omega = q.dag()
    # Omega* control
    H_Omega_star = q

    ts = np.arange(0, len(epsilon))
    alpha_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], alpha)
    alpha_sq_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], alpha ** 2)
    abs_alpha_sq_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], np.abs(alpha) ** 2)
    abs_alpha_sq_alpha_spline = qt.interpolate.Cubic_Spline(
        ts[0], ts[-1], np.abs(alpha) ** 2 * alpha
    )
    abs_alpha_4_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], np.abs(alpha) ** 4)
    if Omega is not None:
        Omega_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], Omega)
        Omega_star_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], np.conj(Omega))

    H = [
        H0,
        [H_alpha, alpha_spline],
        [H_alpha.dag(), lambda t, args: np.conj(alpha_spline(t, *args))],
    ]
    if chi_prime != 0 or Ks != 0 or stark_shift != 0:
        H += [[H_abs_alpha_sq, abs_alpha_sq_spline]]
    if chi_prime != 0 or Ks != 0:
        H += [
            [H_alpha_sq, alpha_sq_spline],
            [H_alpha_sq.dag(), lambda t, args: np.conj(alpha_sq_spline(t, *args))],
            [H_abs_alpha_sq_alpha, abs_alpha_sq_alpha_spline],
            [
                H_abs_alpha_sq_alpha.dag(),
                lambda t, args: np.conj(abs_alpha_sq_alpha_spline(t, *args)),
            ],
            [H_abs_alpha_4, abs_alpha_4_spline],
        ]
    if Omega is not None:
        H.extend([[H_Omega, Omega_spline], [H_Omega_star, Omega_star_spline]])

    loss_ops = []
    if gamma_down_qubit > 0:
        loss_ops.append(np.sqrt(gamma_down_qubit) * q)
    if gamma_up_qubit > 0:
        loss_ops.append(np.sqrt(gamma_up_qubit) * q.dag())
    if gamma_phi_qubit > 0:
        loss_ops.append(np.sqrt(gamma_phi_qubit) * q.dag() * q)
    if kappa > 0 and kappa_cop:
        loss_ops.append(np.sqrt(kappa) * a)
    if kappa_up > 0:
        loss_ops.append(np.sqrt(kappa_up) * a.dag())
    if kappa_phi > 0:
        loss_ops.append(np.sqrt(kappa_phi) * a.dag() * a)

    if output:
        print("Running mesolve:")
    progress_bar = True if output else None
    result = qt.mesolve(H, rho0, ts, loss_ops, expect, progress_bar=progress_bar)

    return result, alpha


# Here, I will simulate directly with a large hilbert space, no frame transformation.
# Will have all the same inputs as the above method, but just compute differently
def simulate_direct(
    epsilon,
    rho0,
    delta=0,
    chi=2 * np.pi * 1e-6 * 28,
    chi_prime=2 * np.pi * 1e-9 * 0,
    Ks=2 * np.pi * 1e-9 * 0,
    kappa=0,
    Kq=-2 * np.pi * 1e-3 * 2 * 192,
    Omega=None,
    expect=None,
    stark_shift=True,
    dispersive_term=True,
    gamma_down_qubit=0,
    gamma_up_qubit=0,
    gamma_phi_qubit=0,
    kappa_up=0,
    output=True,
    qubit_detune=0,
    kappa_phi=0,
):
    # alpha will include classical loss.
    if output:
        print("Constructing Hamiltonian")
    N = rho0.dims[0][1]
    N2 = rho0.dims[0][0]

    a = qt.tensor(qt.identity(N2), qt.destroy(N))
    q = qt.tensor(qt.destroy(N2), qt.identity(N))

    d = +1 if dispersive_term else 0
    H0 = (
        d * delta * a.dag() * a
        + d * chi * a.dag() * a * q.dag() * q
        + chi_prime * a.dag() ** 2 * a ** 2 * q.dag() * q
        + Ks * a.dag() ** 2 * a ** 2
        + Kq * q.dag() ** 2 * q ** 2
        + qubit_detune * q.dag() * q
    )
    # epsilon and epsilon_* control
    H_epsilon = a
    # Omega control
    H_Omega = q
    ts = np.arange(0, len(epsilon))
    epsilon_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], epsilon)
    if Omega is not None:
        Omega_spline = qt.interpolate.Cubic_Spline(ts[0], ts[-1], Omega)

    H = [
        H0,
        [H_epsilon, epsilon_spline],
        [H_epsilon.dag(), lambda t, args: np.conj(epsilon_spline(t, *args))],
    ]
    if Omega is not None:
        H.extend(
            [
                [H_Omega, Omega_spline],
                [H_Omega.dag(), lambda t, args: np.conj(Omega_spline(t, *args))],
            ]
        )

    loss_ops = []
    if gamma_down_qubit > 0:
        loss_ops.append(np.sqrt(gamma_down_qubit) * q)
    if gamma_up_qubit > 0:
        loss_ops.append(np.sqrt(gamma_up_qubit) * q.dag())
    if gamma_phi_qubit > 0:
        loss_ops.append(np.sqrt(gamma_phi_qubit) * q.dag() * q)
    if kappa > 0:
        loss_ops.append(np.sqrt(kappa) * a)
    if kappa_up > 0:
        loss_ops.append(np.sqrt(kappa_up) * a.dag())
    if kappa_phi > 0:
        loss_ops.append(np.sqrt(kappa_phi) * a.dag() * a)

    if output:
        print("Running mesolve:")
    progress_bar = True if output else None
    result = qt.mesolve(H, rho0, ts, loss_ops, expect, progress_bar=progress_bar)

    return result


def wigner(rho, xvec, yvec=None):
    if yvec is None:
        yvec = xvec
    return (np.pi / 2.0) * qt.wigner(rho, xvec, yvec, g=2)


# todo: can make this faster...
# Use symmetry
# and use diagonalizaed construction of displacement ops
def characteristic_function(rho, xvec, yvec=None):
    yvec = xvec if yvec is None else yvec
    N = rho.dims[0][0]
    a = qt.destroy(N)

    X, Y = np.meshgrid(xvec, yvec)

    def CF(beta):
        return qt.expect((beta * a.dag() - np.conj(beta) * a).expm(), rho)

    CF = np.vectorize(CF)
    Z = CF(X + 1j * Y)
    return Z


# note: for now, only working with pure states
# todo: can easily extend to rho.
def characteristic_function_tf(psi, betas):
    from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

    N = psi.dims[0][0]
    # dummy opt object for calculation
    params = {"optimization_type": "calculation", "N_cav": N}
    opt = BatchOptimizer(**params)
    return opt.characteristic_function(psi=psi, betas=betas)


def characteristic_function_rho_tf(rho, betas):
    from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

    N = rho.dims[0][0]
    # dummy opt object for calculation
    params = {"optimization_type": "calculation", "N_cav": N}
    opt = BatchOptimizer(**params)
    return opt.characteristic_function_rho(rho=rho, betas=betas)


# here, cf is assumed to be a 2d complex function.
# betas is a 1D vector, we will assume for now that cf square.
# later, I could instead provide betas via a 2d list of points correspoinding
# to any cf and it should still work.
def fit_characteristic_function_to_model(
    cf, betas_x, model_function, p0, betas_y=None, bounds=None, gtol=1e-6
):
    betas_y = betas_x if betas_y is None else betas_y
    betas_real, betas_imag = np.meshgrid(betas_x, betas_y)
    betas_complex = betas_real + 1j * betas_imag

    def cost_function(p0):
        cf_predicted = model_function(betas_complex, *p0)
        abs_diff_sq = np.abs(cf - cf_predicted) ** 2
        return np.sum(abs_diff_sq)

    result = minimize(cost_function, p0, bounds=bounds, options={"gtol": gtol})
    return result.x


def fit_characteristic_function_cuts_to_model(
    cf_cut_q, cf_cut_p, betas, model_function, p0, gtol=1e-6, bounds=None
):
    betas_concat = np.concatenate([betas, 1j * betas])
    cf_concat = np.concatenate([cf_cut_q, cf_cut_p])

    def cost_function(p0):
        cf_concat_predicted = model_function(betas_concat, *p0)
        abs_diff_sq = np.abs(cf_concat - cf_concat_predicted) ** 2
        return np.sum(abs_diff_sq)

    result = minimize(cost_function, p0, bounds=bounds, options={"gtol": gtol})
    return result.x


def fit_complex_1d_function(f, xs, model_function, p0, gtol=1e-6, bounds=None):
    def cost_function(p0):
        f_predicted = model_function(xs, *p0)
        abs_diff_sq = np.abs(f - f_predicted) ** 2
        return np.sum(abs_diff_sq)

    result = minimize(cost_function, p0, bounds=bounds, options={"gtol": gtol})
    return result.x


def complex_exp_decay_detuned(t, amp, f0, tau, phase, offset=0):
    # recall positive freq in H means negative phase accumulation (clockwise)
    return (
        amp * np.exp(-1j * (2 * np.pi * f0 * t + phase)) * np.exp(-1 * t / tau) + offset
    )


# some model characteristic functions
# They should all be in the form
# model_cf(beta, *real_p0)
def coherent_state_cf(beta, alpha_real, alpha_imag):
    alpha = alpha_real + 1j * alpha_imag
    return np.exp(-np.abs(beta) ** 2 / 2.0) * np.exp(
        beta * np.conj(alpha) - np.conj(beta) * alpha
    )


def thermal_cf(beta, n_th):
    return np.exp(-(n_th + 1 / 2.0) * np.abs(beta) ** 2)


def fock_cf(beta, fock):
    return np.exp(-np.abs(beta) ** 2 / 2.0) * eval_laguerre(fock, np.abs(beta) ** 2)


def displaced_fock_cf(beta, alpha_real, alpha_imag, fock):
    alpha = alpha_real + 1j * alpha_imag
    return np.exp(beta * np.conj(alpha) - np.conj(beta) * alpha) * fock_cf(beta, fock)


def displaced_thermal_cf(beta, alpha_real, alpha_imag, n_th):
    alpha = alpha_real + 1j * alpha_imag
    return np.exp(-(n_th + 1 / 2.0) * np.abs(beta) ** 2) * np.exp(
        beta * np.conj(alpha) - np.conj(beta) * alpha
    )


def displaced_model_cf(model_func, beta, alpha_real, alpha_imag, p0_model):
    alpha = alpha_real + 1j * alpha_imag
    cf = model_func(beta, p0_model)
    return np.exp(beta * np.conj(alpha) - np.conj(beta) * alpha) * cf


def squeezed_vacuum(beta, r):
    return np.exp(-0.5 * np.abs(beta * np.cosh(r) + np.conj(beta) * np.sinh(r)) ** 2)


def displaced_rotated_squeezed_vacuum(beta, r, theta, alpha_real, alpha_imag):
    alpha = alpha_real + 1j * alpha_imag
    return np.exp(
        -0.5
        * np.abs(beta * np.cosh(r) + np.exp(1j * theta) * np.conj(beta) * np.sinh(r))
        ** 2
    ) * np.exp(beta * np.conj(alpha) - np.conj(beta) * alpha)


def wigner_displaced_squeezed_vacuum(alpha, r, beta_real, beta_imag):
    exp = (
        -2 * np.exp(2 * r) * (np.real(alpha) - beta_real) ** 2
        - 2 * np.exp(-2 * r) * (np.imag(alpha) - beta_imag) ** 2
    )
    return (2 / np.pi) * np.exp(exp)


# The unitary gates as used for qutip
def D(alpha, N_cav):
    a = qt.tensor(qt.identity(2), qt.destroy(N_cav))
    q = qt.tensor(qt.destroy(2), qt.identity(N_cav))
    if np.abs(alpha) == 0:
        return qt.tensor(qt.identity(2), qt.identity(N_cav))
    return (alpha * a.dag() - np.conj(alpha) * a).expm()


def R(phi, theta, N_cav):
    sx = qt.tensor(qt.sigmax(), qt.identity(N_cav))
    sy = qt.tensor(qt.sigmay(), qt.identity(N_cav))
    if theta == 0:
        return qt.tensor(qt.identity(2), qt.identity(N_cav))
    # return (-1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()
    return np.cos(theta / 2.0) - 1j * (np.cos(phi) * sx + np.sin(phi) * sy) * np.sin(
        theta / 2.0
    )


def CD(beta, N_cav):
    if np.abs(beta) == 0:
        return R(0, np.pi, N_cav)
    # return self.R(0,np.pi)*((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()
    # temp removing pi pulse from CD for analytic opt testing
    # return ((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()
    zz = qt.tensor(qt.ket2dm(qt.basis(2, 0)), qt.identity(N_cav))
    oo = qt.tensor(qt.ket2dm(qt.basis(2, 1)), qt.identity(N_cav))
    return R(0, np.pi, N_cav) * (D(beta / 2.0, N_cav) * zz + D(-beta / 2.0, N_cav) * oo)
    # includes pi rotation

    # alec 10/8/2020: Removing pi rotation to compare with tensorflow version.
    # return self.D(beta / 2.0) * zz + self.D(-beta / 2.0) * oo


def U_block(beta, phi, theta, N_cav):
    return CD(beta, N_cav) * R(phi, theta, N_cav)


def U_circuit(betas, phis, thetas, N_cav):
    U = qt.tensor(qt.identity(2), qt.identity(N_cav))
    for beta, phi, theta in zip(betas, phis, thetas):
        U = U_block(beta, phi, theta, N_cav) * U
    return U


def unitary_circuit_sim(psi0, betas, phis, thetas, N_cav):
    psis = [psi0]
    for beta, phi, theta in zip(betas, phis, thetas):
        psi = U_block(beta, phi, theta, N_cav) * psis[-1]
        psis.append(psi)
    return psis


def circuit_sim_density_matrix(rho0, betas, phis, thetas, N_cav):
    rhos = [rho0]
    for beta, phi, theta in zip(betas, phis, thetas):
        U = U_block(beta, phi, theta, N_cav)
        rho = U * rhos[-1] * U.dag()
        rhos.append(rho)
    return rhos


# The situation where we turn off the cavity drive
# (stil have qubit pi pulses during CD)
def unitary_circuit_sim_qubit_only(psi0, betas, phis, thetas, N_cav):
    betas = np.zeros_like(betas)
    return unitary_circuit_sim(psi0, betas, phis, thetas, N_cav)


# The situation where we turn off the Rs
# (stil have qubit pi pulses during CD)
def unitary_circuit_sim_CD_only(psi0, betas, phis, thetas, N_cav):
    thetas = np.zeros_like(thetas)
    return unitary_circuit_sim(psi0, betas, phis, thetas, N_cav)


def tensor_fidelities(rho, rho_target, full=True):
    f_storage = (qt.fidelity(rho_target.ptrace(1), rho.ptrace(1))) ** 2
    f_q = (qt.fidelity(rho_target.ptrace(0), rho.ptrace(0))) ** 2
    f_tot = (qt.fidelity(rho_target, rho)) ** 2
    print("storage state squared fidelity: %.6f" % f_storage)
    print("qubit state squared fidelity: %.6f" % f_q)
    print("total state squared fidelity: %.6f" % f_tot)
    return {"f_storage": f_storage, "f_qubit": f_q, "f_total": f_tot}


def norm_and_purities(rho):
    norm = rho.norm()
    purity = np.real((rho ** 2).tr())
    print("norm: %.6f" % norm)
    print("purity: %.6f" % purity)


def tensor_norm_and_purities(rho):
    rhos = rho.ptrace(1)
    rhoq = rho.ptrace(0)
    norm = rho.norm()
    purity = np.real((rho ** 2).tr())
    puritys = np.real((rhos ** 2).tr())
    purityq = np.real((rhoq ** 2).tr())
    print("norm: %.6f" % norm)
    print("purity: %.6f" % purity)
    print("purity storage: %.6f" % puritys)
    print("purity qubit: %.6f" % purityq)


def expect(states):
    rho0 = states[0]
    N = rho0.dims[0][1]
    N2 = rho0.dims[0][0]

    a = qt.tensor(qt.identity(N2), qt.destroy(N))
    q = qt.tensor(qt.destroy(N2), qt.identity(N))

    sx = q + q.dag()
    sy = 1j * (q.dag() - q)
    sz = 1 - 2 * q.dag() * q

    n = a.dag() * a
    I = (a + a.dag()) / 2.0
    Q = 1j * (a.dag() - a) / 2.0

    sx_expect = np.array([qt.expect(sx, rho) for rho in states])
    sy_expect = np.array([qt.expect(sy, rho) for rho in states])
    sz_expect = np.array([qt.expect(sz, rho) for rho in states])

    n_expect = np.array([qt.expect(n, rho) for rho in states])
    I_expect = np.array([qt.expect(I, rho) for rho in states])
    Q_expect = np.array([qt.expect(Q, rho) for rho in states])

    n_sq_expect = np.array([qt.expect(n * n, rho) for rho in states])
    I_sq_expect = np.array([qt.expect(I * I, rho) for rho in states])
    Q_sq_expect = np.array([qt.expect(Q * Q, rho) for rho in states])

    return {
        "sx": sx_expect,
        "sy": sy_expect,
        "sz": sz_expect,
        "n": n_expect,
        "n_sq": n_sq_expect,
        "I": I_expect,
        "I_sq": I_sq_expect,
        "Q": Q_expect,
        "Q_sq": Q_sq_expect,
    }


def expect_displaced(states, alphas):
    rho0 = states[0]
    N = rho0.dims[0][1]
    N2 = rho0.dims[0][0]

    a = qt.tensor(qt.identity(N2), qt.destroy(N))
    q = qt.tensor(qt.destroy(N2), qt.identity(N))

    sx = q + q.dag()
    sy = 1j * (q.dag() - q)
    sz = 1 - 2 * q.dag() * q

    n = a.dag() * a
    I = (a + a.dag()) / 2.0
    Q = 1j * (a.dag() - a) / 2.0

    sx_expect = np.array([qt.expect(sx, rho) for rho in states])
    sy_expect = np.array([qt.expect(sy, rho) for rho in states])
    sz_expect = np.array([qt.expect(sz, rho) for rho in states])

    n_expect = np.array(
        [
            qt.expect(n + alpha * a.dag() + np.conj(alpha) * a, rho)
            + np.abs(alpha) ** 2
            for rho, alpha in zip(states, alphas)
        ]
    )
    I_expect = np.array(
        [qt.expect(I, rho) + np.real(alpha) for rho, alpha in zip(states, alphas)]
    )
    Q_expect = np.array(
        [qt.expect(Q, rho) + np.imag(alpha) for rho, alpha in zip(states, alphas)]
    )

    n_sq_expect = np.array(
        [
            qt.expect(
                (n + alpha * a.dag() + np.conj(alpha) * a + np.abs(alpha) ** 2) ** 2,
                rho,
            )
            for rho, alpha in zip(states, alphas)
        ]
    )

    return {
        "sx": sx_expect,
        "sy": sy_expect,
        "sz": sz_expect,
        "n": n_expect,
        "n_sq": n_sq_expect,
        "I": I_expect,
        "Q": Q_expect,
    }


def plot_expect(states):
    expects = expect(states)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))
    plot1 = ["sx", "sy", "sz"]
    plot2 = ["n", "I", "Q"]
    for n, p in enumerate([plot1, plot2]):
        for name in p:
            e = expects[name]
            axs[n].plot(e, "-", label=name)
        axs[n].grid()
        axs[n].legend(frameon=False)


def fft(complex_pulse, dt=1):
    df = 1 / dt  # sampling frequency in GHz
    N = len(complex_pulse)
    freqs = np.fft.fftfreq(N) * df
    fourier = np.fft.fft(complex_pulse)
    return freqs, fourier


def real_fft(real_pulse, dt=1):
    df = 1 / dt  # sampling frequency in GHz
    N = len(real_pulse)
    freqs = np.fft.rfftfreq(N) * df
    fourier = np.fft.rfft(real_pulse)
    return freqs, fourier


def plot_fft(pulse, max_f_MHz=20, dt=1):
    freqs, fourier = fft(pulse, dt)
    plt.figure(figsize=(8, 6))
    df = 1 / (dt * len(pulse))  # bin spacing of the fft
    plt.bar(
        freqs * 1e3,
        np.abs(fourier),
        align="center",
        width=0.8 * 1e3 * df,
        label="abs(fft)",
    )
    plt.xlim([-max_f_MHz, max_f_MHz])
    # plt.plot(freqs * 1e3,np.real(fourier), label='real')
    # plt.plot(freqs * 1e3,np.imag(fourier), label='imag')
    plt.xlabel("freq MHz")
    plt.legend()
    plt.show()


def plot_cf(
    xvec_data,
    data_cf,
    yvec_data=None,
    sample_betas=None,
    v=1.0,
    title="",
    grid=True,
    bwr=False,
):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 6)
    )
    cmap = "bwr" if bwr else "seismic"
    axs[0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap=cmap,
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    if sample_betas is not None:
        axs[0].scatter(
            np.real(sample_betas), np.imag(sample_betas), marker="x", color="black"
        )
    if grid:
        axs[0].grid()
    axs[0].set_xlabel("Re(beta)")
    axs[0].set_ylabel("Im(beta)")
    axs[0].set_title("Real")
    axs[0].set_axisbelow(True)
    axs[1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap=cmap,
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    if grid:
        axs[1].grid()
    axs[1].set_xlabel("Re(beta)")
    axs[1].set_ylabel("Im(beta)")
    axs[1].set_title("Imag")
    axs[1].set_axisbelow(True)

    fig.suptitle(title)
    fig.tight_layout()


# for now, only for real part.
def plot_cf_sampled(
    sample_betas, C_vals, beta_extent_real=[-5, 5], beta_extent_imag=[-5, 5], v=1.0,
):
    dummy_xvec = np.linspace(beta_extent_real[0], beta_extent_real[1], 11)
    dummy_yvec = np.linspace(beta_extent_real[0], beta_extent_real[1], 11)
    dummy_data = np.zeros(shape=(len(dummy_xvec), len(dummy_yvec)))

    dx = dummy_xvec[1] - dummy_xvec[0]
    dy = dummy_yvec[1] - dummy_yvec[0]
    extent = (
        dummy_xvec[0] - dx / 2.0,
        dummy_xvec[-1] + dx / 2.0,
        dummy_xvec[0] - dy / 2.0,
        dummy_xvec[-1] + dy / 2.0,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(5, 5))
    ax.imshow(
        np.real(dummy_data),
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    ax.scatter(
        np.real(sample_betas),
        np.imag(sample_betas),
        marker="o",
        s=20,
        c=C_vals,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
    )


def plot_cf_and_marginals(xvec_data, data_cf, yvec_data=None, v=1.0, title=""):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10, 14)
    )
    axs[0, 0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 0].grid()
    axs[0, 0].set_xlabel("Re(beta)")
    axs[0, 0].set_ylabel("Im(beta)")
    axs[0, 0].set_title("Real")
    axs[0, 1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 1].grid()
    axs[0, 1].set_xlabel("Re(beta)")
    axs[0, 1].set_ylabel("Im(beta)")
    axs[0, 1].set_title("Imag")

    mid_idx_data_I = int(len(xvec_data) / 2)
    mid_idx_data_Q = int(len(yvec_data) / 2)

    axs[1, 0].plot(
        xvec_data, np.real(data_cf[mid_idx_data_Q, :]), ".", color="blue", label="re"
    )
    axs[1, 0].plot(
        xvec_data, np.imag(data_cf[mid_idx_data_Q, :]), ".", color="red", label="im"
    )
    axs[1, 0].grid()
    axs[1, 0].legend(frameon=False)
    axs[1, 0].set_xlabel("Re(beta)")
    axs[1, 0].set_ylabel("CF")
    axs[1, 0].set_title("q cut")
    axs[1, 1].plot(
        yvec_data, np.real(data_cf[:, mid_idx_data_I]), ".", color="blue", label="re"
    )
    axs[1, 1].plot(
        yvec_data, np.imag(data_cf[:, mid_idx_data_I]), ".", color="red", label="im"
    )
    axs[1, 1].grid()
    axs[1, 1].legend(frameon=False)
    axs[1, 1].set_xlabel("Im(beta)")
    axs[1, 1].set_ylabel("CF")
    axs[1, 1].set_title("p cut")
    fig.suptitle(title)
    # fig.tight_layout()


def plot_data_and_model_cf(
    xvec_data,
    data_cf,
    xvec_model,
    model_cf,
    yvec_data=None,
    yvec_model=None,
    residuals=False,
    residual_multiplier=10,
    v=1.0,
    title="",
):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    dx_model = xvec_model[1] - xvec_model[0]
    yvec_model = xvec_model if yvec_model is None else yvec_model
    dy_model = yvec_model[1] - yvec_model[0]
    extent_model = (
        xvec_model[0] - dx_model / 2.0,
        xvec_model[-1] + dx_model / 2.0,
        yvec_model[0] - dy_model / 2.0,
        yvec_model[-1] + dy_model / 2.0,
    )
    mid_idx_data_I = int(len(xvec_data) / 2)
    mid_idx_model_I = int(len(yvec_model) / 2)
    mid_idx_data_Q = int(len(xvec_data) / 2)
    mid_idx_model_Q = int(len(yvec_model) / 2)
    nrows = 3 if residuals else 2
    ysize = 14 if residuals else 10
    fig, axs = plt.subplots(
        nrows=nrows, ncols=3, sharex=False, sharey=False, figsize=(16, ysize)
    )
    axs[0, 0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 0].grid()
    axs[0, 0].set_xlabel("Re(beta)")
    axs[0, 0].set_ylabel("Im(beta)")
    axs[0, 0].set_title("Real data")
    axs[0, 1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 1].grid()
    axs[0, 1].set_xlabel("Re(beta)")
    axs[0, 1].set_ylabel("Im(beta)")
    axs[0, 1].set_title("Imag data")
    axs[1, 0].imshow(
        np.real(model_cf),
        origin="lower",
        extent=extent_model,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[1, 0].grid()
    axs[1, 0].set_xlabel("Re(beta)")
    axs[1, 0].set_ylabel("Im(beta)")
    axs[1, 0].set_title("Real model")
    axs[1, 1].imshow(
        np.imag(model_cf),
        origin="lower",
        extent=extent_model,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("Re(beta)")
    axs[1, 1].set_ylabel("Im(beta)")
    axs[1, 1].set_title("Imag model")
    axs[0, 2].plot(
        xvec_data, np.real(data_cf[mid_idx_data_Q, :]), ".", color="blue", label="re"
    )
    axs[0, 2].plot(xvec_model, np.real(model_cf[mid_idx_model_Q, :]), color="blue")
    axs[0, 2].plot(
        xvec_data, np.imag(data_cf[mid_idx_data_Q, :]), ".", color="red", label="im"
    )
    axs[0, 2].plot(xvec_model, np.imag(model_cf[mid_idx_model_Q, :]), color="red")
    axs[0, 2].grid()
    axs[0, 2].legend(frameon=False)
    axs[0, 2].set_xlabel("Re(beta)")
    axs[0, 2].set_ylabel("CF")
    axs[0, 2].set_title("q cut")
    axs[1, 2].plot(
        yvec_data, np.real(data_cf[:, mid_idx_data_I]), ".", color="blue", label="re"
    )
    axs[1, 2].plot(yvec_model, np.real(model_cf[:, mid_idx_model_I]), color="blue")
    axs[1, 2].plot(
        yvec_data, np.imag(data_cf[:, mid_idx_data_I]), ".", color="red", label="im"
    )
    axs[1, 2].plot(yvec_model, np.imag(model_cf[:, mid_idx_model_I]), color="red")
    axs[1, 2].grid()
    axs[1, 2].legend(frameon=False)
    axs[1, 2].set_xlabel("Im(beta)")
    axs[1, 2].set_ylabel("CF")
    axs[1, 2].set_title("p cut")
    # to plot residuals, the data and model must be sampled the same
    if residuals:
        real_residual = np.real(data_cf) - np.real(model_cf)
        imag_residual = np.imag(data_cf) - np.imag(model_cf)
        axs[2, 0].imshow(
            residual_multiplier * real_residual,
            origin="lower",
            extent=extent_data,
            cmap="seismic",
            vmin=-v,
            vmax=+v,
            interpolation=None,
        )
        axs[2, 0].grid()
        axs[2, 0].set_xlabel("Re(beta)")
        axs[2, 0].set_ylabel("Im(beta)")
        axs[2, 0].set_title("Real residual %dx" % residual_multiplier)
        axs[2, 1].imshow(
            residual_multiplier * imag_residual,
            origin="lower",
            extent=extent_data,
            cmap="seismic",
            vmin=-v,
            vmax=+v,
            interpolation=None,
        )
        axs[2, 1].grid()
        axs[2, 1].set_xlabel("Re(beta)")
        axs[2, 1].set_ylabel("Im(beta)")
        axs[2, 1].set_title("Imag residual %dx" % residual_multiplier)
    fig.suptitle(title)
    fig.tight_layout()


# todo: bring in function from CD control analysis that was a nice pulse plotter
def plot_pulse_basic(cavity_pulse, qubit_pulse, title=""):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.plot(np.real(cavity_pulse))
    plt.plot(np.imag(cavity_pulse))
    plt.plot(np.real(qubit_pulse))
    plt.plot(np.imag(qubit_pulse))


def nonlinear_dac(complex_signal, b=0, c=0, d=0):
    theta = np.angle(complex_signal)
    mag_in = np.abs(complex_signal)
    mag_out = mag_in + b * mag_in ** 2 + c * mag_in ** 3 + d * mag_in ** 4
    return mag_out * np.exp(1j * theta)


def wigner_marginal(w_data, xvec, yvec=None, I=True, normalize=True):
    if normalize:
        print("normalizing marginal.")
    yvec = xvec if yvec is None else yvec
    dy = yvec[1] - yvec[0]
    dx = xvec[1] - xvec[0]
    if I:
        # integrate along y
        w_marginal = np.sum(w_data, axis=0) * dy
        norm = np.sum(w_marginal) * dx if normalize else 1.0
    else:
        # integrate along x
        w_marginal = np.sum(w_data, axis=1) * dx
        norm = np.sum(w_marginal) * dy if normalize else 1.0
    return w_marginal / norm


def plot_wigner(psi, xvec=np.linspace(-5, 5, 41), ax=None, grid=True, invert=False):
    W = wigner(psi, xvec)
    s = -1 if invert else +1
    plot_wigner_data(s * W, xvec, ax=ax, grid=grid)


def plot_wigner_data(
    W,
    xvec=np.linspace(-5, 5, 41),
    ax=None,
    grid=True,
    yvec=None,
    cut=0,
    vmin=-1,
    vmax=+1,
):
    yvec = xvec if yvec is None else yvec
    if cut > 0:
        xvec = xvec[cut:-cut]
        yvec = yvec[cut:-cut]
        W = W[cut:-cut, cut:-cut]
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    extent = (
        xvec[0] - dx / 2.0,
        xvec[-1] + dx / 2.0,
        yvec[0] - dy / 2.0,
        yvec[-1] + dy / 2.0,
    )
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(
        W,
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation=None,
    )
    # plt.colorbar()
    if grid:
        ax.grid()


def plot_wigner_data_marginals(w_data, xvec, yvec=None, grid=True, norms=True, cut=0):
    yvec = xvec if yvec is None else yvec
    if cut > 0:
        xvec = xvec[cut:-cut]
        yvec = yvec[cut:-cut]
        w_data = w_data[cut:-cut, cut:-cut]
    w_marginal_q = wigner_marginal(w_data, xvec, yvec)
    w_marginal_p = wigner_marginal(w_data, xvec, yvec, I=False)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xvec, w_marginal_q, ".")
    axs[1].plot(yvec, w_marginal_p, ".")
    if grid:
        axs[0].grid()
        axs[1].grid()
    q_title = "P(q)"
    p_title = "P(p)"
    if norms:
        dx = xvec[1] - xvec[0]
        dy = yvec[1] - yvec[0]
        norm_q = np.sum(w_marginal_q) * dx
        norm_p = np.sum(w_marginal_p) * dy
        q_title += " norm: %.3f" % norm_q
        p_title += " norm: %.3f" % norm_p
    axs[0].set_title(q_title)
    axs[1].set_title(p_title)


# note: could steal target_C_vals from the sampling.
def fidelity_from_sampled_CF(
    betas, sampled_C_real, C_target_values,
):
    # using batch optimizer to quickly calculate. It does the pre-diagonalization...

    betas = np.array(betas)
    sampled_C_real = np.array(sampled_C_real)
    C_target_values = np.array(C_target_values)
    norm = np.pi

    overlap = (
        norm
        * (1.0 / np.pi)
        * (1.0 / float(len(betas)))
        * np.sum(sampled_C_real / C_target_values)
    )
    return overlap


# Note: currently hard coded with Alec's parameters.
# todo: re-write for general case.
def process_tomo(cf_half, pulse_scale_I, pulse_scale_Q):
    x, y = np.meshgrid(pulse_scale_I, pulse_scale_Q)
    pulse_scale_complex = x + 1j * y
    # phase correction
    a = 0.062623780865
    cf = np.exp(1j * a * np.abs(pulse_scale_complex) ** 2) * cf_half
    """ could use the following for amplitude correction, but it was not something I measured, so I don't
    think it's so valid.
    #amplitude correction
    x_full, y_full = np.meshgrid(np.concatenate([-1*pulse_scale_I[:0:-1], pulse_scale_I]), pulse_scale_Q)
    pulse_scale_complex = x_full + 1j*y_full
    a =  0.985171878299
    b =  0.004644556712
    amp = a*np.abs(pulse_scale_complex) + b*np.abs(pulse_scale_complex)**2
    betas = np.exp(1j*np.angle(pulse_scale_complex))*amp
    """

    # The full cf
    cf_full = np.zeros((cf.shape[0], cf.shape[1] * 2 - 1), dtype=np.complex64)
    mid_idx = int(len(cf) / 2)
    # cf_full[:,:mid_idx+1] = np.conj(cf_phase_corrected)[:,::-1]
    cf_full[:, mid_idx:] = cf
    cf_full[:, :mid_idx] = np.conj(cf)[::-1, :0:-1]

    betas_I = np.concatenate([-1 * pulse_scale_I[:0:-1], pulse_scale_I])
    betas_Q = pulse_scale_Q
    return cf_full, betas_I, betas_Q


def process_tomo_cut(cf_cut, pulse_scales):
    # phase correction
    a = 0.062623780865
    cf = np.exp(1j * a * np.abs(pulse_scales) ** 2) * cf_cut
    return cf
