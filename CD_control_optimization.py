#%%
# a minimial implementation of a discrete version of grape
# consisting of a sequence of conditional displacement,
# displacement, and rotation pulses, with
# tuneable parameters

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from CD_control.helper_functions import plot_wigner
from scipy.optimize import minimize, basinhopping
import scipy.optimize
from datetime import datetime
from scipy.special import genlaguerre

try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

#%%
# TODO: Handle cases when phi, theta outside range.
# custom step-taking class for basinhopping optimization.
# TODO: make the step sizes changeable
class MyTakeStep(object):
    def __init__(self, CD_control_obj, stepsize=1):
        self.CD_control_obj = CD_control_obj
        self.stepsize = stepsize
        self.N_blocks = CD_control_obj.N_blocks
        self.beta_r_step_size = CD_control_obj.beta_r_step_size
        self.beta_theta_step_size = CD_control_obj.beta_theta_step_size
        self.alpha_r_step_size = (
            CD_control_obj.alpha_r_step_size
            if CD_control_obj.use_displacements
            else 0.0
        )
        self.alpha_theta_step_size = CD_control_obj.alpha_theta_step_size
        self.phi_step_size = CD_control_obj.phi_step_size
        self.theta_step_size = CD_control_obj.theta_step_size
        self.use_displacements = CD_control_obj.use_displacements

    def __call__(self, x):
        s = self.stepsize
        step_array = np.concatenate(
            [
                np.random.uniform(
                    -s * self.beta_r_step_size, s * self.beta_r_step_size, self.N_blocks
                ),  # betas_r
                np.random.uniform(
                    -s * self.beta_theta_step_size,
                    s * self.beta_theta_step_size,
                    self.N_blocks,
                ),  # betas_theta
                np.random.uniform(
                    -s * self.alpha_r_step_size,
                    s * self.alpha_r_step_size,
                    self.N_blocks,
                ),  # alphas_r
                np.random.uniform(
                    -s * self.alpha_theta_step_size,
                    s * self.alpha_theta_step_size,
                    self.N_blocks,
                ),  # alphas_theta
                np.random.uniform(
                    -s * self.phi_step_size, s * self.phi_step_size, self.N_blocks
                ),  # phis
                np.random.uniform(
                    -s * self.theta_step_size, s * self.theta_step_size, self.N_blocks,
                ),  # thetas
            ]
        )
        # todo: could make normalize angles a helper function.
        x = x + step_array
        return x


# custom basinhopping bounds for constrained global optimization
class MyBounds(object):
    def __init__(self, CD_control_obj):
        self.max_beta = CD_control_obj.max_beta
        self.max_alpha = CD_control_obj.max_alpha
        self.N_blocks = CD_control_obj.N_blocks
        self.no_CD_end = CD_control_obj.no_CD_end

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        beta_r_min_constraint = bool(np.all(x[: self.N_blocks] >= -self.max_beta))
        beta_r_max_constraint = bool(np.all(x[: self.N_blocks] <= self.max_beta))
        if self.no_CD_end:
            beta_r_max_constraint = bool(
                np.all(x[self.N_blocks - 1 : self.N_blocks] <= 0)
            )
        alpha_r_min_constraint = bool(
            np.all(x[2 * self.N_blocks : (3 * self.N_blocks)] >= -self.max_alpha)
        )
        alpha_r_max_constraint = bool(
            np.all(x[2 * self.N_blocks : (3 * self.N_blocks)] <= self.max_alpha)
        )

        return (
            beta_r_min_constraint
            and beta_r_max_constraint
            and alpha_r_min_constraint
            and alpha_r_max_constraint
        )


class OptFinishedException(Exception):
    def __init__(self, msg, CD_control_obj):
        super(OptFinishedException, self).__init__(msg)
        # CD_control_obj.save()
        # can save data here...


class CD_control:

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        initial_state=None,
        target_state=None,
        target_unitary=None,
        N_blocks=1,
        unitary_optimization=False,
        unitary_initial_states=None,
        unitary_final_states=None,
        unitary_learning_rate=1,
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        max_alpha=5,
        max_beta=5,
        saving_directory=None,
        name="CD_control",
        term_fid_intermediate=0.97,
        term_fid=0.999,
        beta_r_step_size=1,
        beta_theta_step_size=np.pi,
        alpha_r_step_size=0.5,
        alpha_theta_step_size=np.pi,
        phi_step_size=np.pi,
        theta_step_size=np.pi,
        analytic=True,
        beta_penalty_multiplier=0,
        minimizer_options={},
        basinhopping_kwargs={},
        save_all_minima=False,
        use_displacements=True,
        no_CD_end=True,
        circuits=[],
        N_cav=None,
        N_qb=None,
        CD_control_init_obj=None,
    ):
        if CD_control_init_obj is not None:
            N = CD_control_init_obj.N
            N2 = CD_control_init_obj.N2
            no_CD_end = (
                CD_control_init_obj.no_CD_end if no_CD_end is None else no_CD_end
            )
            CD_control_init_obj.concat_controls(N_blocks)  # take the first N_blocks
            betas = CD_control_init_obj.betas_full if betas is None else betas
            alphas = CD_control_init_obj.alphas_full if alphas is None else alphas
            phis = CD_control_init_obj.phis_full if phis is None else phis
            thetas = CD_control_init_obj.thetas_full if thetas is None else thetas

            # presumably the same prob to optimize over as in the initialization..
            initial_state = (
                CD_control_init_obj.initial_state_original
                if initial_state is None
                else initial_state
            )
            target_state = (
                CD_control_init_obj.target_state_original
                if target_state is None
                else target_state
            )
            target_unitary = (
                CD_control_init_obj.target_unitary_original
                if target_unitary is None
                else target_unitary
            )

            unitary_initial_states = (
                CD_control_init_obj.unitary_initial_states
                if unitary_initial_states is None
                else unitary_initial_states
            )
            unitary_final_states = (
                CD_control_init_obj.unitary_final_states
                if unitary_final_states is None
                else unitary_final_states
            )

        self.initial_state = initial_state
        self.target_state = target_state
        self.target_unitary = target_unitary
        self.unitary_optimization = unitary_optimization
        self.unitary_initial_states = unitary_initial_states
        self.unitary_final_states = unitary_final_states
        self.unitary_learning_rate = unitary_learning_rate
        self.N_blocks = N_blocks
        self.betas = (
            np.array(betas, dtype=np.complex128)
            if betas is not None
            else np.zeros(N_blocks, dtype=np.complex128)
        )
        self.alphas = (
            np.array(alphas, dtype=np.complex128)
            if alphas is not None
            else np.zeros(N_blocks, dtype=np.complex128)
        )
        self.phis = (
            np.array(phis, dtype=np.float64)
            if phis is not None
            else np.zeros(N_blocks, dtype=np.float64)
        )
        self.thetas = (
            np.array(thetas, dtype=np.float64)
            if thetas is not None
            else np.zeros(N_blocks, dtype=np.float64)
        )
        self.max_alpha = max_alpha if use_displacements else 0.0
        self.max_beta = max_beta
        self.saving_directory = saving_directory
        self.name = name
        self.term_fid_intermediate = term_fid_intermediate
        self.term_fid = term_fid
        self.beta_r_step_size = beta_r_step_size
        self.beta_theta_step_size = beta_theta_step_size
        self.alpha_r_step_size = alpha_r_step_size
        self.alpha_theta_step_size = alpha_theta_step_size
        self.phi_step_size = phi_step_size
        self.theta_step_size = theta_step_size
        self.analytic = analytic
        self.beta_penalty_multiplier = beta_penalty_multiplier
        self.minimizer_options = minimizer_options
        self.basinhopping_kwargs = basinhopping_kwargs
        # maximium number of iterations in L-BFGS-B.
        if "maxiter" not in self.minimizer_options:
            self.minimizer_options["maxiter"] = 1e3
        # note: ftol is relative percent difference in fidelity before optimization stops
        if "ftol" not in self.minimizer_options:
            self.minimizer_options["ftol"] = 1e-8
        # gtol is like the maximum gradient before optimization stops.
        if "gtol" not in self.minimizer_options:
            self.minimizer_options["gtol"] = 1e-8

        if "niter" not in self.basinhopping_kwargs:
            self.basinhopping_kwargs["niter"] = 50
        if "T" not in self.basinhopping_kwargs:
            self.basinhopping_kwargs["T"] = 0.1

        self.save_all_minima = save_all_minima
        self.circuits = list(circuits)
        self.use_displacements = use_displacements
        self.no_CD_end = no_CD_end

        if self.initial_state is not None:
            self.N_cav = self.initial_state.dims[0][1]
            self.N_qb = self.initial_state.dims[0][0]
        elif self.target_state is not None:
            self.N_cav = self.target_state.dims[0][1]
            self.N_qb = self.target_state.dims[0][0]
        else:
            self.N_cav = N
            self.N_qb = N2

        self.init_operators(self.N_cav, self.N_qb)

    def init_operators(self, N_cav, N_qb):
        self.N_cav = N_cav
        self.N_qb = N_qb
        self.I = qt.tensor(qt.identity(self.N_qb), qt.identity(self.N_cav))
        self.a = qt.tensor(qt.identity(self.N_qb), qt.destroy(self.N_cav))
        self.q = qt.tensor(qt.destroy(self.N_qb), qt.identity(self.N_cav))
        self.sz = 1 - 2 * self.q.dag() * self.q
        self.sx = self.q + self.q.dag()
        self.sy = 1j * (self.q.dag() - self.q)
        self.n = self.a.dag() * self.a

    def randomize(self, beta_scale=None, alpha_scale=None):
        beta_scale = self.max_beta if beta_scale is None else beta_scale
        alpha_scale = self.max_alpha if alpha_scale is None else alpha_scale
        ang_beta = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_beta = np.random.uniform(0, beta_scale, self.N_blocks)
        ang_alpha = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        rho_alpha = np.random.uniform(0, alpha_scale, self.N_blocks)
        phis = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        thetas = np.random.uniform(0, np.pi, self.N_blocks)
        self.betas = np.array(np.exp(1j * ang_beta) * rho_beta, dtype=np.complex128)
        if self.no_CD_end:
            self.betas[-1] = 0.0 + 1j * 0.0
        if self.use_displacements:
            self.alphas = np.array(
                np.exp(1j * ang_alpha) * rho_alpha, dtype=np.complex128
            )
        else:
            self.alphas = np.zeros(self.N_blocks, dtype=np.complex128)
        self.phis = np.array(phis, dtype=np.float64)
        self.thetas = np.array(thetas, dtype=np.float64)

    def D(self, alpha):
        if np.abs(alpha) == 0:
            return qt.tensor(qt.identity(self.N_qb), qt.identity(self.N_cav))
        return (alpha * self.a.dag() - np.conj(alpha) * self.a).expm()

    def D2(self, alpha):
        dim = self.N_cav
        x_mat = np.zeros((self.N_cav, self.N_cav), dtype=np.complex128)
        for m in range(self.N_cav):
            x_mat[m, m] = genlaguerre(m, 0)(np.abs(alpha) ** 2)
            for n in range(0, m):  # scan over lower triangle, n < m
                x_mn = (
                    np.sqrt(factorial(n) / factorial(m))
                    * (alpha) ** (m - n)
                    * genlaguerre(n, m - n)(np.abs(alpha) ** 2)
                )
                x_mat[m, n] = x_mn

            for n in range(m + 1, dim):  # scan over upper triangle, m < n
                x_mn = (
                    np.sqrt(factorial(m) / factorial(n))
                    * (-np.conj(alpha)) ** (n - m)
                    * genlaguerre(m, n - m)(np.abs(alpha) ** 2)
                )
                x_mat[m, n] = x_mn
        x_mat = x_mat * np.exp(-np.abs(alpha) ** 2 / 2.0)
        D = qt.Qobj(x_mat)
        return qt.tensor(qt.identity(self.N_qb), D)

    def CD(self, beta):
        if np.abs(beta) == 0:
            return qt.tensor(qt.identity(self.N_qb), qt.identity(self.N_cav))
        # return self.R(0,np.pi)*((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()
        # temp removing pi pulse from CD for analytic opt testing
        # return ((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()
        zz = qt.tensor(qt.ket2dm(qt.basis(self.N_qb, 0)), qt.identity(self.N_cav))
        oo = qt.tensor(qt.ket2dm(qt.basis(self.N_qb, 1)), qt.identity(self.N_cav))
        return self.R(0, np.pi) * (self.D(beta / 2.0) * zz + self.D(-beta / 2.0) * oo)
        # includes pi rotation

        # alec 10/8/2020: Removing pi rotation to compare with tensorflow version.
        # return self.D(beta / 2.0) * zz + self.D(-beta / 2.0) * oo

    # TODO: is it faster with non-exponential form?
    def R(self, phi, theta):
        if theta == 0:
            return qt.tensor(qt.identity(self.N_qb), qt.identity(self.N_cav))
        # return (-1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()
        return np.cos(theta / 2.0) - 1j * (
            np.cos(phi) * self.sx + np.sin(phi) * self.sy
        ) * np.sin(theta / 2.0)

    def dalpha_r_dD(self, alpha):
        r = np.abs(alpha)
        return (
            (1.0 / r) * (alpha * self.a.dag() - np.conj(alpha) * self.a) * self.D(alpha)
        )

    def dalpha_theta_dD(self, alpha):
        r = np.abs(alpha)
        return (
            1.0j
            * (alpha * self.a.dag() + np.conj(alpha) * self.a - r ** 2)
            * self.D(alpha)
        )

    def dbeta_r_dCD(self, beta):
        r = np.abs(beta)
        return (
            (0.5 / r)
            * (-self.sz * (beta * self.a.dag() - np.conj(beta) * self.a))
            * self.CD(beta)
        )

    def dbeta_theta_dCD(self, beta):
        r = np.abs(beta)
        return (
            (0.5j)
            * (-self.sz * (beta * self.a.dag() + np.conj(beta) * self.a) - r ** 2 / 2)
            * self.CD(beta)
        )

    def dtheta_dR(self, phi, theta):
        # return (-1j/2.0)*(self.sx*np.cos(phi) + self.sy*np.sin(phi))*self.R(phi, theta)
        return -0.5 * (
            np.sin(theta / 2.0)
            + 1j * (np.cos(phi) * self.sx + np.sin(phi) * self.sy) * np.cos(theta / 2.0)
        )

    def dphi_dR(self, phi, theta):
        return (
            1j * (np.sin(phi) * self.sx - np.cos(phi) * self.sy) * np.sin(theta / 2.0)
        )

    def U_block(self, beta, alpha, phi, theta):
        U = self.CD(beta) * self.D(alpha) * self.R(phi, theta)
        return U

    def U_i_block(self, i, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        return self.U_block(betas[i], alphas[i], phis[i], thetas[i])

    def U_tot(self, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        U = self.I
        for i in range(self.N_blocks):
            U = self.U_i_block(i, betas, alphas, phis, thetas) * U
        return U

    def forward_states(
        self, betas=None, alphas=None, phis=None, thetas=None, initial_state=None
    ):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        initial_state = (
            initial_state if initial_state is not None else self.initial_state
        )
        psi_fwd = [initial_state]
        # blocks
        for i in range(self.N_blocks):
            psi_fwd.append(self.R(phis[i], thetas[i]) * psi_fwd[-1])
            psi_fwd.append(self.D(alphas[i]) * psi_fwd[-1])
            psi_fwd.append(self.CD(betas[i]) * psi_fwd[-1])
        return psi_fwd

    def reverse_states(self, betas, alphas, phis, thetas, target_state=None):
        target_state = target_state if target_state is not None else self.target_state
        target_state = qt.Qobj(target_state)
        psi_bwd = [target_state.dag()]
        # blocks
        for i in np.arange(self.N_blocks)[::-1]:
            psi_bwd.append(psi_bwd[-1] * self.CD(betas[i]))
            psi_bwd.append(psi_bwd[-1] * self.D(alphas[i]))
            psi_bwd.append(psi_bwd[-1] * self.R(phis[i], thetas[i]))
        return psi_bwd

    def fid_and_grad_fid(
        self,
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        initial_state=None,
        target_state=None,
        unitary_fid=False,
    ):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        initial_state = (
            initial_state if initial_state is not None else self.initial_state
        )
        target_state = target_state if target_state is not None else self.target_state
        psi_fwd = self.forward_states(
            betas, alphas, phis, thetas, initial_state=initial_state
        )
        psi_bwd = self.reverse_states(
            betas, alphas, phis, thetas, target_state=target_state
        )

        dbeta_r = np.zeros(self.N_blocks, dtype=np.complex128)
        dbeta_theta = np.zeros(self.N_blocks, dtype=np.complex128)
        dalpha_r = np.zeros(self.N_blocks, dtype=np.complex128)
        dalpha_theta = np.zeros(self.N_blocks, dtype=np.complex128)
        dphi = np.zeros(self.N_blocks, dtype=np.complex128)
        dtheta = np.zeros(self.N_blocks, dtype=np.complex128)

        for i in range(self.N_blocks):
            for j in [1, 2, 3]:
                k = 3 * i + j
                if j == 1:
                    dphi[i] = (
                        psi_bwd[-(k + 1)]
                        * self.dphi_dR(phis[i], thetas[i])
                        * psi_fwd[k - 1]
                    ).full()[0][0]
                    dtheta[i] = (
                        psi_bwd[-(k + 1)]
                        * self.dtheta_dR(phis[i], thetas[i])
                        * psi_fwd[k - 1]
                    ).full()[0][0]
                if j == 2 and self.use_displacements:
                    dalpha_r[i] = (
                        psi_bwd[-(k + 1)] * self.dalpha_r_dD(alphas[i]) * psi_fwd[k - 1]
                    ).full()[0][0]
                    dalpha_theta[i] = (
                        psi_bwd[-(k + 1)]
                        * self.dalpha_theta_dD(alphas[i])
                        * psi_fwd[k - 1]
                    ).full()[0][0]
                if j == 3:
                    dbeta_r[i] = (
                        psi_bwd[-(k + 1)] * self.dbeta_r_dCD(betas[i]) * psi_fwd[k - 1]
                    ).full()[0][0]
                    dbeta_theta[i] = (
                        psi_bwd[-(k + 1)]
                        * self.dbeta_theta_dCD(betas[i])
                        * psi_fwd[k - 1]
                    ).full()[0][0]

        # TODO: move this after unitary_fid, after testing
        overlap = (psi_bwd[0] * psi_fwd[-1]).full()[0][0]
        if overlap == 0:
            return (
                0,
                np.zeros(self.N_blocks, dtype=np.float64),
                np.zeros(self.N_blocks, dtype=np.float64),
                np.zeros(self.N_blocks, dtype=np.float64),
                np.zeros(self.N_blocks, dtype=np.float64),
                np.zeros(self.N_blocks, dtype=np.float64),
                np.zeros(self.N_blocks, dtype=np.float64),
            )

        if unitary_fid:
            return overlap, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta

        fid = np.abs(overlap) ** 2

        dbeta_r = (
            2 * np.abs(overlap) * np.real(overlap * np.conj(dbeta_r)) / np.abs(overlap)
        )

        dbeta_theta = (
            2
            * np.abs(overlap)
            * np.real(overlap * np.conj(dbeta_theta))
            / np.abs(overlap)
        )

        dalpha_r = (
            2 * np.abs(overlap) * np.real(overlap * np.conj(dalpha_r)) / np.abs(overlap)
        )
        dalpha_theta = (
            2
            * np.abs(overlap)
            * np.real(overlap * np.conj(dalpha_theta))
            / np.abs(overlap)
        )
        dphi = 2 * np.abs(overlap) * np.real(overlap * np.conj(dphi)) / np.abs(overlap)
        dtheta = (
            2 * np.abs(overlap) * np.real(overlap * np.conj(dtheta)) / np.abs(overlap)
        )

        return fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta

    def final_state(
        self, betas=None, alphas=None, phis=None, thetas=None, initial_state=None
    ):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        psi = initial_state if initial_state is not None else self.initial_state
        for i in range(self.N_blocks):
            psi = self.R(phis[i], thetas[i]) * psi
            if self.use_displacements:
                psi = self.D(alphas[i]) * psi
            psi = self.CD(betas[i]) * psi
        return psi

    def fidelity(self, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        overlap = (
            self.target_state.dag() * self.final_state(betas, alphas, phis, thetas)
        ).full()[0][0]
        return np.abs(overlap) ** 2

    def unitary_forward_states(self, betas, alphas, phis, thetas):
        U_fwd = [self.I]
        # blocks
        for i in range(self.N_blocks):
            U_fwd.append(self.R(phis[i], thetas[i]) * U_fwd[-1])
            U_fwd.append(self.D(alphas[i]) * U_fwd[-1])
            U_fwd.append(self.CD(betas[i]) * U_fwd[-1])
        return U_fwd

    def unitary_reverse_states(self, betas, alphas, phis, thetas):
        U_bwd = [self.target_unitary.dag()]
        # blocks
        for i in np.arange(self.N_blocks)[::-1]:
            U_bwd.append(U_bwd[-1] * self.CD(betas[i]))
            U_bwd.append(U_bwd[-1] * self.D(alphas[i]))
            U_bwd.append(U_bwd[-1] * self.R(phis[i], thetas[i]))
        return U_bwd

    def unitary_fid_and_grad_fid_stochastic(
        self,
        unitary_initial_states=None,
        unitary_final_states=None,
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        testing=False,
    ):  # stochastic gradient descent
        unitary_initial_states = (
            self.unitary_initial_states
            if unitary_initial_states is None
            else unitary_initial_states
        )
        unitary_final_states = (
            self.unitary_final_states
            if unitary_final_states is None
            else unitary_final_states
        )

        i = np.random.randint(len(unitary_initial_states))
        unitary_initial_state = [unitary_initial_states[i]]
        unitary_final_state = [unitary_final_states[i]]
        return self.unitary_fid_and_grad_fid_approx(
            unitary_initial_states=unitary_initial_state,
            unitary_final_states=unitary_final_state,
            betas=betas,
            alphas=alphas,
            phis=phis,
            thetas=thetas,
            testing=testing,
        )

    def unitary_fid_and_grad_fid_approx(
        self,
        unitary_initial_states=None,
        unitary_final_states=None,
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        testing=False,
    ):
        unitary_initial_states = (
            self.unitary_initial_states
            if unitary_initial_states is None
            else unitary_initial_states
        )
        # unitary_final_states = (
        #     self.unitary_final_states
        #     if unitary_final_states is None
        #     else unitary_final_states
        # )

        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas

        U_circuit = self.U_tot(betas, alphas, phis, thetas)
        D = self.N_cav * self.N_qb
        overlap = (self.target_unitary.dag() * U_circuit).tr()
        fid = np.abs((1 / D) * overlap) ** 2

        num_states = len(unitary_initial_states)
        approx_overlap, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        for i in range(num_states):
            initial_state = unitary_initial_states[i]
            target_state = (
                unitary_final_states[i]
                if unitary_final_states is not None
                else self.target_unitary * initial_state
            )  # if the initial state is an eigenstate of target_unitary, the target_state = eigenvalue*initial_state
            (
                approx_overlap_i,
                dbeta_r_i,
                dbeta_theta_i,
                dalpha_r_i,
                dalpha_theta_i,
                dphi_i,
                dtheta_i,
            ) = self.fid_and_grad_fid(
                initial_state=initial_state, target_state=target_state, unitary_fid=True
            )
            approx_overlap += approx_overlap_i
            dbeta_r += dbeta_r_i
            dbeta_theta += dbeta_theta_i
            dalpha_r += dalpha_r_i
            dalpha_theta += dalpha_theta_i
            dphi += dphi_i
            dtheta += dtheta_i
        approx_fid = np.abs(1.0 / num_states * approx_overlap) ** 2
        scaling_factor = (
            (2.0 / D) * np.conj(overlap) * (1 / num_states) * self.unitary_learning_rate
        )
        dbeta_r = np.real(scaling_factor * dbeta_r)
        dbeta_theta = np.real(scaling_factor * dbeta_theta)
        dalpha_r = np.real(scaling_factor * dalpha_r)
        dalpha_theta = np.real(scaling_factor * dalpha_theta)
        dphi = np.real(scaling_factor * dphi)
        dtheta = np.real(scaling_factor * dtheta)
        if testing:
            return (
                fid,
                approx_fid,
                dbeta_r,
                dbeta_theta,
                dalpha_r,
                dalpha_theta,
                dphi,
                dtheta,
            )
        return fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta

    def unitary_fid_and_grad_fid(self, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas

        U_fwd = self.unitary_forward_states(betas, alphas, phis, thetas)
        U_bwd = self.unitary_reverse_states(betas, alphas, phis, thetas)

        D = self.N_cav * self.N_qb
        overlap = (U_bwd[0] * U_fwd[-1]).tr()
        fid = np.abs((1 / D) * overlap) ** 2

        dbeta_r = np.zeros(self.N_blocks, dtype=np.complex128)
        dbeta_theta = np.zeros(self.N_blocks, dtype=np.complex128)
        dalpha_r = np.zeros(self.N_blocks, dtype=np.complex128)
        dalpha_theta = np.zeros(self.N_blocks, dtype=np.complex128)
        dphi = np.zeros(self.N_blocks, dtype=np.complex128)
        dtheta = np.zeros(self.N_blocks, dtype=np.complex128)

        for i in range(self.N_blocks):
            for j in [1, 2, 3]:
                k = 3 * i + j
                if j == 1:
                    dphi[i] = (
                        U_fwd[k - 1]
                        * U_bwd[-(k + 1)]
                        * self.dphi_dR(phis[i], thetas[i])
                    ).tr()
                    dtheta[i] = (
                        U_fwd[k - 1]
                        * U_bwd[-(k + 1)]
                        * self.dtheta_dR(phis[i], thetas[i])
                    ).tr()
                if j == 2:
                    dalpha_r[i] = (
                        U_fwd[k - 1] * U_bwd[-(k + 1)] * self.dalpha_r_dD(alphas[i])
                    ).tr()
                    dalpha_theta[i] = (
                        U_fwd[k - 1] * U_bwd[-(k + 1)] * self.dalpha_theta_dD(alphas[i])
                    ).tr()
                if j == 3:
                    dbeta_r[i] = (
                        U_fwd[k - 1] * U_bwd[-(k + 1)] * self.dbeta_r_dCD(betas[i])
                    ).tr()
                    dbeta_theta[i] = (
                        U_fwd[k - 1] * U_bwd[-(k + 1)] * self.dbeta_theta_dCD(betas[i])
                    ).tr()

        scalar = 2.0 / (D ** 2) * np.conj(overlap)
        dbeta_r = np.real(scalar * dbeta_r)
        dbeta_theta = np.real(scalar * dbeta_theta)

        if self.use_displacements:  # if not they are left at 0
            dalpha_r = np.real(scalar * dalpha_r)
            dalpha_theta = np.real(scalar * dalpha_theta)
        else:
            dalpha_r = np.zeros(len(dalpha_r), dtype=np.float64)
            dalpha_theta = np.zeros(len(dalpha_theta), dtype=np.float64)

        dphi = np.real(scalar * dphi)
        dtheta = np.real(scalar * dtheta)

        return fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta

    def unitary_fidelity(self, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        U_circuit = self.U_tot(betas, alphas, phis, thetas)
        D = self.N_cav * self.N_qb
        overlap = (self.target_unitary.dag() * U_circuit).tr()
        return np.abs((1 / D) * overlap) ** 2

        # i is state after each operation. Set i = 0 for initial state, i = -1 for final state

    def plot_state(self, i=0, contour=True, fig=None, ax=None, max_alpha=6, cbar=True):
        state = self.forward_states()[i]
        plot_wigner(
            state, contour=contour, fig=fig, ax=ax, max_alpha=max_alpha, cbar=cbar
        )

    def plot_target_state(
        self, contour=True, fig=None, ax=None, max_alpha=6, cbar=True
    ):
        plot_wigner(
            self.target_state,
            contour=contour,
            fig=fig,
            ax=ax,
            max_alpha=max_alpha,
            cbar=cbar,
        )

    def plot_projected_state(
        self,
        i=0,
        p="g",
        contour=True,
        fig=None,
        ax=None,
        max_alpha=6,
        cbar=True,
        normalize=True,
    ):
        state = self.forward_states()[i]
        rho = qt.ket2dm(state)
        if p == "g":
            ket = qt.tensor(qt.basis(self.N_qb, 0), qt.identity(self.N_cav))
            rho_proj = ket.dag() * rho * ket
        elif p == "e":
            ket = qt.tensor(qt.basis(self.N_qb, 1), qt.identity(self.N_cav))
            rho_proj = ket.dag() * rho * ket
        elif p == "+":
            ket = qt.tensor(
                (qt.basis(self.N_qb, 0) + qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            rho_proj = ket.dag() * rho * ket
        elif p == "-":
            ket = qt.tensor(
                (qt.basis(self.N_qb, 0) - qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            rho_proj = ket.dag() * rho * ket
        elif p == "+-":
            ketp = qt.tensor(
                (qt.basis(self.N_qb, 0) - qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            ketm = qt.tensor(
                (qt.basis(self.N_qb, 0) - qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            rho_proj = ketp.dag() * rho * ketm
        elif p == "-+":
            ketm = qt.tensor(
                (qt.basis(self.N_qb, 0) - qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            ketp = qt.tensor(
                (qt.basis(self.N_qb, 0) - qt.basis(self.N_qb, 1)) / np.sqrt(2.0),
                qt.identity(self.N_cav),
            )
            rho_proj = ketm.dag() * rho * ketp
        elif p == "ge":
            ketg = qt.tensor(qt.basis(self.N_qb, 0), qt.identity(self.N_cav))
            kete = qt.tensor(qt.basis(self.N_qb, 1), qt.identity(self.N_cav))
            rho_proj = ketg.dag() * rho * kete
        elif p == "eg":
            ketg = qt.tensor(qt.basis(self.N_qb, 0), qt.identity(self.N_cav))
            kete = qt.tensor(qt.basis(self.N_qb, 1), qt.identity(self.N_cav))
            rho_proj = kete.dag() * rho * ketg

        if normalize:
            rho_proj = rho_proj.unit()
        plot_wigner(
            rho_proj,
            tensor_state=False,
            contour=contour,
            fig=fig,
            ax=ax,
            max_alpha=max_alpha,
            cbar=cbar,
        )

    def plot_projected_states_ge(
        self, i=0, contour=True, max_alpha=6, cbar=True, fig=None, normalize=True
    ):
        if fig is None:
            fig = plt.figure(figsize=(9, 3.5), dpi=200)
        axs = fig.subplots(1, 2)
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="g",
            fig=fig,
            ax=axs[0],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="e",
            fig=fig,
            ax=axs[1],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        axs[0].set_title(r"$\rho_{gg}$")
        axs[1].set_title(r"$\rho_{ee}$")
        axs[1].set_ylabel("")

    def plot_projected_states_pm(
        self, i=0, contour=True, max_alpha=6, cbar=True, fig=None, normalize=True
    ):
        if fig is None:
            fig = plt.figure(figsize=(9, 3.5), dpi=200)
        axs = fig.subplots(1, 2)
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="+",
            fig=fig,
            ax=axs[0],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="-",
            fig=fig,
            ax=axs[1],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        axs[0].set_title(r"$\rho_{++}$")
        axs[1].set_title(r"$\rho_{--}$")
        axs[1].set_ylabel("")

    def plot_full_projection(
        self, i=0, contour=True, max_alpha=6, cbar=True, fig=None, normalize=True
    ):
        if fig is None:
            fig = plt.figure(figsize=(8, 6), dpi=200)
        axs = fig.subplots(2, 2)
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="g",
            fig=fig,
            ax=axs[0, 0],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="ge",
            fig=fig,
            ax=axs[0, 1],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="eg",
            fig=fig,
            ax=axs[1, 0],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        self.plot_projected_state(
            contour=contour,
            i=i,
            p="e",
            fig=fig,
            ax=axs[1, 1],
            max_alpha=max_alpha,
            cbar=cbar,
            normalize=normalize,
        )
        axs[0, 0].set_title(r"$\rho_{gg}$")
        axs[0, 1].set_title(r"$\rho_{ge}$")
        axs[1, 0].set_title(r"$\rho_{eg}$")
        axs[1, 1].set_title(r"$\rho_{ee}$")
        axs[0, 1].set_ylabel("")
        axs[1, 1].set_ylabel("")
        axs[0, 0].set_xlabel("")
        axs[0, 1].set_xlabel("")

    # for the optimization, we will flatten the parameters
    # the will be, in order,
    # [betas_r, betas_theta, alphas_r, alphas_theta,  phis, thetas]
    def cost_function(self, parameters):
        betas, alphas, phis, thetas = self.unflatten_parameters(parameters)
        # TODO: later, implement more sophisticated saving and real time information.
        if self.unitary_optimization:
            f = self.unitary_fidelity(betas, alphas, phis, thetas)
        else:
            f = self.fidelity(betas, alphas, phis, thetas)
        if self.bpm > 0:
            betas_r = np.abs(betas)
            beta_penalty = self.bpm * np.sum(betas_r)
            print("\rfid: %.4f beta penalty: %.4f" % (f, beta_penalty), end="")
        else:
            print("\rfid: %.4f" % f, end="")
            beta_penalty = 0
        fn = f - beta_penalty
        # TODO: Maybe we can have a bit worse f if we want lower betas.
        # I should save parameters when the cost function is the lowest, not the fidelity.
        # but do we stop when the cost function reaches the term? Or the fidelity?
        if fn > self.best_fn:
            self.best_fn = fn
            self.betas = betas
            self.alphas = alphas
            self.phis = phis
            self.thetas = thetas
        if self.tf is not None and f >= self.term_fid:
            raise OptFinishedException("Requested fidelity obtained", self)
        return -fn

    # TODO: include final rotation and displacement
    # TODO: Would it be easier for things instead to specify |beta| and angle(beta) instead of
    # doing things with beta_r and beta_i?
    # TODO: Gauge degree of freedom
    def cost_function_analytic(self, parameters):
        betas, alphas, phis, thetas = self.unflatten_parameters(parameters)
        if self.unitary_optimization == "approximate":
            fid_grads = self.unitary_fid_and_grad_fid_approx
        elif self.unitary_optimization == "stochastic":
            fid_grads = self.unitary_fid_and_grad_fid_stochastic
        elif self.unitary_optimization:
            fid_grads = self.unitary_fid_and_grad_fid
        else:
            fid_grads = self.fid_and_grad_fid
        (f, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta,) = fid_grads(
            betas, alphas, phis, thetas
        )
        gradf = np.concatenate(
            [dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta]
        )
        # todo: instead, optimize |beta| and phase(beta). Same for alpha
        if self.bpm > 0:
            betas_r = np.abs(betas)
            beta_penalty = self.bpm * np.sum(betas_r)
            grad_beta_penalty = (
                self.bpm
                * np.concatenate(  # todo: fix gradient of beta penalty, maybe use soft relu penalty for acceptable range of beta?
                    [-1.0 * betas_r / np.abs(betas_r), np.zeros(5 * self.N_blocks),]
                )
            )
            print("\rfid: %.4f beta penalty: %.4f" % (f, beta_penalty), end="")
        else:
            print("\rfid: %.4f" % f, end="")
            beta_penalty = 0
            grad_beta_penalty = 0
        fn = f - beta_penalty
        if fn > self.best_fn:
            self.best_fn = fn
            self.betas = betas
            self.alphas = alphas
            self.phis = phis
            self.thetas = thetas
        if self.tf is not None and f >= self.tf:
            raise OptFinishedException("Requested fidelity obtained", self)
        return (-fn, -(gradf - grad_beta_penalty))

    # TODO: if I only care about the cavity state, I can optimize on the partial trace of the
    # cavity. Then, in the experiment, I can measure and reset the qubit.
    # For example, if I'm interested in creating a cat I can get to
    # (|alpha> + |-alpha>)|g> + (|alpha> - |-alpha>)|e>
    # then somhow I can optimize that the cavity state conditioned on g or e is only
    # a unitary operaton away from each other, which would allow me to implement feedback for
    # preperation in the experiment.

    def flatten_parameters(self, betas=None, alphas=None, phis=None, thetas=None):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        return np.array(
            np.concatenate(
                [
                    np.abs(self.betas),
                    np.angle(self.betas),
                    np.abs(self.alphas),
                    np.angle(self.alphas),
                    self.phis,
                    self.thetas,
                ]
            ),
            dtype=np.float64,
        )

    def unflatten_parameters(self, parameters):
        betas_r = parameters[0 : self.N_blocks]
        betas_theta = parameters[self.N_blocks : 2 * self.N_blocks]
        alphas_r = parameters[2 * self.N_blocks : (3 * self.N_blocks)]
        alphas_theta = parameters[(3 * self.N_blocks) : (4 * self.N_blocks)]
        phis = parameters[(4 * self.N_blocks) : (5 * self.N_blocks)]
        thetas = parameters[(5 * self.N_blocks) :]
        alphas = alphas_r * np.exp(1j * alphas_theta)
        betas = betas_r * np.exp(1j * betas_theta)
        return betas, alphas, phis, thetas

    def optimize(self):
        init_params = self.flatten_parameters()
        bounds = np.concatenate(
            [
                [(0, self.max_beta) for _ in range(self.N_blocks - 1)]
                + [(0, 0 if self.no_CD_end else self.max_beta)],
                [(-np.inf, np.inf) for _ in range(self.N_blocks)],
                [(0, self.max_alpha) for _ in range(self.N_blocks)],
                [(-np.inf, np.inf) for _ in range(self.N_blocks)],
                [(-np.inf, np.inf) for _ in range(self.N_blocks)],
                [(-np.inf, np.inf) for _ in range(self.N_blocks)],
            ]
        )
        cost_function = (
            self.cost_function_analytic if self.analytic else self.cost_function
        )

        def callback_fun(x, f, accepted):
            betas, alphas, phis, thetas = self.unflatten_parameters(x)
            self.print_info(betas, alphas, phis, thetas, human=True)
            if self.save_all_minima:
                betas, _, _, _ = self.unflatten_parameters(x)
                betas_r = np.abs(betas)
                beta_penalty = self.bpm * np.sum(betas_r)
                fid = (
                    -f + beta_penalty
                )  # easiest is to just add back the penalty to get fidelity
                self.circuits.append(np.concatenate([np.array([fid]), np.array(x)]))
            self.basinhopping_num += 1
            print(
                " basin #%d at min %.4f. accepted: %d"
                % (self.basinhopping_num, f, int(accepted))
            )

        try:
            mytakestep = MyTakeStep(self)
            mybounds = MyBounds(self)
            print("\n\nStarting optimization.\n\n")
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "jac": self.analytic,
                "bounds": bounds,
                "options": self.minimizer_options,
            }
            basinhopping_kwargs = self.basinhopping_kwargs
            self.bpm = self.beta_penalty_multiplier
            self.basinhopping_num = 0
            self.best_fn = 0
            # self.bpm = 0
            self.tf = self.term_fid
            # don't use beta penalty in the first round
            # The first round of optimization: Basinhopping
            print("First optimization round:")
            print("N_blocks: " + str(self.N_blocks))
            print("ftol: " + str(minimizer_kwargs["options"]["ftol"]))
            print("gtol: " + str(minimizer_kwargs["options"]["gtol"]))
            print("niter: " + str(basinhopping_kwargs["niter"]))
            print("beta penality multipler: " + str(self.bpm))
            print("starting step size: " + str(mytakestep.stepsize))
            print("term fid: " + str(self.tf))
            basinhopping(
                cost_function,
                x0=[init_params],
                minimizer_kwargs=minimizer_kwargs,
                take_step=mytakestep,
                accept_test=mybounds,
                callback=callback_fun,
                **basinhopping_kwargs
            )
        except OptFinishedException:
            print("\n\ndesired intermediate fidelity reached.\n\n")
            # if self.unitary_optimization:
            #    fid = self.unitary_fidelity()
            # else:
            #    fid = self.fidelity()
            # print("Best fidelity found: " + str(fid))
        else:
            print("\n\nFirst optimization failed to reach desired fidelity.\n\n")
        self.normalize_angles()
        self.print_info(human=True)

        """
        try:
            print("Second optimization round:")
            init_params = self.flatten_parameters()
            self.bpm = self.beta_penalty_multiplier
            minimizer_kwargs['options']['ftol'] = self.minimizer_options['ftol']/10.0
            minimizer_kwargs['options']['gtol'] = self.minimizer_options['gtol']/10.0
            mytakestep.stepsize = 0.25
            self.tf = self.term_fid
            print("First optimization round:")
            print("ftol: " + str(minimizer_kwargs['options']['ftol']))
            print("gtol: " + str(minimizer_kwargs['options']['gtol']))
            print("niter: " + str(basinhopping_kwargs['niter']))
            print("beta penality multipler: " + str(self.bpm))
            print("starting step size: " + str(mytakestep.stepsize))
            print("term fid: " + str(self.tf))
            init_params
            basinhopping(cost_function, x0=[init_params],
                         minimizer_kwargs=minimizer_kwargs,
                         take_step=mytakestep, accept_test=mybounds,
                         callback=callback_fun, **basinhopping_kwargs)
        except OptFinishedException as e:
            print("\n\ndesired fidelity reached.\n\n")
            fid = self.fidelity()
            print('Best fidelity found: ' + str(fid))
        else:
            print("\n\noptimization failed to reach desired fidelity.\n\n")
            fid = self.fidelity()
            print('Best fidelity found: ' + str(fid))
        """

    # when parameters is specificed, normalize_angles is being used
    # during the optimization. In this case, it normalizes the parameters
    # and returns the parameters without updating self.
    # if parameters not specified, just normalize self's angles.
    # todo: make faster with numpy...
    def normalize_angles(self, phis=None, thetas=None):
        do_return = phis is not None
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        thetas_new = []
        for theta in thetas:
            while theta < -np.pi:
                theta = theta + 2 * np.pi
            while theta > np.pi:
                theta = theta - 2 * np.pi
            thetas_new.append(theta)
        thetas = np.array(thetas_new)
        phis_new = []
        for phi in phis:
            while phi < -np.pi:
                phi = phi + 2 * np.pi
            while phi > np.pi:
                phi = phi - 2 * np.pi
            phis_new.append(phi)
        phis = np.array(phis_new)
        if do_return:
            return phis, thetas
        else:
            self.thetas = thetas
            self.phis = phis

    def save(self):
        datestr = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        filestring = self.saving_directory + self.name + "_" + datestr
        filename_np = filestring + ".npz"
        filename_qt = filestring + ".qt"
        np.savez(
            filename_np,
            betas=self.betas,
            alphas=self.alphas,
            phis=self.phis,
            thetas=self.thetas,
            max_alpha=self.max_alpha,
            max_beta=self.max_beta,
            name=self.name,
            circuits=self.circuits,
        )
        print("\n\nparameters saved as: " + filename_np)
        qt.qsave([self.initial_state, self.target_state], filename_qt)
        print("states saved as: " + filename_qt)
        self.print_info()
        # print('name for loading:' + filestring)
        return filestring

    def load(self, filestring):
        filename_np = filestring + ".npz"
        filename_qt = filestring + ".qt"
        f = np.load(filename_np)
        betas, alphas, phis, thetas, max_alpha, max_beta, name = (
            f["betas"],
            f["alphas"],
            f["phis"],
            f["thetas"],
            f["max_alpha"],
            f["max_beta"],
            str(f["name"]),
        )
        circuits = f["circuits"] if ["circuits"] in f else []
        print("loaded parameters from:" + filename_np)
        f.close()
        states = qt.qload(filename_qt)
        initial_state, target_state = states[0], states[1]
        print("loaded states from:" + filename_qt)
        self.__init__(
            initial_state=initial_state,
            target_state=target_state,
            N_blocks=len(betas),
            betas=betas,
            alphas=alphas,
            phis=phis,
            thetas=thetas,
            max_alpha=max_alpha,
            max_beta=max_beta,
            name=name,
            circuits=circuits,
        )
        self.print_info()

    def print_info(self, betas=None, alphas=None, phis=None, thetas=None, human=True):
        betas = self.betas if betas is None else betas
        alphas = self.alphas if alphas is None else alphas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        phis, thetas = self.normalize_angles(phis, thetas)
        f = (
            self.unitary_fidelity(betas, alphas, phis, thetas)
            if self.unitary_optimization
            else self.fidelity(betas, alphas, phis, thetas)
        )
        if human:
            with np.printoptions(precision=5, suppress=True):
                print("\n\n" + str(self.name))
                print("N_blocks:     " + str(self.N_blocks))
                print("betas:        " + str(betas))
                print("alphas:       " + str(alphas))
                print("phis (deg):   " + str(phis * 180.0 / np.pi))
                print("thetas (deg): " + str(thetas * 180.0 / np.pi))
                print("Fidelity:     %.5f" % f)
                print("\n")
        else:
            print("\n\n" + str(self.name))
            print("N_blocks: " + repr(self.N_blocks))
            print("betas: " + repr(betas))
            print("alphas: " + repr(alphas))
            print("phis: " + repr(phis))
            print("thetas: " + repr(thetas))
            print("Fidelity: " + repr(f))
            print("\n")


# %%
