from CD_control.CD_control_optimization import *
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import binarytree as bt


class CD_control_init(CD_control):

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        max_N=10,
        initial_state=None,
        target_state=None,
        target_unitary=None,
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
        beta_r_step_size=2,
        beta_theta_step_size=2 * np.pi,
        alpha_r_step_size=1,
        alpha_theta_step_size=2 * np.pi,
        phi_step_size=2 * np.pi,
        theta_step_size=np.pi,
        analytic=True,
        beta_penalty_multiplier=0,
        minimizer_options={},
        basinhopping_kwargs={},
        save_all_minima=False,
        use_displacements=True,
        no_CD_end=True,
        circuits=[],
        N=None,
        N2=None,
        init_order=None,
    ):
        if "niter" not in basinhopping_kwargs:
            basinhopping_kwargs["niter"] = 5
        # note: ftol is relative percent difference in fidelity before optimization stops
        if "ftol" not in minimizer_options:
            minimizer_options["ftol"] = 1e-8  # lower than usual
        # gtol is like the maximum gradient before optimization stops.
        if "gtol" not in minimizer_options:
            minimizer_options["gtol"] = 1e-8  # lower than usual
        self.init_order = init_order

        CD_control.__init__(
            self,
            initial_state=initial_state,
            target_state=target_state,
            target_unitary=target_unitary,
            N_blocks=2,
            unitary_optimization=unitary_optimization,
            unitary_initial_states=unitary_initial_states,
            unitary_final_states=unitary_final_states,
            unitary_learning_rate=unitary_learning_rate,
            betas=betas,
            alphas=alphas,
            phis=phis,
            thetas=thetas,
            max_alpha=max_alpha,
            max_beta=max_beta,
            saving_directory=saving_directory,
            name=name,
            term_fid_intermediate=term_fid_intermediate,
            term_fid=term_fid,
            beta_r_step_size=beta_r_step_size,
            beta_theta_step_size=beta_theta_step_size,
            alpha_r_step_size=alpha_r_step_size,
            alpha_theta_step_size=alpha_theta_step_size,
            phi_step_size=phi_step_size,
            theta_step_size=theta_step_size,
            analytic=analytic,
            beta_penalty_multiplier=beta_penalty_multiplier,
            minimizer_options=minimizer_options,
            basinhopping_kwargs=basinhopping_kwargs,
            save_all_minima=save_all_minima,
            use_displacements=use_displacements,
            no_CD_end=True,
            circuits=circuits,
            N=N,
            N2=N2,
        )
        self.max_N = max_N
        self.initial_state_original = self.initial_state
        self.target_state_original = self.target_state
        self.target_unitary_original = self.target_unitary
        self.reset_init()

    def reset_init(self):
        self.fids_reached = [0 for _ in range(self.max_N)]
        self.params = {
            "betas": [0 for _ in range(self.max_N)],
            "alphas": [0 for _ in range(self.max_N)],
            "phis": [0 for _ in range(self.max_N)],
            "thetas": [0 for _ in range(self.max_N)],
        }
        if self.init_order is None:
            self.unitary_initialization_order = [
                x.val for x in bt.build(list(range(self.max_N))).inorder
            ]
        else:
            self.unitary_initialization_order = self.init_order
        self.unitaries = [
            qt.tensor(qt.identity(self.N), qt.identity(self.N2))
            for _ in range(self.max_N)
        ]
        # unitary order will represent the order unitaries are initialized
        # for example, for N_blocks = 7, unitaries will be numbered:
        # |psi_t> = U_6*U_5*U_4*U_3*U_2*U_1*U_0|psi_i>, where U_0 is the first unitary acting on the state
        # and U_6 is the final.
        # unitary_init_order will be [3,1,4,0,5,2,6], meaning first, U_3 will be optimized, then U_1, and so on.

        # [3, 1, 4, 0, 5, 2, 6] corresponds to psi_f.dag() U_6 U_2 U_5 U_0 U_4 U_1 U_3 psi_i,
        # where each j index in U_j reprents the order in which the blocks were added in to the greedy optimization
        # i.e.
        # psi_f.dag() U_0 psi_i,
        # psi_f.dag() U_0 U_1 psi_i,
        # psi_f.dag() U_2 U_0 U_1 psi_i,
        # psi_f.dag() U_2 U_0 U_4 U_1 psi_i, => optimizing U_4, and psi_f'.dag() U_4 psi_i'is the state transfer problem
        # ordering can be set arbitrarily, doesn't have to correspond to a binary tree

    def binary_initialize(self):
        self.reset_init()
        for n in range(self.max_N):
            unitary_initialization_idx = self.unitary_initialization_order[n]
            print("\n\noptimizing U%d\n\n" % unitary_initialization_idx)
            self.initial_state = self.initial_state_original
            self.target_state = self.target_state_original
            for U in self.unitaries[:unitary_initialization_idx]:
                self.initial_state = U * self.initial_state
            for U in self.unitaries[:unitary_initialization_idx:-1]:
                self.target_state = U.dag() * self.target_state
            self.betas = np.array([0 + 0j, 0 + 0j], dtype=np.complex128)
            self.alphas = np.array([0 + 0j, 0 + 0j], dtype=np.complex128)
            self.phis = np.array([0, 0], dtype=np.float64)
            self.thetas = np.array([0, 0], dtype=np.float64)
            self.randomize(alpha_scale=0, beta_scale=0.2)
            self.print_info()
            self.optimize()
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            self.params["betas"][unitary_initialization_idx] = self.betas[0]
            self.params["alphas"][unitary_initialization_idx] = self.alphas
            self.params["phis"][unitary_initialization_idx] = self.phis
            self.params["thetas"][unitary_initialization_idx] = self.thetas
            print("params:")
            print(self.params)
            self.fids_reached[n] = self.fidelity()
            self.unitaries[unitary_initialization_idx] = self.U_tot()
            self.N_reached = n + 1  # number of blocks optimized using binaryinit

    def binary_initialize_unitary(self):
        self.reset_init()
        for n in range(self.max_N):
            ind = self.ind_order.index(n)
            inds_i = self.ind_order[:ind]
            inds_f = self.ind_order[ind + 1 :]
            U_i = self.I
            U_f = self.I
            for index in inds_i:
                if index in self.Ucs:
                    U_i = self.Ucs[index] * U_i
            for index in inds_f:
                if index in self.Ucs:
                    U_f = self.Ucs[index] * U_f

            # Tr[U_t.dag() U_f U_n U_i] = Tr[(U_i U_t.dag() U_f) U_n]
            # => U_t' = U_f.dag() U_t U_i.dag()
            self.target_unitary = U_f.dag() * self.target_unitary_original * U_i.dag()
            self.randomize(alpha_scale=0.2, beta_scale=3)
            self.optimize()
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            self.params["betas"][n] = self.betas[0]
            self.params["alphas"][n] = self.alphas[0]
            self.params["phis"][n] = self.phis[0]
            self.params["thetas"][n] = self.thetas[0]
            self.fids_reached[n] = self.unitary_fidelity()
            self.Ucs[n] = self.U_tot()
            self.N_reached = n + 1  # number of blocks optimized using binaryinit

    def concat_controls(self, include_N=None):
        include_N = include_N if include_N is not None else self.N_reached
        self.betas_full = []
        self.alphas_full = []
        self.thetas_full = []
        self.phis_full = []

        for i in self.ind_order:  # TODO adapt for Unitary initialization
            if i in self.params["betas"] and i < include_N:
                self.betas_full.append(self.params["betas"][i])
                self.alphas_full.append(self.params["alphas"][i])
                self.thetas_full.append(self.params["thetas"][i])
                self.phis_full.append(self.params["phis"][i])
