from CD_GRAPE.cd_grape_optimization import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import binarytree as bt


class CD_grape_init(CD_grape):

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
        name="CD_grape",
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
    ):
        CD_grape.__init__(
            self,
            initial_state=initial_state,
            target_state=target_state,
            target_unitary=target_unitary,
            N_blocks=1,
            unitary_optimization=False,
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
            no_CD_end=False,
            circuits=circuits,
            N=N,
            N2=N2,
        )
        self.max_N = max_N
        self.initial_state_original = self.initial_state
        self.final_state_original = self.final_state
        self.reset_init()

    def reset_init(self):
        self.fids_reached = {}
        self.Ucs = {}  # key: index representing order unitary was added in, val: U
        self.params = {"betas": {}, "alphas": {}, "phis": {}, "thetas": {}}
        self.ind_order = [x.val for x in bt.build(list(range(self.max_N))).inorder]

    def binary_initialize(self):
        self.reset_init()
        for n in range(self.max_N):  # TODO adapt for Unitary initialization
            ind = self.ind_order.index(n)
            inds_i = self.ind_order[ind + 1 :]
            inds_f = self.ind_order[:ind]
            U_i = self.I
            U_f = self.I
            for index in inds_i[::-1]:
                if index in self.Ucs:
                    U_i = self.Ucs[index] * U_i
            for index in inds_f:
                if index in self.Ucs:
                    U_f = self.Ucs[index] * U_f
            self.initial_state = U_i * self.initial_state_original
            self.final_state = U_f * self.final_state_original
            self.randomize(alpha_scale=0.2, beta_scale=3)
            self.optimize()
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            self.params["betas"][n] = self.betas[0]
            self.params["alphas"][n] = self.alphas[0]
            self.params["phis"][n] = self.phis[0]
            self.params["thetas"][n] = self.thetas[0]
            self.fids_reached[n] = self.fidelity()
            self.N_reached = n + 1

    def concat_controls(self, include_N=None):
        include_N = include_N if include_N is not None else self.N_reached
        self.betas_full = []
        self.alphas_full = []
        self.thetas_full = []
        self.phis_full = []

        for i in self.ind_order[::-1]:  # TODO adapt for Unitary initialization
            if i in self.params["betas"] and i < include_N:
                self.betas_full.append(self.params["betas"][i])
                self.alphas_full.append(self.params["alphas"][i])
                self.thetas_full.append(self.params["thetas"][i])
                self.phis_full.append(self.params["phis"][i])
