from CD_control.CD_control_tf import CD_control_tf
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import binarytree as bt


class CD_control_init(CD_control_tf):

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(
        self,
        max_N=10,
        initial_state=None,
        target_state=None,
        target_unitary=None,
        N_blocks=1,
        betas=None,
        alphas=None,
        phis=None,
        thetas=None,
        max_alpha=5,
        max_beta=5,
        saving_directory=None,
        name="CD_control",
        term_fid=0.999,
        beta_penalty_multiplier=0,
        use_displacements=True,
        unitary_optimization=False,
        no_CD_end=True,
        ind_order=None,
    ):
        CD_control_tf.__init__(
            self,
            initial_state=initial_state,
            target_state=target_state,
            target_unitary=target_unitary,
            N_blocks=1,
            max_alpha=max_alpha,
            max_beta=max_beta,
            saving_directory=saving_directory,
            name=name,
            term_fid=term_fid,
            beta_penalty_multiplier=beta_penalty_multiplier,
            use_displacements=use_displacements,
            unitary_optimization=unitary_optimization,
            no_CD_end=True,
        )

        self.max_N = max_N
        self.initial_state_original = self.initial_state
        self.target_state_original = self.target_state
        self.target_unitary_original = self.target_unitary
        self.ind_order = ind_order

        if self.ind_order is None:
            self.reset_init()

    def reset_init(self):
        self.fids_reached = {}
        self.Ucs = {}  # key: index representing order unitary was added in, val: U
        self.params = {"betas": {}, "alphas": {}, "phis": {}, "thetas": {}}

        # TODO: make balanced
        self.ind_order = [x.val for x in bt.build(list(range(self.max_N))).inorder]
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

            # (psi_f.dag() U_f) U_n (U_i psi_i) => psi_f' = U_f.dag() * psi_f
            self.initial_state = U_i * self.initial_state_original
            self.target_state = U_f.dag() * self.target_state_original
            self.randomize(alpha_scale=0.2, beta_scale=3)
            self.optimize()
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            self.params["betas"][n] = self.betas[0]
            self.params["alphas"][n] = self.alphas[0]
            self.params["phis"][n] = self.phis[0]
            self.params["thetas"][n] = self.thetas[0]
            self.fids_reached[n] = self.fidelity()
            self.Ucs[n] = self.U_tot()
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
