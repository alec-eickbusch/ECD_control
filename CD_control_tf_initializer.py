from CD_control.CD_control_tf import CD_control_tf
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import binarytree as bt
import CD_control.tf_quantum as tfq
import tensorflow as tf


class CD_control_init_tf(CD_control_tf):

    # a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(self, ind_order=None, **kwargs):
        CD_control_tf.__init__(self, **kwargs)

        self.max_N = self.N_blocks
        self.N_reached = None
        self.initial_state_original = self.initial_state
        self.target_state_original = self.target_state
        self.target_unitary_original = self.target_unitary
        self.ind_order = ind_order

    def reset_binary_init(self):
        self.N_blocks = 1
        self.no_CD_end = False
        self.fids_reached = {}
        self.Ucs = {}  # key: index representing order unitary was added in, val: U
        self.params = {"betas": {}, "alphas": {}, "phis": {}, "thetas": {}}

        # TODO: make balanced
        self.ind_order = (
            self.ind_order
            if self.ind_order is not None
            else [x.val for x in bt.build(list(range(self.max_N))).inorder]
        )
        # [3, 1, 4, 0, 5, 2, 6] corresponds to psi_f.dag() U_6 U_2 U_5 U_0 U_4 U_1 U_3 psi_i,
        # where each j index in U_j reprents the order in which the blocks were added in to the greedy optimization
        # i.e.
        # psi_f.dag() U_0 psi_i,
        # psi_f.dag() U_0 U_1 psi_i,
        # psi_f.dag() U_2 U_0 U_1 psi_i,
        # psi_f.dag() U_2 U_0 U_4 U_1 psi_i, => optimizing U_4, and psi_f'.dag() U_4 psi_i'is the state transfer problem
        # ordering can be set arbitrarily, doesn't have to correspond to a binary tree

    def binary_initialize(self):
        self.unitary_optimization = False
        self.reset_binary_init()
        for n in range(self.max_N):
            print("==========================================================")
            print("Number of Blocks being used : " + str(n + 1))
            ind = self.ind_order.index(n)
            inds_i = self.ind_order[:ind]
            inds_f = self.ind_order[ind + 1 :]
            U_i = self.I
            U_f = self.I
            for index in inds_i:
                if index in self.Ucs:
                    U_i = self.Ucs[index] @ U_i
            for index in inds_f:
                if index in self.Ucs:
                    U_f = self.Ucs[index] @ U_f

            # (psi_f.dag() U_f) U_n (U_i psi_i) => psi_f' = U_f.dag() @ psi_f
            self.initial_state = U_i @ self.initial_state_original
            self.target_state = tf.linalg.adjoint(U_f) @ self.target_state_original
            self.randomize(beta_scale=3)
            self.optimize(epochs=100, epoch_size=10, dloss_stop=1e-6)
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            betas, phis, thetas = self.get_numpy_vars()
            (
                self.params["betas"][n],
                self.params["phis"][n],
                self.params["thetas"][n],
            ) = (betas[0], phis[0], thetas[0])
            self.fids_reached[n] = self.state_fidelity(
                self.betas_rho, self.betas_angle, self.phis, self.thetas
            ).numpy()
            self.Ucs[n] = self.U_tot(
                self.betas_rho, self.betas_angle, self.phis, self.thetas
            )
            self.N_reached = n + 1  # number of blocks optimized using binaryinit
            print("==========================================================")

    def binary_initialize_unitary(self):
        self.unitary_optimization = True
        self.reset_binary_init()
        for n in range(self.max_N):
            print("==========================================================")
            print("Number of Blocks being used : " + str(n + 1))
            ind = self.ind_order.index(n)
            inds_i = self.ind_order[:ind]
            inds_f = self.ind_order[ind + 1 :]
            U_i = self.I
            U_f = self.I
            for index in inds_i:
                if index in self.Ucs:
                    U_i = self.Ucs[index] @ U_i
            for index in inds_f:
                if index in self.Ucs:
                    U_f = self.Ucs[index] @ U_f

            # Tr[U_t.dag() U_f U_n U_i] = Tr[(U_i U_t.dag() U_f) U_n]
            # => U_t' = U_f.dag() U_t U_i.dag()
            self.target_unitary = (
                tf.linalg.adjoint(U_f)
                @ self.target_unitary_original
                @ tf.linalg.adjoint(U_i)
            )
            self.randomize(alpha_scale=0.2, beta_scale=3)
            self.optimize(epochs=100, epoch_size=10, dloss_stop=1e-6)
            self.print_info()
            # TODO add some stop condition if max fid isn't reached
            betas, phis, thetas = self.get_numpy_vars()
            (
                self.params["betas"][n],
                self.params["phis"][n],
                self.params["thetas"][n],
            ) = (betas[0], phis[0], thetas[0])
            self.fids_reached[n] = self.unitary_fidelity(
                self.betas_rho, self.betas_angle, self.phis, self.thetas
            ).numpy()
            self.Ucs[n] = self.U_tot(
                self.betas_rho, self.betas_angle, self.phis, self.thetas
            ).numpy()
            self.N_reached = n + 1  # number of blocks optimized using binaryinit
            print("==========================================================")

    def concat_controls(self, include_N=None):
        include_N = include_N if include_N is not None else self.N_reached
        self.betas_full = []
        self.thetas_full = []
        self.phis_full = []

        for i in self.ind_order:  # TODO adapt for Unitary initialization
            if i in self.params["betas"] and i < include_N:
                self.betas_full.append(self.params["betas"][i])
                self.thetas_full.append(self.params["thetas"][i])
                self.phis_full.append(self.params["phis"][i])
        self.betas_full = np.array(self.betas_full)
        self.thetas_full = np.array(self.thetas_full)
        self.phis_full = np.array(self.phis_full)

    def state_initialize_unitary(self):
        print("==========================================================")
        print("Single State Transfer Unitary Initializatoin\n")
        self.initial_state = self.initial_state_original
        self.target_state = self.target_state_original
        self.unitary_optimization = False
        self.randomize(beta_scale=3)
        self.optimize(epochs=100, epoch_size=10, dloss_stop=1e-6)
        self.print_info()
        betas, phis, thetas = self.get_numpy_vars()
        self.betas_full = betas
        self.phis_full = phis
        self.thetas_full = thetas
        print("==========================================================")

    def multi_state_initialize_unitary(self, states):
        print("==========================================================")
        print("Multi State Transfer Unitary Initializatoin\n")
        self.set_unitary_fidelity_state_basis(states)
        self.unitary_optimization = "states"
        self.randomize(beta_scale=3)
        self.optimize(epochs=100, epoch_size=10, dloss_stop=1e-6)
        self.print_info()
        betas, phis, thetas = self.get_numpy_vars()
        self.betas_full = betas
        self.phis_full = phis
        self.thetas_full = thetas
        print("==========================================================")

    def initialized_obj(self, unitary_optimization=None, **kwargs):
        return CD_control_tf(
            target_unitary=self.target_unitary_original,
            initial_state=self.initial_state_original,
            target_state=self.target_state_original,
            unitary_optimization=unitary_optimization
            if unitary_optimization is not None
            else self.unitary_optimization,
            P_cav=self.P_cav,
            betas=self.betas_full,
            thetas=self.thetas_full,
            phis=self.phis_full,
            N_blocks=self.N_blocks if self.N_reached is None else self.N_reached,
            no_CD_end=self.no_CD_end,
            name=self.name,
            term_fid=self.term_fid,
            saving_directory=self.saving_directory,
            **kwargs
        )

