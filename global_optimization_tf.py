import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import qutip as qt
from datetime import datetime
from CD_control.CD_control_tf import CD_control_tf


class Global_optimizer_tf(CD_control_tf):
    def multistart_optimize(self, N_multistart=10, beta_scale=1.0, **kwargs):
        # initial randomization
        all_losses = []
        loss = 0
        best_loss = 0
        i = 0
        term_loss = np.log(1 - self.term_fid)
        while (i < N_multistart) and (loss > term_loss):
            print("\nMultistart N: %d / %d\n" % (i, N_multistart))
            self.randomize(beta_scale)
            losses = self.optimize(**kwargs)
            all_losses.append(losses)
            loss = losses[-1]
            if loss < best_loss:
                best_loss = loss
                best_betas, best_phis, best_thetas = self.get_numpy_vars()
            i += 1
        print(
            "Best found: loss =  %.4f fid = %.4f" % (best_loss, 1 - np.exp(best_loss))
        )
        self.set_tf_vars(best_betas, best_phis, best_thetas)
        self.print_info()
        return all_fids

    def N_blocks_sweep(
        self,
        min_N_blocks=2,
        max_N_blocks=12,
        multistart=True,
        N_multistart=10,
        beta_scale=1.0,
        **kwargs
    ):
        fid = 0.0
        N_blocks = min_N_blocks
        while (N_blocks < max_N_blocks) and (fid < self.term_fid):
            print("\nN_blocks: %d\n" % N_blocks)
            self.N_blocks = N_blocks
            self.randomize(beta_scale)
            fid = (
                self.multistart_optimize(N_multistart, **kwargs)
                if multistart
                else self.optimize(*kwargs)
            )
            N_blocks += 1
        self.print_info()
        return fid

    # oftentimes, after an optimization, values of angles will be close to 90,
    # and values of betas will be close to 0.
    # find values close to these values, "pin" them to the expected values,
    # then optimize the others.

    # TODO: could also pin angle of beta if it's 0 or 90...
    def pin_optimize(self, tolerance=0.01, **kwargs):
        betas, phis, thetas = self.get_numpy_vars()
        beta_mask = np.ones(self.N_blocks)
        phi_mask = np.ones(self.N_blocks)
        theta_mask = np.ones(self.N_blocks)
        phi_mask[0] = 0
        if self.no_CD_end:
            beta_mask[-1] = 0

        for i, beta in enumerate(betas):
            if np.abs(beta) <= tolerance:
                betas[i] = 0.0
                beta_mask[i] = 0

        # todo: could do this in much more efficint way... :D
        for i, phi in enumerate(phis):
            if np.abs(phi) <= tolerance:
                phis[i] = 0.0
                phi_mask[i] = 0
            if np.abs(phi - np.pi / 2.0) <= tolerance:
                phis[i] = np.pi / 2.0
                phi_mask[i] = 0
            if np.abs(phi - np.pi) <= tolerance:
                phis[i] = np.pi
                phi_mask[i] = 0
            if np.abs(phi + np.pi / 2.0) <= tolerance:
                phis[i] = -np.pi / 2.0
                phi_mask[i] = 0
            if np.abs(phi + np.pi) <= tolerance:
                phis[i] = -np.pi
                phi_mask[i] = 0

        for i, theta in enumerate(thetas):
            if np.abs(theta) <= tolerance:
                thetas[i] = 0.0
                theta_mask[i] = 0
            if np.abs(theta - np.pi / 2.0) <= tolerance:
                thetas[i] = np.pi / 2.0
                theta_mask[i] = 0
            if np.abs(theta - np.pi) <= tolerance:
                thetas[i] = np.pi
                theta_mask[i] = 0
            if np.abs(theta + np.pi / 2.0) <= tolerance:
                thetas[i] = -np.pi / 2.0
                theta_mask[i] = 0
            if np.abs(theta + np.pi) <= tolerance:
                thetas[i] = -np.pi
                theta_mask[i] = 0

        self.set_tf_vars(betas, phis, thetas)
        self.optimize(
            beta_mask=beta_mask, phi_mask=phi_mask, theta_mask=theta_mask, **kwargs
        )
