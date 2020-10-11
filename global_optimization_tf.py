import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import qutip as qt
from datetime import datetime
from CD_control.CD_control_tf import CD_control_tf


class Global_optimizer_tf(CD_control_tf):
    def multistart_optimize(self, N_multistart=10, beta_scale=1.0, **kwargs):
        # initial randomization
        fid = 0.0
        best_fid = fid
        i = 0
        while (i < N_multistart) and (fid < self.term_fid):
            print("\nMultistart N: %d / %d\n" % (i, N_multistart))
            self.randomize(beta_scale)
            fid = self.optimize(*kwargs)
            if fid > best_fid:
                best_fid = fid
                best_betas, best_phis, best_thetas = self.get_numpy_vars()
            i += 1
        print("Best fid found: %.4f" % best_fid)
        self.set_tf_vars(best_betas, best_phis, best_thetas)
        self.print_info()
        return fid

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
                self.multistart_optimize(N_multistart, *kwargs)
                if multistart
                else self.optimize(*kwargs)
            )
            N_blocks += 1
        self.print_info()
        return fid

