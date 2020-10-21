import numpy as np


class OptimizationSweeps:
    def __init__(opt_object):
        self.opt_object = opt_object

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
