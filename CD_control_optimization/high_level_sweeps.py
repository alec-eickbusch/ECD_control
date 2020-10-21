import numpy as np


class OptimizationSweeps:
    def __init__(self, opt_object):
        self.opt_object = opt_object

    def N_blocks_sweep(
        self,
        min_N_blocks=2,
        max_N_blocks=12,
        terminate=True,
        do_print=True,
    ):
        best_fid = 0.0
        N_blocks = min_N_blocks

        timestamps = []

        while (N_blocks <= max_N_blocks) and (
            best_fid < self.opt_object.parameters["term_fid"]
        ):
            if do_print:
                print("\nN_blocks: %d\n" % N_blocks)
            self.opt_object.modify_parameters(N_blocks=N_blocks)
            timestamps.append(self.opt_object.optimize())
            if terminate:
                best_fid = self.opt_object.best_fidelity()
            N_blocks += 1
        return timestamps
