import numpy as np
import h5py
import datetime
from .batch_optimizer import TIMESTAMP_FORMAT


class OptimizationSweeps:
    def __init__(self, opt_object):
        self.opt_object = opt_object
        self.filename = self.opt_object.filename
        self.sweep_names = []

    def N_blocks_sweep(
        self,
        min_N_blocks=2,
        max_N_blocks=12,
        beta_scale_function=None,
        alpha_scale_function=None,
        terminate=True,
        do_prints=True,
    ):
        best_fid = 0.0
        N_blocks = min_N_blocks
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        sweep_name = "sweep N_blocks " + timestamp
        self.sweep_names.append(sweep_name)
        sweep_param_name = "Number of Blocks"
        timestamps = []
        beta_scale_function = (
            beta_scale_function
            if beta_scale_function is not none
            else lambda N_blocks: self.opt_object.beta_scale
        )
        beta_scale_function = (
            alpha_scale_function
            if alpha_scale_function is not none
            else lambda N_blocks: self.opt_object.beta_scale
        )
        print("\nstarting N blocks sweep")
        while (N_blocks <= max_N_blocks) and (
            best_fid < self.opt_object.parameters["term_fid"]
        ):
            print("\nN_blocks: %d" % N_blocks)
            print("N blocks sweep filename: " + self.filename)
            print("N blocks sweep name: " + sweep_name + "\n")
            beta_scale = beta_scale_function(N_blocks)
            alpha_scale = alpha_scale_function(N_blocks)
            self.opt_object.modify_parameters(
                N_blocks=N_blocks, beta_scale=beta_scale, alpha_scale=alpha_scale
            )
            timestamps.append(self.opt_object.optimize(do_prints=do_prints))
            best_circuit = self.opt_object.best_circuit()
            if N_blocks == min_N_blocks:
                self.save_sweep_data(
                    sweep_name,
                    timestamps,
                    best_circuit,
                    append=False,
                    sweep_param_name=sweep_param_name,
                    sweep_param_value=N_blocks,
                )
            else:
                self.save_sweep_data(
                    sweep_name,
                    timestamps,
                    best_circuit,
                    append=True,
                    sweep_param_name=sweep_param_name,
                    sweep_param_value=N_blocks,
                )
            if terminate:
                best_fid = best_circuit["fidelity"]
            N_blocks += 1
        print("\n N_blocks sweep finished.")
        print("N blocks sweep filename: " + self.filename)
        print("N blocks sweep name: " + sweep_name + "\n")
        return sweep_name

    # as your running larger sweeps, you can accumulate data.
    def save_sweep_data(
        self,
        sweep_name,
        timestamps,
        circuit_data,
        append=False,
        sweep_param_name="sweep_parameter",
        sweep_param_value=0,
        **kwargs
    ):
        timestamp_str = timestamps[0]
        if len(timestamps) > 1:
            for t in timestamps:
                timestamp_str = timestamp_str + "," + t
        if not append:
            with h5py.File(self.filename, "a") as f:
                grp = f.create_group(sweep_name)
                for parameter, value in kwargs.items():
                    grp.attrs[parameter] = value
                grp.attrs["timestamps"] = timestamps
                grp.attrs["sweep_param_name"] = sweep_param_name
                grp.create_dataset(
                    "fidelities",
                    chunks=True,
                    data=[circuit_data["fidelity"]],
                    maxshape=(None,),
                )
                grp.create_dataset(
                    "sweep_param_values",
                    chunks=True,
                    data=[sweep_param_value],
                    maxshape=(None,),
                )
        else:
            with h5py.File(self.filename, "a") as f:
                f[sweep_name]["fidelities"].resize(
                    f[sweep_name]["fidelities"].shape[0] + 1, axis=0
                )
                f[sweep_name]["sweep_param_values"].resize(
                    f[sweep_name]["sweep_param_values"].shape[0] + 1, axis=0
                )
                f[sweep_name]["fidelities"][-1] = circuit_data["fidelity"]
                f[sweep_name]["sweep_param_values"][-1] = sweep_param_value
                f[sweep_name].attrs["timestamps"] = timestamps