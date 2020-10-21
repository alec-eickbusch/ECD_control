import numpy as np
import h5py
import datetime
from .batch_optimizer import TIMESTAMP_FORMAT


class OptimizationSweeps:
    def __init__(self, opt_object, filename=None):
        self.opt_object = opt_object
        self.filename = filename if filename is not None else self.opt_object.filename

    def N_blocks_sweep(
        self,
        min_N_blocks=2,
        max_N_blocks=12,
        terminate=True,
        do_print=True,
        name="N_blocks_sweep",
    ):
        best_fid = 0.0
        N_blocks = min_N_blocks
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        sweep_name = name + " " + timestamp
        timestamps = []

        while (N_blocks <= max_N_blocks) and (
            best_fid < self.opt_object.parameters["term_fid"]
        ):
            if do_print:
                print("\nN_blocks: %d\n" % N_blocks)
            self.opt_object.modify_parameters(N_blocks=N_blocks)
            timestamps.append(self.opt_object.optimize())
            best_circuit = self.opt_object.best_circuit()
            if N_blocks == min_N_blocks:
                self.save_sweep_data(sweep_name, timestamps, best_circuit, append=False)
            else:
                self.save_sweep_data(sweep_name, timestamps, best_circuit, append=True)
            if terminate:
                best_fid = best_circuit["fidelity"]
            N_blocks += 1
        return timestamps

    # as your running larger sweeps, you can accumulate data.
    def save_sweep_data(
        self, sweep_name, timestamps, circuit_data, append=False, **kwargs
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
                grp.create_dataset(
                    "fidelities",
                    chunks=True,
                    data=[circuit_data["fidelity"]],
                    maxshape=(None,),
                )
                grp.create_dataset(
                    "betas",
                    data=[circuit_data["betas"]],
                    chunks=True,
                    maxshape=(
                        None,
                        len(circuit_data["betas"]),
                    ),
                )
                grp.create_dataset(
                    "alphas",
                    data=[circuit_data["alphas"]],
                    chunks=True,
                    maxshape=(
                        None,
                        len(circuit_data["alphas"]),
                    ),
                )
                grp.create_dataset(
                    "phis",
                    data=[circuit_data["phis"]],
                    chunks=True,
                    maxshape=(
                        None,
                        len(circuit_data["phis"]),
                    ),
                )
                grp.create_dataset(
                    "thetas",
                    data=[circuit_data["thetas"]],
                    chunks=True,
                    maxshape=(
                        None,
                        len(circuit_data["thetas"]),
                    ),
                )
        else:
            with h5py.File(self.filename, "a") as f:
                f[sweep_name]["fidelities"].resize(
                    f[sweep_name]["fidelities"].shape[0] + 1, axis=0
                )
                f[sweep_name]["betas"].resize(
                    f[sweep_name]["betas"].shape[0] + 1, axis=0
                )
                f[sweep_name]["alphas"].resize(
                    f[sweep_name]["alphas"].shape[0] + 1, axis=0
                )
                f[sweep_name]["phis"].resize(f[sweep_name]["phis"].shape[0] + 1, axis=0)
                f[sweep_name]["thetas"].resize(
                    f[sweep_name]["thetas"].shape[0] + 1, axis=0
                )

                f[sweep_name]["fidelities"][-1] = circuit_data["fidelity"]
                f[sweep_name]["betas"][-1] = circuit_data["betas"]
                f[sweep_name]["alphas"][-1] = circuit_data["alphas"]
                f[sweep_name]["phis"][-1] = circuit_data["phis"]
                f[sweep_name]["thetas"][-1] = circuit_data["thetas"]
                f[sweep_name].attrs["timestamps"] = timestamps
