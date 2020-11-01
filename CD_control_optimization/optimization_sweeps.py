import numpy as np
import h5py
import datetime
from .batch_optimizer import TIMESTAMP_FORMAT


class OptimizationSweeps:
    def __init__(self, sweep_param_names=[], filename="sweep"):
        # setup filename
        self.filename = filename
        path = self.filename.split(".")
        if len(path) < 2 or (len(path) == 2 and path[-1] != ".h5"):
            self.filename = path[0] + ".h5"

        # Setup Sweep
        timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)
        self.sweep_param_names = ["N_blocks"] + sweep_param_names
        sweep_param_names_str = " ".join(self.sweep_param_names)
        self.sweep_name = "sweep " + sweep_param_names_str + " " + timestamp

    def N_blocks_sweep(
        self,
        opt_object,
        sweep_param_values=[],
        min_N_blocks=2,
        max_N_blocks=12,
        beta_scale_function=None,
        alpha_scale_function=None,
        terminate=True,
        do_prints=True,
    ):
        if len(sweep_param_values) != len(self.sweep_param_names) - 1:
            raise Exception(
                "Please enter matching sweep_param_values to "
                + str(self.sweep_param_names)
                + "!"
            )
        opt_object.filename = self.filename

        # Initialize
        best_fid = 0.0
        N_blocks = min_N_blocks

        # N_block dependent random initialization
        beta_scale_function = (
            beta_scale_function
            if beta_scale_function is not None
            else lambda N_blocks: opt_object.parameters["beta_scale"]
        )
        alpha_scale_function = (
            alpha_scale_function
            if alpha_scale_function is not None
            else lambda N_blocks: opt_object.parameters["alpha_scale"]
        )

        sweep_param_values_base = sweep_param_values
        # Loop through N blocks
        print("\nstarting N blocks sweep")
        while (N_blocks <= max_N_blocks) and (
            best_fid < opt_object.parameters["term_fid"]
        ):
            sweep_param_values = [N_blocks] + sweep_param_values_base
            if self.is_already_optimized(
                sweep_param_values
            ):  # useful when restarting sweep
                N_blocks += 1
                continue
            print(
                "\n"
                + str(self.sweep_param_names)
                + " sweep: "
                + str(sweep_param_values)
            )
            print(str(self.sweep_param_names) + " sweep filename: " + self.filename)
            print(
                str(self.sweep_param_names) + " sweep name: " + self.sweep_name + "\n"
            )

            beta_scale = beta_scale_function(N_blocks)
            alpha_scale = alpha_scale_function(N_blocks)
            opt_object.modify_parameters(
                N_blocks=N_blocks, beta_scale=beta_scale, alpha_scale=alpha_scale
            )

            timestamp = opt_object.optimize(do_prints=do_prints)
            all_fidelities = opt_object.all_fidelities()

            self.save_sweep_data(
                timestamp, all_fidelities, sweep_param_values=sweep_param_values,
            )

            if terminate:
                best_fid = np.max(all_fidelities)

            N_blocks += 1
        print("\n N_blocks sweep finished.")
        print("N blocks sweep filename: " + self.filename)
        print("N blocks sweep name: " + self.sweep_name + "\n")

    def is_already_optimized(self, sweep_param_values):
        with h5py.File(self.filename, "a") as f:
            if self.sweep_name in f:
                return np.any(
                    np.equal(
                        np.array(f[self.sweep_name]["sweep_param_values"]),
                        sweep_param_values,
                    ).all(1)
                )  # checks if row of sweep_param_values is in np array f[self.sweep_name]["sweep_param_values"] already
        return False

    # as your running larger sweeps, you can accumulate data.
    def save_sweep_data(self, timestamp, fidelities, sweep_param_values=[], **kwargs):
        # Updating: timestamps, sweep_params_labels, fidelities
        # No need to update: sweep_params_names
        with h5py.File(self.filename, "a") as f:
            if self.sweep_name not in f:
                grp = f.create_group(self.sweep_name)
                for parameter, value in kwargs.items():
                    grp.attrs[parameter] = value
                grp.attrs["sweep_param_names"] = self.sweep_param_names

                grp.attrs["timestamps"] = [timestamp]
                grp.create_dataset(
                    "fidelities", chunks=True, data=[fidelities], maxshape=(None, None),
                )
                grp.create_dataset(
                    "sweep_param_values",
                    chunks=True,
                    data=[sweep_param_values],
                    maxshape=(None, None),
                )
            else:
                f[self.sweep_name].attrs["timestamps"] = list(
                    f[self.sweep_name].attrs["timestamps"]
                ) + [timestamp]

                f[self.sweep_name]["fidelities"].resize(
                    f[self.sweep_name]["fidelities"].shape[0] + 1, axis=0
                )
                f[self.sweep_name]["sweep_param_values"].resize(
                    f[self.sweep_name]["sweep_param_values"].shape[0] + 1, axis=0
                )
                f[self.sweep_name]["fidelities"][-1] = fidelities
                f[self.sweep_name]["sweep_param_values"][-1] = sweep_param_values
