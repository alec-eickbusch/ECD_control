Can instead just loop through the parameters


self.opt_data[timestamp]["timestamp"] = timestamp
        self.opt_data[timestamp]["leaning_rate"] = learning_rate
        self.opt_data[timestamp]["epoch_size"] = epoch_size
        self.opt_data[timestamp]["epochs"] = epochs
        self.opt_data[timestamp]["save_all_parameters"] = save_all_parameters
        self.opt_data[timestamp]["comments"] = comments
        # data as part of the current object
        self.opt_data[timestamp]["name"] = self.parameters["name"]
        self.opt_data[timestamp]["optimization_type"] = self.parameters[
            "optimization_type"
        ]
        self.opt_data[timestamp]["use_displacements"] = self.parameters[
            "use_displacements"
        ]
        self.opt_data[timestamp]["N_blocks"] = self.parameters["N_blocks"]
        self.opt_data[timestamp]["N_multistart"] = self.parameters["N_multistart"]
        self.opt_data[timestamp]["no_CD_end"] = self.parameters["no_CD_end"]
        self.opt_data[timestamp]["N_cav"] = self.parameters["N_cav"]
        self.opt_data[timestamp]["term_fid"] = self.parameters["term_fid"]
        self.opt_data[timestamp]["dfid_stop"] = self.parameters["dfid_stop"]