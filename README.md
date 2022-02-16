# Echoed Conditional Displacement (ECD) Control



Welcome to the Echoed Conditional Displacement (*ECD*) control package! Built with Tensorflow in python. 

ECD control is a fast, echoed, gate-based approach to the quantum control of an oscillator with weak dispersive coupling to a qubit.

Based on the paper **Fast universal control of an oscillator with a weak dispersive coupling to a qubit (2021)** [arXiv:2111.06414](https://arxiv.org/abs/2111.06414).

This repository can be used to optimize circuit parameters and generate ECD pulse sequences to be used in an experiment.

---
## Requirements
[qutip](https://qutip.org/) (4.0.0 or later), [Tensorflow](https://www.tensorflow.org/) (2.3.0 or later), [h5py](https://www.h5py.org/) (working with 3.1.0)

---
## Installation
To install, clone this repository and run:
```sh
$ pip install -e ECD_control
```



---
## Usage

Given a quantum control problem, optimization is performed in two steps:

1.  **ECD_optimization**
    Optimization of ECD circuit parameters (betas, phis, thetas) for a quantum control problem. This step does not depend on device-specic parameters. Built with tensorflow
    

2. **ECD_pulse_construction**
    Given device-specific parameters, this step complies oscillator and qubit pulses from the ECD circuit parameters found in step 1.

**Please see examples folder for more information. Current documentation is contained in these examples.**

---

For any issues, comments, or questions, please open a github issue or contact: [alec.eickbusch@yale.edu](mailto:alec.eickbusch@yale.edu).
Note: The codebase, examples, and documentation is under active developement.

