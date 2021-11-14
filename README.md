# Echoed Conditional Displacement (ECD) Control

 Fast universal control of an oscillator with a weak dispersive coupling to a qubit.

--- 

Welcome to the ECD control optimization package. Built with Tensorflow in python.

This repo can be used to optimize circuit parameters and generate ECD pulse sequences to be used in an experiment. 
Requirements: qutip (4.0.0 or later), Tensorflow (2.3.0 or later), h5py

### See examples folder for basic usage

--- 
Two modules are contained in this git:

1. **ECD_optimization**
    Optimization of conditional displacement and qubit rotation circuit parameters for a quantum control problem. Built with Tensorflow.
    
2. **ECD_pulse_construction**
    Complies oscillator ($\varepsilon(t)$) and qubit ($\Omega()) pulse sequences from a set of ECD parameters

contact: [alec.eickbusch@yale.edu](mailto:alec.eickbusch@yale.edu)

 The codebase, examples, and documentation is under active developement. 

