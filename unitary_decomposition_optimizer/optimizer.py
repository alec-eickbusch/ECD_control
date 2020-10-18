from .local_optimizer import LocalOptimizer
from .global_optimizer import GlobalOptimizer
from .initalizer import Initalizer
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import binarytree as bt
import unitary_decomposition_optimizer.tf_quantum as tfq
import tensorflow as tf


class Optimizer(Initalizer, GloberOptimizer, VisualizationMixin):
    """
    Main optimization class for interface with the user.

    Attributes
    ----------
    todo.

    Methods
    -------
    todo
    """

    # TODO: use typehinting?
    # question: how to handle multiple types in the docstrings?
    def __init__(
        self,
        optimization_type="state transfer",
        target_unitary=None,
        P_cav=None,
        initial_states=None,
        target_states=None,
        expectation_operators=None,
        target_expectation_values=None,
        max_N_blocks=20,
        desired_epsilon=1e-4,
        saving_directory=None,
        name="CD_control",
        max_beta=10,
        max_alpha=10,
        use_displacements=True,
        no_CD_end=True,
        optimize_expectation=False,
        O=None,
    ):
        """
        initialization of optimization object

        Parameters
        ----------
        optimization type: str, {'state transfer', 'unitary', 'expectation'}
            picks the type of optimization problem.
        target_unitary : qutip object, optional
            the target unitary for unitary optimization
        P_cav : int, optional
            the number of fock states to include in the
            projector matrix for unitary optimization
        initial_states : list of qutip objects, optional
            set of initial states for state transfer optimization
            each initial state should have an associated final state
        target_states : list of qutip objects, optional
            set of target states for state transfer optimization
            each target state has an associated initial state
        expectation_operators: list of qutip objects, optional
            set of operators to optimize the expectation value of.
        target_expectation_values: list of floats, optional
            target values for the expectation values
        max_N_blocks : int, optional
            the maximum number of blocks to be tried
            in the quantum circuit
        desired_epsilon: float, optional
            the desired error in the decomposition of the target operation.
            For example, if desired_epsilon = 1e-3, then the optimizer
            will attempt to construct a quantum circuit which has a
            fidelity of 1-epsilon = 0.999 to the target operation
        saving_directory: str, optional
            Where to save optimizer data.
            example: r'C:\\opt_data\\'
        name: str, optional
            name of the optimization task
        max_beta: float, optional
            maximum conditional displacement amplitude to use during optimization
        max_alpha: float, optional
            maximum displacement amplitude to use during optimization
        use_displacements: bool, optional
            if true, alphas will be optimized. If false, they will be locked to 0.
        no_CD_end: bool, optional
            if true, the final block will not contain a conditional displacement.
        """
        self.optimization_type = optimization_type
        if self.optimization_type == "state transfer":
            self.initial_states = initial_states
            self.target_states = target_states
        elif self.optimization_type == "unitary":
            self.target_unitary = target_unitary
            self.P_cav = P_cav
        elif self.optimization_type == "expectation":
            self.expectation_operators = expectation_operators
            self.target_expectation_values = target_expectation_values
            self.initial_states = initial_states
        else:
            raise ValueError(
                "optimization_type must be one of \{'state transfer', 'unitary', 'expectation'\}"
            )
        self.name = name
        self.saving_directory = saving_directory
        self.max_N_blocks = max_N_blocks
        self.desired_epsilon = desired_epsilon
        self.use_displacements = use_displacements
        self.no_CD_end = no_CD_end
        self.local_optimizers = []

        # here, run the local optimization tasks.
        # later, can use mpi to request resources
        # and run in parallel.
        # for now, they are run in series.
        def _run_local_optimizers(self):
            for local_optimizer in self.local_optimizers:
                local_optimizer.run()
