#%%
import sys

sys.path.append("../../")
from CD_control.CD_control_optimization import CD_control
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

#%%
N = 50  # cavity hilbert space
N2 = 2  # qubit hilbert space
beta = 2  # conditional dispalcement
N_blocks = 1  # only doing 1 conditional displacement
initial_state = qt.tensor(qt.basis(N, 0), (qt.basis(N2, 0) + qt.basis(N2, 1)).unit())
target_state = (
    qt.tensor(qt.coherent(N, beta / 2.0), qt.basis(N2, 1))
    + qt.tensor(qt.coherent(N, -beta / 2.0), qt.basis(N2, 0))
).unit()
# note above that the |e> state will shift the cavity right, and the |g> state
# will shift the cavity left. this is due to the pi pulse during the CD
# which flips the qubit polarity.
CD_control_obj = CD_control(
    initial_state, target_state, N_blocks, name="Conditional Displacement"
)
#%% We can plot the initial and target states (qubit traced out)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_initial_state()
plt.title("initial state")
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_target_state()
plt.title("target state")
#%% First, we trivially get to the target state with a single conditional displacement
CD_control_obj.alphas = [0, 0]
CD_control_obj.betas = [-beta]
CD_control_obj.phis = [0, 0]
CD_control_obj.thetas = [0, 0]
CD_control_obj.print_info()
#%% And we can plot the final cavity state with these parameters
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("final state")
#%% Now, we can convert these parameters to a pulse we can run on the experiment
# first, creating a system object
epsilon_m = 2 * np.pi * 1e-3 * 300.0  # maximum displacement rate
alpha0 = 50  # maximum displacement before things break down
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2 * np.pi) * Ec_GHz
chi_MHz = 0.03
chi = 2 * np.pi * 1e-3 * chi_MHz
sigma = 6  # sigma for gaussian pulses
chop = 4  # chop for gaussian pulses
buffer_time = 4  # time between discrete pulses
ring_up_time = 16  # Time to ring up for large CD pulses
sys = System(
    chi=chi,
    Ec=Ec,
    alpha0=alpha0,
    sigma=sigma,
    chop=chop,
    epsilon_m=epsilon_m,
    buffer_time=buffer_time,
    ring_up_time=ring_up_time,
)
#%% Now, an analysis object
analysis_obj = CD_control_analysis(CD_control_obj, sys)
#%% Given these parameters, we can plot the composite pulse (fastest CD given these parameters)
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
#%% And now we can simulate this pulse on the system
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, target_state)
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
#%%
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("Simulated final state")
#%%
# Now, showing we can get to the same result through optimization.
CD_control_obj.randomize(alpha_scale=0.1, beta_scale=1.0)
print("Randomized parameters:")
CD_control_obj.print_info()
# The alpha and beta scale are scales for the random initialization.
CD_control_obj.optimize()
print("after optimization:")
CD_control_obj.print_info()
# you may notice that there is now an unnecessary cavity displacement (then displace back)
# or qubit rotation in the parameters. This is because the penalties are not yet implemented.
# %%  And finally we can simulate the pulse on the system.
# Since the parameters are connected to the CD_control_obj, we don't have to
# re-initialize or anything.
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, target_state)
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("Simulated final state")

# %%
