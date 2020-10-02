#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../../")
from CD_GRAPE.cd_grape_optimization import CD_grape
from CD_GRAPE.helper_functions import plot_pulse, plot_wigner
from CD_GRAPE.analysis import System, CD_grape_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 20 #cavity hilbert space 
N2 = 2 #qubit hilbert space
N_blocks = 1
term_fid = 0.99
#max alpha and beta are the maximum values of alpha and beta for optimization
max_alpha = 4
max_beta = 20
name = "unitary testing"
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_grape_parameters\\"
#TODO: can we control the step size in the local optimizations?
#TODO: Are there few enough parameters we can do more of a global search? 
# Just sweep each parameter over its range?
cd_grape_obj = CD_grape(N_blocks = N_blocks, unitary_optimization=True, no_CD_end=False,
                    name=name, term_fid=term_fid, analytic=False,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    beta_r_step_size=0.5, alpha_r_step_size=0.2,
                    saving_directory=saving_directory,
                    use_displacements=True, N = N, N2=N2,
                    beta_penalty_multiplier=0,
                    minimizer_options = {'gtol': 1e-4, 'ftol':1e-4}, basinhopping_kwargs={'niter':1000, 'T':0.01})
#%%
beta = 2.0
cd_grape_obj.target_unitary = cd_grape_obj.CD(beta)
# parity = (1j*np.pi*cd_grape_obj.a.dag()*cd_grape_obj.a*cd_grape_obj.sz).expm()
# cd_grape_obj.target_unitary = parity
n = N*N2//3
unitary_eigvals = np.array(cd_grape_obj.target_unitary.eigenstates()[0])[:n]
cd_grape_obj.unitary_initial_states = cd_grape_obj.target_unitary.eigenstates()[1][:n]
cd_grape_obj.unitary_final_states = cd_grape_obj.unitary_initial_states*unitary_eigvals
cd_grape_obj.unitary_learning_rate = 1
#%% Doing the optimization
#The alpha and beta scale are scales for the random initialization.
cd_grape_obj.randomize(alpha_scale=0.2, beta_scale=3)
#cd_grape_obj.betas = [-2.0]
print("Randomized parameters:")
cd_grape_obj.print_info()
cd_grape_obj.optimize()
print("after optimization:")
cd_grape_obj.print_info()

#%% Now, we can convert these parameters to a pulse we can run on the experiment
#first, creating a system object
epsilon_m = 2*np.pi*1e-3*300.0  # maximum displacement rate
alpha0 = 50  # maximum displacement before things break down
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2*np.pi) * Ec_GHz
chi_MHz = 0.03
chi = 2*np.pi*1e-3*chi_MHz
sigma = 6  # sigma for gaussian pulses
chop = 4  # chop for gaussian pulses
buffer_time = 4  # time between discrete pulses
ring_up_time = 16  # Time to ring up for large CD pulses
sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
             sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
             ring_up_time=ring_up_time)
analysis_obj = CD_grape_analysis(cd_grape_obj, sys)
#%%  The composite pulse
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
#%%
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, target_state)
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")

#%% Going step by step
cd_grape_obj.N_blocks = 0
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, cd_grape_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), cd_grape_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), cd_grape_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(cd_grape_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
cd_grape_obj.N_blocks = 1
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, cd_grape_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), cd_grape_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), cd_grape_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(cd_grape_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
cd_grape_obj.N_blocks = 2
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, cd_grape_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), cd_grape_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), cd_grape_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(cd_grape_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
cd_grape_obj.N_blocks = 3
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, cd_grape_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), cd_grape_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), cd_grape_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(cd_grape_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
cd_grape_obj.N_blocks = 4
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, cd_grape_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), cd_grape_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), cd_grape_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(cd_grape_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%% Finally, we can save our parameters
savefile = cd_grape_obj.save()
#%% You can also load up parameters
cd_grape_obj.load(savefile)
# %%
