#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../")
from CD_control.CD_control_tf import CD_control_tf
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
from CD_control.global_optimization_tf import Global_optimizer_tf
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 100 #cavity hilbert space 
# alpha = 2 + 1j #cat alpha
N_blocks = 20
# initial_state = qt.tensor(qt.basis(2,0),qt.basis(N,0))
# target_state = qt.tensor(qt.basis(2,0), (qt.coherent(N,alpha) + qt.coherent(N,-alpha)).unit())
a = qt.tensor(qt.identity(2), qt.destroy(N))
q = qt.tensor(qt.destroy(2), qt.identity(N))
sz = 1 - 2 * q.dag() * q
targ = (1j*np.pi*a.dag()*a*sz).expm()
term_fid = 0.998
#max alpha and beta are the maximum values of alpha and beta for optimization
name = "Parity"
saving_directory = "/"
CD_control_obj = Global_optimizer_tf(target_unitary=targ, unitary_optimization=True, P_cav=10,
                    N_blocks=N_blocks,
                    name=name, term_fid=term_fid,
                    saving_directory=saving_directory)

#%%
CD_control_obj.multistart_optimize()
#%%
CD_control_obj.N_blocks_sweep()
#%% Doing the optimization
#The alpha and beta scale are scales for the random initialization.
CD_control_obj.randomize(beta_scale = 2.0)
print("Randomized parameters:")
CD_control_obj.print_info()
#%%
CD_control_obj.optimize(learning_rate = 0.01, epoch_size=200, epochs=100, df_stop=1e-10)
#%%
CD_control_obj.print_info()
CD_control_obj.optimize()
print("after optimization:")
CD_control_obj.print_info()
#%% plotting the final state
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("final state")
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
analysis_obj = CD_control_analysis(CD_control_obj, sys)
#%%  The composite pulse
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
#%%
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, target_state)
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")

#%% Going step by step
CD_control_obj.N_blocks = 0
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, CD_control_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(CD_control_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
CD_control_obj.N_blocks = 1
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, CD_control_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(CD_control_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
CD_control_obj.N_blocks = 2
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, CD_control_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(CD_control_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
CD_control_obj.N_blocks = 3
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, CD_control_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(CD_control_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%%
CD_control_obj.N_blocks = 4
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
fid = qt.fidelity(psif, CD_control_obj.final_state())
fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(CD_control_obj.final_state().ptrace(1))
b.add_states(psif.ptrace(1))
b.show()
#%% Finally, we can save our parameters
savefile = CD_control_obj.save()
#%% You can also load up parameters
CD_control_obj.load(savefile)
# %%
