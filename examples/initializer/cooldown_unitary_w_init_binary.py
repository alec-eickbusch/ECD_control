#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../../")
from CD_control.CD_control_tf import CD_control_tf
from CD_control.CD_control_tf_initializer import CD_control_init_tf
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import CD_control.tf_quantum as tfq
import tensorflow as tf
#%%
N = 50 #cavity hilbert space 
N_blocks = 20 #5
targ = np.array(qt.tensor(qt.identity(2),qt.identity(N)))
targ[0,0] = 1
targ[1,1] = 0
targ[1,N] = 1
targ[N,1] = 1
targ[N,N] = 0
targ[N+1,N+1] = 1
targ = qt.Qobj(targ,dims=[[2,N],[2,N]])
term_fid = 0.9999
#max alpha and beta are the maximum values of alpha and beta for optimization
name = "Cooldown"
saving_directory = "/"
CD_control_obj = CD_control_init_tf(N_blocks=N_blocks,target_unitary=targ, 
                    initial_state=qt.tensor(qt.basis(2,0),qt.basis(N,1)), target_state=qt.tensor(qt.basis(2,1),qt.basis(N,0)),
                    unitary_optimization=True, P_cav=2,
                    name=name, term_fid=term_fid,
                    saving_directory=saving_directory)

#%% 
CD_control_obj.binary_initialize_unitary()
CD_control_obj.concat_controls()

#%% Normal Unitary Fidelity Optimization
CD_control_unitary_obj = CD_control_obj.initialized_obj(unitary_optimization=True)
CD_control_unitary_obj.optimize(epochs=100, epoch_size=10, dloss_stop=1e-6)


# #%% 
# CD_control_obj.concat_controls()
# #%% 
# CD_control_obj = CD_control(CD_control_init_obj=CD_control_obj_init, N_blocks=2,
#                     name=name, term_fid=term_fid,
#                     max_alpha = max_alpha, max_beta=max_beta,
#                     saving_directory=saving_directory,
#                     basinhopping_kwargs={'T':0.1},
#                     save_all_minima = True,
#                     use_displacements=True, analytic=True)
# CD_control_obj.fidelity()
# #%% We can plot the initial and target states (qubit traced out)
# plt.figure(figsize=(5,5), dpi=200)
# CD_control_obj.plot_initial_state()
# plt.title("initial state")
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_target_state()
# plt.title("target state")
# #%% Doing the optimization
# #The alpha and beta scale are scales for the random initialization.
# # CD_control_obj.randomize(alpha_scale=0.2, beta_scale=1)
# print("Randomized parameters:")
# CD_control_obj.print_info()
# CD_control_obj.optimize()
# print("after optimization:")
# CD_control_obj.print_info()
# #%% plotting the final state
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("final state")
# #%% Now, we can convert these parameters to a pulse we can run on the experiment
# #first, creating a system object
# epsilon_m = 2*np.pi*1e-3*300.0  # maximum displacement rate
# alpha0 = 50  # maximum displacement before things break down
# Ec_GHz = 0.19267571  # measured anharmonicity
# Ec = (2*np.pi) * Ec_GHz
# chi_MHz = 0.03
# chi = 2*np.pi*1e-3*chi_MHz
# sigma = 6  # sigma for gaussian pulses
# chop = 4  # chop for gaussian pulses
# buffer_time = 4  # time between discrete pulses
# ring_up_time = 16  # Time to ring up for large CD pulses
# sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
#              sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
#              ring_up_time=ring_up_time)
# analysis_obj = CD_control_analysis(CD_control_obj, sys)
# #%%  The composite pulse
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# #%%
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, target_state)
# print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")

# #%% Going step by step
# CD_control_obj.N_blocks = 0
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, CD_control_obj.final_state())
# fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
# fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
# print("\n\nSimulated fidelity to final state: %.5f" % fid)
# print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
# print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")
# b = qt.Bloch()
# b.add_states(CD_control_obj.final_state().ptrace(1))
# b.add_states(psif.ptrace(1))
# b.show()
# #%%
# CD_control_obj.N_blocks = 1
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, CD_control_obj.final_state())
# fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
# fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
# print("\n\nSimulated fidelity to final state: %.5f" % fid)
# print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
# print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")
# b = qt.Bloch()
# b.add_states(CD_control_obj.final_state().ptrace(1))
# b.add_states(psif.ptrace(1))
# b.show()
# #%%
# CD_control_obj.N_blocks = 2
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, CD_control_obj.final_state())
# fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
# fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
# print("\n\nSimulated fidelity to final state: %.5f" % fid)
# print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
# print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")
# b = qt.Bloch()
# b.add_states(CD_control_obj.final_state().ptrace(1))
# b.add_states(psif.ptrace(1))
# b.show()
# #%%
# CD_control_obj.N_blocks = 3
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, CD_control_obj.final_state())
# fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
# fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
# print("\n\nSimulated fidelity to final state: %.5f" % fid)
# print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
# print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")
# b = qt.Bloch()
# b.add_states(CD_control_obj.final_state().ptrace(1))
# b.add_states(psif.ptrace(1))
# b.show()
# #%%
# CD_control_obj.N_blocks = 4
# e, O = analysis_obj.composite_pulse()
# plt.figure(figsize=(8, 4), dpi=200)
# plot_pulse(e, O)
# psif = sys.simulate_pulse_trotter(e, O, initial_state)
# fid = qt.fidelity(psif, CD_control_obj.final_state())
# fid_c = qt.fidelity(psif.ptrace(0), CD_control_obj.final_state().ptrace(0))
# fid_q = qt.fidelity(psif.ptrace(1), CD_control_obj.final_state().ptrace(1))
# print("\n\nSimulated fidelity to final state: %.5f" % fid)
# print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
# print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
# plt.figure(figsize=(5, 5), dpi=200)
# CD_control_obj.plot_final_state()
# plt.title("cd grape final state")
# plt.figure(figsize=(5, 5), dpi=200)
# plot_wigner(psif)
# plt.title("constructed pulse final state")
# b = qt.Bloch()
# b.add_states(CD_control_obj.final_state().ptrace(1))
# b.add_states(psif.ptrace(1))
# b.show()
# #%% Finally, we can save our parameters
# savefile = CD_control_obj.save()
# #%% You can also load up parameters
# CD_control_obj.load(savefile)
# # %%

# %%
