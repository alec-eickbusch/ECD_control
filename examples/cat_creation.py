#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../")
from CD_control.CD_control_tf import CD_control_tf
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
from CD_control.global_optimization_tf import Global_optimizer_tf
import CD_control.tf_quantum as tfq
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 80 #cavity hilbert space 
alpha = 2 + 1j #cat alpha
N_blocks = 5
no_CD_end = True
initial_state = qt.tensor(qt.basis(2,0),qt.basis(N,0))
target_state = qt.tensor(qt.basis(2,0), (qt.coherent(N,alpha) + qt.coherent(N,-alpha)).unit())
term_fid = 0.9999
#%%
CD_control_obj = CD_control_tf(initial_state, target_state, N_blocks=N_blocks,
                    term_fid = term_fid, no_CD_end=True)
#%% We can plot the initial and target states (qubit traced out)
plt.figure(figsize=(5,5), dpi=200)
CD_control_obj.plot_initial_state()
plt.title("initial state")
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_target_state()
plt.title("target state")
#%% First, a basic optimization
CD_control_obj.randomize(beta_scale=1.0)
losses = CD_control_obj.optimize(epochs = 100, epoch_size=10,dloss_stop=1e-6)
#%%
#by default, the loss function is 1-log(fid)
fids = 1 - np.exp(losses)
plt.semilogy(1-fids)
plt.xlabel('epoch')
plt.ylabel('1-Fidelity')
plt.title("Optimization")

#%%
CD_control_obj.plot_final_state()
#%% Now, we can perform global optimizations
N_blocks = 3 #fewer N blocks for demonstration
global_opt_obj = Global_optimizer_tf(initial_state, target_state, N_blocks=N_blocks,
                    term_fid = term_fid, no_CD_end=True)
#%% A multi-start optimization
losses = global_opt_obj.multistart_optimize(N_multistart=20, beta_scale=1.0, epochs = 200, epoch_size=10,dloss_stop=1e-6)
plt.figure()
for loss in losses:
    fids = 1 - np.exp(loss)
    plt.semilogy(1-fids)
plt.xlabel('epoch')
plt.ylabel('1-Fidelity')
plt.title("Multistart Optimization")
#%% Sweeping N_blocks


#%%
CD_control_obj.randomize(beta_scale=1.0)
betas, phis, thetas = CD_control_obj.get_numpy_vars()
thetas[0] = np.pi/2.0
betas[0] = 2*alpha
CD_control_obj.set_tf_vars(betas, phis, thetas)
beta_mask = np.ones(N_blocks)
beta_mask[0] = 0
theta_mask = beta_mask
#beta_mask[-1] = 0
CD_control_obj.optimize(beta_mask=beta_mask, theta_mask=theta_mask, learning_rate=1.0)
#%%
CD_control_obj.term_fid = 0.999
fids = CD_control_obj.multistart_optimize(N_multistart = 10, beta_scale=0.5, learning_rate = 0.1,
df_stop=-1, epoch_size=1, epochs= 300, do_prints=False)
#%%
CD_control_obj.term_fid = 0.998
fids = CD_control_obj.pin_optimize(tolerance=0.1, learning_rate=0.01, df_stop=-1, epoch_size=1, epochs=1000, do_prints=False)
#%%

plt.figure()
for fid in fids:
    plt.plot(fid)

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
qubit_T1_us = 60.0
qubit_T2_us = 60.0
sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
             sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
             ring_up_time=ring_up_time, qubit_T1_us=qubit_T1_us,
             qubit_T2_us=qubit_T2_us)
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

#%%
psif = sys.simulate_pulse_master_equation(e, O, initial_state, use_qubit_T1=True, use_qubit_T2=True)
fid = qt.fidelity(psif, target_state)
fid_c = qt.fidelity(psif.ptrace(0), target_state.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), target_state.ptrace(1))
zero = qt.tensor(qt.basis(2,0),qt.identity(N))
fid_zero = qt.fidelity(zero.dag()*psif*zero, target_state.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
print("Simulated zero post selected cavity fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")

#%% Going step by step
max_N_blocks = 1
e, O = analysis_obj.composite_pulse(max_N_blocks=max_N_blocks)
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_master_equation(e, O, initial_state)
psif_cd = tfq.tf2qt(CD_control_obj.state(i=max_N_blocks))
fid = qt.fidelity(psif, psif_cd)
fid_c = qt.fidelity(psif.ptrace(0), psif_cd.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), psif_cd.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif_cd)
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(psif_cd.ptrace(0))
b.add_states(psif.ptrace(0))
b.show()

#%% Going step by step
max_N_blocks = 2
e, O = analysis_obj.composite_pulse(max_N_blocks=max_N_blocks)
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
psif_cd = tfq.tf2qt(CD_control_obj.state(i=max_N_blocks))
fid = qt.fidelity(psif, psif_cd)
fid_c = qt.fidelity(psif.ptrace(0), psif_cd.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), psif_cd.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif_cd)
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(psif_cd.ptrace(0))
b.add_states(psif.ptrace(0))
b.show()
#%% Going step by step
max_N_blocks = 3
e, O = analysis_obj.composite_pulse(max_N_blocks=max_N_blocks)
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
psif_cd = tfq.tf2qt(CD_control_obj.state(i=max_N_blocks))
fid = qt.fidelity(psif, psif_cd)
fid_c = qt.fidelity(psif.ptrace(0), psif_cd.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), psif_cd.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif_cd)
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(psif_cd.ptrace(0))
b.add_states(psif.ptrace(0))
b.show()

# %%
#%% Going step by step
max_N_blocks = 5
e, O = analysis_obj.composite_pulse(max_N_blocks=max_N_blocks)
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)
psif = sys.simulate_pulse_trotter(e, O, initial_state)
psif_cd = tfq.tf2qt(CD_control_obj.state(i=max_N_blocks))
fid = qt.fidelity(psif, psif_cd)
fid_c = qt.fidelity(psif.ptrace(0), psif_cd.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), psif_cd.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif_cd)
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(psif_cd.ptrace(0))
b.add_states(psif.ptrace(0))
b.show()

# %%
