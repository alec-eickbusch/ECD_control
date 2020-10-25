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
from qutip.visualization import plot_fock_distribution
#%%
N = 50 #cavity hilbert space 
N_blocks = 8 #5
targ = np.array(qt.tensor(qt.identity(2),qt.identity(N)))
targ[0,0] = 1
targ[1,1] = 0
targ[1,N] = 1
targ[N,1] = 1
targ[N,N] = 0
targ[N+1,N+1] = 1
targ = qt.Qobj(targ,dims=[[2,N],[2,N]])
term_fid = 0.9999

initial_state = qt.tensor(qt.basis(2,0),qt.basis(N,1))
target_state = qt.tensor(qt.basis(2,1),qt.basis(N,0))
#max alpha and beta are the maximum values of alpha and beta for optimization
name = "Cooldown"
saving_directory = "/"

#%%
CD_control_obj = CD_control_init_tf(N_blocks=N_blocks,target_unitary=targ, 
                    initial_state=initial_state, target_state=target_state,
                    unitary_optimization=True, P_cav=2, no_CD_end=False,
                    name=name, term_fid=term_fid,
                    saving_directory=saving_directory)

#%% 
states = tf.stack([
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,0))),
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,1))),
    ])
CD_control_obj.multi_state_initialize_unitary(states)

#%% Test Initialization
CD_control_obj.plot_initial_state()
CD_control_obj.plot_final_state()
CD_control_obj.plot_target_state()


#%% Thermal State
n_thermal = 0.1

thermal_state_dm_qt = qt.tensor(qt.ket2dm(qt.basis(2,0)),qt.thermal_dm(N,n_thermal))
thermal_state_dm = tfq.qt2tf(thermal_state_dm_qt)
U = CD_control_obj.U_tot(CD_control_obj.betas_rho, CD_control_obj.betas_angle, CD_control_obj.phis, CD_control_obj.thetas)
final_state_dm =  U @ thermal_state_dm @ tf.linalg.adjoint(U)
final_state_dm = tfq.tf2qt(final_state_dm, matrix=True)
vacuum_state_dm  = qt.tensor(qt.ket2dm(qt.basis(2,0)),qt.ket2dm(qt.basis(N,0)))
thermal_state_dm = thermal_state_dm_qt

fig, axes = plt.subplots(1, 2, figsize=(12,3))
plot_fock_distribution(thermal_state_dm, fig=fig, ax=axes[0], title="Thermal Cavity State");
plot_fock_distribution(final_state_dm, fig=fig, ax=axes[1], title="Cooled Cavity State");
print("Qubit Thermal State Fidelity with vacuum: " + str(qt.fidelity(vacuum_state_dm.ptrace(0), thermal_state_dm.ptrace(0))))
print("Cavity Thermal State Fidelity with vacuum: " + str(qt.fidelity(vacuum_state_dm.ptrace(1), thermal_state_dm.ptrace(1))))
print("Qubit Cooled State Fidelity with cooled: " + str(qt.fidelity(vacuum_state_dm.ptrace(0), final_state_dm.ptrace(0))))
print("Cavity Cooled State Fidelity with cooled: " + str(qt.fidelity(vacuum_state_dm.ptrace(1), final_state_dm.ptrace(1))))

fig.tight_layout()
plt.show()

#%% Unitary Fidelity Optimization 
CD_control_unitary_obj = CD_control_obj.initialized_obj(unitary_optimization=True)

#%%
CD_control_unitary_obj.optimize(epochs=1000, epoch_size=10, dloss_stop=1e-6)

#%% 
CD_control_unitary_obj.plot_initial_state()
CD_control_unitary_obj.plot_final_state()
CD_control_unitary_obj.plot_target_state()


#%%
N = 50 #cavity hilbert space 
N_blocks = 5
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
CD_control_obj_small = CD_control_init_tf(N_blocks=N_blocks,target_unitary=targ, 
                    initial_state=qt.tensor(qt.basis(2,0),qt.basis(N,1)), target_state=qt.tensor(qt.basis(2,1),qt.basis(N,0)),
                    unitary_optimization=True, P_cav=2, no_CD_end=False,
                    name=name, term_fid=term_fid,
                    saving_directory=saving_directory)

#%% 
states = tf.stack([
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,0))),
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,1))),
    ])
CD_control_obj_small.multi_state_initialize_unitary(states)

#%% Test Initialization
CD_control_obj_small.plot_initial_state()
CD_control_obj_small.plot_final_state()
CD_control_obj_small.plot_target_state()

#%% Unitary Fidelity Optimization with States Basis
CD_control_unitary_obj_small = CD_control_obj_small.initialized_obj(unitary_optimization=True)

#%%
CD_control_unitary_obj_small.optimize(epochs=1000, epoch_size=10, dloss_stop=1e-6)

#%% 
CD_control_unitary_obj_small.plot_initial_state()
CD_control_unitary_obj_small.plot_final_state()
CD_control_unitary_obj_small.plot_target_state()


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

#%% 
def convert(control_obj):
    sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
                sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
                ring_up_time=ring_up_time)
    return sys, CD_control_analysis(control_obj, sys)

#%%
sys, analysis_obj = convert(CD_control_obj)
e, O = analysis_obj.composite_pulse()
plt.figure(figsize=(8, 4), dpi=200)
plot_pulse(e, O)


#%%
final_state_dm_exp = sys.simulate_pulse_trotter(e, O, thermal_state_dm_qt, density_matrix=True)

#%%
final_state_dm_exp2 = sys.simulate_pulse_master_equation(e, O, thermal_state_dm_qt)
# fid = qt.fidelity(psif, target_state)

#%%
print("Qubit Thermal State Fidelity with vacuum: " + str(qt.fidelity(vacuum_state_dm.ptrace(0), thermal_state_dm_qt.ptrace(0))))
print("Cavity Thermal State Fidelity with vacuum: " + str(qt.fidelity(vacuum_state_dm.ptrace(1), thermal_state_dm_qt.ptrace(1))))
print("Qubit Cooled State Fidelity with cooled: " + str(qt.fidelity(vacuum_state_dm.ptrace(0), final_state_dm_exp.ptrace(0))))
print("Cavity Cooled State Fidelity with cooled: " + str(qt.fidelity(vacuum_state_dm.ptrace(1), final_state_dm_exp.ptrace(1))))
print("Qubit Cooled State Fidelity with cooled (loss): " + str(qt.fidelity(vacuum_state_dm.ptrace(0), final_state_dm_exp2.ptrace(0))))
print("Cavity Cooled State Fidelity with cooled (loss): " + str(qt.fidelity(vacuum_state_dm.ptrace(1), final_state_dm_exp2.ptrace(1))))


#%%
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_unitary_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")

#%% Going step by step
final_state_opt = tfq.tf2qt(CD_control_unitary_obj.final_state())
fid = qt.fidelity(psif, final_state_opt)
fid_c = qt.fidelity(psif.ptrace(0), final_state_opt.ptrace(0))
fid_q = qt.fidelity(psif.ptrace(1), final_state_opt.ptrace(1))
print("\n\nSimulated fidelity to final state: %.5f" % fid)
print("Simulated QUBIT fidelity to final state: %.5f" % fid_q)
print("Simulated CAVITY fidelity to final state: %.5f\n\n" % fid_c)
plt.figure(figsize=(5, 5), dpi=200)
CD_control_unitary_obj.plot_final_state()
plt.title("cd grape final state")
plt.figure(figsize=(5, 5), dpi=200)
plot_wigner(psif)
plt.title("constructed pulse final state")
b = qt.Bloch()
b.add_states(final_state_opt.ptrace(0))
b.add_states(psif.ptrace(0))
b.show()
#%% Finally, we can save our parameters
savefile = CD_control_obj.save()
#%% You can also load up parameters
CD_control_obj.load(savefile)
# %%

# %%
