#%%
from CD_GRAPE.cd_grape_optimization import CD_grape
from CD_GRAPE.helper_functions import plot_pulse, plot_wigner
from CD_GRAPE.analysis import System, CD_grape_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 20 #cavity hilbert space 
N2 = 2 #qubit hilbert space
fock = 2 #fock state to create
N_blocks = 6
initial_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
target_state = qt.tensor(qt.basis(N, fock), qt.basis(N2, 0))
term_fid = 0.99
#max alpha and beta are the maximum values of alpha and beta for optimization
max_alpha = 5
max_beta = 5
name = "Fock creation"
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_grape_parameters\\"
cd_grape_obj = CD_grape(initial_state, target_state, N_blocks,\
                    name='Fock creation', term_fid=term_fid,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    saving_directory=saving_directory)
#%% We can plot the initial and target states (qubit traced out)
plt.figure(figsize=(5,5), dpi=200)
cd_grape_obj.plot_initial_state()
plt.title("initial state")
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_target_state()
plt.title("target state")
#%% Doing the optimization
#The alpha and beta scale are scales for the random initialization.
cd_grape_obj.randomize(alpha_scale=0.5, beta_scale=1)
print("Randomized parameters:")
cd_grape_obj.print_info()
cd_grape_obj.optimize()
#%%
print("after optimization:")
cd_grape_obj.print_info()
#%% plotting the final state
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
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
plot_wigner(psif)
plt.title("Simulated final state")

#%% 
fid = qt.fidelity(psif, target_state)
print("\n\nSimulated fidelity to target state: %.5f\n\n" % fid)

#%% Finally, we can save our parameters
savefile = cd_grape_obj.save()
#%% You can also load up parameters
#Note that the "savefile" is the filename without the .npz or .qt extention.
cd_grape_obj.load(savefile)
# %%
