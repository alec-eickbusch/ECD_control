#%%
from CD_GRAPE.cd_grape import CD_grape
#from CD_GRAPE.basic_pulses import *
#from CD_GRAPE.helper_functions import *
from CD_GRAPE.analysis import System, CD_grape_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 50 #cavity hilbert space 
N2 = 2 #qubit hilbert space
beta = 2 #conditional dispalcement
N_blocks = 1 #only doing 1 conditional displacement
initial_state = qt.tensor(qt.basis(N,0),(qt.basis(N2,0) + qt.basis(N2,1)).unit())
target_state = (qt.tensor(qt.coherent(N, beta/2.0), qt.basis(N2, 1)) +\
     qt.tensor(qt.coherent(N, -beta/2.0), qt.basis(N2, 0))).unit()
#note above that the |e> state will shift the cavity right, and the |g> state
#will shift the cavity left. this is due to the pi pulse during the CD
#which flips the qubit polarity.
cd_grape_obj = CD_grape(initial_state, target_state, N_blocks, name='Conditional Displacement')
#%% We can plot the initial and target states (qubit traced out)
plt.figure(figsize=(5,5), dpi=200)
cd_grape_obj.plot_initial_state()
plt.title("initial state")
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_target_state()
plt.title("target state")
#%% First, we trivially get to the target state with a single conditional displacement
cd_grape_obj.alphas = [0,0]
cd_grape_obj.betas = [beta]
cd_grape_obj.phis = [0,0]
cd_grape_obj.thetas = [0,0]
cd_grape_obj.print_info()
#%% And we can plot the final cavity state with these parameters
plt.figure(figsize=(5, 5), dpi=200)
cd_grape_obj.plot_final_state()
plt.title("final state")
#%% Now, we can convert these parameters to a pulse we can run on the experiment
epsilon_m = 2*np.pi*1e-3*300.0 #maximum displacement rate
alpha0 = 50 #maximum displacement before things break down 
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2*np.pi) * Ec_GHz
chi_MHz = 0.03
chi = 2*np.pi*1e-3*chi_MHz
sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
             sigma=3, chop=4, epsilon_m=epsilon_m, buffer_time=4,
             ring_up_time=16)


#%%
#Now, showing we can get to the same result through optimization.
cd_grape_obj.randomize(alpha_scale = 0.1, beta_scale=1.0)
print("Randomized parameters:")
cd_grape_obj.print_info()
#The alpha and beta scale are scales for the random initialization. 
cd_grape_obj.optimize()
print("after optimization:")
cd_grape_obj.print_info()
#you may notice that there is now an unnecessary cavity displacement (then displace back)
#or qubit rotation in the parameters. This is because the penalties are not yet implemented.
#%%


N = 50
N2 = 2
alpha0 = 60
epsilon_m = 2*np.pi*1e-3*400
chi = 2*np.pi*1e-3*0.03
sigma = 4
chop = 4
ring_up_time = 4
buffer_time = 0
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2*np.pi) * Ec_GHz
sys = System(chi=chi, Ec=Ec, alpha0=alpha0, epsilon_m=epsilon_m,
             sigma=sigma, chop=chop, buffer_time=buffer_time, ring_up_time=ring_up_time)
N_blocks = 4
saving_directory = "C:\\Users\\Alec Eickbusch\\CD_grape_data\\"
max_alpha = 5
max_beta = 5
initial_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
target_state = qt.tensor(qt.basis(N, 1), qt.basis(N2, 0))
name = 'fock_1'
term_fid = 0.999
#a = CD_grape(initial_state=initial_state, target_state=target_state,\
#            max_alpha = max_alpha, max_beta = max_beta, N_blocks=N_blocks,
##            saving_directory=saving_directory, name=name,
 #           term_fid=term_fid)
#a.randomize(alpha_scale=0.1,beta_scale = 1)
#a.optimize()
#a.save()
#savefile = "C:\\Users\\Alec Eickbusch\\CD_grape_data\\cat_2_20200904_11_38_18"
#a.load(savefile)
a = CD_grape()
savefile = "Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=2_N_blocks_4_20200903_16_46_56"
a.load(savefile)
psi0 = a.initial_state
analysis = CD_grape_analysis(a, sys)
e, O = analysis.composite_pulse()
plot_pulse(e,O)
print("final fid: " + str(a.fidelity()))
#%% First, only looking at the first sequence
blocks = 1
a.N_blocks = blocks
e, O = analysis.composite_pulse()
plot_pulse(e,O)
print("betass: " + str(a.betas[:blocks]))
print("alphas: " + str(a.alphas[:blocks]))
print("thetas: " + str(a.thetas[:blocks]))
print("phis: " + str(a.phis[:blocks]))
#%%
psif = sys.simulate_pulse_trotter(e,O,psi0)
#psif_discrete = a.U_i_block(0)*psi0
psif_discrete = a.final_state()
fid = qt.fidelity(psif,psif_discrete)
print("fid: %.4f" % fid)
# %%
plt.figure()
plot_wigner(psif)
plt.figure()
plot_wigner(psif_discrete)
print(psif.ptrace(1))
print(psif_discrete.ptrace(1))
#%%
#%%
blocks = 2
a.N_blocks = blocks
e, O = analysis.composite_pulse()
plot_pulse(e, O)
print("betass: " + str(a.betas[:blocks]))
print("alphas: " + str(a.alphas[:blocks]))
print("thetas: " + str(a.thetas[:blocks]))
print("phis: " + str(a.phis[:blocks]))
#%%
psif = sys.simulate_pulse_trotter(e, O, psi0)
#psif_discrete = a.U_i_block(0)*psi0
psif_discrete = a.final_state()
fid = qt.fidelity(psif, psif_discrete)
print("fid: %.4f" % fid)
# %%
plt.figure()
plot_wigner(psif)
plt.figure()
plot_wigner(psif_discrete)
print(psif.ptrace(1))
print(psif_discrete.ptrace(1))
#%%
blocks = 3
a.N_blocks = blocks
e, O = analysis.composite_pulse()
plot_pulse(e, O)
print("betass: " + str(a.betas[:blocks]))
print("alphas: " + str(a.alphas[:blocks]))
print("thetas: " + str(a.thetas[:blocks]))
print("phis: " + str(a.phis[:blocks]))
#%%
psif = sys.simulate_pulse_trotter(e, O, psi0)
#psif_discrete = a.U_i_block(0)*psi0
psif_discrete = a.final_state()
fid = qt.fidelity(psif, psif_discrete)
print("fid: %.4f" % fid)
# %%
plt.figure()
plot_wigner(psif)
plt.figure()
plot_wigner(psif_discrete)
print(psif.ptrace(1))
print(psif_discrete.ptrace(1))
#%%
blocks = 4
a.N_blocks = blocks
e, O = analysis.composite_pulse()
plot_pulse(e, O)
print("betass: " + str(a.betas[:blocks]))
print("alphas: " + str(a.alphas[:blocks]))
print("thetas: " + str(a.thetas[:blocks]))
print("phis: " + str(a.phis[:blocks]))
#%%
psif = sys.simulate_pulse_trotter(e, O, psi0, use_kerr=True)
#psif_discrete = a.U_i_block(0)*psi0
psif_discrete = a.final_state()
fid = qt.fidelity(psif, psif_discrete)
print("fid: %.4f" % fid)
# %%
plt.figure()
plot_wigner(psif)
plt.figure()
plot_wigner(psif_discrete)
print(psif.ptrace(1))
print(psif_discrete.ptrace(1))

# %%
fid = qt.fidelity(psif,a.target_state)
print("fid: %.4f" % fid)

#%% Saving this for the experiment
datestr = datetime.now().strftime('%Y%m%d_%H_%M_%S')
exp_pulse_dir = r'Y:\Data\Tennessee2020\20200318_cooldown\pulses\\' + datestr + r'\\'
if not os.path.exists(exp_pulse_dir):
    os.makedirs(exp_pulse_dir)
time_str = datetime.now().strftime('%Y%m%d_%Hh_%Mm_%Ss')
exp_pulse_filename = exp_pulse_dir + name + '_' + time_str + '.npz'
np.savez(exp_pulse_filename, Omega=O, epsilon = e, dt=1)
print("Pulse saved as: " + str(exp_pulse_filename))

# %%
