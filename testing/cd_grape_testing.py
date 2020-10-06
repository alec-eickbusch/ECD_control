#%%
from CD_control.CD_control import *
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
from CD_control.analysis import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from datetime import datetime

#%%
N = 50
N2 = 2
alpha0 = 60
epsilon_m = 2 * np.pi * 1e-3 * 400
chi = 2 * np.pi * 1e-3 * 0.03
sigma = 4
chop = 4
ring_up_time = 4
buffer_time = 0
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2 * np.pi) * Ec_GHz
sys = System(
    chi=chi,
    Ec=Ec,
    alpha0=alpha0,
    epsilon_m=epsilon_m,
    sigma=sigma,
    chop=chop,
    buffer_time=buffer_time,
    ring_up_time=ring_up_time,
)
N_blocks = 4
saving_directory = "C:\\Users\\Alec Eickbusch\\CD_control_data\\"
max_alpha = 5
max_beta = 5
initial_state = qt.tensor(qt.basis(N, 0), qt.basis(N2, 0))
target_state = qt.tensor(qt.basis(N, 1), qt.basis(N2, 0))
name = "fock_1"
term_fid = 0.999
# a = CD_control(initial_state=initial_state, target_state=target_state,\
#            max_alpha = max_alpha, max_beta = max_beta, N_blocks=N_blocks,
##            saving_directory=saving_directory, name=name,
#           term_fid=term_fid)
# a.randomize(alpha_scale=0.1,beta_scale = 1)
# a.optimize()
# a.save()
# savefile = "C:\\Users\\Alec Eickbusch\\CD_control_data\\cat_2_20200904_11_38_18"
# a.load(savefile)
a = CD_control()
savefile = "Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_control\\optimization_tests_20200903\\cat n=2_N_blocks_4_20200903_16_46_56"
a.load(savefile)
psi0 = a.initial_state
analysis = CD_control_analysis(a, sys)
e, O = analysis.composite_pulse()
plot_pulse(e, O)
print("final fid: " + str(a.fidelity()))
#%% First, only looking at the first sequence
blocks = 1
a.N_blocks = blocks
e, O = analysis.composite_pulse()
plot_pulse(e, O)
print("betass: " + str(a.betas[:blocks]))
print("alphas: " + str(a.alphas[:blocks]))
print("thetas: " + str(a.thetas[:blocks]))
print("phis: " + str(a.phis[:blocks]))
#%%
psif = sys.simulate_pulse_trotter(e, O, psi0)
# psif_discrete = a.U_i_block(0)*psi0
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
# psif_discrete = a.U_i_block(0)*psi0
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
# psif_discrete = a.U_i_block(0)*psi0
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
# psif_discrete = a.U_i_block(0)*psi0
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
fid = qt.fidelity(psif, a.target_state)
print("fid: %.4f" % fid)

#%% Saving this for the experiment
datestr = datetime.now().strftime("%Y%m%d_%H_%M_%S")
exp_pulse_dir = r"Y:\Data\Tennessee2020\20200318_cooldown\pulses\\" + datestr + r"\\"
if not os.path.exists(exp_pulse_dir):
    os.makedirs(exp_pulse_dir)
time_str = datetime.now().strftime("%Y%m%d_%Hh_%Mm_%Ss")
exp_pulse_filename = exp_pulse_dir + name + "_" + time_str + ".npz"
np.savez(exp_pulse_filename, Omega=O, epsilon=e, dt=1)
print("Pulse saved as: " + str(exp_pulse_filename))

# %%
