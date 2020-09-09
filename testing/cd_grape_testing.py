#%%
from CD_GRAPE.cd_grape import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
from CD_GRAPE.analysis import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 50
N2 = 2
alpha0 = 30
epsilon_m = 2*np.pi*1e-3*200
chi = 2*np.pi*1e-3*0.03
sigma = 6
chop = 4
ring_up_time = 16
buffer_time = 8
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2*np.pi) * Ec_GHz
sys = System(chi=chi, Ec=Ec, alpha0=alpha0, epsilon_m=epsilon_m,
             sigma=sigma, chop=chop, buffer_time=buffer_time, ring_up_time=ring_up_time)
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

# %%
