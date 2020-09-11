#%%
from CD_GRAPE.cd_grape_optimization import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
from CD_GRAPE.analysis import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 30
N2 = 2
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
N_blocks = 0
initial_state = qt.tensor(qt.basis(N, 0), qt.basis(N2, 0))
cd_grape_obj = CD_grape(initial_state, N_blocks=N_blocks)
sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
             sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
             ring_up_time=ring_up_time)
analysis_obj = CD_grape_analysis(cd_grape_obj, sys)
#%%
num_test = 4
thetas = [np.random.uniform(low=0, high=np.pi) for _ in range(num_test)]
phis = [np.random.uniform(low=-np.pi, high=np.pi) for _ in range(num_test)]
#phis = [0 for _ in range(num_test)]
#thetas = [0,np.pi/2.0, np.pi]
#phis = [np.pi/4.0,np.pi/2.0,np.pi/2.0]
for phi, theta in zip(phis,thetas):
    print("\n phi pi units: %.3f" % (phi/np.pi))
    print("\n theta pi units: %.3f" % (theta/np.pi))
    desired_state = cd_grape_obj.R(phi, theta)*initial_state
    cd_grape_obj.phis= [phi]
    cd_grape_obj.thetas = [theta]
    e, O = analysis_obj.composite_pulse()
    psif = sys.simulate_pulse_trotter(e, O, initial_state)
    fid = qt.fidelity(psif, desired_state)
    print('\n\nfid = %f\n\n' % fid)
   # plt.figure()
    #plot_wigner(psif)
    print(psif.ptrace(1))
    #plt.figure()
    #plot_wigner(desired_state)
    print(desired_state.ptrace(1))
    b = qt.Bloch()
    b.add_states(desired_state.ptrace(1))
    b.add_states(psif.ptrace(1))
    b.show()
   # plt.figure()
    #plot_pulse(e, O)
# %%
