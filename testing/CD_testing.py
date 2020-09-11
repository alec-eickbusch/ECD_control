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
N_blocks = 1
initial_state = qt.tensor(qt.basis(N, 0), (qt.basis(N2, 0) + qt.basis(N2,1))/np.sqrt(2))
cd_grape_obj = CD_grape(initial_state, N_blocks=N_blocks)
sys = System(chi=chi, Ec=Ec, alpha0=alpha0,
             sigma=sigma, chop=chop, epsilon_m=epsilon_m, buffer_time=buffer_time,
             ring_up_time=ring_up_time)
analysis_obj = CD_grape_analysis(cd_grape_obj, sys)
#%%
num_test = 4
betas = [np.random.uniform(low=-3, high=3) + 1j*np.random.uniform(low=-2, high=2)
          for _ in range(num_test)]
thetas = [np.random.uniform(low = 0, high=np.pi) for _ in range(num_test)]
phis = [np.random.uniform(low=-np.pi, high=np.pi) for _ in range(num_test)]
for beta, phi, theta in zip(betas, phis, thetas):
    desired_state = cd_grape_obj.CD(beta)*cd_grape_obj.R(phi, theta)*initial_state
    #desired_state = (qt.tensor(qt.coherent(N, -beta/2.0), qt.basis(N2, 0)) +\
                    #qt.tensor(qt.coherent(N, beta/2.0), qt.basis(N2, 1))).unit()
    cd_grape_obj.betas = [beta]
    cd_grape_obj.phis = [phi,0]
    cd_grape_obj.thetas = [theta,0]
    e, O = analysis_obj.composite_pulse()
    psif = sys.simulate_pulse_trotter(e, O, initial_state)
    fid = qt.fidelity(psif, desired_state)
    print('fid = %f' % fid)
    plt.figure()
    plot_wigner(psif)
    print(psif.ptrace(1))
    plt.figure()
    plot_wigner(desired_state)
    print(desired_state.ptrace(1))
    b = qt.Bloch()
    b.add_states(desired_state.ptrace(1))
    b.add_states(psif.ptrace(1))
    b.show()
    plt.figure()
    plot_pulse(e, O)
# %%
