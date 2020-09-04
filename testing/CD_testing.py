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
alpha0 = 60
epsilon_m = 2*np.pi*1e-3*400
chi = 2*np.pi*1e-3*0.03
sigma = 6
chop = 4
ring_up_time = 8
buffer_time = 0
Ec_GHz = 0.19267571  # measured anharmonicity
Ec = (2*np.pi) * Ec_GHz
sys = System(chi=chi, Ec=Ec, alpha0=alpha0, epsilon_m=epsilon_m,
             sigma=sigma, chop=chop, buffer_time=buffer_time, ring_up_time=ring_up_time)
psi0 = qt.tensor(qt.basis(N,0), (qt.basis(N2,0) + np.exp(1j*np.pi/3.0)*qt.basis(N2,1)).unit())
a = CD_grape(initial_state=psi0)
#%%
betas = [1+1j,2+1j,5+2j]
for beta in betas:
    e,O = fastest_CD(beta,alpha0=alpha0, epsilon_m=epsilon_m,\
        chi=chi, buffer_time=buffer_time, sigma_q = sigma,\
        chop_q = chop, ring_up_time=ring_up_time)
    psif = sys.simulate_pulse_trotter(e,O,psi0)
    desired_state = a.CD(beta)*psi0
    fid = qt.fidelity(psif,desired_state)
    print('fid = %f' % fid)
    plt.figure()
    plot_wigner(psif)
    print(psif.ptrace(1))
    plt.figure()
    plot_wigner(desired_state)
    print(desired_state.ptrace(1))
    #plot_pulse(e,O)
# %%

