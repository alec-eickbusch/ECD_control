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
psi0 = qt.tensor(qt.coherent(N,5), qt.basis(N2,0))
a = CD_grape(initial_state=psi0)
#%%
thetas = [np.random.uniform(low=0,high=np.pi) for _ in range(6)]
phis = [np.random.uniform(low=-np.pi, high=np.pi) for _ in range(6)]
#thetas = [np.pi]
#phis = [0]
for theta,phi in zip(thetas,phis):
    print("\n\n theta: %.3f phi: %.3f\n\n" %(theta,phi))
    O = rotate(theta,phi,sigma=sigma,chop=chop)
    e= np.zeros_like(O)
    psif = sys.simulate_pulse_trotter(e, O, psi0)
    desired_state = a.R(phi, theta)*psi0
    fid = qt.fidelity(psif, desired_state)
    print("\n\nfid = %f\n\n" % fid)
    #plt.figure()
    #plot_wigner(psif)
    print(psif.ptrace(1))
    #plt.figure()
    #plot_wigner(desired_state)
    print(desired_state.ptrace(1))
    #plot_pulse(e,O)
# %%
