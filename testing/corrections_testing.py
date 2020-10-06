#%%
%load_ext autoreload
%autoreload 2
from CD_control.CD_control_optimization import CD_control
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
from CD_control.corrections import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%

N = 80 #cavity hilbert space 
N2 = 2 #qubit hilbert space
beta = 2 #conditional dispalcement
N_blocks = 1 #only doing 1 conditional displacement
initial_state = qt.tensor(qt.basis(N,0),(qt.basis(N2,0) + qt.basis(N2,1)).unit())
target_state = (qt.tensor(qt.coherent(N, beta/2.0), qt.basis(N2, 1)) +\
     qt.tensor(qt.coherent(N, -beta/2.0), qt.basis(N2, 0))).unit()
#note above that the |e> state will shift the cavity right, and the |g> state
#will shift the cavity left. this is due to the pi pulse during the CD
#which flips the qubit polarity.
CD_control_obj = CD_control(initial_state, target_state, N_blocks, name='Conditional Displacement')
#%% We can plot the initial and target states (qubit traced out)
plt.figure(figsize=(5,5), dpi=200)
CD_control_obj.plot_initial_state()
plt.title("initial state")
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_target_state()
plt.title("target state")
#%% First, we trivially get to the target state with a single conditional displacement
CD_control_obj.alphas = [0,0]
CD_control_obj.betas = [beta]
CD_control_obj.phis = [0,0]
CD_control_obj.thetas = [0,0]
CD_control_obj.print_info()
#%% And we can plot the final cavity state with these parameters
plt.figure(figsize=(5, 5), dpi=200)
CD_control_obj.plot_final_state()
plt.title("final state")
#%% Testing the corrections
alpha = 0.25 - 0.1j
angle = np.pi/12.0
r = (1j*angle*CD_control_obj.a.dag()*CD_control_obj.a).expm()
d = CD_control_obj.D(alpha)
state2 = d*r*CD_control_obj.final_state()
plot_wigner(state2)
plt.title('distorted state')
# %%
c = OptimalCorrections(distorted_state=state2, target_state=CD_control_obj.target_state,\
     d_alpha = 0.5, d_theta = np.pi/4.0)
#%%
c.grid_search(pts = 21)

c.optimize()
# %%
#plot_wigner(c.target_state)
#plot_wigner(c.corrected_state([-np.real(alpha),-np.imag(alpha),-angle]))
plot_wigner(c.corrected_state())
# %%

# %%
