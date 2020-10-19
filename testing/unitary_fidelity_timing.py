#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../")
from CD_control.CD_control_tf import CD_control_tf
from CD_control.CD_control_tf_initializer import CD_control_init_tf
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import CD_control.tf_quantum as tfq
import tensorflow as tf
import timeit

#%%
N = 50 #cavity hilbert space 
N_blocks = 20 #5
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
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped

#%% 
np.random.seed(0)
CD_control_obj.randomize(beta_scale=3)
betas_rho = CD_control_obj.betas_rho
betas_angle = CD_control_obj.betas_angle, 
phis = CD_control_obj.phis
thetas = CD_control_obj.thetas

U_targ = CD_control_obj.U_tot(betas_rho, betas_angle, phis, thetas)@tfq.qt2tf(qt.tensor(qt.rand_unitary(2, density=.01),qt.rand_unitary(N, density=.01)))
CD_control_obj.target_unitary = U_targ
#%%
wrap = wrapper(CD_control_obj.unitary_fidelity_state_decomp, betas_rho, betas_angle, phis, thetas)
wrap2 = wrapper(CD_control_obj.unitary_fidelity, betas_rho, betas_angle, phis, thetas)

states = tf.stack([
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,0))),
    tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,1))),
    ])
CD_control_obj.set_unitary_fidelity_state_basis(states)
fid = CD_control_obj.unitary_fidelity_state_decomp(betas_rho, betas_angle, phis, thetas)

print("Few States")
print("Fid: " + str(fid.numpy()))
print("Time: " + str(timeit.timeit(wrap, number=100)))

full_states = []
for i in range(N):
    full_states.append(tfq.qt2tf(qt.tensor(qt.basis(2,0),qt.basis(N,i))))
    full_states.append(tfq.qt2tf(qt.tensor(qt.basis(2,1),qt.basis(N,i))))

CD_control_obj.set_unitary_fidelity_state_basis(full_states)
fid = CD_control_obj.unitary_fidelity_state_decomp(betas_rho, betas_angle, phis, thetas)
print("\nFull States")
print("Fid: " + str(fid.numpy()))
print("Time: " + str(timeit.timeit(wrap, number=100)))

fid = CD_control_obj.unitary_fidelity(betas_rho, betas_angle, phis, thetas)
print("\nAnalytic Unitary Fidelity")
print("Fid: " + str(fid.numpy()))
print("Time: " + str(timeit.timeit(wrap2, number=100)))


# %%

