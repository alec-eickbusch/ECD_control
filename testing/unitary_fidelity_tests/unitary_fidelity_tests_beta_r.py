#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../../")
import importlib
from CD_control.CD_control_optimization import *
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

#%%
N = 20 #cavity hilbert space 
N2 = 2 #qubit hilbert space
N_blocks = 8
term_fid = 0.99
#max alpha and beta are the maximum values of alpha and beta for optimization
max_alpha = 1
max_beta = 4
name = "unitary testing"
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_control_parameters\\"
#TODO: can we control the step size in the local optimizations?
#TODO: Are there few enough parameters we can do more of a global search? 
# Just sweep each parameter over its range?
CD_control_obj = CD_control(N_blocks = N_blocks, unitary_optimization=True,
                    name=name, term_fid=term_fid, analytic=False,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    beta_r_step_size=0.5, alpha_r_step_size=0.2,
                    saving_directory=saving_directory,
                    use_displacements=True, N = N, N2=N2,
                    beta_penalty_multiplier=0, no_CD_end=False,
                    minimizer_options = {'gtol': 1e-4, 'ftol':1e-4}, basinhopping_kwargs={'niter':1000, 'T':0.01})
# %% Setup Target
# U_targ = CD_control_obj.CD(2)*CD_control_obj.D(1)*CD_control_obj.R(1,1)
# U_targ = qt.tensor(qt.destroy(N).dag(), qt.identity(N2))
# U_targ = (1j*np.pi*CD_control_obj.a.dag()*CD_control_obj.a*CD_control_obj.sz).expm()
CD_control_obj.randomize(alpha_scale=0.2, beta_scale=3)
U_targ = CD_control_obj.U_tot()*qt.tensor(qt.rand_unitary(N, density=.01),qt.rand_unitary(N2, density=.01))

CD_control_obj.target_unitary = U_targ

# #%% Doing the optimization
# #The alpha and beta scale are scales for the random initialization.
# #CD_control_obj.betas = [-2.0]
# print("Randomized parameters:")
# CD_control_obj.print_info()
# CD_control_obj.optimize()
# print("after optimization:")
# CD_control_obj.print_info()
#%% After some optimization
print(CD_control_obj.unitary_fidelity())
# %% Setup CD_control_obj_new
CD_control_obj_new = CD_control(betas=CD_control_obj.betas,alphas=CD_control_obj.alphas,phis=CD_control_obj.phis,
                    thetas=CD_control_obj.thetas,N_blocks = N_blocks, unitary_optimization=True, target_unitary=U_targ,
                    name=name, term_fid=term_fid, analytic=False,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    beta_r_step_size=0.5, alpha_r_step_size=0.2,
                    saving_directory=saving_directory,
                    use_displacements=True, N = N, N2=N2,
                    beta_penalty_multiplier=0,
                    minimizer_options = {'gtol': 1e-4, 'ftol':1e-4}, basinhopping_kwargs={'niter':1000, 'T':0.01})

# %%
print(CD_control_obj.unitary_fidelity())
print(CD_control_obj_new.unitary_fidelity())
# %% num grad func
def calc_num_gradient(dx, betas=None, alphas=None, phis=None, thetas=None):
    if betas is not None:
        CD_control_obj_new.betas = betas
    if alphas is not None:
        CD_control_obj_new.alphas = alphas
    if phis is not None:
        CD_control_obj_new.phis = phis
    if thetas is not None:
        CD_control_obj_new.thetas = thetas
    diff = CD_control_obj_new.unitary_fidelity() - CD_control_obj.unitary_fidelity()
    num_gradient = (diff)/(dx)
    # Reset
    CD_control_obj_new.betas = CD_control_obj.betas
    CD_control_obj_new.alphas = CD_control_obj.alphas
    CD_control_obj_new.phis = CD_control_obj.phis
    CD_control_obj_new.thetas = CD_control_obj.thetas
    return num_gradient, diff

# %% \del_{alpha_r_k} F
k = 1
dx = 1e-7
beta_r_k = np.abs(CD_control_obj.betas[k])
beta_theta_k = np.angle(CD_control_obj.betas[k])
beta_new = np.copy(CD_control_obj.betas)
beta_new[k] = (beta_r_k + dx)*np.exp(1j*beta_theta_k)
num_gradient, diff = calc_num_gradient(dx, betas=beta_new)
print(diff)
print(num_gradient)
# %% Analytic Unitary Fidelity
fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta = CD_control_obj.unitary_fid_and_grad_fid()
print(dbeta_r[k])
print("Precentage Diff: " + str((dbeta_r[k] - num_gradient)/num_gradient*100) + "%")

# %% Approx Unitary Fidelity (with fock states)
unitary_initial_states = []
for i in range(N*N2):
    unitary_initial_states.append(qt.tensor(qt.basis(N,i%N), qt.basis(N2,i//N)))
fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = CD_control_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states, testing=True)

print(fid)
print(afid)
print(adbeta_r[k])
print("Precentage Diff: " + str((adbeta_r[k] - num_gradient)/num_gradient*100) + "%")

# %% Approx Unitary Fidelity (with eigenstates)
n = N*N2
unitary_eigvals = np.array(CD_control_obj.target_unitary.eigenstates()[0])[:n]
unitary_initial_states = CD_control_obj.target_unitary.eigenstates()[1][:n]
unitary_final_states = unitary_initial_states*unitary_eigvals

fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = CD_control_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states,unitary_final_states=unitary_final_states, testing=True)
print(fid)
print(afid)
print(adbeta_r[k])
print("Precentage Diff: " + str((adbeta_r[k] - num_gradient)/num_gradient*100) + "%")

# %% Approx Unitary Fidelity (with random states)

unitary_initial_states = []
for i in range(N*N2):
    unitary_initial_states.append(qt.tensor(qt.rand_ket_haar(N=N), qt.rand_ket_haar(N=N2)))
unitary_initial_state_weights = np.ones(len(unitary_initial_states))
fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = CD_control_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states,unitary_initial_state_weights=unitary_initial_state_weights, testing=True)
print(fid)
print(afid)
print(adbeta_r[k])
print("Precentage Diff: " + str((adbeta_r[k] - num_gradient)/num_gradient*100) + "%")
# %%
