#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../")
import importlib
from CD_GRAPE.cd_grape_optimization import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

#%%
N = 20 #cavity hilbert space 
N2 = 2 #qubit hilbert space
N_blocks = 7
term_fid = 0.99
#max alpha and beta are the maximum values of alpha and beta for optimization
max_alpha = 1
max_beta = 4
name = "unitary testing"
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_grape_parameters\\"
#TODO: can we control the step size in the local optimizations?
#TODO: Are there few enough parameters we can do more of a global search? 
# Just sweep each parameter over its range?
cd_grape_obj = CD_grape(N_blocks = N_blocks, unitary_optimization=True,
                    name=name, term_fid=term_fid, analytic=False,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    beta_r_step_size=0.5, alpha_r_step_size=0.2,
                    saving_directory=saving_directory,
                    use_displacements=True, N = N, N2=N2,
                    beta_penalty_multiplier=0,
                    minimizer_options = {'gtol': 1e-4, 'ftol':1e-4}, basinhopping_kwargs={'niter':1000, 'T':0.01})
# %% Setup Target
U_targ = cd_grape_obj.CD(2)*cd_grape_obj.D(1)*cd_grape_obj.R(1,1)
# U_targ = qt.tensor(qt.destroy(N).dag(), qt.identity(N2))
# U_targ = (1j*np.pi*cd_grape_obj.a.dag()*cd_grape_obj.a*cd_grape_obj.sz).expm()

# Checking Fidelity Metric Normalization
D = N*N2
overlap = (U_targ.dag() * U_targ).tr()
fid =  np.abs((1 / D) * overlap) ** 2
print(fid)

cd_grape_obj.target_unitary = U_targ

#%% Doing the optimization
#The alpha and beta scale are scales for the random initialization.
cd_grape_obj.randomize(alpha_scale=0.2, beta_scale=3)
#cd_grape_obj.betas = [-2.0]
print("Randomized parameters:")
cd_grape_obj.print_info()
cd_grape_obj.optimize()
print("after optimization:")
cd_grape_obj.print_info()
#%% After some optimization
print(cd_grape_obj.unitary_fidelity())
# %% Setup cd_grape_obj_new
cd_grape_obj_new = CD_grape(betas=cd_grape_obj.betas,alphas=cd_grape_obj.alphas,phis=cd_grape_obj.phis,
                    thetas=cd_grape_obj.thetas,N_blocks = N_blocks, unitary_optimization=True, target_unitary=U_targ,
                    name=name, term_fid=term_fid, analytic=False,\
                    max_alpha = max_alpha, max_beta=max_beta,
                    beta_r_step_size=0.5, alpha_r_step_size=0.2,
                    saving_directory=saving_directory,
                    use_displacements=True, N = N, N2=N2,
                    beta_penalty_multiplier=0,
                    minimizer_options = {'gtol': 1e-4, 'ftol':1e-4}, basinhopping_kwargs={'niter':1000, 'T':0.01})

# %%
print(cd_grape_obj.unitary_fidelity())
print(cd_grape_obj_new.unitary_fidelity())
# %% num grad func
def calc_num_gradient(dx, betas=None, alphas=None, phis=None, thetas=None):
    if betas is not None:
        cd_grape_obj_new.betas = betas
    if alphas is not None:
        cd_grape_obj_new.alphas = alphas
    if phis is not None:
        cd_grape_obj_new.phis = phis
    if thetas is not None:
        cd_grape_obj_new.thetas = thetas
    diff = cd_grape_obj_new.unitary_fidelity() - cd_grape_obj.unitary_fidelity()
    num_gradient = (diff)/(dx)
    # Reset
    cd_grape_obj_new.betas = cd_grape_obj.betas
    cd_grape_obj_new.alphas = cd_grape_obj.alphas
    cd_grape_obj_new.phis = cd_grape_obj.phis
    cd_grape_obj_new.thetas = cd_grape_obj.thetas
    return num_gradient, diff

# %% \del_{alpha_r_k} F
k = 1
dx = 1e-5
alpha_r_k = np.abs(cd_grape_obj.alphas[k])
alpha_theta_k = np.angle(cd_grape_obj.alphas[k])
alphas_new = np.copy(cd_grape_obj.alphas)
alphas_new[k] = (alpha_r_k + dx)*np.exp(1j*alpha_theta_k)
num_gradient, diff = calc_num_gradient(dx, alphas=alphas_new)
print(diff)
print(num_gradient)
# %%
fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta = cd_grape_obj.unitary_fid_and_grad_fid()
print(dalpha_r[k])
print("Precentage Diff: " + str((dalpha_r[k] - num_gradient)/num_gradient*100) + "%")

# %%
