#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../../")
import importlib
from CD_GRAPE.cd_grape_optimization import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from tqdm import tqdm

#%%
N = 20 #cavity hilbert space 
N2 = 2 #qubit hilbert space
N_blocks = 8
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
cd_grape_obj.randomize(alpha_scale=0.2, beta_scale=3)
U_targ = cd_grape_obj.U_tot()*qt.tensor(qt.rand_unitary(N, density=.01),qt.rand_unitary(N2, density=.01))
cd_grape_obj.target_unitary = U_targ

#%% Check Fidelity < 1
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

k = 2
dx = 1e-7
beta_r_k = np.abs(cd_grape_obj.betas[k])
beta_theta_k = np.angle(cd_grape_obj.betas[k])
beta_new = np.copy(cd_grape_obj.betas)
beta_new[k] = (beta_r_k + dx)*np.exp(1j*beta_theta_k)
num_gradient, diff = calc_num_gradient(dx, betas=beta_new)
analytic_fid = cd_grape_obj.unitary_fidelity()
print(diff)
print(num_gradient)

# %% Analytic Unitary Fidelity
fid, dbeta_r, dbeta_theta, dalpha_r, dalpha_theta, dphi, dtheta = cd_grape_obj.unitary_fid_and_grad_fid()
analytic_grad = dbeta_r[k]
print("Precentage Diff: " + str((dbeta_r[k] - num_gradient)/num_gradient*100) + "%")

#%% Setup Vars

data = {}
D = N*N2
data['N'] = np.array([D//20,D//10,D//5,D//2,D*5//8,D*6//8,D*7//8,D])
# data['num_grad'] = num_gradient*np.ones(data['N'])
# data['analytic_grad'] = analytic_grad*np.ones(data['N'])
# data['analytic_fid'] = fid*np.ones(data['N'])
data['approx_eig_fid'] = []
data['approx_eig_grad'] = []
data['approx_fock_fid'] = []
data['approx_fock_grad'] = []
data['approx_random_fid'] = []
data['approx_random_grad'] = []




# %% Approx Unitary Fidelity (with fock states)
def approx_grad_fid_fock(D, printing=False):
    unitary_initial_states = []
    for i in range(D):
        unitary_initial_states.append(qt.tensor(qt.basis(N,i%N), qt.basis(N2,i//N)))
    fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = cd_grape_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states, testing=True)
    
    percent_diff_fid = np.abs((afid - analytic_fid)/analytic_fid*100)
    percent_diff_grad = np.abs((adbeta_r[k] - num_gradient)/num_gradient*100)
    data['approx_fock_fid'].append(percent_diff_fid)
    data['approx_fock_grad'].append(percent_diff_grad)

    if printing:
        print("==== fock ====")
        print(fid)
        print(afid)
        print(adbeta_r[k])
        print("Precentage Diff in Grad from Num: " + str(percent_diff_grad) + "%")

# %% Approx Unitary Fidelity (with eigenstates)
def approx_grad_fid_eig(D, printing=False):
    unitary_eigvals = np.array(cd_grape_obj.target_unitary.eigenstates()[0])[:D]
    unitary_initial_states = cd_grape_obj.target_unitary.eigenstates()[1][:D]
    unitary_final_states = unitary_initial_states*unitary_eigvals

    fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = cd_grape_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states,unitary_final_states=unitary_final_states, testing=True)

    percent_diff_fid = np.abs((afid - analytic_fid)/analytic_fid*100)
    percent_diff_grad = np.abs((adbeta_r[k] - num_gradient)/num_gradient*100)
    data['approx_eig_fid'].append(percent_diff_fid)
    data['approx_eig_grad'].append(percent_diff_grad)

    if printing:
        print("==== eig ====")
        print(fid)
        print(afid)
        print(adbeta_r[k])
        print("Precentage Diff in Grad from Num: " + str(percent_diff_grad) + "%")

# %% Approx Unitary Fidelity (with random states)

def approx_grad_fid_random(D, printing=False):
    unitary_initial_states = []
    for i in range(D):
        unitary_initial_states.append(qt.tensor(qt.rand_ket_haar(N=N), qt.rand_ket_haar(N=N2)))
    fid, afid, adbeta_r, adbeta_theta, adalpha_r, adalpha_theta, adphi, adtheta = cd_grape_obj.unitary_fid_and_grad_fid_approx(unitary_initial_states=unitary_initial_states, testing=True)

    percent_diff_fid = np.abs((afid - analytic_fid)/analytic_fid*100)
    percent_diff_grad = np.abs((adbeta_r[k] - num_gradient)/num_gradient*100)
    data['approx_random_fid'].append(percent_diff_fid)
    data['approx_random_grad'].append(percent_diff_grad)

    if printing:
        print("==== random ====")
        print(fid)
        print(afid)
        print(adbeta_r[k])
        print("Precentage Diff in Grad from Num: " + str(percent_diff_grad) + "%")
# %% Run Loop

for D in tqdm(data['N']):
    approx_grad_fid_fock(D)
    approx_grad_fid_eig(D)
    approx_grad_fid_random(D)

# %% Plot
fig = plt.figure(figsize=(12,8))
plt.xlabel("Number of States")
plt.ylabel("Percent Difference from Numerical Gradient")        
plt.scatter(data['N'], data['approx_fock_grad'], marker='o',color='g',label="Fock States")
plt.scatter(data['N'], data['approx_random_grad'], color='b',label="Random States")
plt.scatter(data['N'], data['approx_eig_grad'], marker='.',color='r', label="Eigenstates")
plt.legend()
plt.savefig('data/NumericalFidelityComparison.png')
plt.yscale('log')
plt.savefig('data/NumericalFidelityComparison_log.png')

fig = plt.figure(figsize=(12,8))
plt.xlabel("Number of States")
plt.ylabel("Percent Difference from Gradient Fidelity")        
plt.scatter(data['N'], data['approx_fock_fid'], marker='o', color='g',label="Fock States")
plt.scatter(data['N'], data['approx_random_fid'], color='b', label="Random States")
plt.scatter(data['N'], data['approx_eig_fid'], marker='.', color='r', label="Eigenstates")
plt.legend()
plt.savefig('data/NumericalGradientComparison.png')
plt.yscale('log')
plt.savefig('data/NumericalGradientComparison_log.png')

plt.show()


# %%
