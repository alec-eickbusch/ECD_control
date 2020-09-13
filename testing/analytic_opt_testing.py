#%%
%load_ext autoreload
%autoreload 2
import importlib
from CD_GRAPE.cd_grape_optimization import *
from CD_GRAPE.basic_pulses import *
from CD_GRAPE.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 150  # cavity hilbert space
N2 = 2  # qubit hilbert space
N_blocks = 1
alpha = 1
initial_state = qt.tensor(qt.basis(N, 0), qt.basis(N2, 0))
target_state = qt.tensor(qt.coherent(N,alpha), qt.basis(N2,0))
term_fid = 0.99
#max alpha and beta are the maximum values of alpha and beta for optimization
max_alpha = 5
max_beta = 5
name = "displace"
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_grape_parameters\\"
cd_grape_obj = CD_grape(initial_state, target_state, N_blocks,
                        name=name, term_fid=term_fid,
                        max_alpha=max_alpha, max_beta=max_beta,
                        saving_directory=saving_directory)

#%% First thing is to make a basic test of the derivative functions
alpha = -3j - 3
dalpha = 1e-5
real_der_num = ((cd_grape_obj.D(alpha+dalpha) - cd_grape_obj.D(alpha))/dalpha)
real_der_ana = cd_grape_obj.dalphar_dD_mul(alpha)*cd_grape_obj.D(alpha)
diff = real_der_num - real_der_ana
diff = diff[:50, :50] #they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
#%%
dalpha = 1e-4*1j
imag_der_num = ((cd_grape_obj.D(alpha+dalpha) - cd_grape_obj.D(alpha))/np.abs(dalpha))
imag_der_ana = cd_grape_obj.dalphai_dD_mul(alpha)*cd_grape_obj.D(alpha)
diff = imag_der_num - imag_der_ana
diff = diff[:50, :50]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
#%%
beta = 4 + 2j
dbeta = 1e-6
real_der_num = ((cd_grape_obj.CD(beta + dbeta) - cd_grape_obj.CD(beta))/np.abs(dbeta))
real_der_ana = cd_grape_obj.dbetar_dCD_mul(beta)*cd_grape_obj.CD(beta)
diff = real_der_num - real_der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
beta = 2 - 3j
dbeta = 1e-6*1j
imag_der_num = ((cd_grape_obj.CD(beta + dbeta) -
                 cd_grape_obj.CD(beta))/np.abs(dbeta))
imag_der_ana = cd_grape_obj.dbetai_dCD_mul(beta)*cd_grape_obj.CD(beta)
diff = imag_der_num - imag_der_ana
diff = diff[:30, :30]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
theta = 5
phi = - -0.365
dtheta = 1e-3
der_num = ((cd_grape_obj.R(phi, theta + dtheta) -
                 cd_grape_obj.R(phi, theta))/np.abs(dtheta))
der_ana = cd_grape_obj.dtheta_dR_mul(phi, theta)*cd_grape_obj.R(phi, theta)
diff = der_num - der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
init_params = \
    np.array(np.concatenate([np.real(cd_grape_obj.alphas), np.imag(cd_grape_obj.alphas),
                             np.real(cd_grape_obj.betas), np.imag(
                                 cd_grape_obj.betas),
                             cd_grape_obj.phis, cd_grape_obj.thetas]), dtype=np.float64)

f, df = cd_grape_obj.cost_function_analytic(init_params)

test_d = np.zeros_like(init_params)
test_d[4] = 0.001
f2, df2 = cd_grape_obj.cost_function_analytic(init_params + test_d)

test_grad = (f2-f)/0.001
print(df)
print(test_grad)
print(f)
print(f2)

# %%
