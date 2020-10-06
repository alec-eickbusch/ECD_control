#%%
%load_ext autoreload
%autoreload 2
import sys
sys.path.append("../../")
import importlib
from CD_control.CD_control_optimization import *
from CD_control.basic_pulses import *
from CD_control.helper_functions import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 60  # cavity hilbert space
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
saving_directory = "C:\\Users\\Alec Eickbusch\\Documents\\CD_control_parameters\\"
CD_control_obj = CD_control(initial_state, target_state, N_blocks,
                        name=name, term_fid=term_fid,
                        max_alpha=max_alpha, max_beta=max_beta,
                        saving_directory=saving_directory)

#%% Test d_r D(r e^(i theta)) derivative
alpha = -3j - 3
alpha_r = np.abs(alpha)
alpha_theta = np.angle(alpha)
dalpha_r = 1e-5
alpha_new = (alpha_r + dalpha_r)*np.exp(1j*alpha_theta)
real_der_num = ((CD_control_obj.D(alpha_new) - CD_control_obj.D(alpha))/dalpha_r)
real_der_ana = CD_control_obj.dalpha_r_dD(alpha)
diff = real_der_num - real_der_ana
diff = diff[:50, :50] #they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))

#%% Test d_theta D(r e^(i theta)) derivative
alpha = -3j - 3
alpha_r = np.abs(alpha)
alpha_theta = np.angle(alpha)
dalpha_theta = 1e-5
alpha_new = (alpha_r)*np.exp(1j*(alpha_theta+dalpha_theta))
real_der_num = ((CD_control_obj.D(alpha_new) - CD_control_obj.D(alpha))/dalpha_theta)
real_der_ana = CD_control_obj.dalpha_theta_dD(alpha)
diff = real_der_num - real_der_ana
divide = real_der_num * np.reciprocal(real_der_ana)
diff = diff[:50, :50] #they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))

#%%
beta = 4 + 2j
beta_r = np.abs(beta)
beta_theta = np.angle(beta)
dbeta_r = 1e-6
beta_new = (beta_r + dbeta_r)*np.exp(1j*beta_theta)
real_der_num = ((CD_control_obj.CD(beta_new) - CD_control_obj.CD(beta))/np.abs(dbeta_r))
real_der_ana = CD_control_obj.dbeta_r_dCD(beta)
diff = real_der_num - real_der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))

#%%
beta = 4 + 2j
beta_r = np.abs(beta)
beta_theta = np.angle(beta)
dbeta_theta = 1e-6
beta_new = (beta_r)*np.exp(1j*(beta_theta+dbeta_theta))
real_der_num = ((CD_control_obj.CD(beta_new) - CD_control_obj.CD(beta))/np.abs(dbeta_theta))
real_der_ana = CD_control_obj.dbeta_theta_dCD(beta)
diff = real_der_num - real_der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))

#%%
theta = -0.2
phi =  4.2
dtheta = 1e-3
der_num = ((CD_control_obj.R(phi, theta + dtheta) -
                 CD_control_obj.R(phi, theta))/np.abs(dtheta))
der_ana = CD_control_obj.dtheta_dR(phi, theta)
diff = der_num - der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))
#%%
theta = 2
phi = -.5
dphi = 1e-3
der_num = ((CD_control_obj.R(phi + dphi, theta) -
            CD_control_obj.R(phi, theta))/np.abs(dphi))
der_ana = CD_control_obj.dphi_dR(phi, theta)
diff = der_num - der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
print(np.max(np.abs(real_der_num)))
# %%
