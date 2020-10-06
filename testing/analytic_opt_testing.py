#%%
%load_ext autoreload
%autoreload 2
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

#%% First thing is to make a basic test of the derivative functions
alpha = -3j - 3
dalpha = 1e-5
real_der_num = ((CD_control_obj.D(alpha+dalpha) - CD_control_obj.D(alpha))/dalpha)
real_der_ana = CD_control_obj.dalphar_dD(alpha)
diff = real_der_num - real_der_ana
diff = diff[:50, :50] #they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
#%%
dalpha = 1e-4*1j
imag_der_num = ((CD_control_obj.D(alpha+dalpha) - CD_control_obj.D(alpha))/np.abs(dalpha))
imag_der_ana = CD_control_obj.dalphai_dD(alpha)
diff = imag_der_num - imag_der_ana
diff = diff[:50, :50]  # they differ near the edges of the hilbert space
print(repr(diff))
print(np.max(np.abs(diff)))
#%%
beta = 4 + 2j
dbeta = 1e-6
real_der_num = ((CD_control_obj.CD(beta + dbeta) - CD_control_obj.CD(beta))/np.abs(dbeta))
real_der_ana = CD_control_obj.dbetar_dCD(beta)
diff = real_der_num - real_der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
beta = 2 - 3j
dbeta = 1e-6*1j
imag_der_num = ((CD_control_obj.CD(beta + dbeta) -
                 CD_control_obj.CD(beta))/np.abs(dbeta))
imag_der_ana = CD_control_obj.dbetai_dCD(beta)
diff = imag_der_num - imag_der_ana
diff = diff[:30, :30]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
theta = -0.2
phi =  4.2
dtheta = 1e-3
der_num = ((CD_control_obj.R(phi, theta + dtheta) -
                 CD_control_obj.R(phi, theta))/np.abs(dtheta))
der_ana = CD_control_obj.dtheta_dR(phi, theta)
diff = der_num - der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))
#%%
theta = 2
phi = -.5
dphi = 1e-3
der_num = ((CD_control_obj.R(phi + dphi, theta) -
            CD_control_obj.R(phi, theta))/np.abs(dphi))
der_ana = CD_control_obj.dphi_dR(phi, theta)
diff = der_num - der_ana
diff = diff[:10, :10]  # they differ near the edges of the hilbert space
#print(repr(diff))
print(np.max(np.abs(diff)))