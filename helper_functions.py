#%%
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16,'pdf.fonttype':42,'ps.fonttype':42})
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad
#%%
def plot_wigner(state, xvec = np.linspace(-5,5,81), tensor_state=True):
    if tensor_state:
        state = qt.ptrace(state, 0)
    W = qt.wigner(state,xvec*np.sqrt(2),xvec*np.sqrt(2))
    plt.figure(figsize=(6,5))
    plt.pcolormesh(xvec,xvec,W, cmap='seismic', vmin=-1, vmax=+1)
    plt.axhline(0,linestyle='--', color='black',alpha=0.4)
    plt.axvline(0,linestyle='--', color='black',alpha=0.4)
    plt.colorbar()
    plt.grid()

def plot_pulse(epsilon, Omega):
    ts = np.arange(len(epsilon))
    plt.plot(ts,1e3*np.real(epsilon)/2/np.pi,label='Re(epsilon)')
    plt.plot(ts,1e3*np.imag(epsilon)/2/np.pi,label='Im(epsilon)')
    plt.plot(ts,20*1e3*np.real(Omega)/2/np.pi,label='20*Re(Omega)')
    plt.plot(ts,20*1e3*np.imag(Omega)/2/np.pi,label='20*Im(Omega)')
    plt.ylabel('drive amplitude (MHz)')
    plt.xlabel('t (ns)')
    plt.legend()

def interp(data_array, dt=1):
    ts = np.arange(0,len(data_array))*dt
    return interp1d(ts,data_array,kind='cubic', bounds_error=False) #can test different kinds

#still having problems with this version
'''
def alpha_from_epsilon(epsilon_array, dt=1, K = 0, alpha_init=0+0j, kappa_cavity=0):
    t_eval = np.linspace(0,len(epsilon_array)*dt-dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    dalpha_dt = lambda t, alpha : -1j*epsilon(t) - (kappa_cavity/2)*alpha# - 1j*4*K*np.abs(alpha)**2 * alpha
    alpha = solve_ivp(dalpha_dt,(0,len(epsilon_array)*dt-dt),y0=[alpha_init],\
                      method='RK23',t_eval=t_eval).y[0]
    return alpha
'''
def alpha_from_epsilon(epsilon):
    alpha = -1j*np.cumsum(epsilon)
    return alpha
# %%
