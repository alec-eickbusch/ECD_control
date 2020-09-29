#%%
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
plt.rcParams.update({'font.size': 16,'pdf.fonttype':42,'ps.fonttype':42})
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad
#%%

def plot_wigner(state, tensor_state=True, contour=False, fig=None, ax=None,max_alpha=6):
    xvec = np.linspace(-max_alpha,max_alpha,81)
    if fig is None:
        fig = plt.figure(figsize=(6,5))
    if ax is None:
        ax = fig.subplots()
    if tensor_state:
        state = qt.ptrace(state, 0)
    W = (np.pi/2.0)*qt.wigner(state,xvec,xvec, g=2)
    dx = xvec[1] - xvec[0]
    
    if contour:
        levels = np.linspace(-1.1, 1.1, 102)
        im = ax.contourf(xvec, xvec, W, cmap='seismic',
                        vmin=-1, vmax=+1, levels = levels)
    else:
        im = ax.pcolormesh(xvec-dx/2.0, xvec-dx/2.0, W, cmap='seismic',
                    vmin=-1, vmax=+1)
    ax.axhline(0, linestyle='-', color='black', alpha=0.8)
    ax.axvline(0, linestyle='-', color='black', alpha=0.8)
    ax.set_xlabel(r'Re$(\alpha)$')
    ax.set_ylabel(r'Im$(\alpha)$')
    ax.grid()
    #ax.set_title(title)
    #TODO: Gaurentee that the wigners are square!
    fig.tight_layout()
    fig.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    ticks = np.linspace(-1, 1, 5)
    fig.colorbar(im, cax=cbar_ax, ticks=ticks)
    cbar_ax.set_title(r'$\frac{\pi}{2} W(\alpha)$', pad=20)

def plot_pulse(epsilon, Omega):
    ts = np.arange(len(epsilon))
    plt.plot(ts,1e3*np.real(epsilon)/2/np.pi,label='Re(epsilon)')
    plt.plot(ts,1e3*np.imag(epsilon)/2/np.pi,label='Im(epsilon)')
    plt.plot(ts,10*1e3*np.real(Omega)/2/np.pi,label='10*Re(Omega)')
    plt.plot(ts,10*1e3*np.imag(Omega)/2/np.pi,label='10*Im(Omega)')
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
