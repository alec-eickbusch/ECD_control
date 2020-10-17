#%%
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

plt.rcParams.update({"font.size": 16, "pdf.fonttype": 42, "ps.fonttype": 42})
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

#%%


def interp(data_array, dt=1):
    ts = np.arange(0, len(data_array)) * dt
    return interp1d(
        ts, data_array, kind="cubic", bounds_error=False
    )  # can test different kinds


# still having problems with this version
"""
def alpha_from_epsilon(epsilon_array, dt=1, K = 0, alpha_init=0+0j, kappa_cavity=0):
    t_eval = np.linspace(0,len(epsilon_array)*dt-dt, len(epsilon_array))
    epsilon = interp(epsilon_array, dt)
    dalpha_dt = lambda t, alpha : -1j*epsilon(t) - (kappa_cavity/2)*alpha# - 1j*4*K*np.abs(alpha)**2 * alpha
    alpha = solve_ivp(dalpha_dt,(0,len(epsilon_array)*dt-dt),y0=[alpha_init],\
                      method='RK23',t_eval=t_eval).y[0]
    return alpha
"""


def alpha_from_epsilon(epsilon):
    alpha = -1j * np.cumsum(epsilon)
    return alpha


# %%
