import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16,'pdf.fonttype':42,'ps.fonttype':42})
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

