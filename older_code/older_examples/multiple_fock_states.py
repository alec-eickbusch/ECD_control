#%%
%load_ext autoreload
%autoreload 2
from CD_control.CD_control_optimization import CD_control
from CD_control.helper_functions import plot_pulse, plot_wigner
from CD_control.analysis import System, CD_control_analysis
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
#later, for a test like this, put each fock state on a different node
target_fock_values = np.arange(1,11)
max_N_blocks = 15
term_fid = 0.99
niter = 50
#%% The object shared by all optimizations
N = 80  # cavity hilbert space
N2 = 2  # qubit hilbert space
initial_state = qt.tensor(qt.basis(N, 0), qt.basis(N2, 0))
max_alpha = 6
max_beta = 8
alpha_scale = 0.5
beta_scale = 2
#all else will use default parameters in CD grape
saving_directory = "Z:\\Data\\Tennessee2020\\20200915_cooldown\\fock_state_CD_control_optimizations\\"
CD_control_obj = CD_control(initial_state=initial_state, term_fid=term_fid,
                        max_alpha=max_alpha, max_beta=max_beta,
                        saving_directory=saving_directory,
                        basinhopping_kwargs={'niter': niter},
                        save_all_minima=True)
#%%
savefile_list = {}
fidelities = {}
N_blocks_used = {}
for fock in target_fock_values:
    savefile_list[fock] = []
    fidelities[fock] = []
    N_blocks_used[fock] = []
    sucessful = False
    N_blocks = 1
    target_state = qt.tensor(qt.basis(N, fock), qt.basis(N2, 0))
    CD_control_obj.target_state = target_state
    best_fid = 0
    while (not sucessful) and (N_blocks <= max_N_blocks):
        name = 'fock %d N_blocks %d' % (fock, N_blocks)
        print("\n\n" + name + "\n\n")
        CD_control_obj.N_blocks = N_blocks
        CD_control_obj.name = name
        CD_control_obj.randomize(alpha_scale=alpha_scale, beta_scale=beta_scale)
        CD_control_obj.optimize()
        savefile_list[fock].append(CD_control_obj.save())
        fid = CD_control_obj.fidelity()
        fidelities[fock].append(fid)
        N_blocks_used[fock].append(N_blocks)
        sucessful = (fid >= term_fid)
        N_blocks += 1
#%%
print(savefile_list)
print(fidelities)
print(N_blocks_used)

        
# %%
plt.figure(figsize=(8,6),dpi=200)
plt.axhline(0.99,linestyle='-',color='black', alpha=0.5)
for fock, fids in fidelities.items():
    plt.plot(N_blocks_used[fock],fids, '--.', label='fock %d' % fock)
plt.xlabel('N blocks')
plt.ylabel('Optimized fidelity')

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
figname = saving_directory + 'fock_results.png'
plt.savefig(saving_directory + 'fock_results.png', filetype='png')
print("fig saved as: " + figname)
# %%
