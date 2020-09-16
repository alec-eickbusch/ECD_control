#%%
%load_ext autoreload
%autoreload 2
from CD_GRAPE.cd_grape_optimization import CD_grape
from CD_GRAPE.helper_functions import plot_pulse, plot_wigner
from CD_GRAPE.analysis import System, CD_grape_analysis
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
saving_directory = "Z:\\Data\\Tennessee2020\\20200915_cooldown\\fock_state_CD_grape_optimizations\\"
cd_grape_obj = CD_grape(initial_state=initial_state, term_fid=term_fid,
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
    cd_grape_obj.target_state = target_state
    best_fid = 0
    while (not sucessful) and (N_blocks <= max_N_blocks):
        name = 'fock %d N_blocks %d' % (fock, N_blocks)
        print("\n\n" + name + "\n\n")
        cd_grape_obj.N_blocks = N_blocks
        cd_grape_obj.name = name
        cd_grape_obj.randomize(alpha_scale=alpha_scale, beta_scale=beta_scale)
        cd_grape_obj.optimize()
        savefile_list[fock].append(cd_grape_obj.save())
        fid = cd_grape_obj.fidelity()
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
