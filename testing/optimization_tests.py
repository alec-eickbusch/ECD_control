#%%
from CD_GRAPE.cd_grape import *
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
#%%
N = 50
N2 = 2
#%%
target_states = {
    'fock 1': qt.tensor(qt.basis(N,1),qt.basis(N2,0)),
    'fock 2': qt.tensor(qt.basis(N,2),qt.basis(N2,0)),
    'fock 3': qt.tensor(qt.basis(N,3),qt.basis(N2,0)),
    'fock 4': qt.tensor(qt.basis(N,4),qt.basis(N2,0)),
    'fock 5': qt.tensor(qt.basis(N,5),qt.basis(N2,0)),
    'cat n=1': qt.tensor((qt.coherent(N,np.sqrt(1)) + qt.coherent(N,-np.sqrt(1))).unit(),qt.basis(N2,0)),
    'cat n=2': qt.tensor((qt.coherent(N,np.sqrt(1)) + qt.coherent(N,-np.sqrt(1))).unit(),qt.basis(N2,0)),
    'cat n=3': qt.tensor((qt.coherent(N,np.sqrt(3)) + qt.coherent(N,-np.sqrt(3))).unit(),qt.basis(N2,0)),
    'cat n=4': qt.tensor((qt.coherent(N,np.sqrt(4)) + qt.coherent(N,-np.sqrt(4))).unit(),qt.basis(N2,0)),
    'cat n=5': qt.tensor((qt.coherent(N,np.sqrt(5)) + qt.coherent(N,-np.sqrt(5))).unit(),qt.basis(N2,0)),
    '0 + 4': qt.tensor((qt.basis(N,0) + qt.basis(N,4))/np.sqrt(2.0),qt.basis(N2,0))
}
initial_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
saving_directory = "Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\"
#%%
savefiles = []
fidelities = []
N_blocks_attempt = [3,4,5,6]
term_fid = 0.99
for name, target_state in target_states.items():
    for N_blocks in N_blocks_attempt:
        name2 = name + '_N_blocks_' + str(N_blocks)
        print("\n\n" + name2 + "\n\n")
        a = CD_grape(initial_state, target_state, N_blocks, max_alpha=4,max_beta = 4, name=name2,\
                     saving_directory=saving_directory, term_fid = term_fid)
        a.randomize(alpha_scale=0.5, beta_scale = 1)
        fid = a.optimize()
        fidelities.append(fid)
        f = a.save()
        savefiles.append(f)
        print('fidelities:' + str(fidelities))
        print('savefiles:' + str(savefiles))
