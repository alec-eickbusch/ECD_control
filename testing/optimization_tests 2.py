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
#%%
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
#%% Analysis
savefiles = ['Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 1_N_blocks_3_20200903_12_42_40', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 1_N_blocks_4_20200903_12_49_55', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 1_N_blocks_5_20200903_12_51_20', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 1_N_blocks_6_20200903_12_54_49', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 2_N_blocks_3_20200903_12_58_08', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 2_N_blocks_4_20200903_13_08_57', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 2_N_blocks_5_20200903_13_36_32', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 2_N_blocks_6_20200903_13_41_53', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 3_N_blocks_3_20200903_13_44_17', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 3_N_blocks_4_20200903_13_46_14', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 3_N_blocks_5_20200903_14_03_27', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 3_N_blocks_6_20200903_14_17_06', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 4_N_blocks_3_20200903_14_22_05', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 4_N_blocks_4_20200903_14_26_48', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 4_N_blocks_5_20200903_14_50_20', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 4_N_blocks_6_20200903_15_20_45', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 5_N_blocks_3_20200903_15_24_19', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 5_N_blocks_4_20200903_15_32_23', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 5_N_blocks_5_20200903_15_45_46', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 5_N_blocks_6_20200903_16_20_42', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=1_N_blocks_3_20200903_16_24_45', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=1_N_blocks_4_20200903_16_29_22', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=1_N_blocks_5_20200903_16_34_48', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=1_N_blocks_6_20200903_16_39_41', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=2_N_blocks_3_20200903_16_44_26', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=2_N_blocks_4_20200903_16_46_56', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=2_N_blocks_5_20200903_16_49_36', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=2_N_blocks_6_20200903_16_51_45', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=3_N_blocks_3_20200903_16_57_09', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=3_N_blocks_4_20200903_17_02_31', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=3_N_blocks_5_20200903_17_07_29', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=3_N_blocks_6_20200903_17_17_55', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=4_N_blocks_3_20200903_17_19_36', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=4_N_blocks_4_20200903_17_25_57', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=4_N_blocks_5_20200903_17_30_19', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=4_N_blocks_6_20200903_17_58_35', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=5_N_blocks_3_20200903_18_00_51', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=5_N_blocks_4_20200903_18_02_31', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=5_N_blocks_5_20200903_18_22_56', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\cat n=5_N_blocks_6_20200903_18_28_45', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\0 + 4_N_blocks_3_20200903_18_31_41', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\0 + 4_N_blocks_4_20200903_18_37_25', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\0 + 4_N_blocks_5_20200903_18_53_18', 'Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\0 + 4_N_blocks_6_20200903_19_06_49']
a = CD_grape()
fids = []
fids_inner = []
fid_names = []
i = 0
for savefile in savefiles:
    a.load(savefile)
    fid = a.fidelity()
    fids_inner.append(fid)
    i+=1
    if i%4 == 0:
        fids.append(fids_inner)
        fids_inner = []
        fid_names.append(str(a.name)[:-11])


# %%
plt.figure(figsize=(8,8),dpi=200)
N_blocks = [3,4,5,6]
for i,fids_inner in enumerate(fids):
    plt.semilogy(N_blocks,1-np.array(fids_inner), '--.', label=fid_names[i])
plt.legend()
plt.xticks(N_blocks)
plt.grid()
plt.title('Discrete CD Optimization')
plt.xlabel('Number of Conditional Displacements')
plt.ylabel('Optimization infidelity')
savefile = saving_directory + 'result_fids.png'
plt.savefig(savefile, filetype='png')
print('plot saved as:' + str(savefile))
# %%
