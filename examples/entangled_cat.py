#%%
import numpy as np
import qutip as qt 
from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

N = 40
alpha = 2.0
psi_t = (qt.tensor(qt.basis(2,1),qt.coherent(N,-alpha)) + qt.tensor(qt.basis(2,0),qt.coherent(N,alpha))).unit()

opt_params = {
'N_blocks' : 1, 
'N_multistart' : 20, 
'epochs' : 200, 
'epoch_size' : 10, 
'learning_rate' : 0.01, 
'term_fid' : 0.999, 
'dfid_stop' : 1e-6, 
'beta_scale' : 4.0, 
'initial_states' : [qt.tensor(qt.basis(2,0),qt.basis(N,0))], 
'target_states' : [psi_t],
'name' : 'cat alpha=%.3f' % alpha,
'filename' : None,
}

opt = BatchOptimizer(**opt_params)
opt.optimize()

#can print info, including the best circuit found.
opt.print_info()
# %%
