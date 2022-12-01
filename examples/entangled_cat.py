#%%
import numpy as np
import qutip as qt 
from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

N = 40
alpha = 2.0

#first, target centered cat
psi_t = (qt.tensor(qt.basis(2,1),qt.coherent(N,-alpha)) + qt.tensor(qt.basis(2,0),qt.coherent(N,alpha))).unit()

opt_params = {
'N_blocks' : 1, 
'N_multistart' : 50, 
'epochs' : 100, 
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
#%%
opt = BatchOptimizer(**opt_params)
opt.optimize()
#%%
#now, target displaced cat
mean_displacement = 1.0 - 2.0j
psi_t_displaced = qt.tensor(qt.identity(2), qt.displace(N, mean_displacement))*psi_t

#optimizing without the final displacement - it won't work
opt_params.update({
'target_states' : [psi_t_displaced],
'include_final_displacement':False
})
opt = BatchOptimizer(**opt_params)
opt.optimize()
# %%
#optimizing with the final displacement included - will now optimize to a high fidelity
opt_params.update({
'include_final_displacement':True
})
opt = BatchOptimizer(**opt_params)
opt.optimize()

# %%
