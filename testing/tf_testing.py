#%%
%load_ext autoreload
%autoreload 2
from CD_control.CD_control_tf import CD_control_tf
from CD_control.CD_control_optimization import CD_control
import qutip as qt
import numpy as np
import tensorflow as tf
#%%
N = 80
psi_i = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
psi_t = qt.tensor(qt.basis(2, 0), qt.basis(N, 1))
#%%
N_blocks = 4
betas = np.array([
    np.random.uniform(low=-3.0,high=3.0) + 1j*np.random.uniform(low=-3.0,high=3.0)
    for _ in range(N_blocks)]
    )
phis = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
thetas = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
#%%
obj_tf = CD_control_tf(initial_state=psi_i, target_state=psi_t, N_blocks = N_blocks, Bs=betas/2.0)
obj = CD_control(initial_state=psi_i, target_state=psi_t, N_blocks=N_blocks,\
                    use_displacements=False, analytic=True,
                    no_CD_end=False, betas=betas)
#%%
d_tf = (obj_tf.construct_displacement_operators(obj_tf.Bs)).numpy()
#%%
d = np.array([obj.D(obj.betas[i]/2.0).ptrace(1).full()/2.0 for i in range(N_blocks)])

#%%
cut = 50
print(np.max(np.abs((d_tf - d)[:,:cut,:cut])))
# %%
b_tf = obj_tf.construct_block_operators(Bs = obj_tf.Bs, Phis = obj_tf.Phis, Thetas=obj_tf.Thetas).numpy()
#%%
b = np.array([obj.U_i_block(i).full() for i in range(N_blocks)])
#%%
cut = 50
print(np.max(np.abs((b_tf - b)[:,:cut,:cut])))
#%%
# %%
import CD_control.tf_quantum as tfq
ds = obj_tf.construct_displacement_operators(obj_tf.Bs)
ds_dag = tf.linalg.adjoint(ds)
Phis = tf.cast(tfq.matrix_flatten(obj_tf.Phis), dtype=tf.complex64)
Thetas = tf.cast(tfq.matrix_flatten(obj_tf.Thetas), dtype=tf.complex64)
#%%
exp = tf.math.exp(tf.constant(1j, dtype=tf.complex64) * Phis)
exp_dag = tf.linalg.adjoint(exp)
cos = tf.math.cos(Thetas)
sin = tf.math.sin(Thetas)
#%%
# constructing the blocks of the matrix
ul = cos * ds
ll = exp * sin * ds_dag
ur = tf.constant(-1, dtype=tf.complex64) * exp_dag * sin * ds
lr = cos * ds_dag
#%%
blocks = tf.concat([tf.concat([ul, ur], 2), tf.concat([ll, lr], 2)], 1)
# %%
