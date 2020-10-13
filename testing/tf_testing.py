#%%
%load_ext autoreload
%autoreload 2
from CD_control.CD_control_tf import CD_control_tf
from CD_control.CD_control_optimization import CD_control
import CD_control.tf_quantum as tfq
import qutip as qt
import numpy as np
import tensorflow as tf
#%%
x = tf.cast(tf.random.uniform((300,1)), dtype=tf.complex64) + 1j*tf.cast(tf.random.uniform((300,1)), dtype=tf.complex64)
print(x)
y = tf.linalg.adjoint(x)
print(y)
#%%
N = 100
psi_i = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
alpha = 2 + 1j #cat alpha
N_blocks = 5
psi_t = qt.tensor(qt.basis(2,0), (qt.coherent(N,alpha) + qt.coherent(N,-alpha)).unit())
#%%
N_blocks = 6
betas = np.array([
    np.random.uniform(low=-1,high=1) + 1j*np.random.uniform(low=-1,high=1)
    for _ in range(N_blocks)]
    )
phis = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
phis[0] = 0
thetas = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
#%%
obj_tf = CD_control_tf(initial_state=psi_i, target_state=psi_t,
                 N_blocks = N_blocks, term_fid=0.999,
                 betas=betas, phis=phis, thetas=thetas, no_CD_end=False)
obj_tf.print_info()
#%%
obj = CD_control(initial_state=psi_i, target_state=psi_t, N_blocks=N_blocks,
                    use_displacements=False, analytic=True,
                    no_CD_end=False, betas=betas, phis=phis, thetas=thetas)
obj.print_info()


#%%
psif = obj.final_state()
psif_tf = obj_tf.final_state(obj_tf.betas_rho, obj_tf.betas_angle, obj_tf.phis, obj_tf.thetas)
#%%
psit = obj.target_state
psit_tf = obj_tf.target_state
#%%
psit_dag = psit.dag()
psit_tf_dag = tf.linalg.adjoint(psit_tf)
#%%

#%%
obj_tf.plot_final_state()
#%%
obj.plot_state(i=-1)
#%%
obj_tf.optimize(learning_rate = 0.01, epoch_size=20, epochs=40)
#%%
d_tf = (obj_tf.construct_displacement_operators(obj_tf.betas_rho, obj_tf.betas_angle)).numpy()
#%%
d = np.array([obj.D(obj.betas[i]).ptrace(1).full()/2.0 for i in range(N_blocks)])

#%%
cut = 80
print(np.max(np.abs((d_tf - d)[:,:cut,:cut])))
# %%
b_tf = obj_tf.construct_block_operators(betas_rho = obj_tf.betas_rho, betas_angle=obj_tf.betas_angle,
                                 phis = obj_tf.phis, thetas=obj_tf.thetas).numpy()
#%%
b = np.array([obj.U_i_block(i).full() for i in range(N_blocks)])
#%%
cut = 90
print(np.max(np.abs((b_tf - b)[:,:cut,:cut])))
#%%
overlap = obj_tf.state_overlap(obj_tf.betas_rho, obj_tf.betas_angle, obj_tf.phis, obj_tf.thetas)
#%%
fid_tf = obj_tf.state_fidelity(obj_tf.betas_rho, obj_tf.betas_angle, obj_tf.phis, obj_tf.thetas)
print(fid_tf)
#%%
fid = obj.fidelity()
print(fid)
#%%
plt.figure()
obj.plot_state(i=-1)
plt.figure()
obj_tf.plot_final_state()

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
t = tf.constant([[[1, 0],
        [0, 1]],
       [[1, 2],
        [3, 4]]], dtype=tf.complex64)
U = tf.eye(2,2, dtype=tf.complex64)
pi = obj_tf.initial_state
#%%
for M in tf.reverse(t, axis=[0]):
    pi = M @ pi
# %%
pi = obj_tf.initial_state
b0 = b_tf[0]
pt = obj_tf.target_state
pt_dag = tf.linalg.adjoint(pt)
# %%
overlap = pt_dag @ b0 @ pi
fid = tf.cast(overlap * tf.math.conj(overlap), dtype=tf.float32)
# %%
