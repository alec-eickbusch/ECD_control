#%%
%load_ext autoreload
%autoreload 2
from CD_control.CD_control_tf import CD_control_tf
from CD_control.CD_control_optimization import CD_control
import qutip as qt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#%%
N = 150
focks = np.arange(1,20)
max_N_blocks = 80
term_fid = 1-1e-4
#%%
psi_i = qt.tensor(qt.basis(2, 0), qt.basis(N, 0))
fids = {}
N_blocks_used = {}
epochs = 30
epoch_size = 200
learning_rate = 0.005
df_stop=1e-5
for fock in focks:
    print("\n\n fock: %d\n\n" % fock)
    fids[fock] = []
    N_blocks_used[fock] = []
    psi_t = qt.tensor(qt.basis(2, 0), qt.basis(N, fock))
    for N_blocks in np.arange(1,max_N_blocks):
        print("\nFock: %d N_blocks:%d\n" % (fock,N_blocks))
        obj_tf = CD_control_tf(initial_state=psi_i, target_state=psi_t,
                 N_blocks = N_blocks, term_fid=term_fid)
        obj_tf.randomize(beta_scale = 0.25*fock)
        f = obj_tf.optimize(learning_rate = learning_rate, 
        epoch_size=epoch_size, epochs=epochs, df_stop = df_stop)
        fids[fock].append(f)
        N_blocks_used[fock].append(N_blocks)
        if f >= term_fid:
            break
#%%
focks_used = np.arange(1,8)
fids_final = []
#%%
for fock in focks_used:
    fids_final[fock] = np.squeeze(np.array(fids[fock]))
#%%printing the data
print("\n Fids: \n")
print(fids)
print("\n N_blocks_used: \n")
print(N_blocks_used)
#%%
plt.figure(figsize = (8,6))
for fock in fids.keys():
    print(fock)
    print(fids[fock])
    print(N_blocks_used[fock])
    fids[fock] = np.squeeze(np.array(fids[fock]))
    plt.semilogy(N_blocks_used[fock], 1-np.array(fids[fock]), '--.', label=fock)
plt.legend()
#%%
plt.figure()
for fock, N_blocks in N_blocks_used.items():
    plt.scatter(fock, N_blocks[-1])
#%%
N_blocks = 12
betas = np.array([
    np.random.uniform(low=-1,high=1) + 1j*np.random.uniform(low=-1,high=1)
    for _ in range(N_blocks)]
    )
phis = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
thetas = np.array([
    np.random.uniform(low=-np.pi,high=np.pi) for _ in range(N_blocks)
])
#%%
obj_tf = CD_control_tf(initial_state=psi_i, target_state=psi_t,
                 N_blocks = N_blocks,
                 betas=betas,phis=phis, thetas=thetas)
#%%
obj = CD_control(initial_state=psi_i, target_state=psi_t, N_blocks=N_blocks,
                    use_displacements=False, analytic=True,
                    no_CD_end=False, betas=betas, phis=phis, thetas=thetas)
#%%
obj_tf.optimize(learning_rate = 0.01, epoch_size=100, epochs=100)
#%%
d_tf = (obj_tf.construct_displacement_operators(obj_tf.betas_rho, obj_tf.betas_angle)).numpy()
#%%
d = np.array([obj.D(obj.betas[i]).ptrace(1).full()/2.0 for i in range(N_blocks)])

#%%
cut = 40
print(np.max(np.abs((d_tf - d)[:,:cut,:cut])))
# %%
b_tf = obj_tf.construct_block_operators(betas_rho = obj_tf.betas_rho, betas_angle=obj_tf.betas_angle,
                                 phis = obj_tf.phis, thetas=obj_tf.thetas).numpy()
#%%
b = np.array([obj.U_i_block(i).full() for i in range(N_blocks)])
#%%
cut = 50
print(np.max(np.abs((b_tf - b)[:,:cut,:cut])))
#%%
overlap = obj_tf.state_overlap(obj_tf.betas_rho, obj_tf.betas_angle, obj_tf.phis, obj_tf.thetas)
#%%
fid_tf = obj_tf.state_fidelity(obj_tf.betas_rho, obj_tf.betas_angle, obj_tf.phis, obj_tf.thetas)
print(fid_tf)
#%%
fid = obj.fidelity()
print(fid)

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
