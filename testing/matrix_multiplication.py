#%%
import tensorflow as tf
import numpy as np
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def mult_bin(a):
    while len(a) > 1:
        if len(a) % 2 == 1:
            a[-2] = np.matmul(a[-2], a[-1])
            a = a[:-1]
        a = np.matmul(a[::2, ...], a[1::2, ...])
    return a


def mult(a):
    ans = a[0]
    for i in range(1, len(a)):
        ans = ans @ a[i]
    return ans


@tf.function
def mult_tf(a):
    ans = a[0]
    for i in range(1, len(a)):
        ans = ans @ a[i]
    return ans


@tf.function
def mult_bin_tf(a):
    while len(a) > 1:
        if len(a) % 2 == 1:
            a = tf.concat([a[:-2], [tf.matmul(a[-2], a[-1])]], 0)
        a = tf.matmul(a[::2, ...], a[1::2, ...])
    return a


#%%
np.random.seed(0)
matrices = np.array(np.random.rand(2 ** 10 + 3, 2, 2), dtype=np.float32)

#%% may have to rerun this cell
bin_time = wrapper(mult_bin, matrices.copy())
reg_time = wrapper(mult, matrices.copy())

print(reg_time())
print(bin_time())
print(timeit.timeit(reg_time, number=100))
print(timeit.timeit(bin_time, number=100))
with tf.device("CPU:0"):
    mat = tf.constant(matrices.copy(), dtype=tf.float32)
    mat2 = tf.constant(matrices.copy(), dtype=tf.float32)
    reg_time_tf = wrapper(mult_tf, mat)
    bin_time_tf = wrapper(mult_bin_tf, mat2)
    print(reg_time_tf())
    print(bin_time_tf())
    print(timeit.timeit(reg_time_tf, number=100))
    print(timeit.timeit(bin_time_tf, number=100))

# %%
