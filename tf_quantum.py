#%%
# Code here is various things taken from Henry Liu and Vlad Sivak as well as some of my own additions
# Keep up to date with their versions, and later just import from their
# soon to come tf-quantum library.
from distutils.version import LooseVersion

import tensorflow as tf
import qutip as qt

if LooseVersion(tf.__version__) >= "2.2":
    diag = tf.linalg.diag
else:
    import numpy as np

    diag = np.diag  # k=1 option is broken in tf.linalg.diag in TF 2.1 (#35761)

#%%
# todo: handle case when passed an already tf object.
def qt2tf(qt_object, dtype=tf.complex64):
    return tf.constant(qt_object.full(), dtype=dtype)


def tf2qt(tf_object, matrix=False):
    if not matrix:
        return qt.Qobj(
            tf_object.numpy(), dims=[[2, int(tf_object.numpy().shape[0] / 2)], [1, 1]]
        )
    else:
        return qt.Qobj(
            tf_object.numpy(),
            dims=[
                [2, int(tf_object.numpy().shape[0] / 2)],
                [2, int(tf_object.numpy().shape[0] / 2)],
            ],
        )


def matrix_flatten(tensor):
    """Takes a tensor of arbitrary shape and returns a "flattened" vector of matrices.
    This is useful to get the correct broadcasting shape for batch operations.
    Args:
        tensor (Tensor([x, y, ...])): A tensor with arbitrary shape
    Returns:
        Tensor([numx * numy * ..., 1, 1]): A "flattened" vector of matrices
    """
    tensor = tf.reshape(tensor, [-1])
    tensor = tf.reshape(tensor, shape=[tensor.shape[0], 1, 1])
    return tensor


def identity(N, dtype=tf.complex64):
    """Returns an identity operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN identity operator
    """
    return tf.eye(N, dtype=dtype)


def destroy(N, dtype=tf.complex64):
    """Returns a destruction (lowering) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    a = diag(tf.sqrt(tf.range(1, N, dtype=tf.float64)), k=1)
    return tf.cast(a, dtype=dtype)


def create(N, dtype=tf.complex64):
    """Returns a creation (raising) operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN creation operator
    """
    # Preserve max precision in intermediate calculations until final cast
    return tf.cast(tf.linalg.adjoint(destroy(N, dtype=tf.complex128)), dtype=dtype)


def num(N, dtype=tf.complex64):
    """Returns the number operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN number operator
    """
    return tf.cast(diag(tf.range(0, N)), dtype=dtype)


def position(N, dtype=tf.complex64):
    """Returns the position operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
        dtype (tf.dtypes.DType, optional): Returned dtype. Defaults to c64.
    Returns:
        Tensor([N, N], dtype): NxN position operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast((a_dag + a) / sqrt2, dtype=dtype)


def momentum(N, dtype=tf.complex64):
    """Returns the momentum operator in the Fock basis.
    Args:
        N (int): Dimension of Hilbert space
    Returns:
        Tensor([N, N], c64): NxN momentum operator
    """
    # Preserve max precision in intermediate calculations until final cast
    sqrt2 = tf.sqrt(tf.constant(2, dtype=tf.complex128))
    a_dag = create(N, dtype=tf.complex128)
    a = destroy(N, dtype=tf.complex128)
    return tf.cast(1j * (a_dag - a) / sqrt2, dtype=dtype)


# %%
