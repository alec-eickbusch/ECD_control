# The unitary gates as used for qutip
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

def D(alpha, N_cav):
    a = qt.tensor(qt.identity(2), qt.destroy(N_cav))
    # q = qt.tensor(qt.destroy(2), qt.identity(N_cav))
    if np.abs(alpha) == 0:
        return qt.tensor(qt.identity(2), qt.identity(N_cav))
    return (alpha * a.dag() - np.conj(alpha) * a).expm()


def R(phi, theta, N_cav):
    sx = qt.tensor(qt.sigmax(), qt.identity(N_cav))
    sy = qt.tensor(qt.sigmay(), qt.identity(N_cav))
    id = qt.tensor(qt.identity(2), qt.identity(N_cav))
    if theta == 0:
        return id
    if isinstance(theta, (list, tuple, np.ndarray)):
        U = id
        for t, p in zip(theta, phi):
            U = (
                np.cos(t / 2.0)
                - 1j * (np.cos(p) * sx + np.sin(p) * sy) * np.sin(t / 2.0)
            ) * U

        return U
    # return (-1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()
    return np.cos(theta / 2.0) * id - 1j * (
        np.cos(phi) * sx + np.sin(phi) * sy
    ) * np.sin(theta / 2.0)


def ECD(beta, N_cav):
    if np.abs(beta) == 0:
        return qt.tensor(qt.identity(2), qt.identity(N_cav))
    sx = qt.tensor(qt.sigmax(), qt.identity(N_cav))
    sz = qt.tensor(qt.sigmaz(), qt.identity(N_cav))
    a = qt.tensor(qt.identity(2), qt.destroy(N_cav))
    return sx * ((beta * a.dag() - np.conj(beta) * a) * (sz / 2.0)).expm()
    # includes pi rotation


def U_block_ECD(beta, phi, theta, N_cav):
    return ECD(beta, N_cav) * R(phi, theta, N_cav)


def U_circuit_ECD(betas, phis, thetas, N_cav):
    U = qt.tensor(qt.identity(2), qt.identity(N_cav))
    for beta, phi, theta in zip(betas, phis, thetas):
        U = U_block_ECD(beta, phi, theta, N_cav) * U
    return U


def wigner(rho, xvec, yvec=None, g=2):
    if yvec is None:
        yvec = xvec
    # return (np.pi / 2.0) * qt.wigner(rho, xvec, yvec, g=2)
    N = rho.dims[0][0]
    max_radius = np.sqrt(np.max(xvec * g / 2.0) ** 2 + np.max(yvec * g / 2.0) ** 2)
    if N < 0.8 * max_radius**2:
        print(
            "Warning: calculating Wigner with N = %d and max radius=%.3f"
            % (N, max_radius)
        )
    return qt.wigner(rho, xvec, yvec, g=g)

def plot_wigner(psi, xvec=np.linspace(-5, 5, 151), ax=None, grid=True, invert=False):
    W = wigner(psi, xvec)
    s = -1 if invert else +1
    plot_wigner_data(s * W, xvec, ax=ax, grid=grid)


def plot_wigner_data(
    W,
    xvec=np.linspace(-5, 5, 101),
    ax=None,
    grid=True,
    yvec=None,
    cut=0,
    vmin=-2 / np.pi,
    vmax=+2 / np.pi,
):
    yvec = xvec if yvec is None else yvec
    if cut > 0:
        xvec = xvec[cut:-cut]
        yvec = yvec[cut:-cut]
        W = W[cut:-cut, cut:-cut]
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    extent = (
        xvec[0] - dx / 2.0,
        xvec[-1] + dx / 2.0,
        yvec[0] - dy / 2.0,
        yvec[-1] + dy / 2.0,
    )
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(
        W,
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation=None,
    )
    # plt.colorbar()
    if grid:
        ax.grid()