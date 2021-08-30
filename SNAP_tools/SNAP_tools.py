#%%

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

#%%
def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def rotate(theta, phi=0, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    amp = 1 / energy
    wave = (1 + 0j) * wave
    return (theta / (2.0)) * amp * np.exp(1j * phi) * wave


# displace cavity by an amount alpha
def disp_gaussian(alpha, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx=dt)
    wave = (1 + 0j) * wave
    return (
        (np.abs(alpha) / energy) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave
    )


# sigma and chop are for the selective pulses
# sigma1 and chop 1 are for the first pi pulse.
def SNAP_gate(thetas, chi=-2 * np.pi * 1e-3 * 1, sigma=125, chop=4, sigma1=8, chop1=4):
    pi1 = rotate(np.pi, sigma=sigma1, chop=chop1)
    pi2 = rotate(np.pi, sigma=sigma, chop=chop)

    pulse2 = np.zeros_like(pi2)
    pulse1 = np.zeros_like(pi2)
    ts2 = np.arange(len(pulse2))
    for n, theta in enumerate(thetas):
        if theta == 0:
            continue
        omega = n * chi
        pulse1 = pulse1 + np.exp(1j * omega * ts2) * pi2
        pulse2 = pulse2 + np.exp(1j * omega * ts2) * np.exp(1j * theta) * pi2
    return np.concatenate([pi2, pulse2])


def SNAP_circuit(
    alphas,
    thetas_list,
    chi=-2 * np.pi * 1e-3 * 1,
    sigma=125,
    chop=4,
    sigma1=8,
    chop1=4,
    sigma_disp=8,
    chop_disp=4,
):
    epsilon = []
    Omega = []
    for alpha, thetas in zip(alphas, thetas_list):
        d = disp_gaussian(alpha, sigma_disp, chop_disp)
        epsilon.append(d)
        Omega.append(np.zeros_like(d))
        if np.max(np.abs(thetas)) > 0:
            o = SNAP_gate(thetas, chi, sigma, chop, sigma1, chop1)
            Omega.append(o)
            epsilon.append(np.zeros_like(o))
    return np.concatenate(epsilon), np.concatenate(Omega)
