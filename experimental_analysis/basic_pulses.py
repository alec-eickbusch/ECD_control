#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:18:08 2019

@author: Alec Eickbusch
"""

import numpy as np
from experimental_analysis.helper_functions import alpha_from_epsilon, plot_pulse

#%%
def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    P = np.exp(-(ts ** 2) / (2.0 * sigma ** 2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)


def gaussian_deriv_wave(sigma, chop=4):
    ts = np.linspace(-chop / 2 * sigma, chop / 2 * sigma, chop * sigma)
    ofs = np.exp(-ts[0] ** 2 / (2 * sigma ** 2))
    return (
        (0.25 / sigma ** 2) * ts * np.exp(-(ts ** 2) / (2.0 * sigma ** 2)) / (1 - ofs)
    )


def ring_up_smootherstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 6 * ts ** 5 - 15 * ts ** 4 + 10 * ts ** 3


def ring_up_smoothstep(length):
    dt = 1.0 / length
    ts = np.arange(length) * dt
    return 3 * ts ** 2 - 2 * ts ** 3


def polynomial_ring_up(
    length, reverse=False, negative=False, ring_up_type="smootherstep"
):
    if length == 0:
        return np.array([])
    if ring_up_type == "smoothstep":
        wave = ring_up_smoothstep(length)
    elif ring_up_type == "smootherstep":
        wave = ring_up_smootherstep(length)
    else:
        raise ValueError("Type must be 'smoothstep' or 'smootherstep', not %s" % type)
    if reverse:
        wave = wave[::-1]
    if negative:
        wave = -1 * wave
    return wave


def trapezoid_pulse(ring_up_time, flat_time):
    return np.concatenate(
        [
            polynomial_ring_up(ring_up_time, ring_up_type="smootherstep"),
            np.ones(flat_time),
            polynomial_ring_up(
                ring_up_time, reverse=True, negative=False, ring_up_type="smootherstep"
            ),
        ]
    )


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


def fastest_disp_gaussian(
    alpha, epsilon_m=2 * np.pi * 1e-3 * 400, initial_sigma=32, chop=4, min_sigma=2
):
    def valid(sigma):
        epsilon = disp_gaussian(alpha, sigma, chop)
        return (np.max(np.abs(epsilon)) < epsilon_m) and (sigma >= min_sigma)

    sigma = int(initial_sigma)
    # reduce the sigma until the pulse is too large then back off by 1.
    while valid(sigma):
        sigma = int(sigma - 1)
    sigma = int(sigma + 1)
    return disp_gaussian(alpha, sigma, chop)


def disp_trapezoid(alpha, ring_up_time=8, flat_time=8, dt=1):
    wave = trapezoid_pulse(ring_up_time, flat_time)
    energy = np.trapz(wave, dx=dt)
    wave = (1 + 0j) * wave
    return (
        (np.abs(alpha) / energy) * np.exp(1j * (np.pi / 2.0 + np.angle(alpha))) * wave
    )


def fastest_disp_trapezoid(
    alpha, epsilon_m=2 * np.pi * 1e-3 * 400, ring_up_time=8, initial_flat_time=100
):
    def valid(flat_time):
        if flat_time == -1:
            return False
        epsilon = disp_trapezoid(alpha, ring_up_time, flat_time)
        return np.max(np.abs(epsilon)) < epsilon_m

    flat_time = int(initial_flat_time)
    # reduce the flat_time until the pulse is too large then back off by 1.
    while valid(flat_time):
        flat_time = int(flat_time - 1)
    flat_time = int(flat_time + 1)
    return disp_trapezoid(alpha, ring_up_time, flat_time)


def fastest_CD(
    beta,
    alpha0=60,
    epsilon_m=2 * np.pi * 1e-3 * 400,
    chi=2 * np.pi * 1e-3 * 0.03,
    buffer_time=0,
    sigma_q=6,
    chop_q=4,
    ring_up_time=8,
    qubit_pi_pulse=None,
):
    def beta_bare(
        alpha0,
    ):  # calculate the beta from just the displacement part of the CD
        epsilon_bare = np.concatenate(
            [
                fastest_disp_trapezoid(
                    alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
                ),
                fastest_disp_trapezoid(
                    -1 * alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
                ),
            ]
        )
        alpha_bare = np.abs(alpha_from_epsilon(epsilon_bare))
        beta_bare = np.abs(
            2 * chi * np.sum(alpha_bare)
        )  # extra factor of 2 for second half of pulse
        return beta_bare

    # return beta_bare
    while np.abs(beta_bare(alpha0)) > np.abs(beta):
        alpha0 = alpha0 * 0.99  # step down alpha0 by 1%
    betab = beta_bare(alpha0)
    total_time = (np.abs(beta) - np.abs(betab)) / (alpha0 * chi)
    zero_time = int(round(total_time / 2))
    alpha_angle = np.angle(beta) + np.pi / 2.0  # todo: is it + or - pi/2?
    alpha0 = alpha0 * np.exp(1j * alpha_angle)

    epsilon = np.concatenate(
        [
            fastest_disp_trapezoid(
                alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
            ),
            np.zeros(zero_time),
            fastest_disp_trapezoid(
                -1 * alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
            ),
            np.zeros(buffer_time + sigma_q * chop_q + buffer_time),
            fastest_disp_trapezoid(
                -1 * alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
            ),
            np.zeros(zero_time),
            fastest_disp_trapezoid(
                alpha0, epsilon_m=epsilon_m, ring_up_time=ring_up_time
            ),
        ]
    )
    # note: above, displacement back is shorter than initial displacement

    alpha = alpha_from_epsilon(np.pad(epsilon, 40, mode="constant"))
    beta2 = chi * np.sum(np.abs(alpha))

    epsilon = epsilon * np.abs(beta) / np.abs(beta2)
    # for some reason, CD is still off by like 2 percent sometimes here ?

    # alpha = alpha_from_epsilon(np.pad(epsilon,40))
    # beta3 = chi*(np.sum(alpha[0:int(len(alpha)/2)]) - np.sum(alpha[int(len(alpha)/2):]))
    # return alpha, beta3

    total_len = int(len(epsilon))
    qubit_delay = int(total_len / 2 - sigma_q * chop_q / 2)
    pi_pulse = (
        qubit_pi_pulse
        if qubit_pi_pulse is not None
        else rotate(np.pi, sigma=sigma_q, chop=chop_q)
    )
    Omega = np.concatenate(
        [
            np.zeros(qubit_delay),
            pi_pulse,
            np.zeros(total_len - qubit_delay - sigma_q * chop_q),
        ]
    )

    return epsilon, Omega


"""

def CD(beta, alpha0 = 60, chi=2*np.pi*1e-3*0.03, sigma=24, chop=4, sigma_q=4, chop_q=6, buffer_time=0):
    #will have to think about CDs where it's too to have any zero time given beta bare
    
    #beta = 2*chi*int(alpha)

    #intermederiate pulse used to find the energy in the displacements
    epsilon_bare = np.concatenate([
    disp(alpha0, sigma, chop),
    disp(-1*alpha0, sigma, chop)])
    alpha_bare = np.abs(alpha_from_epsilon(epsilon_bare))
    beta_bare = np.abs(2*2*chi*np.sum(alpha_bare)) #extra factor of 2 for second half of pulse

    total_time = (np.abs(beta) - np.abs(beta_bare))/(2*alpha0*chi)
    zero_time = int(round(total_time/2))
    alpha0 = (np.abs(beta) - np.abs(beta_bare))/(2*2*zero_time*chi) # a slight readjustment due to the rounding
    alpha_angle = np.angle(beta) + np.pi/2.0 #todo: is it + or - pi/2?
    alpha0 = alpha0*np.exp(1j*alpha_angle)
    
    epsilon = np.concatenate([
    disp(alpha0, sigma, chop),
    np.zeros(zero_time),
    disp(-1*alpha0, sigma, chop),
    np.zeros(buffer_time + sigma_q*chop_q + buffer_time),
    disp(-1*alpha0, sigma, chop),
    np.zeros(zero_time),
    disp(alpha0, sigma, chop)])

    alpha = alpha_from_epsilon(np.pad(epsilon,40))
    beta2 = np.abs(2*chi*np.sum(np.abs(alpha)))

    return alpha, beta2
    epsilon = epsilon*(np.abs(beta)/np.abs(beta2))
    #for some reason, CD is still off by like 2 percent sometimes here. 

    alpha = alpha_from_epsilon(np.pad(epsilon,40))
    beta3 = 2*chi*np.sum(np.abs(alpha))

    Omega = np.concatenate([
        np.zeros(zero_time + 2*sigma*chop + buffer_time),
        rotate(np.pi,sigma=sigma_q,chop=chop_q),
        np.zeros(zero_time + 2*sigma*chop + buffer_time)
    ])

    return epsilon, Omega, beta3, alpha, beta_bare
"""
# %%

# %% testing
if __name__ == "__main__":
    e, O, b = fastest_CD(1j, min_sigma=2, chop=4, sigma_q=4, chop_q=6, buffer_time=6)
    plt.figure()
    plt.plot(1e3 * np.imag(e) / 2 / np.pi)
    plt.plot(1e3 * O / 2 / np.pi)
    plt.ylim([-450, 450])
    plt.grid()
    print(b)
    print(len(e))
    print(len(O))
# %%
