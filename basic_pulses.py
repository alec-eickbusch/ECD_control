#%%
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:18:08 2019

@author: Alec Eickbusch
"""

import numpy as np
import qutip as qt
from helper_functions import alpha_from_epsilon

#%%
def gaussian_wave(sigma, chop=4):
    ts = np.linspace(-chop/2*sigma, chop/2*sigma, chop*sigma)
    P = np.exp(-ts**2 / (2.0 * sigma**2))
    ofs = P[0]
    return (P - ofs) / (1 - ofs)
#rotate qubit angle theta around axis phi on the bloch sphere
def rotate(theta, phi=0, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx = dt)
    amp_pi = np.pi / (2*energy)
    wave = (1 + 0j)*wave
    return (theta/np.pi) * amp_pi * np.exp(1j*phi) * wave

#displace cavity by an amount alpha
def disp(alpha, sigma=8, chop=6, dt=1):
    wave = gaussian_wave(sigma=sigma, chop=chop)
    energy = np.trapz(wave, dx = dt)
    wave = (1 + 0j)*wave
    return (np.abs(alpha)/energy) * np.exp(1j*(np.pi/2.0 + np.angle(alpha))) * wave

def fastest_disp(alpha, epsilon_m=2*np.pi*1e-3*400, initial_sigma = 32, chop=4, min_sigma=2):
    def valid(sigma):
        epsilon = disp(alpha, sigma, chop)
        return (np.max(np.abs(epsilon)) < epsilon_m) and (sigma >= min_sigma)
    sigma = int(initial_sigma)
    #reduce the sigma until the pulse is too large then back off by 1.
    while valid(sigma):
        sigma = int(sigma - 1)
    sigma = int(sigma + 1)
    return disp(alpha, sigma, chop)

def fastest_CD(beta, alpha0 = 60, epsilon_m = 2*np.pi*1e-3*400, chi=2*np.pi*1e-3*0.03, buffer_time=0,\
              sigma_q=6, chop_q=4,min_sigma = 2, chop=4):
    beta_bare = 100 #some large number
    def beta_bare(alpha0): #calculate the beta from just the displacement part of the CD
        epsilon_bare = np.concatenate([
        fastest_disp(alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma),
        fastest_disp(-1*alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma)])
        alpha_bare = np.abs(alpha_from_epsilon(epsilon_bare))
        beta_bare = np.abs(2*2*chi*np.sum(alpha_bare)) #extra factor of 2 for second half of pulse
        return beta_bare
    while beta_bare(alpha0) > beta:
        alpha0 = alpha0*0.99 #step down alpha0 by 1%
    
    betab = beta_bare(alpha0)
    total_time = np.abs((beta - betab)/(2*alpha0*chi))
    zero_time = int(round(total_time/2))
    alpha_angle = np.angle(beta) + np.pi/2.0 #todo: is it + or - pi/2?
    alpha0 = alpha0*np.exp(1j*alpha_angle)
    
    epsilon = np.concatenate([
    fastest_disp(alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma),
    np.zeros(zero_time),
    fastest_disp(-1*alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma),
    np.zeros(buffer_time + sigma_q*chop_q + buffer_time),
    fastest_disp(-1*alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma),
    np.zeros(zero_time),
    fastest_disp(alpha0, epsilon_m=epsilon_m, chop=chop, min_sigma=min_sigma)])

    alpha = alpha_from_epsilon(epsilon)
    beta2 = 2*chi*np.sum(np.abs(alpha))

    epsilon = epsilon*(beta/beta2)

    total_len = int(len(epsilon))
    qubit_delay = int(total_len/2 - sigma_q*chop_q/2)
    Omega = np.concatenate([
        np.zeros(qubit_delay),
        rotate(np.pi,sigma=sigma_q,chop=chop_q),
        np.zeros(total_len - qubit_delay - sigma_q*chop_q)
    ])

    return epsilon, Omega, beta2
        


def CD(beta, alpha0 = 60, chi=2*np.pi*1e-3*0.03, sigma=24, chop=4, sigma_q=4, chop_q=6, buffer_time=0):
    #will have to think about CDs where it's too to have any zero time given beta bare
    
    #beta = 2*chi*int(alpha)

    #intermederiate pulse used to find the energy in the displacements
    epsilon_bare = np.concatenate([
    disp(alpha0, sigma, chop),
    disp(-1*alpha0, sigma, chop)])
    alpha_bare = np.abs(alpha_from_epsilon(epsilon_bare))
    beta_bare = np.abs(2*2*chi*np.sum(alpha_bare)) #extra factor of 2 for second half of pulse

    total_time = np.abs((beta - beta_bare)/(2*alpha0*chi))
    zero_time = int(round(total_time/2))
    alpha0 = np.abs((beta - beta_bare)/(2*2*zero_time*chi)) #a slight readjustment due to the rounding
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
    beta2 = 2*chi*np.sum(np.abs(alpha))

    epsilon = epsilon*(beta/beta2)
    #for some reason, CD is still off by like 2 percent sometimes here. 

    alpha = alpha_from_epsilon(np.pad(epsilon,40))
    beta3 = 2*chi*np.sum(np.abs(alpha))

    Omega = np.concatenate([
        np.zeros(zero_time + 2*sigma*chop + buffer_time),
        rotate(np.pi,sigma=sigma_q,chop=chop_q),
        np.zeros(zero_time + 2*sigma*chop + buffer_time)
    ])

    return epsilon, Omega, beta3, alpha, beta_bare
# %%

# %% testing
if __name__ == '__main__':
    e, O, b= fastest_CD(3, min_sigma=2, chop=4, sigma_q = 4, chop_q = 6, buffer_time = 0)
    plt.figure()
    plt.plot(1e3*e/2/np.pi)
    plt.plot(1e3*O/2/np.pi)
    plt.ylim([-450,450])
    plt.grid()
    print(b)
    print(len(e))
    print(len(O))
# %%
