import numpy as np
import qutip as qt
from cd_grape import *
from basic_pulses import CD, rotate, disp

class system:

    #for now will use the same sigma and chop for the displacement and qubit pulses
    #the sigma_cd and chop_cd will be for the gaussian displacement pulses during the conditional displacement
    def __init__(self, chi, alpha0, sigma, chop, sigma_cd, chop_cd, buffer_time = int(0)):
        self.chi = chi
        self.alpha0 = alpha0
        self.sigma = int(sigma)
        self.chop = int(chop)
        self.sigma_cd = int(sigma_cd)
        self.chop_cd = int(chop_cd)

        def CD_pulse(self, beta):
            return CD(beta=beta, alpha0 = self.alpha0, chi=self.chi,\
                    sigma=self.sigma_cd, chop=self.chop_cd, sigma_q=self.sigma,\
                    chop_q=self.chop, buffer_time=self.buffer_time)
        
        def rotate_displace_pulse(self, alpha, phi, theta):
            Omega = rotate(theta=theta, phi=phi, sigma=self.sigma, chop=self.chop)
            epsilon = disp(alpha=alpha, sigma=self.sigma, chop=self.chop)
            return epsilon, Omega


class cd_grape_analysis:

    def __init__(self, cd_grape_object, system):
        self.cd_grape_object = cd_grape_object
        self.system = system

    #a block is defined by a rotation and displacement pulse, followed by a CD pulse.
    #for now pulse block will end with buffer time zeros.
    def pulse_block(self, i):
        alpha = self.cd_grape_object.alphas[i]
        beta = self.cd_grape_object.betas[i]
        phi = self.cd_grape_object.phis[i]
        theta = self.cd_grape_object.thetas[i]

        epsilon_D, Omega_R = self.system.rotate_displace_pulse(alpha, phi, theta)
        epsilon_CD, Omega_CD = self.system.CD_pulse(beta)
        epsilon = np.concatenate([epsilon_D, np.zeros(self.system.buffer_time),
                                  epsilon_CD, np.zeros(self.system.buffer_time)])
        Omega = np.concatenate([Omega_R, np.zeros(self.system.buffer_time),
                                Omega_CD, np.zeros(self.system.buffer_time)])
        return epsilon, Omega

    def pulse_block(self):
        epsilon = []
        Omega = []
        for i in self.cd_grape_object.N_blocks:
            epsilon, Omega = self.pulse_block(i)
            epsilon.append(epsilon)
            Omega.append(Omega)
        epsilon = np.concatenate(epsilon)
        Omega = np.concatenate(Omega)
        return epsilon, Omega



