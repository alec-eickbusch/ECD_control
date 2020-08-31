#a minimial implementation of a discrete version of grape
#consisting of a sequence of conditional displacement, 
#displacement, and rotation pulses, with 
#tuneable parameters

import numpy as np
import qutip as qt 

class CD_grape:

    #a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(self, initial_state, target_state, N_blocks,
                 init_alphas = None, init_betas = None,
                 init_phis = None, init_thetas = None, 
                 max_abs_alpha = 5, max_abs_beta = 5):

        self.initial_state = initial_state
        self.target_state = target_state
        self.N_blocks = N_blocks
        self.alphas = np.array(init_alphas, dtype=np.complex64) if init_alphas is not None \
            else np.zeros(N_blocks, dtype=np.complex64)
        self.betas = np.array(init_betas, dtype=np.complex64) if init_betas is not None \
            else np.zeros(N_blocks, dtype=np.complex64)
        self.phis = np.array(init_phis, dtype=np.float32) if init_phis is not None \
            else np.zeros(N_blocks, dtype=np.float32)
        self.thetas = np.array(init_thetas, dtype=np.float32) if init_thetas is not None \
            else np.zeros(N_blocks, dtype=np.float32)
        self.N = self.initial_state.dims[0][0]
        self.N2 = self.initial_state.dims[0][1]

        self.a = qt.tensor(qt.destroy(self.N), qt.identity(self.N2))
        self.q = qt.tensor(qt.identity(self.N), qt.destroy(self.N2))
        self.sz = 1-2*self.q.dag()*self.q
        self.sx = (self.q+self.q.dag())
        self.sy = 1j*(self.q.dag() - self.q)
        self.n = self.a.dag()*self.a

    def D(self, alpha):
        return (alpha*self.a.dag() - np.conj(alpha)*self.a).expm()
    
    def CD(self, beta):
        return ((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()

    def R(self, phi, theta):
        return (1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()

    def dalphar_dD(self, alpha):
        return (self.a.dag() - self.a - (np.conj(alpha) - alpha)/2.0)*self.D(alpha)

    def dalphai_dD(self, alpha):
        return 1j*(self.a.dag() + self.a - (np.conj(alpha) + alpha)/2.0)*self.D(alpha)

    def dbetar_dCD(self, beta):
        return (self.a.dag() - self.a - (self.sz/2.0)*(np.conj(beta) - beta)/2.0)*self.CD(beta)

    def dbetai_dCD(self, beta):
        return 1j*(self.a.dag() + self.a - (self.sz/2.0)*(np.conj(beta) + beta)/2.0)*self.CD(beta)

    def dtheta_dR(self, phi, theta):
        (1j/2.0)*((np.sinc(theta/np.pi))*(self.sx*np.cos(phi)+self.sy*np.sin(phi))+\
            (1 - np.sinc(theta/np.pi)(self.sx*np.cos(phi)**2 + self.sy*np.sin(phi)**2))\
                *self.R(phi,theta)