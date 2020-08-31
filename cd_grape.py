#%%
#a minimial implementation of a discrete version of grape
#consisting of a sequence of conditional displacement, 
#displacement, and rotation pulses, with 
#tuneable parameters

import numpy as np
import qutip as qt 
from helper_functions import plot_wigner
from scipy.optimize import minimize

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
        self.max_abs_alpha = max_abs_alpha
        self.max_abs_beta = max_abs_beta

        self.a = qt.tensor(qt.destroy(self.N), qt.identity(self.N2))
        self.q = qt.tensor(qt.identity(self.N), qt.destroy(self.N2))
        self.sz = 1-2*self.q.dag()*self.q
        self.sx = (self.q+self.q.dag())
        self.sy = 1j*(self.q.dag() - self.q)
        self.n = self.a.dag()*self.a
        
    def randomize(self):
        ang_alpha = np.random.uniform(-np.pi,np.pi,self.N_blocks)
        rho_alpha = np.random.uniform(-self.max_abs_alpha, self.max_abs_alpha, self.N_blocks)
        ang_beta = np.random.uniform(-np.pi,np.pi,self.N_blocks)
        rho_beta = np.random.uniform(-self.max_abs_beta, self.max_abs_beta, self.N_blocks)
        phis = np.random.uniform(-np.pi, np.pi, self.N_blocks)
        thetas = np.random.uniform(0,np.pi,self.N_blocks)
        self.alphas = np.array(np.exp(1j*ang_alpha)*rho_alpha, dtype=np.complex64)
        self.betas = np.array(np.exp(1j*ang_beta)*rho_beta, dtype=np.complex64)
        self.phis = np.array(phis, dtype=np.complex64)
        self.thetas = np.array(thetas, dtype=np.complex64)

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
        return (1j/2.0)*((np.sinc(theta/np.pi))*(self.sx*np.cos(phi)+self.sy*np.sin(phi))+\
            (1 - np.sinc(theta/np.pi))(self.sx*np.cos(phi)**2 + self.sy*np.sin(phi)**2))\
                *self.R(phi,theta)
    
    def dphi_dR(self, phi, theta):
        return (1j/2.0)*((np.sin(theta))*(self.sy*np.cos(phi)-self.sx*np.sin(phi)) +\
            (1-np.cos(phi))*self.sz + np.cos(phi)*np.sin(phi)*(theta - np.sin(theta))(self.sy - self.sx))\
                *self.R(phi, theta)

    def U_block(self, alpha, beta, phi, theta):
        U = self.CD(beta)*self.D(alpha)*self.R(phi, theta)
        return U

    def U_i_block(self, i, alphas,betas,phis,thetas):
        return self.U_block(alphas[i], betas[i], phis[i], thetas[i])

    def U_tot(self, alphas,betas,phis,thetas):
        U = qt.tensor(qt.identity(self.N),qt.identity(self.N2))
        for i in range(self.N_blocks):
            U = self.U_i_block(i, alphas, betas, phis, thetas) * U
        return U

    def final_state(self, alphas=None,betas=None,phis=None,thetas=None):
        alphas = self.alphas if alphas is None else alphas
        betas = self.betas if betas is None else betas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        U = self.U_tot(alphas, betas, phis, thetas)
        return U*self.initial_state

    def fidelity(self, alphas=None, betas=None, phis = None, thetas=None):
        r= self.target_state.dag() *\
            self.final_state(alphas, betas, phis, thetas)
        return np.real(r.full()[0][0])    
    
    def plot_initial_state(self):
        plot_wigner(self.initial_state)
        
    def plot_final_state(self):
        plot_wigner(self.final_state())
        
    #for the optimization, we will flatten the parameters
    #the will be, in order, 
    #[alphas_r, alphas_i, betas_r, betas_i,  phis, thetas]
    def cost_function(self, parameters):
        alphas_r = parameters[0:self.N_blocks]
        alphas_i = parameters[self.N_blocks:2*self.N_blocks]
        betas_r = parameters[2*self.N_blocks:3*self.N_blocks]
        betas_i = parameters[3*self.N_blocks:4*self.N_blocks]
        phis = parameters[4*self.N_blocks:5*self.N_blocks]
        thetas = parameters[5*self.N_blocks:6*self.N_blocks]
        alphas = alphas_r + 1j*alphas_i
        betas = betas_r + 1j*betas_i
        #temp
        self.alphas = alphas
        self.betas = betas
        self.phis = phis
        self.thetas = thetas
        f = self.fidelity(alphas,betas,phis,thetas)
        if self.last_fidelity != f:
            print('fid: %.3f' % f, end='\r')
        self.last_fidelity = f
        return -f
    
    def optimize(self, maxiter = None):
        self.last_fidelity = -2
        init_params = \
        np.array([np.real(self.alphas),np.imag(self.alphas),
                  np.real(self.betas), np.imag(self.betas),
                  self.phis, self.thetas],dtype=np.float32)
        bounds = np.concatenate(
            [[(-self.max_abs_alpha,self.max_abs_alpha) for _ in range(2*N_blocks)],\
            [(-self.max_abs_beta,self.max_abs_beta) for _ in range(2*N_blocks)],\
            [(-np.pi,np.pi) for _ in range(N_blocks)],\
            [(0,np.pi) for _ in range(N_blocks)]])
        result = minimize(self.cost_function,x0=init_params,method='L-BFGS-B',
                          bounds = bounds, jac=False, options={'disp':True,
                                                               'maxiter':maxiter})
        return result
    
#%%
if __name__ == '__main__':
    N = 50
    N2 = 2
    N_blocks = 6
    init_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
    target_state = qt.tensor(qt.basis(N,1),qt.basis(N2,0))
    init_alphas = np.random.uniform(low=-3,high=+3,size=N_blocks)
    a = CD_grape(init_state, target_state, N_blocks, max_abs_alpha=1,
                 max_abs_beta = 1)
    a.randomize()
    a.plot_initial_state()
    a.plot_final_state()
    print(a.fidelity())


# %%
a.optimize(1000)
# %%
