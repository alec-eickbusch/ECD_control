#%%
#a minimial implementation of a discrete version of grape
#consisting of a sequence of conditional displacement, 
#displacement, and rotation pulses, with 
#tuneable parameters

import numpy as np
import qutip as qt 
from helper_functions import plot_wigner
from scipy.optimize import minimize
import scipy.optimize

class CD_grape:

    #a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(self, initial_state, target_state, N_blocks,
                 init_alphas = None, init_betas = None,
                 init_phis = None, init_thetas = None, 
                 max_abs_alpha = 5, max_abs_beta = 5):

        self.initial_state = initial_state
        self.target_state = target_state
        self.N_blocks = N_blocks
        self.alphas = np.array(init_alphas, dtype=np.complex128) if init_alphas is not None \
            else np.zeros(N_blocks, dtype=np.complex128)
        self.betas = np.array(init_betas, dtype=np.complex128) if init_betas is not None \
            else np.zeros(N_blocks, dtype=np.complex128)
        self.phis = np.array(init_phis, dtype=np.float64) if init_phis is not None \
            else np.zeros(N_blocks, dtype=np.float64)
        self.thetas = np.array(init_thetas, dtype=np.float64) if init_thetas is not None \
            else np.zeros(N_blocks, dtype=np.float64)
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
        self.alphas = np.array(np.exp(1j*ang_alpha)*rho_alpha, dtype=np.complex128)
        self.betas = np.array(np.exp(1j*ang_beta)*rho_beta, dtype=np.complex128)
        self.phis = np.array(phis, dtype=np.complex128)
        self.thetas = np.array(thetas, dtype=np.complex128)

    def D(self, alpha):
        return (alpha*self.a.dag() - np.conj(alpha)*self.a).expm()
    
    def CD(self, beta):
        return ((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()

    def R(self, phi, theta):
        return (1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()

    #derivative multipliers
    def dalphar_dD_mul(self, alpha):
        return (self.a.dag() - self.a - (np.conj(alpha) - alpha)/2.0)

    def dalphai_dD_mul(self, alpha):
        return 1j*(self.a.dag() + self.a - (np.conj(alpha) + alpha)/2.0)

    def dbetar_dCD_mul(self, beta):
        return (self.a.dag() - self.a - (self.sz/2.0)*(np.conj(beta) - beta)/2.0)

    def dbetai_dCD_mul(self, beta):
        return 1j*(self.a.dag() + self.a - (self.sz/2.0)*(np.conj(beta) + beta)/2.0)

    def dtheta_dR_mul(self, phi, theta):
        return (1j/2.0)*((np.sinc(theta/np.pi))*(self.sx*np.cos(phi)+self.sy*np.sin(phi))+\
            (1 - np.sinc(theta/np.pi))*(self.sx*np.cos(phi)**2 + self.sy*np.sin(phi)**2))
    
    def dphi_dR_mul(self, phi, theta):
        return (1j/2.0)*((np.sin(theta))*(self.sy*np.cos(phi)-self.sx*np.sin(phi)) +\
            (1-np.cos(phi))*self.sz + np.cos(phi)*np.sin(phi)*(theta - np.sin(theta))*(self.sy - self.sx))

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

    def forward_states(self, alphas, betas, phis, thetas):
        psi_fwd = [self.initial_state]
        for i in range(self.N_blocks):
            psi_fwd.append(self.R(phis[i],thetas[i])*psi_fwd[-1])
            psi_fwd.append(self.D(alphas[i])*psi_fwd[-1])
            psi_fwd.append(self.CD(betas[i])*psi_fwd[-1])
        return psi_fwd
    
    def reverse_states(self, alphas, betas, phis, thetas):
        psi_bwd = [self.target_state.dag()]
        for i in np.arange(self.N_blocks)[::-1]:
            psi_bwd.append(psi_bwd[-1]*self.CD(betas[i]))
            psi_bwd.append(psi_bwd[-1]*self.D(alphas[i]))
            psi_bwd.append(psi_bwd[-1]*self.R(phis[i],thetas[i]))
        return psi_bwd

    def fid_and_grad_fid(self, alphas, betas, phis, thetas):
        psi_fwd = self.forward_states(alphas, betas, phis, thetas)
        psi_bwd = self.reverse_states(alphas, betas, phis, thetas)
        overlap = (psi_bwd[0]*psi_fwd[-1]).full()[0][0] #might be complex
        fid = np.abs(overlap)**2
        
        dalphar = np.zeros(N_blocks, dtype=np.complex128)
        dalphai = np.zeros(N_blocks, dtype=np.complex128)
        dbetar = np.zeros(N_blocks, dtype=np.complex128)
        dbetai = np.zeros(N_blocks, dtype=np.complex128)
        dphi = np.zeros(N_blocks, dtype=np.complex128)
        dtheta = np.zeros(N_blocks, dtype=np.complex128)

        for i in range(self.N_blocks): #question: do I take the real part?
            for j in [1,2,3]:
                k = 3*i + j
                if j == 1:
                    dphi[i] = (psi_bwd[-(k+1)]*self.dphi_dR_mul(phis[i], thetas[i])*psi_fwd[k]).full()[0][0]
                    dtheta[i] = (psi_bwd[-(k+1)]*self.dtheta_dR_mul(phis[i], thetas[i])*psi_fwd[k]).full()[0][0]
                if j==2:
                    dalphar[i] = (psi_bwd[-(k+1)]*self.dalphar_dD_mul(alphas[i])*psi_fwd[k]).full()[0][0]
                    dalphai[i] = (psi_bwd[-(k+1)]*self.dalphai_dD_mul(alphas[i])*psi_fwd[k]).full()[0][0]
                if j == 3:
                    dbetar[i] = (psi_bwd[-(k+1)]*self.dbetar_dCD_mul(betas[i])*psi_fwd[k]).full()[0][0]
                    dbetai[i] = (psi_bwd[-(k+1)]*self.dbetai_dCD_mul(betas[i])*psi_fwd[k]).full()[0][0]
                    
        dalphar = 2*np.abs(overlap)*np.real(overlap*np.conj(dalphar))/np.abs(overlap)
        dalphai = 2*np.abs(overlap)*np.real(overlap*np.conj(dalphai))/np.abs(overlap)
        dbetar = 2*np.abs(overlap)*np.real(overlap*np.conj(dbetar))/np.abs(overlap)
        dbetai = 2*np.abs(overlap)*np.real(overlap*np.conj(dbetai))/np.abs(overlap)
        dphi = 2*np.abs(overlap)*np.real(overlap*np.conj(dphi))/np.abs(overlap)
        dtheta = 2*np.abs(overlap)*np.real(overlap*np.conj(dtheta))/np.abs(overlap)
 
        return fid, dalphar, dalphai, dbetar, dbetai, dphi, dtheta

    def final_state(self, alphas=None,betas=None,phis=None,thetas=None):
        alphas = self.alphas if alphas is None else alphas
        betas = self.betas if betas is None else betas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        psi = self.initial_state
        for i in range(self.N_blocks):
            psi = self.R(phis[i],thetas[i])*psi
            psi = self.D(alphas[i])*psi
            psi = self.CD(betas[i])*psi
        return psi
    
    def fidelity(self, alphas=None, betas=None, phis = None, thetas=None):
        alphas = self.alphas if alphas is None else alphas
        betas = self.betas if betas is None else betas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        return np.abs((self.target_state.dag() * self.final_state(alphas,betas,phis,thetas)).full()[0][0])**2
    
    def plot_initial_state(self):
        plot_wigner(self.initial_state)
        
    def plot_final_state(self):
        plot_wigner(self.final_state())
    
    def plot_target_state(self):
        plot_wigner(self.target_state)
        
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
        print('fid: %.3f' % f, end='\r')
        return -f

    def cost_function_analytic(self, parameters):
        alphas_r = parameters[0:self.N_blocks]
        alphas_i = parameters[self.N_blocks:2*self.N_blocks]
        betas_r = parameters[2*self.N_blocks:3*self.N_blocks]
        betas_i = parameters[3*self.N_blocks:4*self.N_blocks]
        phis = parameters[4*self.N_blocks:5*self.N_blocks]
        thetas = parameters[5*self.N_blocks:6*self.N_blocks]
        alphas = alphas_r + 1j*alphas_i
        betas = betas_r + 1j*betas_i
        self.alphas = alphas
        self.betas = betas
        self.phis = phis
        self.thetas = thetas
        f, dalphar, dalphai, dbetar, dbetai, dphi, dtheta = self.fid_and_grad_fid(alphas,betas,phis,thetas)
        gradf = np.concatenate([dalphar,dalphai, dbetar, dbetai, dphi, dtheta])
        print('fid: %.3f' % f, end='\r')
        return (-f, -gradf)
    
    def optimize(self, maxiter = 1e4):
        init_params = \
        np.array(np.concatenate([np.real(self.alphas),np.imag(self.alphas),
                  np.real(self.betas), np.imag(self.betas),
                  self.phis, self.thetas]),dtype=np.float64)
        bounds = np.concatenate(
            [[(-self.max_abs_alpha,self.max_abs_alpha) for _ in range(2*N_blocks)],\
            [(-self.max_abs_beta,self.max_abs_beta) for _ in range(2*N_blocks)],\
            [(-np.pi,np.pi) for _ in range(N_blocks)],\
            [(0,np.pi) for _ in range(N_blocks)]])
        result = minimize(self.cost_function,x0=init_params,method='L-BFGS-B',
                          bounds = bounds, jac=False, options={'maxiter':maxiter})
        return result

    def optimize_analytic(self, check=False, maxiter = 1e4, gtol=1e-10, ftol=1e-10):
        init_params = \
        np.array(np.concatenate([np.real(self.alphas),np.imag(self.alphas),
                  np.real(self.betas), np.imag(self.betas),
                  self.phis, self.thetas]),dtype=np.float64)
        bounds = np.concatenate(
            [[(-self.max_abs_alpha,self.max_abs_alpha) for _ in range(2*N_blocks)],\
            [(-self.max_abs_beta,self.max_abs_beta) for _ in range(2*N_blocks)],\
            [(-np.pi,np.pi) for _ in range(N_blocks)],\
            [(0,np.pi) for _ in range(N_blocks)]])
        result = minimize(self.cost_function_analytic,x0=[init_params],method='L-BFGS-B',
                          bounds = bounds, jac=True), options={'maxiter':maxiter,'gtol':gtol,'ftol':ftol})
        return result
    
#%%
if __name__ == '__main__':
    N = 50
    N2 = 2
    N_blocks = 4
    init_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
    target_state = qt.tensor(qt.basis(N,1),qt.basis(N2,0))
    #target_state = qt.tensor((qt.coherent(N,np.sqrt(2)) + qt.coherent(N,-np.sqrt(2))).unit(),qt.basis(N2,0))
    a = CD_grape(init_state, target_state, N_blocks, max_abs_alpha=1,
                 max_abs_beta = 1)
    a.randomize()
    #if 1:
        #a.plot_initial_state()
        #a.plot_final_state()
        #a.plot_target_state()
    print(a.fidelity())
#%% 
init_params = \
        np.array(np.concatenate([np.real(a.alphas),np.imag(a.alphas),
                  np.real(a.betas), np.imag(a.betas),
                  a.phis, a.thetas]),dtype=np.float64)

f, df = a.cost_function_analytic(init_params)

test_d = np.zeros_like(init_params)
test_d[4] = 0.001
f2, df2 = a.cost_function_analytic(init_params + test_d)

test_grad = (f2-f)/0.001
print(df)
print(test_grad)
print(f)
print(f2)
#%%
a.optimize()
# %%
a.optimize_analytic()
# %%
a.plot_final_state()
print('alphas:' + str(a.alphas))
print('betas:' + str(a.betas))
print('phis:' + str(a.phis))
print('thetas:' + str(a.thetas))
# %%
