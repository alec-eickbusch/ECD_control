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
from datetime import datetime


class OptFinishedException(Exception):
    def __init__(self, msg, CD_grape_obj):
        super(OptFinishedException, self).__init__(msg)
        #CD_grape_obj.save()
        #can save data here...

class CD_grape:

    #a block is defined as the unitary: CD(beta)D(alpha)R_phi(theta)
    def __init__(self, initial_state=None, target_state=None, N_blocks = 1,
                 betas = None, alphas = None,
                 phis = None, thetas = None, 
                 max_alpha = 5, max_beta = 5,
                 saving_directory = None, name = 'CD_grape',
                 term_fid = None):

        self.initial_state = initial_state
        self.target_state = target_state
        self.N_blocks = N_blocks
        self.betas = np.array(alphas, dtype=np.complex128) if alphas is not None \
            else np.zeros(N_blocks, dtype=np.complex128)
        self.alphas = np.array(alphas, dtype=np.complex128) if alphas is not None \
            else np.zeros(N_blocks+1, dtype=np.complex128)  
        self.phis = np.array(phis, dtype=np.float64) if phis is not None \
            else np.zeros(N_blocks+1, dtype=np.float64)
        self.thetas = np.array(thetas, dtype=np.float64) if thetas is not None \
            else np.zeros(N_blocks+1, dtype=np.float64)

        self.max_alpha = max_alpha
        self.max_beta = max_beta
        self.saving_directory = saving_directory
        self.name = name 
        self.term_fid = term_fid

        if self.initial_state is not None:
            self.N = self.initial_state.dims[0][0]
            self.N2 = self.initial_state.dims[0][1]
            self.a = qt.tensor(qt.destroy(self.N), qt.identity(self.N2))
            self.q = qt.tensor(qt.identity(self.N), qt.destroy(self.N2))
            self.sz = 1-2*self.q.dag()*self.q
            self.sx = (self.q+self.q.dag())
            self.sy = 1j*(self.q.dag() - self.q)
            self.n = self.a.dag()*self.a
        
    def randomize(self, alpha_scale = None, beta_scale=None):
        alpha_scale = self.max_alpha if alpha_scale is None else alpha_scale
        beta_scale = self.max_beta if beta_scale is None else beta_scale
        ang_alpha = np.random.uniform(-np.pi,np.pi,self.N_blocks+1)
        rho_alpha = np.random.uniform(-alpha_scale, alpha_scale, self.N_blocks+1)
        ang_beta = np.random.uniform(-np.pi,np.pi,self.N_blocks)
        rho_beta = np.random.uniform(-beta_scale, beta_scale, self.N_blocks)
        phis = np.random.uniform(-np.pi, np.pi, self.N_blocks+1)
        thetas = np.random.uniform(0,np.pi,self.N_blocks+1)
        self.alphas = np.array(np.exp(1j*ang_alpha)*rho_alpha, dtype=np.complex128)
        self.betas = np.array(np.exp(1j*ang_beta)*rho_beta, dtype=np.complex128)
        self.phis = np.array(phis, dtype=np.float64)
        self.thetas = np.array(thetas, dtype=np.float64)

    def D(self, alpha):
        return (alpha*self.a.dag() - np.conj(alpha)*self.a).expm()
    
    def CD(self, beta):
        if beta == 0:
            return qt.tensor(qt.identity(self.N),qt.identity(self.N2))
        return self.R(0,np.pi)*((beta*self.a.dag() - np.conj(beta)*self.a)*(self.sz/2.0)).expm()

    def R(self, phi, theta):
        return (1j*(theta/2.0)*(np.cos(phi)*self.sx + np.sin(phi)*self.sy)).expm()

    #derivative multipliers
    #todo: optimization with derivatives
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
        if i == self.N_blocks:
            beta = 0
        else:
            beta = betas[i]
        return self.U_block(alphas[i], beta, phis[i], thetas[i])

    def U_tot(self, alphas,betas,phis,thetas):
        U = qt.tensor(qt.identity(self.N),qt.identity(self.N2))
        for i in range(self.N_blocks + 1):
            U = self.U_i_block(i, alphas, betas, phis, thetas) * U
        return U

    #TODO: work out optimization with the derivatives, include block # N_blocks + 1
    def forward_states(self, alphas, betas, phis, thetas, aux_params):
        psi_fwd = [self.initial_state]
        for i in range(self.N_blocks):
            psi_fwd.append(self.R(phis[i],thetas[i])*psi_fwd[-1])
            psi_fwd.append(self.D(alphas[i])*psi_fwd[-1])
            psi_fwd.append(self.CD(betas[i])*psi_fwd[-1])
        for i in range(len(self.aux_ops)):
            psi_fwd.append((1j*aux_params[i]*self.aux_ops[i]).expm()*psi_fwd[-1])
        return psi_fwd
    
    def reverse_states(self, alphas, betas, phis, thetas, aux_params):
        psi_bwd = [self.target_state.dag()]
        for i in np.arange(len(self.aux_ops))[::-1]:
            psi_bwd.append(psi_bwd[-1]*(1j*aux_params[i]*self.aux_ops[i]).expm())
        for i in np.arange(self.N_blocks)[::-1]:
            psi_bwd.append(psi_bwd[-1]*self.CD(betas[i]))
            psi_bwd.append(psi_bwd[-1]*self.D(alphas[i]))
            psi_bwd.append(psi_bwd[-1]*self.R(phis[i],thetas[i]))
        return psi_bwd

    #TODO: Modify for aux params
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
        for i in range(self.N_blocks + 1):
            psi = self.R(phis[i],thetas[i])*psi
            psi = self.D(alphas[i])*psi
            if i < self.N_blocks:
                psi = self.CD(betas[i])*psi
        return psi
    
    def fidelity(self, alphas=None, betas=None, phis = None, thetas=None):
        alphas = self.alphas if alphas is None else alphas
        betas = self.betas if betas is None else betas
        phis = self.phis if phis is None else phis
        thetas = self.thetas if thetas is None else thetas
        overlap =  (self.target_state.dag()*self.final_state(alphas,betas,phis,thetas)).full()[0][0]
        return np.abs(overlap)**2
    
    def plot_initial_state(self):
        plot_wigner(self.initial_state)
        
    def plot_final_state(self):
        plot_wigner(self.final_state())
    
    def plot_target_state(self):
        plot_wigner(self.target_state)
        
    #for the optimization, we will flatten the parameters
    #the will be, in order, 
    #[betas_r, betas_i, alphas_r, alphas_i,  phis, thetas]
    def cost_function(self, parameters):
        betas_r = parameters[0:self.N_blocks]
        betas_i = parameters[self.N_blocks:2*self.N_blocks]
        alphas_r = parameters[2*self.N_blocks:(3*self.N_blocks + 1)]
        alphas_i = parameters[(3*self.N_blocks + 1):(4*self.N_blocks + 2)]
        phis = parameters[(4*self.N_blocks + 2):(5*self.N_blocks + 3)]
        thetas = parameters[(5*self.N_blocks + 3):]
        alphas = alphas_r + 1j*alphas_i
        betas = betas_r + 1j*betas_i
        #temp.
        #TODO: later, implement more sophisticated saving and real time information.
        self.alphas = alphas
        self.betas = betas
        self.phis = phis
        self.thetas = thetas
        f = self.fidelity(alphas,betas,phis,thetas)
        print('fid: %.3f' % f, end='\r')
        if self.term_fid is not None and f >= self.term_fid:
            raise OptFinishedException('Requested fidelity obtained', self)
        return -f

    #TODO: include final rotation and displacement
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
    
    #TODO: if I only care about the cavity state, I can optimize on the partial trace of the
    #cavity. Then, in the experiment, I can measure and reset the qubit.
    # For example, if I'm interested in creating a cat I can get to
    # (|alpha> + |-alpha>)|g> + (|alpha> - |-alpha>)|e>
    #then somhow I can optimize that the cavity state conditioned on g or e is only 
    #a unitary operaton away from each other, which would allow me to implement feedback for
    #preperation in the experiment.
    #TODO: implement and understand gtol and ftol
    def optimize(self, maxiter = 1e4):
        init_params = \
        np.array(np.concatenate([np.real(self.betas),np.imag(self.betas),
                  np.real(self.alphas), np.imag(self.alphas),
                  self.phis, self.thetas]),dtype=np.float64)
        bounds = np.concatenate(
            [[(-self.max_beta,self.max_beta) for _ in range(2*self.N_blocks)],\
            [(-self.max_alpha,self.max_alpha) for _ in range(2*self.N_blocks + 2)],\
            [(-np.inf,np.inf) for _ in range(self.N_blocks + 1)],\
            [(-np.inf,np.inf) for _ in range(self.N_blocks + 1)]])
        try:
            print("\n\nStarting optimization.\n\n")
            result = minimize(self.cost_function, x0=init_params, method='L-BFGS-B',
                              bounds=bounds, jac=False, options={'maxiter': maxiter})
        except OptFinishedException as e:
            print("\n\ndesired fidelity reached.\n\n")
            print('fidelity: ' + str(self.fidelity()))
        else:
            print("\n\noptimization failed to reach desired fidelity.\n\n")
            print('fidelity: ' + str(self.fidelity()))


    #TODO: understand gtol and ftol
    def optimize_analytic(self, check=False, maxiter = 1e4, gtol=1e-4, ftol=1e-4):
        init_params = \
        np.array(np.concatenate([np.real(self.alphas),np.imag(self.alphas),
                  np.real(self.betas), np.imag(self.betas),
                  self.phis, self.thetas]),dtype=np.float64)
        bounds = np.concatenate(
            [[(-self.max_alpha,self.max_alpha) for _ in range(2*N_blocks)],\
            [(-self.max_beta,self.max_beta) for _ in range(2*N_blocks)],\
            [(-1000,1000) for _ in range(N_blocks)],\
            [(-1000,1000) for _ in range(N_blocks)]])
            #note that for the cyclic variables the bounds are hard to define, you don't want to
            #get stuck at the edge. For now, just give them enough room to move around.
        result = minimize(self.cost_function_analytic,x0=[init_params],method='L-BFGS-B',
                          bounds = bounds, jac=True, options={'maxiter':maxiter,'gtol':gtol,'ftol':ftol})
        return result

    def save(self):
        datestr = datetime.now().strftime('%Y%m%d_%H_%M_%S')
        filestring = self.saving_directory + self.name + '_' + datestr
        filename_np = filestring + '.npz'
        filename_qt = filestring + '.qt'
        np.savez(filename_np, betas=self.betas, alphas=self.alphas, phis=self.phis, thetas=self.thetas,
                 max_alpha = self.max_alpha, max_beta = self.max_beta, name=self.name)
        print('parameters saved as: ' + filename_np)
        qt.qsave([self.initial_state,self.target_state], filename_qt)
        print('states saved as: ' + filename_qt)
       # print('name for loading:' + filestring)
        return filestring

    def load(self, filestring):
        filename_np = filestring + '.npz'
        filename_qt = filestring + '.qt'
        f = np.load(filename_np)
        betas, alphas, phis, thetas, max_alpha, max_beta, name =\
             f['betas'], f['alphas'], f['phis'],f['thetas'], f['max_alpha'], f['max_beta'], f['name']
        print('loaded parameters from:' + filename_np)
        f.close()
        states = qt.qload(filename_qt)
        initial_state, target_state = states[0], states[1]
        print('loaded states from:' + filename_qt)
        self.__init__(initial_state, target_state, len(betas),\
                 betas, alphas, phis, thetas, max_alpha, max_beta, None, name)
        

    
#%%
if __name__ == '__main__':
    N = 40
    N2 = 2
    N_blocks = 1
    init_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
    target_state = qt.tensor(qt.basis(N,1),qt.basis(N2,0))
    #target_state = qt.tensor(qt.basis(N,2),qt.basis(N2,0))
    #target_state = qt.tensor((qt.coherent(N,np.sqrt(2)) + qt.coherent(N,-np.sqrt(2))).unit(),qt.basis(N2,0))
    name = "test_CD"
    saving_directory = "C:\\Users\\Alec Eickbusch\\Desktop\\cd_grape_results\\"
    term_fid = 0.6
    a = CD_grape(init_state, target_state, N_blocks, max_alpha=4,max_beta = 4, name=name,\
                 saving_directory=saving_directory, term_fid=term_fid)
    a.randomize(alpha_scale=0.5, beta_scale = 1)
    fs = a.save()
    b = CD_grape()
    b.load(fs)
    #a.alphas = np.array([ 0.26770958-0.14984806j, 0.43046834+0.2849068j, 0.44571901+0.22912736j, -1.14223082-0.36287999j])
    #a.betas =  np.array([-2.11342464+0.16635473j, 0.18959299-0.50403244j, -0.68346816-0.31073315j, 0.00263728-0.3142331j]) 
    #a.aux_params = np.array([0.12630521, -0.77663552, 0.78854091])
    if 0:
        #a.plot_initial_state()
        a.plot_final_state()
        #a.plot_target_state()
    print(a.fidelity())
    #%% 
    '''
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
    '''
    #%%
    a.optimize()
    a.save()
    # %%
    #a.optimize_analytic()
    # %%
    a.plot_final_state()
    print('betas = np.' + repr(a.betas))
    print('alphas = np.' + repr(a.alphas))
    print('phis = np.' + repr(a.phis))
    print('thetas = np.' + repr(a.thetas))
    # %%
