#%%
import numpy as np
import qutip as qt
from cd_grape import *
from basic_pulses import CD, rotate, disp
#from helper_functions import plot_pulse, alpha_from_epsilon
from tqdm import tqdm
#%%
class System:

    #for now will use the same sigma and chop for the displacement and qubit pulses
    #the sigma_cd and chop_cd will be for the gaussian displacement pulses during the conditional displacement
    def __init__(self, chi, Ec, alpha0, epsilon_m, sigma, chop, buffer_time =0):
        self.chi = chi
        self.Ec = Ec
        self.alpha0 = alpha0
        #sigma and chop for qubit pulse and displacements
        self.sigma = int(sigma)
        self.chop = int(chop)

        #sigma and chop for displacement during the conditional displacement 
        #calculated from provided value of epsilon_m
        self.epsilon_m = epsilon_m

        #buffer time inserted between qubit and cavity pulses. Use small number for best performance.
        self.buffer_time = int(buffer_time)

    def CD_pulse(self, beta):
        return CD(beta=beta, alpha0 = self.alpha0, chi=self.chi,\
                sigma=self.sigma_cd, chop=self.chop_cd, sigma_q=self.sigma,\
                chop_q=self.chop, buffer_time=self.buffer_time)
    
    def rotate_displace_pulse(self, alpha, phi, theta):
        Omega = rotate(theta=theta, phi=phi, sigma=self.sigma, chop=self.chop)
        epsilon = disp(alpha=alpha, sigma=self.sigma, chop=self.chop)
        return epsilon, Omega

    def simulate_pulse_trotter(self, epsilon, Omega, psi0, use_kerr = False,\
                               use_chi_prime = False, use_kappa = False, dt=1):
        alphas = alpha_from_epsilon(epsilon)
        N = psi0.dims[0][0]
        N2 = psi0.dims[0][1]

        a = qt.tensor(qt.destroy(N), qt.identity(N2))
        q = qt.tensor(qt.identity(N), qt.destroy(N2))
        sz = 1-2*q.dag()*q
        sx = (q+q.dag())
        sy = 1j*(q.dag() - q)
        n = a.dag()*a

        chi_qs = self.chi
        #todo: make kerr and chi prime elements of the class
        #if use_kerr:
           # kerr = self.K
        #else:
        kerr = 0
        #if use_chi_prime:
            #chip = self.chi_prime
        #else:
        chip = 0
        #if use_kappa:
            #kappa = self.kappa_cavity
        #else:   
        kappa = 0
        #if stark_shift:
           # ss = 1.0
        #else:
        ss = 0.0

        ts = np.arange(len(Omega))*dt
        print('constructing H')
        H_array = []
        for i in tqdm(range(len(Omega))):
            alpha = alphas[i]
            O = Omega[i]
            H_array.append(-chi_qs*a.dag()*a*q.dag()*q -chi_qs*(alpha*a.dag() + np.conj(alpha)*a)*q.dag()*q - ss*chi_qs * np.abs(alpha)**2*q.dag()*q\
                        - 1j*(kappa/2.0)*alpha*(a.dag() - a) + \
                        -kerr * (a.dag() + np.conj(alpha))**2 * (a + alpha)**2 + \
                    -chip*(a.dag()+np.conj(alpha))**2 * (a + alpha)**2 * q.dag() * q + \
                - (Ec/2.0)*q.dag()**2 * q**2 + np.real(O)*(q+q.dag()) + np.imag(O)*1j*(q.dag() - q))

        psi = psi0
        print('trotterized simulation')
        for H in tqdm(H_array):
            U = (-1j*H*dt).expm()
            psi = U*psi
        return psi


class CD_grape_analysis:

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

        #each CD pulse contains a pi pulse, so we need to flip the dfn of the bloch sphere
        #evey other round.
        if i % 2 == 1:
            theta = -theta
        epsilon_D, Omega_R = self.system.rotate_displace_pulse(alpha, phi, theta)
        epsilon_CD, Omega_CD = self.system.CD_pulse(beta)
        epsilon = np.concatenate([epsilon_D, np.zeros(self.system.buffer_time),
                                  epsilon_CD, np.zeros(self.system.buffer_time)])
        Omega = np.concatenate([Omega_R, np.zeros(self.system.buffer_time),
                                Omega_CD, np.zeros(self.system.buffer_time)])
        return epsilon, Omega

    def composite_pulse(self):
        e = []
        O = []
        for i in range(self.cd_grape_object.N_blocks):
            epsilon, Omega = self.pulse_block(i)
            e.append(epsilon)
            O.append(Omega)
        epsilon = np.concatenate(e)
        Omega = np.concatenate(O)
        return epsilon, Omega


#%% Testing
if __name__ == '__main__':
    N = 20
    N2 = 2
    N_blocks = 1
    init_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
    a = qt.tensor(qt.destroy(N), qt.identity(N2))
    q = qt.tensor(qt.identity(N), qt.destroy(N2))
    sz = 1-2*q.dag()*q
    sx = (q+q.dag())
    sy = 1j*(q.dag() - q)
    aux_ops = [a.dag()*a,sz,sx]
    aux_params = np.array([0,0,0], dtype=np.float64)
    alphas = np.array([0])
    betas =  np.array([1]) 
    phis = [0] 
    thetas = [0]
    aux_params = np.array([0,0,0])
    #aux_bounds = np.array([(-np.pi,np.pi),(-np.pi/2.0,np.pi/2.0),(-np.pi/2.0,np.pi/2.0)])
    aux_bounds = np.array([(-1000,1000) for _ in range(len(aux_ops))])
    #target_state = qt.tensor(qt.basis(N,1),qt.basis(N2,0))
    #target_state = qt.tensor(qt.basis(N,2),qt.basis(N2,0))
    target_state = qt.tensor((qt.coherent(N,np.sqrt(2)) + qt.coherent(N,-np.sqrt(2))).unit(),qt.basis(N2,0))
    a = CD_grape(init_state, target_state, N_blocks, init_alphas=alphas,init_phis=phis, init_thetas=thetas,\
                init_betas=betas, aux_params=aux_params, max_abs_alpha=4,max_abs_beta = 4,
                aux_ops=aux_ops, aux_params_bounds=aux_bounds)
    #a.randomize(alpha_scale=1, beta_scale = 2)
    if 1:
        #a.plot_initial_state()
        a.plot_final_state()
        #a.plot_target_state()
    print(a.fidelity())
    Ec_GHz = 0.19267571 #measured anharmonicity
    Ec = (2*np.pi) * Ec_GHz
    sys = System(chi=2*np.pi*1e-3*0.03, Ec = Ec, alpha0=60,\
         sigma=6, chop=4, sigma_cd=12, chop_cd=4, buffer_time = 8)
    analysis = CD_grape_analysis(a,sys)
    e,O = analysis.composite_pulse()
    #%% 
    plt.figure(figsize = (10,6), dpi=200)
    plot_pulse(e,O)

    psi0 = a.initial_state
    psif = sys.simulate_pulse_trotter(e,O,psi0)

    plot_wigner(psif)


# %%
