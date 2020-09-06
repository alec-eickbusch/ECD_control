#%%
import numpy as np
import qutip as qt
from init_script import *
from CD_GRAPE.cd_grape import *
from CD_GRAPE.basic_pulses import fastest_CD, rotate, disp_gaussian
from CD_GRAPE.helper_functions import plot_pulse, alpha_from_epsilon
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16, 'pdf.fonttype': 42, 'ps.fonttype': 42})
from tqdm import tqdm

#%% It's not so easy to get the waves from the FPGA code so I'll just hard code here
# Maybe make a better solution later

def calibrated_gaussian_pulse(sigma, chop, drag=0):
    i_wave = gaussian_wave(self.sigma, chop=chop)
    q_wave = drag * gaussian_deriv_wave(self.sigma, chop=chop)
    return i_wave + 1j*q_wave

#in the experiment it's calibrated to a unit amp of pi. 
#TODO: use the calibrated pi/2 pulse if it's closer to pi/2!
#I might want to just always use the pi/2 pulse, since it's closer to average rotation
def calibrated_rotate(unit_amp, theta, phi, sigma, chop, drag=0, detune=0):
    amp = (angle / np.pi) * unit_amp
    wave = amp*calibrated_gaussian_pulse(sigma, chop, drag)
    ts = np.arange(len(wave)) * 1e-9
    return wave * np.exp(-2j * np.pi * ts * detune)*np.exp(1j*phase)

def calibrated_displace(unit_amp, alpha, sigma, chop, drag=0, detune=0):
    amp = np.abs(alpha)*unit_amp
    phase = np.anble(alpha)
    wave = amp*unit_amp*calibrated_gaussian_pulse(sigma, chop, drag)
    ts = np.arange(len(wave)) * 1e-9
    return wave * np.exp(-2j * np.pi * ts * detune)*np.exp(1j*phase)
#%%
class FPGA_System:

    #for now will use the same sigma and chop for the displacement and qubit pulses
    #the sigma_cd and chop_cd will be for the gaussian displacement pulses during the conditional displacement
    #note that we want to calibrate small displacements and large displacements for conditional displacements
    #differently. So epsilon_m will be used for the conditional displacements, while the displacements
    #will be taken from the calibrated displacement pulses. I'm curious to see how well this works.
    def __init__(self, qubit, cavity, chi, alpha0, epsilon_m, buffer_time =0,
                 ring_up_time = 8):
        self.qubit = qubit
        self.cavity = cavity
        self.chi = chi
        self.alpha0 = alpha0
        self.ring_up_time = int(ring_up_time)

        #epsilon_m used for conditional displacements
        self.epsilon_m = epsilon_m

        #buffer time inserted between qubit and cavity pulses. Use small number for fastest pulses.
        self.buffer_time = int(buffer_time)

        #parameters for the qubit pulse
        self.q_unit_amp = self.qubit.pulse.unit_amp
        self.q_sigma = self.qubit.pulse.sigma
        self.q_chop = self.qubit.pulse.chop
        self.q_detune = self.qubit.pulse.detune
        self.q_drag = self.qubit.pulse.drag

        #parameters for the cavity pulse
        self.c_unit_amp = self.cavity.displace.unit_amp
        self.c_sigma = self.cavity.displace.sigma
        self.c_chop = self.cavity.displace.chop
        self.c_detune = self.cavity.displace.detune
        self.c_drag = self.cavity.displace.drag

    def CD_pulse(self, beta):
        return fastest_CD(beta, alpha0 = self.alpha0, epsilon_m = self.epsilon_m,\
                         chi=self.chi, buffer_time=self.buffer_time, sigma_q=self.q_sigma,\
                              chop_q=self.q_chop,ring_up_time=self.ring_up_time)
    
    def rotate_displace_pulse(self, alpha, phi, theta):
        #TODO: Can I use the pi/2 pulse if it's closer to that?
        Omega = calibrated_rotate(self.q_unit_amp, theta, phi, self.q_sigma,\
                                 self.q_chop, self.q_drag, self.q_detune)
        epsilon = calibrated_displace(self.c_unit_amp, alpha, self.c_sigma,
                                      self.c_chop, self.c_drag, self.c_detune=0)
        if len(Omega)>len(epsilon):
            diff = int(len(Omega) - len(epsilon))
            epsilon = np.pad(epsilon, (diff/2, diff-diff/2))
        elif len(epsilon)>len(Omega)
            diff = int(len(epsilon) - len(Omega))
            Omega = np.pad(Omega, (diff/2, diff-diff/2))
        return epsilon, Omega

    #TODO: include case where alpha does not start at 0
    def stark_shift_correction(self, epsilon, Omega):
        alphas = alpha_from_epsilon(epsilon)
        stark_shift = -self.chi*np.abs(alphas**2)
        ts = np.arange(len(Omega))
        Omega_corrected = Omega*np.exp(1j*stark_shift*ts)
        return epsilon, Omega_corrected


    def simulate_pulse_trotter(self, epsilon, Omega, psi0, use_kerr = False,\
                               use_chi_prime = False, use_kappa = False, dt=1, pad=20,
                               stark_shift = True):
        epsilon = np.pad(epsilon, 20)
        Omega = np.pad(Omega, 20)
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
        if stark_shift:
            ss = 1.0
        else:
            ss = 0.0

        ts = np.arange(len(Omega))*dt
        H_array = []
        #TODO: make H construction parallel
        for i in tqdm(range(len(Omega)), desc="constructing H"):
            alpha = alphas[i]
            O = Omega[i]
            H_array.append(-chi_qs*a.dag()*a*q.dag()*q -chi_qs*(alpha*a.dag() + np.conj(alpha)*a)*q.dag()*q+ - ss*chi_qs * np.abs(alpha)**2*q.dag()*q\
                        - 1j*(kappa/2.0)*alpha*(a.dag() - a) + \
                        -kerr * (a.dag() + np.conj(alpha))**2 * (a + alpha)**2 + \
                    -chip*(a.dag()+np.conj(alpha))**2 * (a + alpha)**2 * q.dag() * q + \
                - (self.Ec/2.0)*q.dag()**2 * q**2 + np.real(O)*(q+q.dag()) +  np.imag(O)*1j*(q.dag() - q))

        psi = psi0
        for H in tqdm(H_array, desc='trotterized simulation'):
            U = (-1j*H*dt).expm()
            psi = U*psi
        #finally, move psi to displaced frame
        D = qt.tensor(qt.displace(N,alphas[-1]),qt.identity(N2))
        return  D*psi


class CD_grape_analysis:

    def __init__(self, cd_grape_object, system):
        self.cd_grape_object = cd_grape_object
        self.system = system

    #a block is defined by a rotation and displacement pulse, followed by a CD pulse.
    #for now pulse block will end with buffer time zeros.
    def pulse_block(self, i):
        if i == self.cd_grape_object.N_blocks:
            beta = 0
        else:
            beta = self.cd_grape_object.betas[i]
        alpha = self.cd_grape_object.alphas[i]    
        phi = self.cd_grape_object.phis[i]
        theta = self.cd_grape_object.thetas[i]

        #each CD pulse contains a pi pulse, so we need to flip the dfn of the bloch sphere
        #evey other round.
        #if i % 2 == 1:
            #theta = -theta
            #beta = -beta
        epsilon_D, Omega_R = self.system.rotate_displace_pulse(alpha, phi, theta)
        if beta!= 0:
            epsilon_CD, Omega_CD = self.system.CD_pulse(beta)
            epsilon = np.concatenate([epsilon_D, np.zeros(self.system.buffer_time),
                                    epsilon_CD, np.zeros(self.system.buffer_time)])
            Omega = np.concatenate([Omega_R, np.zeros(self.system.buffer_time),
                                    Omega_CD, np.zeros(self.system.buffer_time)])
        else:
            epsilon = np.concatenate([epsilon_D, np.zeros(self.system.buffer_time)])
            Omega = np.concatenate([Omega_R, np.zeros(self.system.buffer_time)])
        return epsilon, Omega

    def composite_pulse(self):
        e = []
        O = []
        for i in range(self.cd_grape_object.N_blocks + 1):
            epsilon, Omega = self.pulse_block(i)
            e.append(epsilon)
            O.append(Omega)
        epsilon = np.concatenate(e)
        Omega = np.concatenate(O)
        return epsilon, Omega
        


#%% Testing
if __name__ == '__main__':
    saving_directory = "C:\\Users\\Alec Eickbusch\\CD_grape_data\\"
    savefile = "C:\\Users\\Alec Eickbusch\\CD_grape_data\\cat_2_20200904_11_38_18"
    N = 60
    N2 = 2
    alpha0 = 60
    epsilon_m = 2*np.pi*1e-3*400.0
    Ec_GHz = 0.19267571 #measured anharmonicity
    Ec = (2*np.pi) * Ec_GHz
    sys = System(chi=2*np.pi*1e-3*0.03, Ec = Ec, alpha0=alpha0,\
         sigma=3, chop=4, epsilon_m = epsilon_m, buffer_time = 4,
         ring_up_time=16)
    a = qt.tensor(qt.destroy(N), qt.identity(N2))
    q = qt.tensor(qt.identity(N), qt.destroy(N2))
    sz = 1-2*q.dag()*q
    sx = (q+q.dag())
    sy = 1j*(q.dag() - q)
    
    #N_blocks = 4
    #init_state = qt.tensor(qt.basis(N,0),qt.basis(N2,0))
    
    #betas = np.array([-1.36234495+0.06757008j,  0.22142574-0.67359083j,
    #   -0.6176627 -0.45383865j,  0.47641324-0.16478542j])
    #alphas = np.array([ 0.09225012-0.05002766j,  0.18467285+0.19299275j,
    #   -0.07992844+0.01357778j, -0.13970461-0.01943885j,
    #   -0.08920646-0.18281873j])
    #phis = np.array([ 0.33571763, -2.0362122 , -1.85860465, -1.02817261, -0.58735507])
    #thetas = np.array([1.60209962, 1.09499735, 2.25532292, 1.59027321, 1.44970562])
    #betas = np.array([1e-5])
    #alphas = np.array([0,0])
    #phis = np.array([np.pi/2.0,0])
    #thetas = np.array([np.pi/3.0,0])
    #betas, alphas, phis, thetas = None, None, None, None

    #target_state = qt.tensor(qt.basis(N,1),qt.basis(N2,0))
    #target_state = qt.tensor(qt.basis(N,2),qt.basis(N2,0))
    #target_state = qt.tensor((qt.coherent(N,np.sqrt(2)) + qt.coherent(N,-np.sqrt(2))).unit(),qt.basis(N2,0))
    #target_state = qt.tensor(qt.coherent(N,1j), qt.basis(N2, 1))
    #a = CD_grape(init_state, target_state, N_blocks, max_alpha=4, max_beta = 4, term_fid= 0.99)
    #a.randomize(alpha_scale=1, beta_scale = 1.5)
    #a.optimize()
    #a.save()
    a = CD_grape()
    a.load(savefile)
    if 1:
        #a.plot_initial_state()
        a.plot_final_state()
        #a.plot_final_state()
        #a.plot_target_state()
    print(a.fidelity())
    
    analysis = CD_grape_analysis(a,sys)
    e,O = analysis.composite_pulse()
    alphas = alpha_from_epsilon(e)
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(np.abs(alphas))
    plt.axhline(alpha0)
    #e2,O2 = sys.stark_shift_correction(e,O)
    #%% 
    if 1:
        plt.figure(figsize = (10,6), dpi=200)
        plot_pulse(e,O)
        plt.plot(np.abs(1e3*e/2/np.pi))
        plt.axhline(1e3*epsilon_m/2/np.pi)
        plt.axhline(-1e3*epsilon_m/2/np.pi)
        #plt.figure(figsize=(10, 6), dpi=200)
        #plot_pulse(e2,O2)
#%%
    psi0 = a.initial_state
    psif = sys.simulate_pulse_trotter(e,O,psi0, stark_shift=False)

    plot_wigner(psif)
    print('discrete sim:')
    print(a.final_state().ptrace(1))
    print('trotter sim:')
    print(psif.ptrace(1))

    fid2 = qt.fidelity(psif, a.final_state())

    print('sim fid: ' + str(fid2))

    fid3 = qt.fidelity(psif.ptrace(0), a.final_state().ptrace(0))

    print('sim fid cav: ' + str(fid3))
#%%
    alphas = alpha_from_epsilon(e)
    plt.figure()
    plt.plot(np.real(alphas),label='re(alpha)')
    plt.plot(np.imag(alphas),label='im(alpha)')
    plt.legend()


# %%
    print('betas = np.' + repr(a.betas))
    print('alphas = np.' + repr(a.alphas))
    print('phis = np.' + repr(a.phis))
    print('thetas = np.' + repr(a.thetas))
