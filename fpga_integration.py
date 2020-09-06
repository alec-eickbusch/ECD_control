#THIS FILE IS PYTHON 2

#%%
import numpy as np
from init_script import *
from basic_pulses import fastest_CD
import matplotlib.pyplot as plt
from fpga_lib.entities.pulses import gaussian_wave, gaussian_deriv_wave
#%% It's not so easy to get the waves from the FPGA code so I'll just hard code here
# Maybe make a better solution later
def calibrated_gaussian_pulse(sigma, chop, drag=0):
    i_wave = gaussian_wave(sigma, chop=chop)
    q_wave = drag * gaussian_deriv_wave(sigma, chop=chop)
    return i_wave + 1j*q_wave

#in the experiment it's calibrated to a unit amp of pi. 
#TODO: use the calibrated pi/2 pulse if it's closer to pi/2!
#I might want to just always use the pi/2 pulse, since it's closer to average rotation
def calibrated_rotate(unit_amp, theta, phi, sigma, chop, drag=0, detune=0):
    amp = (theta / np.pi) * unit_amp
    wave = amp*calibrated_gaussian_pulse(sigma, chop, drag)
    ts = np.arange(len(wave)) * 1e-9
    return wave * np.exp(-2j * np.pi * ts * detune)*np.exp(1j*phi)

def calibrated_displace(unit_amp, alpha, sigma, chop, drag=0, detune=0):
    amp = np.abs(alpha)*unit_amp
    phase = np.angle(alpha)
    wave = amp*unit_amp*calibrated_gaussian_pulse(sigma, chop, drag)
    ts = np.arange(len(wave)) * 1e-9
    return wave * np.exp(-2j * np.pi * ts * detune)*np.exp(1j*phase)
#%%
#To be used with the analysis class!
class FPGA_System:

    #for now will use the same sigma and chop for the displacement and qubit pulses
    #the sigma_cd and chop_cd will be for the gaussian displacement pulses during the conditional displacement
    #note that we want to calibrate small displacements and large displacements for conditional displacements
    #differently. So epsilon_m will be used for the conditional displacements, while the displacements
    #will be taken from the calibrated displacement pulses. I'm curious to see how well this works.
    def __init__(self, savefile, qubit, cavity, chi, alpha0, epsilon_m, buffer_time =0,
                 ring_up_time = 8):
        self.load_parameters(savefile)
        self.N_blocks = len(self.betas)
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

    def load_parameters(self, savefile):
        f = np.load(savefile)
        self.betas, self.alphas, self.phis, self.thetas =\
        f['betas'], f['alphas'], f['phis'],f['thetas']
        print 'loaded parameters from:' + savefile
        f.close()

    def CD_pulse(self, beta):
        pi_pulse = calibrated_rotate(self.q_unit_amp, np.pi, 0, self.q_sigma,\
                                 self.q_chop, self.q_drag, self.q_detune)
        return fastest_CD(beta, alpha0 = self.alpha0, epsilon_m = self.epsilon_m,\
                         chi=self.chi, buffer_time=self.buffer_time, sigma_q=self.q_sigma,\
                              chop_q=self.q_chop,ring_up_time=self.ring_up_time,
                              qubit_pi_pulse = pi_pulse)
    
    def rotate_displace_pulse(self, alpha, phi, theta):
        #TODO: Can I use the pi/2 pulse if it's closer to that?
        Omega = calibrated_rotate(self.q_unit_amp, theta, phi, self.q_sigma,\
                                 self.q_chop, self.q_drag, self.q_detune)
        epsilon = calibrated_displace(self.c_unit_amp, alpha, self.c_sigma,
                                      self.c_chop, self.c_drag, self.c_detune)
        if len(Omega)>len(epsilon):
            diff = int(len(Omega) - len(epsilon))
            epsilon = np.pad(epsilon, (diff/2, diff-diff/2),mode='constant')
        elif len(epsilon)>len(Omega):
            diff = int(len(epsilon) - len(Omega))
            Omega = np.pad(Omega, (diff/2, diff-diff/2),mode='constant')
        return epsilon, Omega

    def pulse_block(self, i):
        if i == self.N_blocks:
            beta = 0
        else:
            beta = self.betas[i]
        alpha = self.alphas[i]    
        phi = self.phis[i]
        theta = self.thetas[i]

        epsilon_D, Omega_R = self.rotate_displace_pulse(alpha, phi, theta)
        if beta!= 0:
            epsilon_CD, Omega_CD = self.CD_pulse(beta)
            epsilon = np.concatenate([epsilon_D, np.zeros(self.buffer_time),
                                    epsilon_CD, np.zeros(self.buffer_time)])
            Omega = np.concatenate([Omega_R, np.zeros(self.buffer_time),
                                    Omega_CD, np.zeros(self.buffer_time)])
        else: #don't need trailing zeros at the end
            epsilon = np.concatenate([epsilon_D])
            Omega = np.concatenate([Omega_R])
        return epsilon, Omega

    def composite_pulse(self):
        e = []
        O = []
        for i in range(self.N_blocks + 1):
            epsilon, Omega = self.pulse_block(i)
            e.append(epsilon)
            O.append(Omega)
        epsilon = np.concatenate(e)
        Omega = np.concatenate(O)
        return epsilon, Omega


#%% Testing
if __name__ == '__main__':
    def plot_pulse(epsilon, Omega):
        ts = np.arange(len(epsilon))
        plt.plot(ts,1e3*np.real(epsilon)/2/np.pi,label='Re(epsilon)')
        plt.plot(ts,1e3*np.imag(epsilon)/2/np.pi,label='Im(epsilon)')
        plt.plot(ts,1e3*np.real(Omega)/2/np.pi,label='10*Re(Omega)')
        plt.plot(ts,1e3*np.imag(Omega)/2/np.pi,label='10*Im(Omega)')
        plt.ylabel('drive amplitude (MHz)')
        plt.xlabel('t (ns)')
        plt.legend()
    savefile = "Z:\\Data\\Tennessee2020\\20200318_cooldown\\CD_grape\\optimization_tests_20200903\\fock 1_N_blocks_4_20200903_22_01_52.npz"
    chi = 2*np.pi*1e-3*0.03
    alpha0=60
    epsilon_m = 2*np.pi*1e-3*400
    buffer_time = 4
    ring_up_time = 200
    sys = FPGA_System(savefile, qubit_alice, storage_alice, chi, alpha0, epsilon_m, buffer_time, ring_up_time)
    sys.N_blocks = 1
    e, O = sys.composite_pulse()
    plt.figure()
    plot_pulse(e,O)
# %%

# %%
