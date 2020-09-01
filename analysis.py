import numpy as np
import qutip as qt
from cd_grape import *

class system:

    #for now will use the same sigma and chop for the displacement and qubit pulses
    #the sigma_cd and chop_cd will be for the gaussian displacement pulses during the conditional displacement
    def __init__(self, chi, alpha_m, sigma, chop, sigma_cd, chop_cd):
        self.chi = chi
        self.alpha_m = alpha_m
        self.sigma = sigma
        self.chop = chop

    
    def composite_pulse(self):


class cd_grape_analysis:

    def __init__(self, cd_grape_object, system):
        self.cd_grape_object = cd_grape_object
        self.system = system

    
