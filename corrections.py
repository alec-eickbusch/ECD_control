import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import minimize, Bounds, basinhopping
from tqdm import tqdm

class OptimalCorrections:
    def __init__(self, distorted_state, target_state, alpha = 0.0, theta = 0.0,
                 d_alpha = 0.5, d_theta = 0.2):
        self.theta = theta
        self.alpha = alpha
        self.d_alpha = d_alpha
        self.d_theta = d_theta
        self.target_state = target_state
        self.distorted_state = distorted_state
        N = self.target_state.dims[0][0]
        if len(self.target_state.dims[0]) == 2:
            N2 = self.target_state.dims[0][1]
        else:
            N2 = 0
        if N2 != 0:
            self.a = qt.tensor(qt.destroy(N), qt.identity(N2))
        else:
            self.a = qt.destroy(N)

    def corrected_state(self, parameters=None):
        if parameters is not None:
            alpha = (parameters[0] + parameters[1]*1j)
            theta = (parameters[2])
        else:
            alpha = self.alpha
            theta = self.theta
        r = (1j*self.a.dag()*self.a*theta).expm()
        d = (alpha*self.a.dag() - np.conj(alpha)*self.a).expm()
        corrected_state = r*d*self.distorted_state
        return corrected_state

    def metric(self, parameters=None, output=True):
        f = np.abs(qt.fidelity(self.target_state, self.corrected_state(parameters)))
        if output:
            print('\rfid: %.4f' % f, end='')
        return -1*f

    #todo: parallelize this
    def grid_search(self, pts = 11):
        alphas_real = np.linspace(self.alpha - self.d_alpha, self.alpha + self.d_alpha, pts)
        alphas_imag = np.linspace(self.alpha - self.d_alpha, self.alpha + self.d_alpha, pts)
        thetas = np.linspace(self.theta - self.d_theta, self.theta + self.d_theta, pts)
        fids = []
        parameters = []
        print("\nstarting slow grid search.")

        print("d_alpha: %.3f d_theta: %.3f grid pts: %dx%dx%d (%d total)" %\
            (self.d_alpha, self.d_theta, pts, pts, pts, pts**3))
        for alphar in tqdm(alphas_real):
            for alphai in alphas_imag:
                for theta in thetas:
                    para = [alphar, alphai, theta]
                    parameters.append(para)
                    fid = -1*self.metric(para, output=False)
                    fids.append(fid)
        best_idx = np.argmax(np.array(fids))
        best_fid = fids[best_idx]
        best_para = parameters[best_idx]
        self.alpha = best_para[0] + 1j*best_para[1]
        self.theta = best_para[2]
        print("\n\n Grid search complete. \n alpha: %.4f + %.4f theta deg: %.4f fidelity: %.4f" % \
            (np.real(self.alpha),np.imag(self.alpha), self.theta*180.0/np.pi, best_fid))

    def optimize(self):
        init_params = np.array([np.real(self.alpha),np.imag(self.alpha),self.theta])
        bounds = Bounds([self.alpha - self.d_alpha,self.alpha - self.d_alpha,self.theta - self.d_theta],\
             [self.alpha + self.d_alpha,self.alpha + self.d_alpha, self.theta + self.d_theta])
        res = minimize(self.metric, init_params, method='BFGS', bounds=bounds,\
             options = {'ftol':1e-9, 'gtol':1e-9})
        self.alpha = res.x[0] + res.x[1]*1j
        self.theta = res.x[2]

        print("\ncorrection alpha: " + str(self.alpha))
        print("correction theta: " + str(self.theta*360/(2*np.pi)) + " degrees")
        fid = np.abs(self.metric())
        print("\nFidelity of optimally corrected state: " + str(fid))
        return res.x


    def opt_correction2(self):
        init_params = np.array([np.real(self.initial_alpha),np.imag(self.initial_alpha),self.initial_theta])
        bounds = Bounds([-3,-3,-np.pi], [3,3, np.pi])
        minimizer_kwargs = {'method':'L-BFGS-B', 'jac':False, 'bounds':bounds,\
                 'options':{'ftol':1e-8, 'gtol':1e-8}}
        basinhopping_kwargs = {'niter':10,'T':0.1}
        mytakestep = MyTakeStepCorrections()
        mybounds = MyBoundsCorrections()
        def callback_fun(x, f, accepted):
            self.basinhopping_num += 1
            print(" basin #%d at min %.4f. accepted: %d" %\
                 (self.basinhopping_num,f, int(accepted)))
            print("alpha: %.3f + %.3fj, thata deg: %.3f" %(x[0],x[1],x[2]*180.0/np.pi))
        self.basinhopping_num = 0
        res = basinhopping(self.metric, x0=init_params,\
                        take_step=mytakestep, accept_test=mybounds,callback=callback_fun,\
                        minimizer_kwargs=minimizer_kwargs, **basinhopping_kwargs)
        #res = minimize(self.metric, guess, method='L-BFGS-B', bounds=bounds, options={'ftol':1e-8, 'gtol':1e-8})
        alpha = res.x[0] + res.x[1]*1j
        theta = res.x[2]

        print("correction alpha: " + str(alpha))
        print("correction theta: " + str(theta*360/(2*np.pi)) + " degrees")
        fid = np.abs(self.metric(res.x))
        print("Fidelity of optimally corrected state: " + str(fid))
        return res.x
