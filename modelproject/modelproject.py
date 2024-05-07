from types import SimpleNamespace
import numpy as np
from scipy import optimize

class RamseyModelClass1():
    ''' Ramsey model class '''

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.ss = SimpleNamespace()
        self.path = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()

    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0
        par.beta = np.nan

        # b. firms
        par.Gamma = np.nan
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30
        par.theta = 0.05
        par.delta = 0.05

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'scpiy'
        par.Tpath = 500

    def allocate(self):
        """ allocate arrays for transition path """

        par = self.par
        path = self.path

        allvarnames = ['Gamma','K','C','rk','w','r','Y','K_lag']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # a. unpack
        K_ss,Y_ss = KY_ss

        # b. solve for C_ss
        ss.C = Y_ss - par.delta*K_ss

        # c. solve for Gamma
        ss.Gamma = par.alpha*Y_ss/K_ss

        # d. solve for r_ss
        ss.r = ss.Gamma - par.delta

        # e. solve for w_ss
        ss.w = (1-par.alpha)*Y_ss

        # f. solve for rk_ss
        ss.rk = ss.r + par.delta

        # g. store
        ss.K = K_ss
        ss.Y = Y_ss

    if do_print:

        print(f'Y_ss = {ss.Y:.4f}')
        print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
        print(f'rk_ss = {ss.rk:.4f}')
        print(f'r_ss = {ss.r:.4f}')
        print(f'w_ss = {ss.w:.4f}')
        print(f'Gamma = {ss.Gamma:.4f}')


def production(par,Gamma,K):
    """ production function (ces)"""

    if par.production_function == 'cobb-douglas':
        Y = Gamma(*par.alpha*K**(-par.theta-1)+(1-par.alpha)(1.0)**(-par.theta-1))**(-1/par.theta)
        rk = Gamma*par.alpha*K_lag**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)
        w = Gamma*(1-par.alpha)*(1.0)**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)
    
    return Y,rk,w

def solve_ss(alpha, c):
    """ Example function. Solve for steady state k. 

    Args:
        c (float): costs
        alpha (float): parameter

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous) and c (exogenous).
    f = lambda k: k**alpha - c
    obj = lambda kss: kss - f(kss)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
    
    return result