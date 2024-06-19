from types import SimpleNamespace
import numpy as np
from scipy import optimize

class RamseyModelClass():

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
        par.sigma = 2.0 # CRRA coefficient
        par.beta = np.nan # discount factor
        par.sigmah = 2.0 # IES of housing
        par.upsilon = 1.0

        # b. housing 
        par.Hbar = 1.0 # fixed housing supply
        par.tauH = 0.05 # property tax
        par.upsilon = 2.0 # housing utility weight

        # b. firms
        par.Gamma = np.nan
        par.production_function = 'cobb-douglas-gov'
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # substitution parameter        
        par.delta = 0.05 # depreciation rate

        par.alphaK = 0.30 # capital weight
        par.alphaG = 0.10 # government capital weight

        # c. initial
        par.K_lag_ini = 1.0

        # d. misc
        par.solver = 'broyden' # solver for the equation system, 'broyden' or 'scipy'
        par.Tpath = 500 # length of transition path, "truncation horizon"
        par.endovar = 3 # number of endogenous variables

    def allocate(self):
        """ allocate arrays for transition path """
        
        par = self.par
        path = self.path

        allvarnames = ['Gamma','K','C','q','tauH','rk','w','r','Y','K_lag', 'KG', 'G']
        for varname in allvarnames:
            path.__dict__[varname] =  np.nan*np.ones(par.Tpath)

    def find_steady_state(self,KY_ss,do_print=True):
        """ find steady state """

        par = self.par
        ss = self.ss

        # h. government capital
        ss.G = par.tauH*par.Hbar
        ss.KG = ss.G

        # a. find A
        ss.K = KY_ss
        Y,_,_ = production(par,1.0,ss.K,ss.G)
        ss.Gamma = 1/Y

        # b. factor prices
        ss.Y,ss.rk,ss.w = production(par,ss.Gamma,ss.K,ss.G)
        assert np.isclose(ss.Y,1.0)

        ss.r = ss.rk-par.delta
        
        # c. implied discount factor
        par.beta = 1/(1+ss.r)

        # d. consumption
        ss.C = ss.Y - par.delta*ss.K - ss.G

        # g. solve for housing market 
        ss.tauH = par.tauH
        ss.H = par.Hbar
        ss.q = 1/(1 - par.beta) * ((par.upsilon*ss.H**(-par.sigmah) - ss.tauH*par.beta*ss.C**(-par.sigma))/ss.C**(-par.sigma))


        if do_print:

            print(f'Y_ss = {ss.Y:.4f}')
            print(f'C_ss = {ss.C:.4f}')
            print(f'q_ss = {ss.q:.4f}')
            print(f'K_ss/Y_ss = {ss.K/ss.Y:.4f}')
            print(f'rk_ss = {ss.rk:.4f}')
            print(f'r_ss = {ss.r:.4f}')
            print(f'w_ss = {ss.w:.4f}')
            print(f'Gamma = {ss.Gamma:.4f}')
            print(f'beta = {par.beta:.4f}')
            print(f'G = {ss.G:.4f}')

    def evaluate_path_errors(self):
        """ evaluate errors along transition path """
       
        par = self.par
        ss = self.ss
        path = self.path

        # a. consumption        
        C = path.C
        C_plus = np.append(path.C[1:],ss.C)
        
        # b. capital
        K = path.K
        K_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)

        # c. housing
        q = path.q
        q_plus = np.append(path.q[1:],ss.q)

        tauH = path.tauH
        tauH_plus = np.append(path.tauH[1:],ss.tauH)

        # d. government capital
        G = path.G
        G_lag = np.insert(path.tauH[1:]*par.Hbar,0,ss.G)
        # G_lag = path.K_lag = np.insert(K[:-1],0,par.K_lag_ini)
        # print(G_lag)

        # c. production and factor prices
        path.Y,path.rk,path.w = production(par,path.Gamma,K_lag,G_lag)
        path.r = path.rk-par.delta
        r_plus = np.append(path.r[1:],ss.r)
        # print(path.G)

        # d. errors (also called MatcalH in the notebook)
        errors = np.nan*np.ones((par.endovar,par.Tpath))
        errors[0,:] = C**(-par.sigma) - par.beta*(1+r_plus)*C_plus**(-par.sigma)
        errors[1,:] = K - ((1-par.delta)*K_lag + (path.Y - C - G))
        errors[2,:] = par.upsilon*par.Hbar**(-par.sigmah) + par.beta*C_plus**(-par.sigma)*(q_plus - tauH_plus) - C**(-par.sigma)*q
        
        return errors.ravel()
        
    def calculate_jacobian(self,h=1e-6):
        """ calculate jacobian """
        
        par = self.par
        ss = self.ss
        path = self.path
        
        # a. allocate
        Njac = par.endovar*par.Tpath
        jac = self.jac = np.nan*np.ones((Njac,Njac))
        
        x_ss = np.nan*np.ones((par.endovar,par.Tpath))
        x_ss[0,:] = ss.C
        x_ss[1,:] = ss.K
        x_ss[2,:] = ss.q
        # x_ss[3,:] = ss.G
        x_ss = x_ss.ravel()

        # b. baseline errors
        path.C[:] = ss.C
        path.K[:] = ss.K
        path.q[:] = ss.q
        # path.G[:] = ss.G
        base = self.evaluate_path_errors()

        # c. jacobian
        for i in range(Njac):
            
            # i. add small number to a single x (single K or C) 
            x_jac = x_ss.copy()
            x_jac[i] += h
            x_jac = x_jac.reshape((par.endovar,par.Tpath))
            
            # ii. alternative errors
            path.C[:] = x_jac[0,:]
            path.K[:] = x_jac[1,:]
            path.q[:] = x_jac[2,:]
            # path.G[:] = x_jac[3,:]
            alt = self.evaluate_path_errors()

            # iii. numerical derivative
            jac[:,i] = (alt-base)/h
        
    def solve(self,do_print=True):
        """ solve for the transition path """

        par = self.par
        ss = self.ss
        path = self.path
        
        # a. equation system
        def eq_sys(x):
            
            # i. update
            x = x.reshape((par.endovar,par.Tpath))
            path.C[:] = x[0,:]
            path.K[:] = x[1,:]
            path.q[:] = x[2,:]
            # path.G[:] = x[3,:]
            
            # ii. return errors
            return self.evaluate_path_errors()

        # b. initial guess
        x0 = np.nan*np.ones((par.endovar,par.Tpath))
        x0[0,:] = ss.C
        x0[1,:] = ss.K
        x0[2,:] = ss.q
        # x0[3,:] = ss.G
        x0 = x0.ravel()

        # c. call solver
        if par.solver == 'broyden':

            x = broyden_solver(eq_sys,x0,self.jac,do_print=do_print)
        
        elif par.solver == 'scipy':
            
            root = optimize.root(eq_sys,x0,method='hybr',options={'factor':1.0})
            # the factor determines the size of the initial step
            #  too low: slow
            #  too high: prone to errors
             
            x = root.x

        else:

            raise NotImplementedError('unknown solver')
            

        # d. final evaluation
        eq_sys(x)
            
def production(par,Gamma,K_lag, G):
    """ production and factor prices """

    # a. production and factor prices
    if par.production_function == 'ces':

        # a. production
        Y = Gamma*( par.alpha*K_lag**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # b. factor prices
        rk = Gamma*par.alpha*K_lag**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)
        w = Gamma*(1-par.alpha)*(1.0)**(-par.theta-1) * (Y/Gamma)**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # a. production
        Y = Gamma*K_lag**par.alpha * (1.0)**(1-par.alpha)

        # b. factor prices
        rk = Gamma*par.alpha * K_lag**(par.alpha-1) * (1.0)**(1-par.alpha)
        w = Gamma*(1-par.alpha) * K_lag**(par.alpha) * (1.0)**(-par.alpha)

    elif par.production_function == 'cobb-douglas-gov':
            
        # a. production
        Y = Gamma*K_lag**par.alphaK * (1.0)**(1-par.alphaK-par.alphaG) * G**(par.alphaG)

        # b. factor prices
        rk = Gamma*par.alphaK * K_lag**(par.alphaK-1) * (1.0)**(1-par.alphaK-par.alphaG) * G**(par.alphaG)
        w = Gamma*(1-par.alphaK-par.alphaG) * K_lag**par.alphaK * (1.0)**(-par.alphaK-par.alphaG) * G**(par.alphaG)

    else:

        raise Exception('unknown type of production function')

    return Y,rk,w            

def broyden_solver(f,x0,jac,tol=1e-8,maxiter=100,do_print=False):
    """ numerical equation system solver using the broyden method 
    
        f (callable): function return errors in equation system
        jac (ndarray): initial jacobian
        tol (float,optional): tolerance
        maxiter (int,optional): maximum number of iterations
        do_print (bool,optional): print progress

    """

    # a. initial
    x = x0.ravel()
    y = f(x)

    # b. iterate
    for it in range(maxiter):
        
        # i. current difference
        abs_diff = np.max(np.abs(y))
        if do_print: print(f' it = {it:3d} -> max. abs. error = {abs_diff:12.8f}')

        if abs_diff < tol: return x
        
        # ii. new x
        dx = np.linalg.solve(jac,-y)
        assert not np.any(np.isnan(dx))
        
        # iii. evaluate
        ynew = f(x+dx)
        dy = ynew-y
        jac = jac + np.outer(((dy - jac @ dx) / np.linalg.norm(dx)**2), dx)
        y = ynew
        x += dx
            
    else:

        raise ValueError(f'no convergence after {maxiter} iterations')        