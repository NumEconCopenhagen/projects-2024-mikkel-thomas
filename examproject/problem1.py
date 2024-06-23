import numpy as np
from types import SimpleNamespace
from scipy import optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})
from matplotlib import cm # for colormaps

def ell_star(par,p):
    return (p*par.A*par.gamma/1)**(1/(1-par.gamma))

def y_star(par,p):
    return par.A*ell_star(par,p)**par.gamma

def pi(par,p):
    return (1-par.gamma)/par.gamma * 1 * (p*par.A*par.gamma/1)**(1/(1-par.gamma))

def c1_star(par,p1, p2,ell):
    return par.alpha * (1*ell + par.T + pi(par,p1) + pi(par,p2))/p1

def c2_star(par,p1, p2,ell):
    return (1-par.alpha) * (1*ell + par.T + pi(par,p1) + pi(par,p2))/(p2 + par.tau)

objective = lambda ell, par, p1, p2: -(np.log(c1_star(par,p1, p2,ell)**par.alpha*c2_star(par,p1, p2,ell)**(1-par.alpha)) - par.nu * ell**(1+par.epsilon)/(1+par.epsilon))

def plot_excess_demand(par):
    p1 = np.linspace(0.2, 2.0, 10)
    p2 = np.linspace(0.2, 2.0, 10)

    excess_demand_ell = np.nan + np.zeros((len(p1), len(p2)))
    excess_deamnd_c1 = np.nan + np.zeros((len(p1), len(p2)))
    excess_demand_c2 = np.nan + np.zeros((len(p1), len(p2)))
    ell_array = np.nan + np.zeros((len(p1), len(p2)))

    for i, p1i in enumerate(p1):
        for j, p2j in enumerate(p2):
            sol_case1 = optimize.minimize_scalar(objective, method='bounded', bounds=(1e-8,10*10^6), args=(par,p1i,p2j))
            ell = sol_case1.x
            ell_array[i,j] = ell
            c1 = c1_star(par,p1i, p2j,ell)
            c2 = c2_star(par,p1i, p2j,ell)
            ell1, ell2 = ell_star(par,p1i), ell_star(par,p2j)

            excess_demand_ell[i,j] = ell - (ell1 + ell2)
            excess_deamnd_c1[i,j] = c1 - y_star(par,p1i)
            excess_demand_c2[i,j] = c2 - y_star(par,p2j)

    fig = plt.figure(figsize=(18,6))
    ax = fig.add_subplot(1,3,1,projection='3d')
    ax.plot_surface(p1,p2,excess_demand_ell,cmap=cm.jet)
    ax.set_xlabel('$p_1$')
    ax.set_ylabel('$p_2$')
    ax.set_title('Excess demand for $\ell$')

    ax2 = fig.add_subplot(1,3,2,projection='3d')
    ax2.plot_surface(p1,p2,excess_deamnd_c1,cmap=cm.jet)
    ax2.set_xlabel('$p_1$')
    ax2.set_ylabel('$p_2$')
    ax2.set_title('Excess demand for $c_1$')

    ax3 = fig.add_subplot(1,3,3,projection='3d')
    ax3.plot_surface(p1,p2,excess_demand_c2,cmap=cm.jet)
    ax3.set_xlabel('$p_1$')
    ax3.set_ylabel('$p_2$')
    ax3.set_title('Excess demand for $c_2$')

    plt.show()

def solve_equilibrium(par):
    """ Function that solves for the equilibrium prices and excess demand for the system of equations given the parameters"""
    def errors(p):
        """ Function that returns the market clearing errors for the system of equations given p"""
        demand_errors = np.zeros(2)
        p1, p2 = p
        sol_case1 = optimize.minimize_scalar(objective, method='bounded', bounds=(1e-8,10*10^6), args=(par,p1,p2))
        ell = sol_case1.x

        demand_errors[0] = c1_star(par,p1, p2,ell) - y_star(par,p1)
        demand_errors[1] = ell - (ell_star(par,p1) + ell_star(par,p2))

        return demand_errors

    x0 = [1, 1]  # Initial guess for the prices
    root = optimize.root(errors,x0,method='hybr',options={'factor':1.0})
    errors_opt = errors(root.x)
    print(f'Optimal prices: p1 = {root.x[0]:.2f}, p2 = {root.x[1]:.2f}')
    print(f'Excess demand for c1: {errors_opt[0]:.6f}')
    print(f'Excess demand for ell: {errors_opt[1]:.6f}')

def errors(p, par):
    """ Function that returns the market clearing errors for the system of equations given p"""
    demand_errors = np.zeros(2)
    p1, p2 = p
    sol_case1 = optimize.minimize_scalar(objective, method='bounded', bounds=(1e-8,10*10^6), args=(par,p1,p2))
    ell = sol_case1.x

    par.T = par.tau*y_star(par,p2)

    demand_errors[0] = c1_star(par,p1, p2,ell) - y_star(par,p1)
    demand_errors[1] = ell - (ell_star(par,p1) + ell_star(par,p2))

    return demand_errors

def swf(par, p1, p2):
    """ Social welfare function"""
    sol_case1 = optimize.minimize_scalar(objective, method='bounded', bounds=(1e-8,10*10^6), args=(par,p1,p2))
    ell = sol_case1.x
    U = (np.log(c1_star(par,p1, p2,ell)**par.alpha*c2_star(par,p1, p2,ell)**(1-par.alpha)) - par.nu * ell**(1+par.epsilon)/(1+par.epsilon))
    y2 = y_star(par,p2)
    return U - par.kappa*y_star(par, p2)

def find_optimal_tau(par):
    def maximize_swf(par):
        """
        Function to be optimized. It returns the negative of the SWF since we are using a minimization
        optimizer to find the maximum value.
        """

        x0 = [1, 1]  # Initial guess for the prices

        def objective(tau):
            par.tau = tau
            # Solve for equilibrium prices with the current value of tau
            root = optimize.root(lambda prices: errors(prices, par), x0, method='hybr', options={'factor': 1.0})
            p1, p2 = root.x
            # Compute the SWF for these prices and the current tau
            swf_value = swf(par, p1, p2)
            # Return the negative SWF because we are minimizing
            return -swf_value

        # Initial guess for tau
        tau_initial_guess = 1
        # Bounds for tau (assuming tau > 0, adjust according to your model's specifics)
        tau_bounds = (1e-8, 1)
        # Optimize
        result = optimize.minimize_scalar(objective, bounds=tau_bounds, method='bounded')

        # Return the optimal tau and the maximum SWF (note the negation to correct the sign)
        return result.x, -result.fun

    optimal_tau, max_swf = maximize_swf(par)
    print(f'Optimal tau: {optimal_tau:.4f}, Maximum SWF: {max_swf:.4f}')
    print(f'Implied T: {par.T:.4f}')

def plot_swf(par):
    tau_list = np.linspace(0, 1, 100)
    swf_list = np.zeros(100)

    x0 = [1, 1]  # Initial guess for the prices
    for i, tau in enumerate(tau_list):
        par.tau = tau
        root = optimize.root(lambda prices: errors(prices, par), x0, method='hybr', options={'factor': 1.0})
        p1, p2 = root.x
        swf_list[i] = swf(par, p1, p2)

    fig = plt.figure(figsize=(6,6))

    ax = fig.add_subplot(1,1,1)
    ax.plot(tau_list, swf_list)
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('SWF')
    ax.set_title('SWF as a function of $\\tau$')