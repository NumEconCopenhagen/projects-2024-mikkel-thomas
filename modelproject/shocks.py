import ramsey_thje
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.grid":True,"grid.color":"black","grid.alpha":"0.25","grid.linestyle":"--"})
plt.rcParams.update({'font.size': 14})

def updatepar(par, parnames, parvals):
    ''' Update parameter values in par of parameters in parnames '''

    for i,parval in enumerate(parvals):
        parname = parnames[i]
        setattr(par,parname,parval) # It gives the attibute parname the new value parval, within the par class
    return par

def shock_1(update_par = False, parnames=None, parvals=None):
    """ 
        simulate a shock to productivity

        Args:
            update_par (bool): if True, update parameters
            parnames (list): list of parameter names to update
            parvals (list): list of parameter values to update

        Returns:
            None (plots the results)
    """
    model = ramsey_thje.RamseyModelClass(do_print=False)
    par = model.par
    ss = model.ss
    path = model.path

    if update_par:
        par = updatepar(par, parnames, parvals)

    model.find_steady_state(KY_ss=4.0, do_print=False)

    # a. set initial value
    par.K_lag_ini = ss.K

    # b. set path
    path.Gamma[:] = ss.Gamma
    path.tauH[:] = ss.tauH
    path.C[:] = ss.C
    path.K[:] = ss.K
    path.q[:] = ss.q
    path.G[:] = ss.G

    # c. check errors
    errors_ss = model.evaluate_path_errors()
    assert np.allclose(errors_ss,0.0)

    model.calculate_jacobian()

    par.K_lag_ini = ss.K # start from steady state
    path.Gamma[:] = ss.Gamma 
    path.tauH[:] = ss.tauH
    path.Gamma[:] = 0.95**np.arange(par.Tpath)*0.1*ss.Gamma + ss.Gamma # shock path
    model.solve(do_print = False) # find transition path

    x_max = 200
    fig = plt.figure(figsize=(3*6,12/1.5))
    fig.suptitle('Capital, housing prices and productivity over time', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(2,2,1)
    ax.set_title('Capital, $K_{t-1}$')
    ax.plot(path.K_lag)
    ax.set_xlim([0,x_max])

    ax = fig.add_subplot(2,2,2)
    ax.plot(path.q)
    ax.set_title('Housing price, $q_{t}$')
    ax.set_xlim([0,x_max])

    ax = fig.add_subplot(2,2,3)
    ax.plot(path.q)
    ax.set_title('Consumption, $c_{t}$')
    ax.set_xlim([0,x_max])

    ax = fig.add_subplot(2,2,4)
    ax.plot(path.Gamma)
    ax.set_title('Productivity, $\Gamma_{t}$')
    ax.set_xlim([0,x_max])

    fig.tight_layout()

    plt.show()

def shock_2(update_par = False, parnames=None, parvals=None):
    """ 
        simulate a transitory anticipated shock to tax rate on housing

        Args:
            update_par (bool): if True, update parameters
            parnames (list): list of parameter names to update
            parvals (list): list of parameter values to update

        Returns:
            None (plots the results)
    """
    model = ramsey_thje.RamseyModelClass(do_print=False)
    par = model.par
    ss = model.ss
    path = model.path

    if update_par:
        par = updatepar(par, parnames, parvals)

    model.find_steady_state(KY_ss=4.0, do_print=False)

    # a. set initial value
    par.K_lag_ini = ss.K

    # b. set path
    path.Gamma[:] = ss.Gamma
    path.tauH[:] = ss.tauH
    path.C[:] = ss.C
    path.K[:] = ss.K
    path.q[:] = ss.q
    path.G[:] = ss.G

    # c. check errors
    errors_ss = model.evaluate_path_errors()
    assert np.allclose(errors_ss,0.0)

    model.calculate_jacobian()

    tax_start = 20

    par.K_lag_ini = ss.K # start from steady state
    path.Gamma[:] = ss.Gamma 
    path.tauH[:] = ss.tauH # reset path
    path.tauH[20:] = 0.95**np.arange(par.Tpath - tax_start)*0.1*ss.tauH + ss.tauH # shock path
    model.solve(do_print = False) # find transition path

    x_max = 200
    fig = plt.figure(figsize=(3*6,12/1.5))
    fig.suptitle('Capital, housing prices, consumption and housing tax over time', fontsize=14, fontweight='bold')
    ax = fig.add_subplot(2,2,1)
    ax.set_title('Capital, $K_{t-1}$')
    ax.plot(path.K_lag)

    ax = fig.add_subplot(2,2,2)
    ax.plot(path.q)
    ax.set_title('Housing price, $q_{t}$')

    ax = fig.add_subplot(2,2,3)
    ax.plot(path.C)
    ax.set_title('Consumption, $C_{t}$')

    ax = fig.add_subplot(2,2,4)
    # ax.plot(path.tauH)
    ax.plot(path.tauH)
    ax.set_title('Housing tax, $tau_{t}$')

    fig.tight_layout()

    fig.tight_layout()

    plt.show()