from types import SimpleNamespace
from scipy import optimize
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        ''' 
        Implementing the utility function of agent A
        Input: x1A and x2A being the consumption of good 1 and 2 for agent A
        Output: Utility of agent A
        '''
        return x1A**self.par.alpha * x2A**(1 - self.par.alpha)

    def utility_B(self,x1B,x2B):
        '''
        Implementing the utility function of agent B
        Input: x1B and x2B being the consumption of good 1 and 2 for agent B
        Output: Utility of agent B
        '''
        return x1B**self.par.beta * x2B**(1 - self.par.beta)

    def demand_A(self,p1):
        '''
        Implementing the demand function of agent A obtained from solving the consumer problem
        Input: p1 being the price of good 1 and the price of good 2 is used as numeraire
        Output: x1A and x2A being the optimal consumption of good 1 and 2 for agent A given prices p1 and p2 = 1
        '''
        return self.par.alpha*(p1*self.par.w1A+self.par.w2A)/p1, (1 - self.par.alpha)*(p1*self.par.w1A+self.par.w2A)
    
    def demand_B(self,p1):
        '''
        Implementing the demand function of agent B obtained from solving the consumer problem
        Input: p1 being the price of good 1 and the price of good 2 is used as numeraire
        Output: x1A and x2A being the optimal consumption of good 1 and 2 for agent B given prices p1 and p2 = 1
        '''
        par = self.par
        x1B = par.beta*(p1*(1 - par.w1A)+(1 - par.w2A))/p1
        x2B = (1 - par.beta)*(p1*(1 - par.w1A)+ 1 - par.w2A) 
        return x1B, x2B
    
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    def pareto_improve(self, x1A, x2A):
        '''
        This function checks if the allocation is pareto improving and then adds the allocation to a list of pareto improvements
        Input: x1A and x2A being the consumption of good 1 and 2 for agent A
        Output: List of pareto improvements
        '''

        par = self.par
        pareto_improvements = []
        init_utilityA = self.utility_A(par.w1A, par.w2A)
        init_utilityB = self.utility_B(1 - par.w1A, 1 - par.w2A)
        for i, c in enumerate(x1A):
            for j, d in enumerate(x2A):
                if self.utility_A(c,d) > init_utilityA and self.utility_B(1-c,1-d) > init_utilityB:
                    pareto_improvements.append((c,d))
        return pareto_improvements
    
    def solve_A_cont(self):
        '''
        Solves the market maker problem for agent A.

        Returns:
        - p1opt (float): Optimal price of good 1 for agent A.
        - utilityA_opt (float): Optimal utility of agent A.
        '''

        # par = self.par
        sol = self.sol    
        
        # a. objective function (to minimize) 
        obj = lambda p1: -self.utility_A(1 - self.demand_B(p1)[0],1 - self.demand_B(p1)[1]) # minimize -> negative of utility
            
        # b. constraints and bounds
        bounds = ((1e-8,None),)
                
        # c. call solver
        p10 = 1.6
        result = optimize.minimize(obj,p10,method='SLSQP',bounds=bounds)
            
        # d. save
        p1opt = result.x[0]
        utilityA_opt = self.utility_A(1 - self.demand_B(p1opt)[0],1 - self.demand_B(p1opt)[1])
        print(f'Optimal price for agent A is: {p1opt:.4f}')
        print(f'The consumption for A is: x1A = {1 - self.demand_B(p1opt)[0]:.4f}, x2A = {1 - self.demand_B(p1opt)[1]:.4f} with utility u(x1A,x2A) = {utilityA_opt:.4f}')
        print(f'The consumption for B is: x1B = {self.demand_B(p1opt)[0]:.4f}, x2B = {self.demand_B(p1opt)[1]:.4f} with utility u(x1B,x2B) = {self.utility_B(self.demand_B(p1opt)[0],self.demand_B(p1opt)[1]):.4f}')
    
        # Return the results to a sol dict
        sol.x1A_opt_cont = 1 - self.demand_B(p1opt)[0]
        sol.x2A_opt_cont = 1 - self.demand_B(p1opt)[1]

    def solve_A_disc(self, p1):

        sol = self.sol
        x1B, x2B = self.demand_B(p1)

        # Applying a boolean mask to ensure that the values of x1B and x2B are within the unit interval
        x1B[x1B > 1] = np.nan
        x2B[x2B > 1] = np.nan

        # Calculate the utility for agent A given agent B's demand
        utilityA = self.utility_A(1 - x1B,1 - x2B)

        # Finding the maximum utility and the corresponding index (Using nanmax and nanargmax to handle NaN values)
        utilityA_max = np.nanmax(utilityA)
        index = np.nanargmax(utilityA)

        # Finding the optimal price for agent A to choose
        p1_opt_A = p1[index]
        print(f'Optimal price on discrete grid for agent A: {p1_opt_A:.4f}')
        print(f'The consumption for A is: x1A  = {1-x1B[index]:.4f}, x2A = {1-x2B[index]:.4f} with utility u(x1A,x2A) =  {utilityA_max:.4f}')
        print(f'The consumption for B is: x1B  = {x1B[index]:.4f}, x2B = {x2B[index]:.4f} with utility u(x1B,x2B) = {self.utility_B(x1B[index],x2B[index]):.4f}')

        sol.x1A_opt_disc = 1-x1B[index]
        sol.x2A_opt_disc = 1-x2B[index]

    def solve_A_pareto(self):
        '''
        Solves the market maker problem for A by maximizing utility subject to a pareto improvement constraint for B.

        Returns:
        - p1opt (float): Optimal price of good 1 for agent A.
        - utilityA_opt (float): Optimal utility of agent A.
        '''
        par = self.par

        # a. objective function (to minimize) 
        obj = lambda xA: -self.utility_A(xA[0],xA[1]) # minimize -> negative of utility
            
        # b. constraints and bounds
        budget_constraint = lambda xA: self.utility_B(1 - xA[0],1 - xA[1]) - self.utility_B(1 - par.w1A, 1 - par.w2A) # violated if negative
        constraints = ({'type':'ineq','fun':budget_constraint})
        bounds = ((1e-8,1),(1e-8,1))
                
        # c. call solver
        x0 = [0.2,0.8]
        result = optimize.minimize(obj,x0,method='SLSQP',constraints=constraints, bounds=bounds)
            
        # d. save
        x1Aopt, x2Aopt = result.x
        
        print(f'The consumption for A is: x1A  = {x1Aopt:.4f}, x2A = {x2Aopt:.4f}')
        print(f'The consumption for B is: x1B  = {1 - x1Aopt:.4f}, x2B = {1 - x2Aopt:.4f}')

        return x1Aopt, x2Aopt

    def solve_social_planner(self):
        '''
        Solves the social planner problem by maximizing the sum of utilities of agent A and B.

        Returns:
        - p1opt (float): Optimal price of good 1 for agent A.
        - utilityA_opt (float): Optimal utility of agent A.
        '''

        # par = self.par
        # sol = model.sol    
        
        # a. objective function (to minimize) 
        obj = lambda xA: -(self.utility_A(xA[0],xA[1]) + self.utility_B(1 - xA[0],1 - xA[1])) # minimize -> negative of utility
            
        # b. constraints and bounds
        # budget_constraint = lambda x: par.m-par.p1*x[0]-par.p2*x[1] # violated if negative
        # constraints = ({'type':'ineq','fun':budget_constraint})
        bounds = ((1e-8,1),(1e-8,1))
                
        # c. call solver
        x0 = [0.75,0.8]
        # print(x0)
        result = optimize.minimize(obj,x0,method='SLSQP',bounds=bounds)
            
        # d. save
        x1Aopt, x2Aopt = result.x
        # utilityA = self.utility_A(x1Aopt,x2Aopt)
        return x1Aopt, x2Aopt
    
    def solve_market_equilibrium(self, w1A):
        p1_eq = np.zeros(len(w1A)) # empty array to store equilibrium prices
        for i in range(len(w1A)):
            obj = lambda p1: self.demand_A(p1)[0]-w1A[i] + self.demand_B(p1)[0]-(1-w1A[i])
            res = optimize.root_scalar(obj,bracket=(1e-8,10),method='bisect')
            p1_eq[i] = res.root
        x1A_eq, x2A_eq = self.demand_A(p1_eq)
        X1B_eq, X2B_eq = self.demand_B(p1_eq)
        return x1A_eq, x2A_eq, X1B_eq, X2B_eq
    
    def random_endowments(self, n_rand, seed = 1):
        '''
        QUESTION 8: Function to generate random endowments for agent A and B and the solve for the market equilibrium
        '''
        np.random.seed(seed)
        par = self.par
        x1A_Q8, x2A_Q8 = np.zeros(n_rand), np.zeros(n_rand)
        for n in range(n_rand):
            rand_omega = np.random.uniform(0,1, size = 2)
            par.w1A, par.w2A = rand_omega
            eq_price = self.solve(do_print = False)
            x1A_Q8[n], x2A_Q8[n] = self.demand_A(eq_price)

        return x1A_Q8, x2A_Q8


    def solve(self, do_print = True):
        '''
        QUESTION 3: Solves for the market equilibrium price and prints the error in equilibrium.
        '''
        obj = lambda p1: self.check_market_clearing(p1)[0] # here the input is a scalar
        res = optimize.root_scalar(obj,bracket=(1e-8,10),method='bisect')
        x = res.root
        error_equilibrium = self.check_market_clearing(res.root)
        if do_print:
            print(f'Error in equilibrium {error_equilibrium} with equilbrium price p_1 = {x:.6f}')
        return x
    
    def plot_all_sol(self, x1A_Q3, x2A_Q3, x1A_c, x2A_c, x1A_Q5B, x2A_Q5B, x1A_sp, x2A_sp, w1bar, w2bar):
        '''QUESTION 6B: Plotting the solutions for all the questions in the project so far'''
        par = self.par
        sol = self.sol
        # a. figure set up
        fig = plt.figure(frameon=True,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # b. scatter plot for equilibrium allocations
        ax_A.scatter(x1A_Q3, x2A_Q3,marker='x',color='red',label=f'Question 3: Walras equilibrium, $U^A = {self.utility_A(x1A_Q3,x2A_Q3):.4f}$, $U^B = {self.utility_B(1-x1A_Q3,1-x2A_Q3):.4f}$')
        ax_A.scatter(self.sol.x1A_opt_disc, self.sol.x2A_opt_disc,marker='x',label=f'Question 4a: Agent A discrete price, $U^A = {self.utility_A(self.sol.x1A_opt_disc,self.sol.x2A_opt_disc):.4f}$, $U^B = {self.utility_B(1-self.sol.x1A_opt_disc,1-self.sol.x2A_opt_disc):.4f}$')
        ax_A.scatter(self.sol.x1A_opt_cont, self.sol.x2A_opt_cont,marker='^',label=f'Question 4a: Agent A continuous price, $U^A = {self.utility_A(self.sol.x1A_opt_cont,self.sol.x2A_opt_cont):.4f}$, $U^B = {self.utility_B(1-self.sol.x1A_opt_cont,1-self.sol.x2A_opt_cont):.4f}$')
        ax_A.scatter(x1A_c, x2A_c,marker='o',color='blue',label=f'Question 5a: Agent A market marker, $U^A = {self.utility_A(x1A_c,x2A_c):.4f}$, $U^B = {self.utility_B(1-x1A_c,1-x2A_c):.4f}$')
        ax_A.scatter(x1A_Q5B, x2A_Q5B,marker='o',color='orange',label=f'Question 5B: Agent A market marker, $U^A = {self.utility_A(x1A_Q5B,x2A_Q5B):.4f}$, $U^B = {self.utility_B(1-x1A_Q5B,1-x2A_Q5B):.4f}$')
        ax_A.scatter(x1A_sp, x2A_sp,marker='D',color='green',label=f'Question 6: Social planner, $U^A = {self.utility_A(x1A_sp,x2A_sp):.4f}$, $U^B = {self.utility_B(1-x1A_sp,1-x2A_sp):.4f}$')
        ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label=f'Endowment, $U^A = {self.utility_A(par.w1A,par.w2A):.4f}$, $U^B = {self.utility_B(1-par.w1A,1-par.w2A):.4f}$')

        # c. limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='lower left', fontsize=10, bbox_to_anchor=(1.2, 0.65));
        ax_A.set_title("Question 6B: Allocations")


    def plot_Q8(self, x1A_Q8, x2A_Q8, w1bar, w2bar, WA1, WA2):
        # a. figure set up
        fig = plt.figure(frameon=True,figsize=(6,6), dpi=100)
        ax_A = fig.add_subplot(1, 1, 1)

        ax_A.set_xlabel("$x_1^A$")
        ax_A.set_ylabel("$x_2^A$")

        temp = ax_A.twinx()
        temp.set_ylabel("$x_2^B$")
        ax_B = temp.twiny()
        ax_B.set_xlabel("$x_1^B$")
        ax_B.invert_xaxis()
        ax_B.invert_yaxis()

        # b. scatter plot for equilibrium allocations
        # ax_A.scatter(x1A_eq, x2A_eq,marker='o',color='blue',label='$(x_1^A,x_1^B)$, $(x_2^A,x_2^B)$')
        # ax_A.scatter(par.w1A,par.w2A,marker='s',color='black',label='original endowment')
        ax_A.scatter(WA1,WA2,marker='o',color='blue',label='random endowment')
        ax_A.scatter(x1A_Q8,x2A_Q8,marker='o',color='orange',label='Market equilibrium')


        # ax_A.scatter(X1B_eq, X2B_eq,marker='o',color='red',label='$x_1^B$, $x_2^B$')

        # c. limits
        ax_A.plot([0,w1bar],[0,0],lw=2,color='black')
        ax_A.plot([0,w1bar],[w2bar,w2bar],lw=2,color='black')
        ax_A.plot([0,0],[0,w2bar],lw=2,color='black')
        ax_A.plot([w1bar,w1bar],[0,w2bar],lw=2,color='black')

        ax_A.set_xlim([-0.1, w1bar + 0.1])
        ax_A.set_ylim([-0.1, w2bar + 0.1])    
        ax_B.set_xlim([w1bar + 0.1, -0.1])
        ax_B.set_ylim([w2bar + 0.1, -0.1])

        ax_A.legend(frameon=True,loc='lower right', fontsize=10);