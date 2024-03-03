from types import SimpleNamespace

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

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
        return x1A**self.par.alpha * x2A**(1 - self.par.beta)

    def utility_B(self,x1B,x2B):
        '''
        Implementing the utility function of agent B
        Input: x1B and x2B being the consumption of good 1 and 2 for agent B
        Output: Utility of agent B
        '''
        return x1B**self.par.alpha * x2B**(1 - self.par.beta)

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