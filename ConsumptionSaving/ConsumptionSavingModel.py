import time
import numpy as np
from scipy import optimize
from statsmodels.distributions.empirical_distribution import ECDF
from numba import njit, prange, double, int64, boolean

# plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# consav
from consav.misc import nonlinspace
from consav.misc import normal_gauss_hermite
from consav import linear_interp
from consav import ModelClass

##########################
# 1. pure Python version #
##########################

class ConsumptionSavingModelClass():
    
    #########
    # setup #
    #########

    def __init__(self,name='baseline',solmethod='EGM',**kwargs):
        """ setup model sub-classes and parameters in .par """

        # a. set baseline parameters
        self.name = name
        self.solmethod = solmethod

        # parameters and grids
        class ParClass: None
        self.par = ParClass()

        # solution
        class SolClass: None
        self.sol = SolClass()

        # simulation
        class SimClass: None
        self.sim = SimClass()

        self.setup()

        # b. update parameters
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val        

    def setup(self):
        """ baseline parameters in .par """

        par = self.par

        # a. demographics
        par.T = 200
        par.TR = par.T # retirement age (end-of-period), no retirement if TR = T
        par.age_min = 25 # only relevant for figures

        # b. preferences
        par.rho = 2
        par.beta = 0.96

        # c. income parameters

        # growth
        par.G = 1.02

        # standard deviations
        par.sigma_xi = 0.1
        par.sigma_psi = 0.1

        # low income shock
        par.low_p = 0.005 # called pi in slides
        par.low_val = 0.0 # called mu in slides

        # life-cycle
        par.L = np.ones(par.T) # if ones then no life-cycle           

        # d. saving and borrowing
        par.R = 1.04
        par.borrowingfac = 0.0

        # e. numerical integration and grids         
        par.a_max = 20.0 # maximum point i grid for a
        par.a_phi = 1.1 # curvature parameters
        par.m_max = 20.0 # maximum point i grid for m
        par.m_phi = 1.1 # curvature parameters

        # number of elements
        par.Nxi  = 8 # number of quadrature points for xi
        par.Npsi = 8 # number of quadrature points for psi
        par.Na = 500 # number of points in grid for a
        par.Nm = 100 # number of points in grid for m

        # f. simulation
        par.sim_mini = 2.5 # initial m in simulation
        par.simN = 100_000 # number of persons in simulation
        par.simT = 100 # number of periods in simulation
        par.simlifecycle = 0 # = 0 simulate infinite horizon model

    def create_grids(self):
        """ create grids and other preperations for solving the model"""

        par = self.par

        # a. perfect foresight or buffer-stock model
        if par.sigma_xi == 0 and par.sigma_psi == 0 and par.low_p == 0: # no risk
            self.model = 'pf' # perfect foresight
        else:
            self.model = 'bs' # buffer-stock

        # b. shocks

        # i. basic GuassHermite
        psi, psi_w = normal_gauss_hermite(sigma=par.sigma_psi,n=par.Npsi)
        xi, xi_w = normal_gauss_hermite(sigma=par.sigma_xi,n=par.Nxi)

        # ii. add low income shock to xi
        if par.low_p > 0:
            
            # a. weights
            xi_w *= (1.0-par.low_p)
            xi_w = np.insert(xi_w,0,par.low_p)

            # b. values
            xi = (xi-par.low_val*par.low_p)/(1.0-par.low_p)
            xi = np.insert(xi,0,par.low_val)

        # iii. vectorize tensor product of shocks and total weight
        psi_vec,xi_vec = np.meshgrid(psi,xi,indexing='ij')
        psi_w_vec,xi_w_vec = np.meshgrid(psi_w,xi_w,indexing='ij')
        
        par.psi_vec = psi_vec.ravel()
        par.xi_vec = xi_vec.ravel()
        par.w = xi_w_vec.ravel()*psi_w_vec.ravel()

        assert 1-np.sum(par.w) < 1e-8 # == summing to 1

        # iv. count number of shock nodes
        par.Nshocks = par.w.size

        # c. minimum a
        if par.borrowingfac == 0:

            par.a_min = np.zeros(par.T) # never any borriwng

        else:

            # using formula from slides 
            psi_min = np.min(par.psi_vec)
            xi_min = np.min(par.xi_vec)
            
            par.a_min = np.nan*np.ones(par.T)
            for t in reversed(range(par.T-1)):
                
                if t >= par.TR-1: # in retirement
                    Omega = 0
                elif t == par.TR-2: # next period is retirement
                    Omega = par.R**(-1)*par.G*par.L[t+1]*psi_min*xi_min
                else: # before retirement
                    Omega = par.R**(-1)*(np.fmin(Omega,par.borrowingfac)+xi_min)*par.G*par.L[t+1]*psi_min

                par.a_min[t] = -np.fmin(Omega,par.borrowingfac)*par.G*par.L[t+1]*psi_min
            
        # d. end-of-period assets and cash-on-hand
        par.grid_a = np.nan*np.ones((par.T,par.Na))
        par.grid_m = np.nan*np.ones((par.T,par.Nm))
        for t in range(par.T):
            par.grid_a[t,:] = nonlinspace(par.a_min[t]+1e-6,par.a_max,par.Na,par.a_phi)
            par.grid_m[t,:] = nonlinspace(par.a_min[t]+1e-6,par.m_max,par.Nm,par.m_phi)        

        # e. conditions
        par.FHW = par.G/par.R
        par.AI = (par.R*par.beta)**(1/par.rho)
        par.GI = par.AI*np.sum(par.w*par.psi_vec**(-1))/par.G
        par.RI = par.AI/par.R        
        par.WRI = par.low_p**(1/par.rho)*par.AI/par.R
        par.FVA = par.beta*np.sum(par.w*(par.G*par.psi_vec)**(1-par.rho))

        # f. fast solution with EGM
        
        # grid_a tiled with the number of shocks
        par.grid_a_tile = np.ones((par.TR,par.Na*par.Nshocks))
        for t in range(par.TR):
            par.grid_a_tile[t,:] = np.tile(par.grid_a[t,:],par.Nshocks)

        # xi, psi and w repeated with the number of grid points for a
        par.xi_vec_rep = np.repeat(par.xi_vec,par.Na)
        par.psi_vec_rep = np.repeat(par.psi_vec,par.Na)
        par.w_rep = np.repeat(par.w,par.Na)

        # g. check for existance of solution
        self.print_and_check_parameters(do_print=False)

    def print_and_check_parameters(self,do_print=True):
        """ print and check parameters """

        par = self.par

        if do_print:
            print(f'FHW = {par.FHW:.3f}, AI = {par.AI:.3f}, GI = {par.GI:.3f}, RI = {par.RI:.3f}, WRI = {par.WRI:.3f}, FVA = {par.FVA:.3f}')

        # check for existance of solution
        if self.model == 'pf' and par.GI >= 1 and par.RI >= 1:
            raise Exception('GI >= 1 and RI >= 1: no solution')

        if self.model == 'bs' and (par.FVA >= 1 or par.WRI >= 1):
            raise Exception('FVA >= 1 or WRI >= 1: no solution')

    def utility(self,c):
        """ utility function """

        return c**(1-self.par.rho)/(1-self.par.rho)
    
    def marg_utility(self,c):
        """ marginal utility function """

        return c**(-self.par.rho)            
    
    def inv_marg_utility(self,u):
        """ inverse marginal utility funciton """

        return u**(-1/self.par.rho)            
                            
    #########
    # solve #
    #########
    
    def solve(self,do_print=True):
        """ gateway for solving the model """

        # a. create (or re-create) grids
        self.create_grids()

        # b. solve
        if self.solmethod in ['EGM','EGMvec']:
            self.solve_EGM(do_print=do_print)
        elif self.solmethod == 'VFI':
            self.solve_VFI(do_print=do_print)
        else:
            raise Exception(f'{self.solmethod} is an unknown solution method')

    def solve_EGM(self,do_print):
        """ solve model using EGM """

        t0 = time.time()
        par = self.par
        sol = self.sol
        
        # a. allocate
        sol.m = np.zeros((par.T,par.Na+1))
        sol.c = np.zeros((par.T,par.Na+1))
        sol.inv_v = np.zeros((par.T,par.Na+1))

        # working memory
        m = np.zeros(par.Na)
        c = np.zeros(par.Na)
        inv_v = np.zeros(par.Na)

        # b. last period (= consume all)
        sol.m[-1,:] = np.linspace(0,par.a_max,par.Na+1)
        sol.c[-1,:] = sol.m[-1,:]
        sol.inv_v[-1,0] = 0
        sol.inv_v[-1,1:] = 1.0/self.utility(sol.c[-1,1:])

        # c. before last period
        for t in reversed(range(par.T-1)):
            
            # i. solve by EGM
            if self.solmethod == 'EGM':
                self.EGM(t,m,c,inv_v)    
            elif self.solmethod == 'EGMvec':
                self.EGMvec(t,m,c,inv_v)    

            # ii. add zero consumption
            sol.m[t,0] = par.a_min[t]
            sol.m[t,1:] = m
            sol.c[t,0] = 0
            sol.c[t,1:] = c
            sol.inv_v[t,0] = 0
            sol.inv_v[t,1:] = inv_v

        if do_print:
            print(f'model solved in {time.time()-t0:.1f} secs')

    def EGM(self,t,m,c,inv_v):
        """ EGM with partly vectorized code """

        par = self.par
        sol = self.sol

        # loop over end-of-period assets
        for i_a in range(par.Na):

            # a. prep
            a = par.grid_a[t,i_a]
            if t+1 <= par.TR-1: # still working in next-period
                fac = par.G*par.L[t]*par.psi_vec
                w = par.w
                xi = par.xi_vec
            else:
                fac = par.G*par.L[t]
                w = 1
                xi = 1
            
            inv_fac = 1.0/fac

            # b. future m and c (vectors)
            m_plus = inv_fac*par.R*a + xi

            c_plus = np.zeros(m_plus.size)
            linear_interp.interp_1d_vec(sol.m[t+1,:],sol.c[t+1,:],m_plus,c_plus)

            inv_v_plus = np.zeros(m_plus.size)
            linear_interp.interp_1d_vec(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus,inv_v_plus)
            v_plus = 1.0/inv_v_plus

            # c. average future marginal utility (number)
            marg_u_plus = self.marg_utility(fac*c_plus)
            avg_marg_u_plus = np.sum(w*marg_u_plus)
            avg_v_plus = np.sum(w*(fac**(1-par.rho))*v_plus)

            # d. current c
            c[i_a] = self.inv_marg_utility(par.beta*par.R*avg_marg_u_plus)

            # e. current m
            m[i_a] = a + c[i_a]

            # f. current v
            if c[i_a] > 0:
                inv_v[i_a] = 1.0/(self.utility(c[i_a]) + par.beta*avg_v_plus)
            else:
                inv_v[i_a] = 0

    def EGMvec(self,t,m,c,inv_v):
        """ EGM with fully vectorized code """

        par = self.par
        sol = self.sol

        # a. prep
        if t+1 <= par.TR-1: # still working in next-period
            a = par.grid_a_tile[t,:]
            fac = par.G*par.L[t]*par.psi_vec_rep
            w = par.w_rep
            xi = par.xi_vec_rep
            Nshocks = par.Nshocks
        else:
            a = par.grid_a
            fac = par.G*par.L[t]
            w = 1
            xi = 1
            Nshocks = par.Nshocks

        inv_fac = 1.0/fac

        # b. future m and c
        m_plus = inv_fac*par.R*a + xi
        
        c_plus = np.zeros(m_plus.size)
        linear_interp.interp_1d_vec(sol.m[t+1,:],sol.c[t+1,:],m_plus,c_plus)
        
        inv_v_plus = np.zeros(m_plus.size)
        linear_interp.interp_1d_vec(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus,inv_v_plus)
        v_plus = 1.0/inv_v_plus

        # c. average future marginal utility
        marg_u_plus = self.marg_utility(fac*c_plus)
        avg_marg_u_plus = np.sum( (w*marg_u_plus).reshape((Nshocks,par.Na) ),axis=0)
        avg_v_plus = np.sum( (w*(fac**(1-par.rho))*v_plus).reshape((Nshocks,par.Na) ),axis=0)

        # d. current c
        c[:] = self.inv_marg_utility(par.beta*par.R*avg_marg_u_plus)

        # e. current m
        m[:] = par.grid_a[t,:] + c

        # f. current v
        I = c > 0
        inv_v[I] = 1.0/(self.utility(c[I]) + par.beta*avg_v_plus[I])
        inv_v[~I] = 0.0

    def solve_VFI(self,do_print):
        """ solve model with VFI """

        t0 = time.time()

        par = self.par
        sol = self.sol

        # a. allocate solution
        sol.m = np.nan*np.ones((par.T,par.Nm))
        sol.c = np.nan*np.ones((par.T,par.Nm))
        sol.inv_v = np.nan*np.ones((par.T,par.Nm))

        # b. last period (= consume all)
        sol.m[-1,:] = par.grid_m[-1,:]
        sol.c[-1,:] = sol.m[-1,:]
        sol.inv_v[-1,:] = 1.0/self.utility(sol.c[-1,:])

        # c. before last period
        for t in reversed(range(par.T-1)):
            for i_m in range(par.Nm):

                m = par.grid_m[t,i_m]                
                result = optimize.minimize_scalar(
                    lambda c: self.value_of_choice(c,t,m),method='bounded',
                    bounds=(0,m))

                sol.c[t,i_m] = result.x
                sol.inv_v[t,i_m]= -1/result.fun
            
            # save grid for m
            sol.m[t,:] = par.grid_m[t,:]

        if do_print:
            print(f'model solved in {time.time()-t0:.1f} secs')

    def value_of_choice(self,c,t,m):
        """ value of choice of c used in VFI """

        par = self.par
        sol = self.sol

        # a. end-of-period assets
        a = m-c

        # b. next-period cash-on-hand
        if t+1 <= par.TR-1: # still working in next-period
            fac = par.G*par.L[t]*par.psi_vec
            w = par.w
            xi = par.xi_vec
        else:
            fac = par.G*par.L[t]
            w = 1
            xi = 1

        m_plus = (par.R/fac)*a + xi            

        # c. continuation value
        inv_v_plus = np.zeros(m_plus.size)
        linear_interp.interp_1d_vec(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus,inv_v_plus)
        v_plus = 1/inv_v_plus
        
        # d. value-of-choice
        total = self.utility(c) + par.beta*np.sum(w*fac**(1-par.rho)*v_plus)
        return -total

    ############
    # simulate #
    ############
    
    def simulate(self,seed=2017):
        """ simulate the model """

        np.random.seed(seed)

        par = self.par
        sim = self.sim

        t0 = time.time()

        # a. allocate
        sim.m = np.nan*np.zeros((par.simN,par.simT))
        sim.c = np.nan*np.zeros((par.simN,par.simT))
        sim.a = np.nan*np.zeros((par.simN,par.simT))
        sim.p = np.nan*np.zeros((par.simN,par.simT))
        sim.y = np.nan*np.zeros((par.simN,par.simT))

        # b. shocks
        _shocki = np.random.choice(par.Nshocks,size=(par.simN,par.simT),p=par.w)
        sim.psi = par.psi_vec[_shocki]
        sim.xi = par.xi_vec[_shocki]

        # c. initial values
        sim.m[:,0] = par.sim_mini 
        sim.p[:,0] = 0.0 

        # d. simulation
        self.simulate_timeloop()

        # e. renomarlized
        sim.P = np.exp(sim.p)
        sim.Y = np.exp(sim.y)
        sim.M = sim.m*sim.P
        sim.C = sim.c*sim.P
        sim.A = sim.a*sim.P

        print(f'model simulated in {time.time()-t0:.1f} secs')

    def simulate_timeloop(self):
        """ simulate model with loop over time """

        par = self.par
        sol = self.sol
        sim = self.sim

        # loop over time
        for t in range(par.simT):
            
            # a. solution
            if par.simlifecycle == 0:
                grid_m = sol.m[0,:]
                grid_c = sol.c[0,:]
            else:
                grid_m = sol.m[t,:]
                grid_c = sol.c[t,:]
            
            # b. consumption
            linear_interp.interp_1d_vec(grid_m,grid_c,sim.m[:,t],sim.c[:,t])
            sim.a[:,t] = sim.m[:,t] - sim.c[:,t]

            # c. next-period states
            if t < par.simT-1:

                if t+1 > par.TR-1:
                    sim.m[:,t+1] = par.R*sim.a[:,t] / (par.G*par.L[t]) +  1
                    sim.p[:,t+1] = np.log(par.G) + np.log(par.L[t]) + sim.p[:,t]
                    sim.y[:,t+1] = sim.p[:,t+1]
                else:
                    sim.m[:,t+1] = par.R*sim.a[:,t] / (par.G*par.L[t]*sim.psi[:,t+1]) + sim.xi[:,t+1]
                    sim.p[:,t+1] = np.log(par.G) + np.log(par.L[t]) + sim.p[:,t] + np.log(sim.psi[:,t+1])   
                    I = sim.xi[:,t+1] > 0
                    sim.y[I,t+1] = sim.p[I,t+1] + np.log(sim.xi[I,t+1])

    ##################
    # solution plots #
    ##################

    def plot_value_function_convergence(self):

        par = self.par
        sol = self.sol

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        for t in [par.T-1, par.T-2, par.T-6, par.T-11, 100, 50, 0]:
            if t > par.T-1 or t < 0: continue
            ax.plot(sol.m[t,:],-sol.inv_v[t,:],label=f'$n = {par.T-t}$')

        # limits
        ax.set_xlim([np.min(par.a_min), 5])
        ax.set_ylim([0, 1])

        # layout
        bbox = {'boxstyle':'square','ec':'white','fc':'white'}
        ax.text(1.5,0.4,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('$-1.0/v_t(m_t)$')
        ax.legend(loc='upper right',frameon=True)

        fig.savefig(f'figs/cons_converge_{self.name}.pdf')

    def plot_consumption_function_convergence(self):

        par = self.par
        sol = self.sol

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        for t in [par.T-1, par.T-2, par.T-6, par.T-11, 100, 50, 0]:
            if t > par.T-1 or t < 0: continue
            ax.plot(sol.m[t,:],sol.c[t,:],label=f'$n = {par.T-t}$')

        # limits
        ax.set_xlim([np.min(par.a_min), 5])
        ax.set_ylim([0, 5])

        # layout
        bbox = {'boxstyle':'square','ec':'white','fc':'white'}
        ax.text(1.5,0.5,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('$-1.0/v_t(m_t)$')
        ax.legend(loc='upper left',frameon=True)

        fig.savefig(f'figs/val_converge_{self.name}.pdf')

    def plot_consumption_function_convergence_age(self):

        par = self.par
        sol = self.sol

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        # consumption function for various ages
        for age in [25, 35, 45, 55, 65, 75, par.T+par.age_min-2, par.T+par.age_min-1]:
            ax.plot(sol.m[age-par.age_min],sol.c[age-par.age_min],label=f'age = {age}')

        # limits
        ax.set_xlim([min(par.a_min), 5])
        ax.set_ylim([0, 5])

        # layout
        bbox = {'boxstyle':'square','ec':'white','fc':'white'}
        ax.text(1.5,0.5,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('$c(m_t)$')
        ax.legend(loc='upper left',frameon=True)

        fig.savefig(f'figs/cons_converge_{self.name}.pdf')

    def plot_consumption_function_pf(self):

        par = self.par
        sol = self.sol

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        # perfect foresight consumption
        c_pf = (1-par.RI)*(sol.m[0,:]+(1-par.FHW)**(-1)-1)   

        # consumption function deviation from perfect foresight
        ax.plot(sol.m[0,:],sol.c[0,:]-c_pf,'-',lw=1.5)

        # limits
        ax.set_xlim([1, 500])
        ylim_now = ax.set_ylim()
        if np.max(np.abs(ylim_now)) < 1e-4:
            ax.set_ylim([-1,1])

        # layout
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('$c(m_t) - c^{PF}(m_t)$')

        fig.savefig(f'figs/cons_converge_pf_{self.name}.pdf')

    def plot_buffer_stock_target(self):

        par = self.par
        sol = self.sol

        # a. find a and avg. m_plus and c_plus
        
        # allocate
        a = np.nan*np.ones(par.Na+1)
        m_plus = np.nan*np.ones(par.Na+1)
        C_plus = np.nan*np.ones(par.Na+1)

        delta_log_C_plus = np.nan*np.ones(par.Na+1)
        delta_log_C_plus_approx_2 = np.nan*np.ones(par.Na+1)

        fac = 1.0/(par.G*par.psi_vec)
        for i_a in range(par.Na+1):

            # a. a and m
            a[i_a] = sol.m[0,i_a]-sol.c[0,i_a]            
            m_plus[i_a] = np.sum(par.w*(fac*par.R*a[i_a] + par.xi_vec))                

            # b. C_plus
            m_plus_vec = fac*par.R*a[i_a] + par.xi_vec            
            c_plus_vec = np.zeros(m_plus_vec.size)
            linear_interp.interp_1d_vec(sol.m[0,:],sol.c[0,:],m_plus_vec,c_plus_vec)
            C_plus_vec = par.G*par.psi_vec*c_plus_vec
            C_plus[i_a] = np.sum(par.w*C_plus_vec)

            # c. approx 
            if self.model == 'bs' and sol.c[0,i_a] > 0:

                delta_log_C_plus[i_a] = np.sum(par.w*(np.log(par.G*C_plus_vec)))-np.log(sol.c[0,i_a])
                var_C_plus = np.sum(par.w*(np.log(par.G*C_plus_vec) - np.log(sol.c[0,i_a]) - delta_log_C_plus[i_a])**2)
                delta_log_C_plus_approx_2[i_a] = par.rho**(-1)*(np.log(par.R*par.beta)) + 2/par.rho*var_C_plus + np.log(par.G)

        # b. find target
        i = np.argmin(np.abs(m_plus-sol.m[0,:]))
        m_target = sol.m[0,i]

        # c. figure 1 - buffer-stock target
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        # limits
        ax.set_xlim([np.min(par.a_min), 5])
        ax.set_ylim([0, 5])

        # layout
        bbox = {'boxstyle':'square','ec':'white','fc':'white'}
        ax.text(2.1,0.25,f'$\\beta = {par.beta:.2f}$, $R = {par.R:.2f}$, $G = {par.G:.2f}$',bbox=bbox)
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('')

        # i. consumption
        ax.plot(sol.m[0,:],sol.c[0,:],'-',lw=1.5,label='$c(m_t)$')  
        ax.legend(loc='upper left',frameon=True)  
        fig.savefig(f'figs/buffer_stock_target_{self.name}_c.pdf')

        # ii. perfect foresight solution
        if par.FHW < 1 and par.RI < 1:

            c_pf = (1-par.RI)*(sol.m[0,:]+(1-par.FHW)**(-1)-1)   
            ax.plot(sol.m[0,:],c_pf,':',lw=1.5,color='black',label='$c^{PF}(m_t)$')

            ax.legend(loc='upper left',frameon=True)
            fig.savefig(f'figs/buffer_stock_target_{self.name}_pf.pdf')

        # iii. a    
        ax.plot(sol.m[0,:],a,'-',lw=1.5,label=r'$a_t=m_t-c^{\star}(m_t)$')
        ax.legend(loc='upper left',frameon=True)
        fig.savefig(f'figs/buffer_stock_target_{self.name}_a.pdf')

        # iv. m_plus
        ax.plot(sol.m[0,:],m_plus,'-',lw=1.5,label='$E[m_{t+1} | a_t]$')
        ax.legend(loc='upper left',frameon=True)
        fig.savefig(f'figs/buffer_stock_target_{self.name}_m_plus.pdf')
        
        # v. 45
        ax.plot([0,5],[0,5],'-',lw=1.5,color='black',label='45 degree')
        ax.legend(loc='upper left',frameon=True)
        fig.savefig(f'figs/buffer_stock_target_{self.name}_45.pdf')

        # vi. target            
        if self.model == 'bs' and par.GI < 1:
            ax.plot([m_target,m_target],[0,5],'--',lw=1.5,color='black',label=f'target = {m_target:.2f}')

        ax.legend(loc='upper left',frameon=True)
        fig.savefig(f'figs/buffer_stock_target_{self.name}.pdf')

        # STOP
        if self.model == 'pf':
            return

        # d. figure 2 - C ratio
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        I = sol.c[0,:] > 0
        ax.plot(sol.m[0,I],(C_plus[I]/sol.c[0,I]),'-',lw=1.5,label='$E[C_{t+1}/C_t]$')
        ax.plot([m_target,m_target],[0,10],'--',lw=1.5,color='black',label='target')
        ax.plot([np.min(par.a_min),500],[par.G,par.G],':',lw=1.5,color='black',label='$G$')
        ax.plot([np.min(par.a_min),500],[(par.R*par.beta)**(1/par.rho),(par.R*par.beta)**(1/par.rho)],
            '-',lw=1.5,color='black',label=r'$(\beta R)^{1/\rho}$')

        # limit     
        ax.set_xlim([np.min(par.a_min),10])
        ax.set_ylim([0.95,1.1])

        # layout
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('$C_{t+1}/C_t$')
        ax.legend(loc='upper right',frameon=True)

        fig.savefig(f'figs/cons_growth_{self.name}.pdf')

        # e. figure 3 - euler approx
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(sol.m[0,:],delta_log_C_plus,'-',lw=1.5,label=r'$E[\Delta \log C_{t+1}]$')                

        ax.plot(sol.m[0,:],par.rho**(-1)*np.log(par.R*par.beta)*np.ones(par.Na+1)+np.log(par.G),'-',lw=1.5,label='1st order approx.')                
        ax.plot(sol.m[0,:],delta_log_C_plus_approx_2,'-',lw=1.5,label='2nd order approx.')                
        ax.plot([m_target,m_target],[-10 ,10],'--',lw=1.5,color='black',label='target')

        # limit     
        ax.set_xlim([np.min(par.a_min),10])
        ax.set_ylim([-0.03,0.12])

        # layout
        ax.set_xlabel('$m_t$')
        ax.set_ylabel(r'$E[\Delta \log C_{t+1}]$')
        ax.legend(loc='upper right',frameon=True)

        fig.savefig(f'figs/euler_approx_{self.name}.pdf')
    
    ####################
    # simulation plots #
    ####################

    def plot_simulate_cdf_cash_on_hand(self):

        par = self.par
        sim = self.sim

        # figure
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        for t in [0,1,2,4,9,29,49,par.simT-1]:
            ecdf = ECDF(sim.m[:,t])
            ax.plot(ecdf.x,ecdf.y,lw=1.5,label=f'$t = {t}$')

        # limits
        ax.set_xlim([np.min(par.a_min),4])

        # layout  
        ax.set_xlabel('$m_t$')
        ax.set_ylabel('CDF')
        ax.legend(loc='upper right',frameon=True)

        fig.savefig(f'figs/sim_cdf_cash_on_hand_{self.name}.pdf')

    def plot_simulate_consumption_growth(self):

        par = self.par
        sim = self.sim

        # 1. consumption growth
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        y = np.mean(np.log(sim.C[:,1:])-np.log(sim.C[:,:-1]),axis=0)
        ax.plot(np.arange(par.simT-1),y,'-',lw=1.5,label=r'$E[\Delta\log(C_t)]$')
    
        y = np.log(np.mean(sim.C[:,1:],axis=0))-np.log(np.mean(sim.C[:,:-1],axis=0))
        ax.plot(np.arange(par.simT-1),y,'-',lw=1.5,
            label=r'$\Delta\log(E[C_t])$')
        
        ax.axhline(np.log(par.G),ls='-',lw=1.5,color='black',label='$\\log(G)$')
        ax.axhline(np.log(par.G)-0.5*par.sigma_psi**2,ls='--',lw=1.5,color='black',label=r'$\log(G)-0.5\sigma_{\psi}^2$')
    
        # layout  
        ax.set_xlabel('time')
        ax.set_ylabel('')
        ax.legend(loc='lower right',frameon=True)
    
        fig.savefig(f'figs/sim_cons_growth_{self.name}.pdf')

        # b. cash-on-hand
        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(np.arange(par.simT),np.mean(sim.m,axis=0),'-',lw=1.5,label='mean')
        ax.plot(np.arange(par.simT),np.percentile(sim.m,25,axis=0),'--',lw=1.5,color='black',label='25th percentile')
        ax.plot(np.arange(par.simT),np.percentile(sim.m,75,axis=0),'--',lw=1.5,color='black',label='75th percentile')

        # layout 
        ax.set_xlabel('time')
        ax.set_ylabel('$m_t$')
        ax.legend(loc='upper right',frameon=True)

        fig.savefig(f'figs/sim_cash_on_hand_{self.name}.pdf')

    ####################
    # life-cycle plots #
    ####################

    def plot_life_cycle_income(self):

        par = self.par
        sim = self.sim

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(par.age_min+np.arange(1,par.simT),np.nanmean(sim.Y[:,1:],axis=0),'-',lw=1.5)

        # layout 
        ax.set_ylabel('income, $Y_t$')
        ax.set_xlabel('age')

        fig.savefig(f'figs/sim_Y_{self.name}.pdf')

    def plot_life_cycle_cashonhand(self):

        par = self.par
        sim = self.sim

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.M,axis=0),'-',lw=1.5)

        # layout 
        ax.set_ylabel('cash-on-hand, $M_t$')        
        ax.set_xlabel('age')

        fig.savefig(f'figs/sim_M_{self.name}.pdf')

    def plot_life_cycle_consumption(self):

        par = self.par
        sim = self.sim

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.C,axis=0),'-',lw=1.5)

        # layout 
        ax.set_ylabel('consumption, $C_t$')         
        ax.set_xlabel('age')

        fig.savefig(f'figs/sim_C_{self.name}.pdf')  

    def plot_life_cycle_assets(self):

        par = self.par
        sim = self.sim

        fig = plt.figure(figsize=(6,4),dpi=100)
        ax = fig.add_subplot(1,1,1)

        ax.plot(par.age_min+np.arange(par.simT),np.mean(sim.A,axis=0),'-',lw=1.5)

        # layout 
        ax.set_ylabel('assets, $A_t$')         
        ax.set_xlabel('age')

        fig.savefig(f'figs/sim_A_{self.name}.pdf')
           

####################
# 2. numba version #
####################

## same results with faster code

class ConsumptionSavingModelClassNumba(ModelClass,ConsumptionSavingModelClass):    
    
    def __init__(self,name='baseline',solmethod='EGM',**kwargs):

        # a. set baseline parameters
        self.name = name
        self.solmethod = solmethod
                
        # b. define subclasses
        parlist = [

            # setup
            ('T',int64),
            ('TR',int64),
            ('age_min',int64),
            ('rho',double),
            ('beta',double),
            ('G',double),
            ('sigma_xi',double),
            ('sigma_psi',double),
            ('low_p',double),
            ('low_val',double),
            ('L',double[:]),
            ('R',double),
            ('borrowingfac',double),
            ('a_max',double),
            ('a_phi',double),
            ('m_max',double),
            ('m_phi',double),
            ('Npsi',int64),
            ('Nxi',int64),
            ('Na',int64),
            ('Nm',int64),
            ('sim_mini',double),
            ('simN',int64),
            ('simT',int64),
            ('simlifecycle',boolean),

            # create grids
            ('psi_vec',double[:]),
            ('psi_w_vec',double[:]),
            ('xi_vec',double[:]),
            ('xi_w_vec',double[:]),
            ('w',double[:]),
            ('Nshocks',int64),
            ('a_min',double[:]),
            ('grid_a',double[:,:]),
            ('grid_m',double[:,:]),
            ('FHW',double),
            ('AI',double),
            ('GI',double),
            ('RI',double),
            ('WRI',double),
            ('FVA',double),
            ('grid_a_tile',double[:,:]),
            ('psi_vec_rep',double[:]),
            ('xi_vec_rep',double[:]),
            ('w_rep',double[:]),

        ]        
        
        sollist = [
            ('m',double[:,:]),
            ('c',double[:,:]),
            ('inv_v',double[:,:]),
        ]

        simlist = [
            ('m',double[:,:]),
            ('c',double[:,:]),
            ('a',double[:,:]),
            ('p',double[:,:]),
            ('y',double[:,:]),
            ('psi',double[:,:]),
            ('xi',double[:,:]),
            ('P',double[:,:]),
            ('Y',double[:,:]),
            ('M',double[:,:]),
            ('C',double[:,:]),
            ('A',double[:,:]),
        ]

        # c. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        self.setup()

        # b. update parameters
        for key,val in kwargs.items():
            setattr(self.par,key,val) # like par.key = val   

    def EGM(self,t,m,c,inv_v):
        """ overwrite method with numba version """

        EGM(self.par,self.sol,t,m,c,inv_v)

    def simulate_timeloop(self):
        """ overwrite method with numba version """

        simulate_timeloop(self.par,self.sol,self.sim)

# jitted utility function
@njit
def utility(par,c):
    return c**(1-par.rho)/(1-par.rho)

@njit
def marg_utility(par,c):
    return c**(-par.rho)            

@njit
def inv_marg_utility(par,u):
    return u**(-1/par.rho)   

# jitted EGM
@njit(parallel=True)
def EGM(par,sol,t,m,c,inv_v):
    """ EGM with fully unrolled loops """

    # loop over end-of-period assets
    for i_a in prange(par.Na):

        a = par.grid_a[t,i_a]
        still_working_next_period = t+1 <= par.TR-1
        Nshocks = par.Nshocks if still_working_next_period else 1

        # loop over shocks
        avg_marg_u_plus = 0
        avg_v_plus = 0
        for i_shock in range(Nshocks):
            
            # a. prep
            if still_working_next_period:
                fac = par.G*par.L[t]*par.psi_vec[i_shock]
                w = par.w[i_shock]
                xi = par.xi_vec[i_shock]
            else:
                fac = par.G*par.L[t]
                w = 1
                xi = 1
        
            inv_fac = 1.0/fac

            # b. future m and c
            m_plus = inv_fac*par.R*a + xi
            c_plus = linear_interp.interp_1d(sol.m[t+1,:],sol.c[t+1,:],m_plus)
            inv_v_plus = linear_interp.interp_1d(sol.m[t+1,:],sol.inv_v[t+1,:],m_plus)
            v_plus = 1.0/inv_v_plus

            # c. average future marginal utility
            marg_u_plus = marg_utility(par,fac*c_plus)
            avg_marg_u_plus += w*marg_u_plus
            avg_v_plus += w*(fac**(1-par.rho))*v_plus

        # d. current c
        c[i_a] = inv_marg_utility(par,par.beta*par.R*avg_marg_u_plus)

        # e. current m
        m[i_a] = a + c[i_a]

        # f. current v
        if c[i_a] > 0:
            inv_v[i_a] = 1.0/(utility(par,c[i_a]) + par.beta*avg_v_plus)
        else:
            inv_v[i_a] = 0

# jitted simulate_timeloop
@njit(parallel=True)
def simulate_timeloop(par,sol,sim):
    """ simulate model with parallization over households """

    # unpack (helps numba)
    m = sim.m
    p = sim.p
    y = sim.y
    c = sim.c
    a = sim.a

    # loop over first households and then time
    for i in prange(par.simN):
        for t in range(par.simT):
            
            # a. solution
            if par.simlifecycle == 0:
                grid_m = sol.m[0,:]
                grid_c = sol.c[0,:]
            else:
                grid_m = sol.m[t,:]
                grid_c = sol.c[t,:]
            
            # b. consumption
            c[i,t] = linear_interp.interp_1d(grid_m,grid_c,m[i,t])
            a[i,t] = m[i,t] - c[i,t]

            # c. next-period
            if t < par.simT-1:

                if t+1 > par.TR-1:
                    m[i,t+1] = par.R*a[i,t] / (par.G*par.L[t]) +  1
                    p[i,t+1] = np.log(par.G) + np.log(par.L[t]) + p[i,t]
                    y[i,t+1] = p[i,t+1]
                else:
                    m[i,t+1] = par.R*a[i,t] / (par.G*par.L[t]*sim.psi[i,t+1]) + sim.xi[i,t+1]
                    p[i,t+1] = np.log(par.G) + np.log(par.L[t]) + p[i,t] + np.log(sim.psi[i,t+1])   
                    if sim.xi[i,t+1] > 0:
                        y[i,t+1] = p[i,t+1] + np.log(sim.xi[i,t+1])