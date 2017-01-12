# -*- coding: utf-8 -*-

import warnings

import numpy as np

from neuralfilters.util.util import as_callable
from neuralfilters.util.linalg import is_psd

class LinearPolicy:
    def __init__( self, A ):
        self.A = as_callable(A)
        self.reset()
        
    def step( self, observation, dt=1 ):
        At = self.A(self.t)
        self.t += dt
        return At @ observation
    
    def reset( self ):
        self.t = 0
    
    @property
    def N_IN(self):
        return self.A(0).shape[1]
        
    @property
    def N_OUT(self):
        return self.A(0).shape[0]
    
    def __repr__(self):
        return "Linear Policy {} -> {}".format(self.N_IN, self.N_OUT)

from scipy import linalg as la
from scipy import integrate

def riccati_rhs( A, B, R_state, R_control ):
    n = A.shape[0]
    invR = la.inv(R_control)
    def f(t,S):
        S = S.reshape((n,n))
        dS = S @ B @ invR @ B.T @ S - A.T @ S - S @ A - R_state
        return dS.reshape((n**2,))
    return f

def tabulate_riccati( T, dt, A, B, R_state, R_control, S_final ):
    n = A.shape[0]
    L = int( np.round(T/dt) )
    S = np.zeros( (L+1, n ,n) )
    f = riccati_rhs( A, B, R_state, R_control )
    riccati = integrate.ode( f )
    riccati.set_initial_value( S_final.reshape((n**2,)), T )
    S[-1] = S_final
    for i in reversed(range(L)):
        S[i] = riccati.integrate(i*dt).reshape((n,n))
        
    return S
    
class FiniteHorizonOptimalLinearPolicy:
    def __init__( self, T, dt, A, B, R_state, R_control, S_final ):
        self.riccati_rhs = riccati_rhs( A, B, R_state, R_control )
        self.S_final = S_final
        self.T = T
        self.dt = dt
        self.optimal_gain_coeff = -la.solve( R_control, B.T )
        self.riccati_sol = tabulate_riccati( T, dt, A, B, R_state, R_control, S_final )
        self.reset()
    
    def reset( self ):
        self.t = 0
        
    def step( self, observation, dt=1 ):
        n = self.optimal_gain_coeff.shape[1]
        # Take value from nearest point in the tabulated solution.
        i = int(np.round(self.t / dt))
        S = self.riccati_sol[i]
        if abs(i - self.t/dt) > 1e-3: # not close enough to the nearest point
            print( "Integrating... (i={}, t/dt={})".format(i,self.t/dt) )
            print(S)
            riccati = integrate.ode( self.riccati_rhs )
            riccati.set_initial_value( S.reshape((n**2,)), i*dt )
            S = riccati.integrate(self.t)
            print(S)
        if not is_psd(S,tol=1e-12):
            warnings.warn('Negative Riccati solution, t={}, S={}'.format(
                                self.t, S))   
        result = self.optimal_gain_coeff @ (S @ observation)
        self.t += dt
        return result


def optimal_linear_ih( A, B, R_control, R_state ):
    """
    Construct the optimal linear solution for the (determenistic or stochastic) 
    infinite horizon time-invariant linear optimal regulator problem.
    The notation is as in Kwakernaak&Sivan (p.255), except 
    R_state, R_control instead of R1, R2 respectively
    (note that R_control = R1 = D^T R3 D).
    """
    
    P = la.solve_continuous_are( A, B, R_state, R_control )
    
    err = A.T @ P + P @ A - P @ B @ la.solve( R_control, B.T @ P ) + R_state
    assert( la.norm(err) < 1e-7 )
    
    F0 = la.solve( R_control, B.T @ P )
    
    return LinearPolicy( -F0 )
    
def optimal_linear_fh( A, B, R_control, R_state, S_final, T, dt ):
    """
    Construct the optimal linear solution for the (determenistic or stochastic) 
    finite horizon linear optimal regulator problem.
    The notation is as in Kwakernaak&Sivan (p.255), except 
    R_state, R_control, S_final instead of R1, R2, P1 respectively
    (note that R_control = R1 = D^T R3 D).
    T is the control horizon, and dt is the step size for pre-tabulation of the 
    solution to the Riccati equation (used to avoid stability issues in solving 
    the equation forward in time or performance issues in repeatedly solving it
    backwards). Ideally, the returned policy should be used with step size dt.
    """
    
    if S_final is None:
        S_final = np.zeros_like(R_state)
    return FiniteHorizonOptimalLinearPolicy( T, dt, A, B, 
                                             R_state, R_control, S_final )
    
