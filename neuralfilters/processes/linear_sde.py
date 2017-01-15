# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import randn

from neuralfilters.processes.process import VectorProcess
from neuralfilters.util.util import overrides, as_callable

def _sde_inc( x, t, A, B, S, u, dt ):
    dWt = np.sqrt(dt) * randn( S.shape[1] )
    return (A @ x + B @ u) * dt + S @ dWt

class LinearSDE_TI(VectorProcess):
    """
    A time-invariant LinearSDE, with state x obeying
        dx(t) = (Ax(t) + Bu(t))dt + SdW(t),
    where W is standard Brownian motion.
    """

    def __init__( self, A, B, S, x0 = None, desc = "Linear SDE" ):
        self.A = A
        self.B = B
        self.S = S
        if np.any(np.array([A.ndim, B.ndim, S.ndim]) != 2):
            raise ValueError()
        self._N_OUT, self._N_IN = B.shape
        self._N_NOISE = S.shape[1]
        if A.shape != (self.N_OUT, self.N_OUT) or S.shape[0] != self.N_OUT:
            raise ValueError()
        if x0 is None:
            x0 = np.zeros( self._N_OUT )
        self.x0 = x0
        self.desc = desc
        self.reset()
        
    @property
    @overrides(VectorProcess)
    def N_IN( self ):
        return self._N_IN

    @property
    @overrides(VectorProcess)
    def N_OUT( self ):
        return self._N_OUT
        
    @overrides(VectorProcess)
    def step( self, u, dt=1 ):
        self.t += dt
        self.x += _sde_inc( self.x, self.t, self.A, self.B, self.S, u, dt )
        return self.x
        
    @overrides(VectorProcess)
    def peek( self ):
        return self.x
        
    @overrides(VectorProcess)
    def reset( self ):
        self.t = 0
        if callable(self.x0):
            self.x = self.x0()
        else:
            self.x = np.array( self.x0, copy=True )
        if self.x.ndim > 1:
            raise ValueError("Non-vector initial condition")
        self.x = self.x.flatten()
    
    def __repr__( self ):
        return self.desc


class LinearSDE(LinearSDE_TI):
    """
    A system with state x that obeys the SDE:
        dx(t) = (A(t)x(t) + B(t)u(t))dt + S(t)dW(t),
    where W is standard Brownian motion.
    """

    def __init__( self, A, B, S, x0 = None, desc = "Linear SDE" ):
        A = as_callable(A)
        B = as_callable(B)
        S = as_callable(S)
        super().__init__( A(0), B(0), S(0), x0, desc )
        self.A = A
        self.B = B
        self.S = S
        
    @overrides(LinearSDE_TI)
    def step( self, u, dt=1 ):
        t = self.t
        self.x += _sde_inc( self.x, t, self.A(t), self.B(t), self.S(t), u, dt )
        self.t += dt
        return self.x
        
def integrator( x0 = 0 ):
    x0 = np.asarray( x0 )
    if x0.ndim > 1:
        raise ValueError()
    x0 = x0.flatten()
    N = x0.shape[0]
    return LinearSDE_TI( np.zeros((N,N)), np.eye(N), 
                         np.zeros((N,0)), "Integrator" )
    

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def simulate( system, dt, L, controller ):
        X = np.zeros(L+1)
        system.reset()
        x = system.peek()
        for i in range(L):
            X[i] = x[0]
            u = controller(x)
            x = system.step( [u], dt )
        X[L] = x[0]
        return X
        
    # TI system
    L = 1000
    dt = 0.01
    T = L*dt
    t = np.arange(0,T+dt,dt)
    s = LinearSDE_TI( np.array([[0,1],[0,0]]), np.array([[0],[1]]), 
                   0.05*np.eye(2), np.array([0.0,1.0]) )
    def u(x): return -x[0]-x[1] # A PD controller
    for i in range(100):
        X = simulate( s, dt, L, u )
        plt.plot( t, X, 'k', alpha=0.1 )
        
    #%%
    
    # time-variant system: a Brownian bridge 
    #     dX_t = X_t/(t-1) dt + dW_t (0<t<1)
    def A(t): return np.array([[1/(t-1)]])
    T = 1
    L = 1000
    dt = T/L
    t = np.arange(0,T+dt,dt)
    s = LinearSDE( A, np.array([[1.0]]), 
                   np.array([[1.0]]), np.array([0.0]) )
    
    for i in range(100):
        X = simulate( s, dt, L, lambda x: 0 )
        plt.plot( t, X, 'k', alpha=0.1 )