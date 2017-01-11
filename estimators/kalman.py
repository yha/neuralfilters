# -*- coding: utf-8 -*-

from processes.process import VectorProcess
from util.util import overrides
import util.linalg

import numpy as np
from scipy import linalg as la

import warnings

class KalmanBucyFilter(VectorProcess):
    """
    Kalman-Bucy filter for the dynamics:
        dx = (Ax + Bu)dt + D1 dw1,
        dy = Cx dt + D2 dw2,
    where w1, w2 are independent processes with the second-order statistics of
    standard Brownian motion (zero mean, uncorrelated increments and unit 
    intensity), D2 has full row rank, y(0) = 0, and x(0) has mean xe(0) and 
    variance Q(0).
    
    The filter computes the posterior mean xe(t) and posterior variance Q(t) 
    of x(t) given y(s) for 0<=s<=t, and is given by:
        d{xe} = (A*xe + B*u)dt + K(dy - C*xe*dt),
        dQ/dt = AQ + QA' + V1 - QC'*inv(V2)*CQ,
    where ' indicates matrix transpose, inv() matrix inverse, and
        V1=D1*D1', V2=D2*D2', K = Q*C'*inv(V2).
        
    The posterior distribution of x(t) is Gaussian (thus characterized by 
    xe(t), Q(t)), when w1,w2,x(0) are Gaussian, and u(t) is linear in 
    {y(s), 0<=s<=t}.
    """
    
    def __init__( self, A, B, C, D1, D2, xe0, Q0 ):
        self.A = A = np.atleast_2d(A)
        self.B = B = np.atleast_2d(B)
        self.C = C = np.atleast_2d(C)
        self.D1 = D1 = np.atleast_2d(D1)
        self.D2 = D2 = np.atleast_2d(D2)
        self.V1 = D1.dot(D1.T)
        self.V2 = D2.dot(D2.T)
        self.CtV2i = C.T @ la.inv(self.V2)
        self.xe0 = xe0
        self.Q0 = Q0
        self._y_prev = 0
        self.reset()
    
    @overrides(VectorProcess)
    def step( self, y, u, dt=1 ):
        xe,Q,A,B,C,V1,CtV2i = self.xe,self.Q,self.A,self.B,self.C,\
                                 self.V1,self.CtV2i
        dy = np.atleast_1d(y) - self._y_prev
        self._y_prev = np.array(y)
        u = np.atleast_1d(u)
        K = Q.dot(CtV2i)
#        print('u:', u)
#        print('xe <<<', self.xe)
#        print('    dy:', dy )
        self.xe += (A @ xe + B @ u) * dt + K @ ( dy - C @ xe * dt )
        self.Q += (A @ Q + Q @ A.T + V1 - Q @ CtV2i @ C @ Q) * dt
#        print('xe >>>', self.xe)
        if not util.linalg.is_psd( self.Q, tol=1e-9 ):
            warnings.warn( "Non-PSD variance encountered in Kalman-Bucy " + 
                           "filter. Step size too large?" )
            self.Q = np.zeros_like(Q)
        return (self.xe,self.Q)
    
    @overrides(VectorProcess)
    def reset( self ):
        self.xe = np.array( np.atleast_1d(self.xe0) )
        self.Q = np.array( np.atleast_2d(self.Q0) )
        print('RESET:', self.xe, self.Q)
        return (self.xe,self.Q)
        
    def __repr__( self ):
        return "Kalman-Bucy filter"
        
    @property
    @overrides(VectorProcess)
    def N_IN( self ):
        return self.B.shape[1]

    @property
    @overrides(VectorProcess)
    def N_OUT( self ):
        return self.B.shape[0]
        

class KalmanBucyMeanFilter(KalmanBucyFilter):
    def step( self, y, u, dt=1 ):
        xe,_ = super().step(y,u)
        return xe