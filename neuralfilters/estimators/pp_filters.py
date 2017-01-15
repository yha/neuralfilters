# -*- coding: utf-8 -*-

import warnings

import numpy as np
from numpy import sqrt, exp, outer, pi
from scipy.linalg import inv, det, eigvals

from neuralfilters.processes.process import Process
from neuralfilters.processes import populations
from neuralfilters.util.util import overrides, wrap_closure
from neuralfilters.util.linalg import is_psd

class GenericPPFilter( Process ):

    def __init__( self, p_terms, c_terms, N_updates, xe0, V0 ):
        self.p_terms = p_terms
        self.c_terms = c_terms
        self.N_updates = N_updates
        self.xe0 = xe0
        self.V0 = V0
        self.reset()
        
    @overrides(Process)
    def step( self, y, u, dt=1 ):
        t = self.t
        
        p1, p2 = self.p_terms( t, self.xe, self.V, u )
        c1, c2 = self.c_terms( t, self.xe, self.V )
        self.xe += (p1 + c1) * dt
        self.V += (p2 + c2) * dt

        if y is not None:
            self.xe, self.V = self.N_updates( t, self.xe, self.V, y )
        
        if not is_psd(self.V):
            warnings.warn( "Non-PSD variance in PP filter. Eigvals: {}".format(
                              eigvals(self.V)) )
            
        self.t += dt
        
        return (self.xe, self.V)
        
    @overrides(Process)
    def reset( self ):
        self.t = 0
        self.xe = np.copy(self.xe0)
        self.V = np.copy(self.V0)
        
    @overrides(Process)
    def peek( self ):
        return (self.xe, self.V)

        
    @overrides(Process)
    def __repr__( self ):
        return "<Generic PP Filter>"

@wrap_closure("Linear dynamics terms")
def linear_p_terms( A, B, D, b=0 ):
    DD = D @ D.T
    def p_terms( t, xe, V, u ):
        #print( xe, V, u, A, B(u), DD )
        return ( A@(xe-b) + B(u), A@V + V@A.T + DD )
    return p_terms
    
@wrap_closure("Guassian population discontinuous updates")
def gaussian_N_updates( W, H ):
    iW = inv(W)
    def N_update( t, xe, V, y ):
        S = inv( W + H@V@H.T )
        return ( xe + V @ H.T @ S @ (y-H@xe), 
                 inv( inv(V) + H.T @ iW @ H ) )
    return N_update
    
@wrap_closure("Mixture population discontinuous updates")
def mixture_N_updates( pops, updates ):
    pop2update = dict(zip(pops,updates))
    def N_update( t, xe, V, source_tagged_y ):
        y, source = source_tagged_y
        return pop2update[source]( t, xe, V, y )
    return N_update
    

@wrap_closure("Uniform-Gaussian continuous terms")
def uniform_gaussian_c_terms():
    def c_terms( t, xe, V ):
        return (0,0)
    return c_terms

@wrap_closure("Gaussian-Gaussian continuous terms")
def gaussian_gaussian_c_terms( c, G, base_rate, W, H ):
    def c_terms( t, xe, V ):
        Z = inv( W + G + H @ V @ H.T )
        diff = H @ xe - c
        VHZ = V @ H.T @ Z
        VHZdiff = VHZ @ diff
        
        g = base_rate * sqrt(det(W)*det(Z)) * exp( -diff@Z@diff / 2 )

        return ( g * VHZdiff, g * (VHZ @ H @ V - outer(VHZdiff,VHZdiff)) )
        
    return c_terms

# much faster than scipy.stats.norm.pdf:
def _normpdf(x): return exp(-x*x/2) / sqrt(2*pi)

@wrap_closure("Interval-Gaussian continuous terms")
def _interval_uniform_gaussian_c_terms( a, b, base_rate, w, H ):
    def c_terms( t, xe, V ):
        VH = V @ H
        HVH = np.asscalar( H @ VH )
        V_w = w + HVH
        sV_w = sqrt(V_w)
        at = (a - xe) / sV_w
        bt = (b - xe) / sV_w
        #Z = special.ndtr(bt) - special.ndtr(at)
        z = _normpdf(bt) - _normpdf(at)
        zt = bt*_normpdf(bt) - at*_normpdf(at)
        
        coeff = base_rate * sqrt(2*pi*w)
        
        return ( coeff * z * VH / sV_w, coeff * zt * np.outer(VH,VH) / V_w )
        
    return c_terms

def interval_uniform_gaussian_c_terms( a, b, base_rate, w, H ):
    if H.ndim == 2:
        if H.shape[0] != 1:
            raise ValueError()
        H = H[0]
    w = np.asscalar(w)
    return _interval_uniform_gaussian_c_terms( a, b, base_rate, w ,H )

@wrap_closure("Mixture population continuous terms")
def mixture_c_terms( component_terms ):
    def c_terms( t, xe, V ):
        xe_term, V_term = 0, 0
        for dxe, dV in (c_terms(t,xe,V) for c_terms in component_terms):
            xe_term += dxe
            V_term += dV
        return (xe_term, V_term)
    return c_terms
    
def build_updates( pop ):

    if isinstance( pop, populations.MixturePopulation ):
        c, N = zip( *(build_updates(comp) for comp in pop.components) )
        c_terms = mixture_c_terms( c )
        N_updates = mixture_N_updates( pop.components, N )
        
    else:
        
        if isinstance( pop, populations.GaussianPopulation ):
            N_updates = gaussian_N_updates( pop.W, pop.H )
        else:
            raise NotImplementedError()
            
        if isinstance( pop, populations.UniformGaussianPopulation ):
            c_terms = uniform_gaussian_c_terms()
        elif isinstance( pop, populations.GaussianGaussianPopulation  ):
            c_terms = gaussian_gaussian_c_terms( pop.c, pop.G, pop.base_rate,
                                                 pop.W, pop.H )
        elif isinstance( pop, populations.IntervalUniformGaussianPopulation ):
            c_terms = interval_uniform_gaussian_c_terms( 
                                                 pop.a, pop.b, pop.base_rate,
                                                 pop.W, pop.H )
        else:
            raise NotImplementedError()
    
    return (c_terms, N_updates)

    
def build_filter( pop, xe0, V0, A, B, D, b=0 ):
    p_terms = linear_p_terms( A, B, D, b )
    c_terms, N_updates = build_updates( pop )
    return GenericPPFilter( p_terms, c_terms, N_updates, xe0, V0 )

