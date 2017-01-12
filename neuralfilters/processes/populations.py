# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.linalg as la
from numpy.random import rand, multivariate_normal

from neuralfilters.processes.process import Process
from neuralfilters.util.util import overrides
from neuralfilters.util.linalg import is_psd

class NeuralPopulation( Process, metaclass = ABCMeta):
    """
    A point process representing a (finite or infinite) population of sensory 
    neurons. 
    The result of step(..) is None if no neuron fired in the time interval, 
    otherwise the mark of the neuron that fired (it is assumed dt is chosen 
    small enough to safely neglect the probability of two neurons firing in 
    the same interval).
    """

    @abstractmethod
    def rate( self, s, x ):
        """Firing rate of a neuron with mark s in response to x"""

    @abstractmethod        
    def total_rate( self, x ):
        """
        The total firing rate from all neurons in response to stimulus x.
        """

        
        
class GaussianPopulation( NeuralPopulation ):        
    """
    A population of Gaussian neurons, marked by preferred location.
    
    The firing rate at preferred location s for input (state) x is given by:
        rate(s,x) = base_rate * exp(-(H*x-s)'inv(W)(H*x-s)/2).
    where "inv" denotes matrix inverse, and "'" matrix transpose
    """
    
    def __init__( self, W, H, base_rate, **kwargs ):
        super().__init__(**kwargs)
        self.W = np.atleast_2d(np.copy(W))
        self.H = np.atleast_2d(np.copy(H))
        self.base_rate = base_rate
        if not is_psd(self.W):
            raise ValueError( "Non-PSD rate function width matrix" )
        self._iW = la.inv(self.W)
        
    @overrides(NeuralPopulation)
    def rate( self, s, x ):
        diff = self.H @ x - s
        return self.base_rate * np.exp( -diff @ self._iW @ diff / 2 )
        
    
class MarkSamplingPopulation( NeuralPopulation ):
    """    
    An abstract neural population implementation based on the following 
    methods, which should be specified by subclasses:
     - total_rate(self,x) computes the total firing rate (from all neurons) 
       for stimulus x.
     - sample_mark(self,x) to randomly select the mark (identity) of the 
       neuron that fired a spike, given that some spike ocurred in response
       to stimulus x.
    
    If the rate for a single neuron with mark s in response to x is given by 
    rate(s,x), and the neuron marks' counting measure is f(ds), then 
    total_rate(..) should compute the integral of rate(s,x)f(ds), and 
    sample_mark(..) should sample randomly from the probability measure 
    rate(s,x)f(ds)/total_rate(x).
    """
    
    def __init__( self, **kwargs ):
        super().__init__(**kwargs)
        self.reset()
        
    @abstractmethod
    def sample_mark( self, x ):
        """
        A random sample of the mark of the firing neuron, given that some 
        spike ocurred. That is, a random sample from the probability measure 
        rate(s,x)f(ds)/total_rate(x), where rate(s,u) is the rate for a single 
        neuron with mark s in response to x, and f(ds) is the neuron marks' 
        counting measure (possibly a countinuous measure for an infinite 
        population).
        """
    
    @overrides(NeuralPopulation)
    def step( self, x, dt=1 ):
        if rand() < self.total_rate(x)*dt:
            self.s = self.sample_mark(x)
        else: 
            self.s = None
        self.t += dt
        return self.s
        
    @overrides(NeuralPopulation)
    def peek( self ):
        return self.s
        
    def reset( self ):
        self.s = None
        self.t = 0
        

class UniformGaussianPopulation( GaussianPopulation, MarkSamplingPopulation ):
    """
    A NeuralPopulation with Gaussian tuning functions and preferred locations 
    uniformly distributed on the entire space.
    The tuning functions have the form of (non-normalized) Gaussians with 
    variance W (m-by-m), and height base_rate. This Gaussian is applied to H*x,
    where x is the input (state, of length n), and H is an m-by-n matrix.
    Neurons share the same matrices H,W, and are marked with their tuning 
    function center (preferred stimulus). The preferred stimuli are uniformly 
    distributed over m-dimensional space. 
    
    The firing rate of each neuron, rate(s,x), is as specified in 
    GaussianPopulation.
    The neuron mark distribution is 
        
        f(ds) = ds.
                
    The total firing rate is given by the integral of rate(s,x)ds which is
        total_rate(x) = base_rate * sqrt((2*pi)^m*det(W)),
    and the mark sampling distribution is normal with mean H*x and variance W,
        rate(s,x)ds/total_rate(x) = 1/sqrt((2*pi)^m*det(W)) 
                                    * exp(-(H*x-s)'inv(W)(H*x-s)/2)
    """
    
    def __init__( self, W, H, base_rate=1.0 ):
        super().__init__( W=W, H=H, base_rate=base_rate )
        m,n = H.shape
        self._total_rate = base_rate * np.sqrt((2*np.pi)**m * la.det(W))
        
    @overrides(MarkSamplingPopulation)
    def total_rate( self, x ):
        return self._total_rate
    
    @overrides(MarkSamplingPopulation)
    def sample_mark( self, x ):
        return multivariate_normal( self.H @ x, self.W )


    def __repr__( self ):
        return "Uniform-Gaussian population"

from scipy.stats import norm, truncnorm
from scipy import special

class IntervalUniformGaussianPopulation( GaussianPopulation, 
                                         MarkSamplingPopulation ):
    """
    A NeuralPopulation with homogenous 1-d Gaussian tuning functions, and 
    preferred stimuli distributed uniformly on an interval.
    The tuning functions have the form of (non-normalized) 1-d Gaussians with 
    variance w. This Gaussian is applied to H*x, where x is the input 
    (state), and H a row vector.
    Neurons share the same matrix H and variance w, and are marked with their 
    tuning  function center (preferred stimulus). The preferred stimuli are 
    uniformly distributed on an interval [a,b]. 
    
    The firing rate of each neuron, rate(s,x), is as specified in 
    GaussianPopulation.
    The neuron mark distribution is 
        f(ds) = 1{[a,b]}(s) ds,
    where 1{[a,b]} is the indicator function for the interval [a,b].

    The total firing rate is given by the integral of rate(s,x)ds over [a,b],
    which is
        total_rate(x) = base_rate * sqrt(2*pi*w) * (ncdf(z(b)) - ncdf(z(a))),
    where ncdf is the cdf of the standard normal distribution, 
    and z(s) = (s-H*x)/sqrt(w).
    The mark sampling distribution is a truncated normal distribution on the
    interval [a,b], with "nominal mean" H*x and "nominal variance" w.
    """
    
    def __init__( self, a, b, w, H, base_rate=1.0 ):
        super().__init__( W=[[w]], H=H, base_rate=base_rate )
        self.a = a
        self.b = b
        self.w = w
        m,n = self.H.shape
        if m != 1:
            raise ValueError()
        self._sw = np.sqrt(w)
        self._rate_factor = base_rate * np.sqrt(2*np.pi*w)

    @overrides(MarkSamplingPopulation)
    def total_rate( self, x ):
        a,b,sw,H = self.a, self.b, self._sw, self.H
        m = np.asscalar( H @ x )
        za, zb = (a-m)/sw, (b-m)/sw
        # special.ndtr is much faster than norm.cdf
        return self._rate_factor * (special.ndtr(zb) - special.ndtr(za))

    @overrides(MarkSamplingPopulation)
    def sample_mark( self, x ):
        a,b,sw,H = self.a, self.b, self._sw, self.H
        m = H @ x
        za, zb = (a-m)/sw, (b-m)/sw
        return truncnorm.rvs( za, zb, m, sw )

    def __repr__( self ):
        return "Uniform-Gaussian population on [{},{}]".format(self.a, self.b)


class GaussianGaussianPopulation( GaussianPopulation, MarkSamplingPopulation):
    """
    A neural population with homogenous Gaussian tuning functions, and Gaussian
    preferred location distribution.
    The tuning functions have the form of (non-normalized) Gaussians with 
    (possibly singular) variance matrix W. This Gaussian is applied to H*x,
    where x is the input (state). 
    All neurons share the same matrices H and W, with the tuning function 
    centers (preferred locations) normally distributed with mean c and variance 
    matrix G. Neurons are marked with their preferred location. 
    A population of a single neuron may be realized as a 
    GaussianGaussianPopulation with G=0.
    
    The firing rate of each neuron, rate(s,x), is as specified in 
    GaussianPopulation.
    The mark distribution is
    
          f(ds) = 1/sqrt((2*pi)^n*det(G)) * exp(-(s-c)'inv(G)(s-c)/2) * ds,
    
    where "det" denotes determinant, "inv" matrix inverse, and "'" matrix
    transpose.
                
    An alternate expression for the rate-density rate(s,x)f(ds)/ds, involving 
    only a single Gaussian in s:
        rate(s,x)f(ds)/ds = 
                base_rate
                * 1/sqrt((2*pi)^n*det(G)) * exp(-(H*x-c)'*inv(W+G)*(H*x-c)/2)
                * exp(-(s-q)'*(inv(W)+inv(G))*(s-q)),
    where q = inv(inv(W)+inv(G))*(inv(W)H*x+inv(G)c) 
            = G*inv(W+G)x + W*inv(W+G)c
                
    The total firing rate is given by the integral of rate(s,x)f(ds) which is:
        total_rate(x) = base_rate * sqrt(det(W)/det(G+W)) 
                                  * exp(-(H*x-c)'*inv(W+G)*(H*x-c)/2),
    and the mark sampling distribution is normal with mean q and precision 
    matrix (inv(W)+inv(G)),
        rate(s,x)f(ds)/total_rate(x) = 
                sqrt((inv(W)+inv(G))/(2*pi)^n)
                * exp(-(s-q)'*(inv(W)+inv(G))*(s-q)) ds
    """
    
    def __init__( self, c, G, W, H, base_rate=1.0 ):
        super().__init__( W=W, H=H, base_rate=base_rate )
        self.c = np.atleast_1d(np.copy(c))
        self.G = G = np.atleast_2d(np.copy(G))
        # use self.W, initialized by super().__init__, 
        # which is W converted to a 2-d array
        W, H = self.W, self.H
        self._K = K = la.inv(G+W)
        self._GKH = G @ K @ H
        self._WK = W @ K
        # self._hGW is the "harmonic sum" inv(inv(G)+inv(W)) = W*inv(G+W)*G.
        # Computed using the latter expression, which generalizes to singular 
        # G or W.
        self._hGW = self._WK @ G
        self._rate_factor = base_rate * np.sqrt(la.det(W)/la.det(G+W))
        #self._rate_factor = base_rate * np.sqrt(la.det(self._WK))
        self._N_OUT, self._N_IN = H.shape
        self.reset()
        
    @overrides(MarkSamplingPopulation)
    def total_rate( self, u ):
        Huc = self.H @ u - self.c
        return self._rate_factor * np.exp( -(Huc @ self._K @ Huc) / 2 )
    
    @overrides(MarkSamplingPopulation)
    def sample_mark( self, u ):
        GKH, WK, hGW, c = self._GKH, self._WK, self._hGW, self.c
        return multivariate_normal( GKH @ u + WK @ c, hGW )
        
    @overrides(Process)
    def __repr__( self ):
        return str(self._N_OUT) + ("-dimensional time-invariant " 
                                   "Gaussian-Gaussian population")


from collections import namedtuple
SourceTaggedMark = namedtuple( 'SourceTaggedMark', ('mark', 'source') )

class MixturePopulation(NeuralPopulation):
    def __init__( self, *components ):
        self.components = components
        self.s = None
        
    @overrides(NeuralPopulation)
    def step( self, u, dt=1 ):
        out, out_pop = None, None
        for pop in self.components:
            y = pop.step(u,dt)
            if y is not None:
                out, out_pop = y, pop
                # Do not return here:
                # For time-varying population it may be important to step all 
                # of them the same amount, even if it is assumed at most one 
                # would spike each step.
        if out is None:
            assert out_pop is None
            self.s = None
        else:
            self.s = SourceTaggedMark(out, out_pop)
        return self.s

    @overrides(NeuralPopulation)
    def peek( self ):
        return self.s
        
    @overrides(NeuralPopulation)
    def rate( self, s, x ):
        y, source = s
        if source not in self.components:
            raise ValueError()
        return source.rate(y,x)
        
    @overrides(NeuralPopulation)
    def total_rate( self, x ):
        return sum( pop.total_rate(x) for pop in self.components )
        
    def reset( self ):
        self.s = None
        
    @overrides(NeuralPopulation)
    def __repr__( self ):
        return "Mixture population: " + ", ".join( pop.__repr__() 
                                                  for pop in self.components )
#%%

import itertools

def _get( list, index, default ):
    return list[index] if -len(list) <= index < len(list) else default

    
class SpikeHistory:
    def __init__(self):
        self.times = []
        self.marks = []
    
    def append( self, time, mark ):
        last = _get( self.times, -1, -np.inf )
        if time <= last: raise ValueError('Non-increasing spike times')
        self.times.append(time)
        self.marks.append(mark)
        
    def iterate( self, dt, T, include_times = False, shift = -0.5 ):
        next_index = 0
        next_time = _get(self.times,0,None)
        for i in itertools.count():
            t = (i+shift) * dt
            if t+dt >= T:
                return
            if next_time is None: # no more spikes
                yield None
            elif t <= next_time < t + dt:
                time, mark = next_time, self.marks[next_index]
                next_index += 1
                next_time = _get( self.times, next_index, None )
                yield (time, mark) if include_times else mark
            else:
                yield None


#%%

if __name__ == '__main__':
    
    from numpy import ma
    import matplotlib.pyplot as plt
    
    H = np.array([[1,0]])
    W = np.array([[1]])
    h = np.array([[1,0]])
    w = 0.25
    a, b = -1, 1
    c, G = map(np.array, ([0], [[4]]))
    base_rate = 50
    ug = UniformGaussianPopulation(W,H,base_rate)
    iug = IntervalUniformGaussianPopulation(a,b,w,h,base_rate)
    gg = GaussianGaussianPopulation(c,G,W,H,base_rate)
    nets = (ug,iug,gg)

    # population densities
    ug_pop = lambda y: np.ones_like(y)
    iug_pop = lambda y: (a <= y) & (y <= b)
    gg_pop = norm(c,np.sqrt(G[0,0])).pdf
    pop_densities = (ug_pop, iug_pop, gg_pop)

    # tuning curves
    tc_f = lambda x,y,h,w: h * np.exp( -(x-y)**2 / (2*w) )
    ug_tc = gg_tc = lambda x,y: tc_f(x,y,base_rate,W[0,0])
    iug_tc = lambda x,y: tc_f(x,y,base_rate,w)
    tcs = (ug_tc, iug_tc, gg_tc)
    
    # mark sampling distributions for input x0
    x0 = (1.0,0)
    y0 = H @ x0
    ug_dist_0 = norm(y0,np.sqrt(W[0,0]))
    iug_dist_0 = truncnorm( (a-y0)/np.sqrt(w), (b-y0)/np.sqrt(w),
                            y0, np.sqrt(w) )
    gg_var = 1 / (1/W[0,0] + 1/G[0,0])
    gg_dist_0 = norm( gg_var*(y0/W[0,0] + c/G[0,0]), np.sqrt( gg_var ) )
    ug_rate_0 = ug.total_rate(x0)
    gg_rate_0 = gg.total_rate(x0)
    ms_dists = (ug_dist_0, iug_dist_0, gg_dist_0)

    #%% mark sampling distribution tests

    import time
    
    T = 100
    dt = 0.001
    L = int(np.round(T/dt))
    #X = np.full( (L, 2), x0 )
    Y = ma.masked_all( (L, len(nets)) )
    from util.progress import ProgressBar
    bar = ProgressBar( T, 40, 'mark sampling distribution test', 't' )
    
    tic = time.clock()
    for i in range(L):
        if i % 100 == 0:
            bar.render(i*dt)
        for k,net in enumerate(nets):
            y = net.step(x0,dt)
            if y is not None:
                Y[i,k] = y
    bar.render(T)
    
    print()
    print('time:',time.clock()-tic)

#%%
    
    im = int(np.round(9/dt))
    iM = int(np.round(10/dt))
    K = len(nets)
    for k in range(K):
        plt.subplot(K,2,2*k+1)
        t = np.r_[0:T:dt]
        I = (~Y[:,k].mask).nonzero()[0]
        I_i = I[ (im<=I) & (I<iM) ]
        plt.scatter(t[I_i],Y[I_i,k], c='y', s=30, label='spikes' )
        ax = plt.subplot(K,2,2*k+2)
        n, bins, _ = plt.hist( Y[I,k], 25, normed=True )
        bins_c = (bins[:-1] + bins[1:]) / 2
        plt.plot( bins_c, ms_dists[k].pdf(bins_c), linewidth=2 )
        ax.text( 1, 1, '$n={}$'.format(len(I)), 
                 ha='right', va='top', transform=ax.transAxes )
    plt.tight_layout()
    
    #%% "tuning curve" tests
    
    inputs = np.linspace(-2,2,11)
    T_hold = 1
    N = 1000
    dt = 0.001
    T = T_hold*N
    I = np.random.randint( len(inputs), size=N )
    X = np.array(inputs)[I]
    L_hold = int(np.round(T_hold/dt))
    L = int(np.round(T/dt))
    Y = ma.masked_all( (L, len(nets)) )
    
    bar = ProgressBar(T, 40, 'tuning curve test', 't')
    for i in range(N):
        bar.render(i*T_hold)
        for k,net in enumerate(nets):
        #for k,net in ((1,nets[1]),):
            #print(k,net)
            for j in range(int(T_hold/dt)):
                #print(X[i],net.total_rate((X[i],0)))
                y = net.step((X[i],0),dt)
                if y is not None:
                    Y[i*L_hold+j,k] = y
                    #print(i,j, ' =>', y)
    bar.render(T)
    
    #%% visualize start of trial
    
    k = 1
    m = 20
    im = 0
    iM = int(np.round(m*T_hold/dt))
    t = np.r_[0:T:dt]
    I_spikes = (~Y[:,k].mask).nonzero()[0]
    I_i = I_spikes[ (im<=I_spikes) & (I_spikes<iM) ]
    plt.scatter(t[I_i],Y[I_i,k], c='y', s=30, label='spikes' )
    plt.step( T_hold*np.arange(m+1), np.concatenate((X[:m],X[m-1:m]),0), 
              where='post' )
    
    #%% compare simulated and target tuning curves
    
    k = 1 # net to test
    
    x_centers = np.array(inputs)
    x_edges = np.concatenate((((3*x_centers[0]-x_centers[1]) / 2,),
                              (x_centers[:-1] + x_centers[1:]) / 2,
                              ((3*x_centers[-1]-x_centers[-2]) / 2,)))
    y_edges = np.linspace(-2,2,17)
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    y_binsize = y_edges[1] - y_edges[0]
    # x (input) and y (preferred location) values for each spike
    I_spikes = ~Y.mask[:,k]
    y_spikes = Y.data[I_spikes,k]
    x_spikes = X[ np.nonzero(I_spikes)[0] // L_hold ]
    
    x_times = np.bincount(I) * T_hold
    
    counts = np.histogram2d( x_spikes, y_spikes, bins=(x_edges,y_edges) )[0]
    rates = counts / x_times[:,None]
    # rate per cell
    rates_norm = rates / (pop_densities[k](y_centers) * y_binsize)[None,:]
    
    plt.pcolormesh( x_edges, y_edges, ma.masked_invalid(rates_norm).T )
    plt.xlim((x_edges[0],x_edges[-1]))
    #plt.pcolormesh( counts.T, cmap='viridis' )
    plt.colorbar()
    plt.show()

    tc = tcs[k](x_centers[:,None],y_centers)
    plt.pcolormesh( x_edges, y_edges, tc.T )
    plt.xlim((x_edges[0],x_edges[-1]))
    plt.colorbar()
    plt.show()
    i = slice(None,None,2)
    plt.plot( x_centers, rates_norm[:,i] )
    plt.gca().set_prop_cycle(None) # reset color cycle
    plt.plot( x_centers, tc[:,i], '--' )