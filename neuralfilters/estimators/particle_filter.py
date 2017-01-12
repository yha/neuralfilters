# -*- coding: utf-8 -*-

import numpy as np

import math
import itertools

from neuralfilters.processes.process import Process
from neuralfilters.util.util import overrides

def _resample( particles, counts ):
    I = np.zeros( len(particles), dtype=int )
    k = 0
    for index,count in enumerate(counts):
        for j in range(count):
            I[k] = index
            k += 1
    return np.array([p.clone() for p in particles[I]])
    
    
def multinomial_resample( weights, particles ):
    n = len(particles)
    counts = np.random.multinomial( n, weights )
    return _resample( particles, counts )
    
def systematic_resample( weights, particles ):
    #print( "weights:", weights )
    n = len(particles)
    cw = np.cumsum(tuple(itertools.chain((0,),weights)))
#    cw = np.cumsum(np.concatenate(([0],weights)))
    partition = n * cw - np.random.rand()
    c_counts = partition.astype(int)
    counts = c_counts[1:] - c_counts[:-1]
    #print( len(counts), sum(counts), counts )
    return _resample( particles, counts )    

class ParticleFilter( Process ):
    
    def __init__( self, dynamics, obs_model, n_particles,
                  resampler=multinomial_resample ):
        self.dynamics = dynamics
        self.obs_model = obs_model
        self.particles = np.array([dynamics.instance() for i in range(n_particles)])
        self.weights = np.full( n_particles, 1/n_particles )
        self.t = 0
        self.last_resample = -math.inf
        self.resampler = resampler

    @overrides(Process)
    def step( self, observation, control, dt=1, resample_dt=None ):
        for i, particle in enumerate( self.particles ):
            x = particle.step( control, dt )
            self.weights[i] *= self.obs_model.likelihood( x, observation, dt )
        self.weights /= np.sum(self.weights)
        assert np.all(self.weights >= 0)
        if resample_dt is None or self.t >= self.last_resample + resample_dt:
            self.particles = self.resampler( self.weights, self.particles )
            n = len(self.particles)
            self.weights = np.full( n, 1/n )
            self.last_resample = self.t
        self.t += dt
        return self.particles
    
    @overrides(Process)
    def reset( self ):
        for particle in self.particles:
            particle.reset()
        self.obs_model.reset()    
    
from processes.linear_sde import LinearSDE_TI    
import scipy.linalg as la


class LinearSdeParticle(LinearSDE_TI):
    def __init__( self,  A, B, S, x0, desc, parent=None ):
        super().__init__( A, B, S, x0, desc )
        self.parent = parent
        
    def clone( self, recordParent=False ):
        return LinearSdeParticle( self.A, self.B, self.S, 
                                  x0 = self.x, # !
                                  desc = self.desc,
                                  parent = self if recordParent else None )
        
class LinearSdeModel:

    def __init__( self, A, B, S, x0 = None, V0 = None, desc = "Linear-SDE" ):
        self.A = np.copy(A)
        self.B = np.copy(B)
        self.S = np.copy(S)
        self.x0 = np.copy(x0)
        self.V0 = np.copy(V0)
        self.desc = desc
        self._N_OUT = self.A.shape[0]
        
    def instance( self ):
        x0_sde = self.x0 + la.sqrtm(self.V0) @ np.random.randn(self._N_OUT)
        id = np.random.randint(2**16)
        return LinearSdeParticle( self.A, self.B, self.S, x0_sde, 
                                  self.desc + "-" + str(id) )
        

class NeuralPopulationModel:
    
    def __init__( self, pop ):
        self.pop = pop
        
    def likelihood( self, x, obs, dt ):
        if obs is None:
            # likelihood(x;obs) = prob. (no spike | x)
            return 1 - self.pop.total_rate(x) * dt
        else:
            # likelihood(x;obs) 
            #       = prob.( spike | x ) * density( obs | x, spike )
            #       = total_rate(x)*dt * [rate(obs;x)*f(obs)/total_rate(x)]
            #       = f(obs) * rate(obs;x) * dt,
            # where f(obs) is the population density.
            # Since f(obs)*dt is independent of x, we return rate(obs;x)
            if self.pop.rate(obs,x) < 0:
                print( "!!!!!", self.pop, obs, x )
            return self.pop.rate(obs,x)
        
    def reset( self ):
        pass
