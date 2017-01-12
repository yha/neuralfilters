# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod, abstractproperty

class Process( metaclass = ABCMeta ):
    @abstractmethod
    def step( self, u, dt=1 ):
        """
        Advance the process by dt time units with control/input u, 
        and returns its new state.
        """
    
    @abstractmethod
    def peek( self ):
        """
        Return the current state.
        """
    
    @abstractmethod
    def reset( self ):
        """
        Reset the plant to its initial state.
        Its new state is determined as if it was just constructed.
        """
        
#    # force defining __repr__
#    @abstractmethod
#    def __repr__(self): pass
    

class VectorProcess( Process, metaclass=ABCMeta ):
    """
    A process with fixed-length numpy vectors as input and output.
    """
    
    @abstractproperty
    def N_IN(self):
        """
        Number of inputs to the plant.
        Controls passed to step(..) should be of length N_in.
        """

    @abstractproperty
    def N_OUT(self):
        """
        Size of the plant state.
        The plant outputs vectors of length N_out.
        """

    @abstractmethod
    def step( self, u, dt=1 ):
        """
        Advance the plant by dt time units with control u, 
        and returns its new state.
        The control should be a numpy vector of length self.N_IN.
        The returned state is a vector of length self.N_OUT.
        """
