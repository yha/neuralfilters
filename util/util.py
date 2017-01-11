import numpy as np

class SeedContextManager(object):
    def __init__( self, seed = None ):
         self._state = np.random.get_state()
         np.random.seed( seed )
    def __enter__( self ):
         return None
    def __exit__( self, *args ):
         np.random.set_state(self._state)

def seed( seed = None ):
    return SeedContextManager(seed)

def as_callable( obj ):
    if callable(obj):
        return obj
    else:
        def obj_f( *args, **kwargs ):
            return obj
        return obj_f

# method decorator
def overrides(parent):
    def overrider(func):
        if func.__name__ not in dir(parent):
            raise ValueError( parent.__name__ + 
                              " has no method/property named " +
                              func.__name__ )
        return func
    return overrider
    


import functools

class ClosureWrapper:
    def __init__( self, func, repr, fields ):
        self._func = func
        self._repr = repr
        self._fields = fields
        functools.update_wrapper(self, func)
        for field,value in fields.items():
            setattr( self, field, value )

    def __call__( self, *args, **kwargs ):
        return self._func( *args, **kwargs )
    
    def __repr__( self ):
        return self._repr(self._func)


import inspect
def wrap_closure( repr_str=None, repr=None ):
    """
    Wrap a function returning a closure to make the closure more like a 
    callable class, with access to closure variables.
    
    The definition
        @wrap_closure(repr_str)
        def f(param1,param2,...):
            def g():
                ...
            return g
        
    is like
    
        def f(param1,param2,...)
            def g():
                ...
            return ClosureWrapper( g, repr, dict(param1=param1,...))
    
    The returned ClosureWrapper delegates calls to g, but additionaly has the 
    arguments passed to f accesible as attributes, and has the given repr str 
    or repr function. 
    Note that changes to f's arguments inside f are not visible.
    """
    if repr is None and repr_str is not None:
        repr = lambda self: repr_str
        
    def wrap(f):
        params = inspect.signature(f).parameters
        param_names = list(params)
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            fields = dict(zip(param_names,args))
            fields.update( (p, kwargs.get(p,params[p].default)) 
                            for p in param_names[len(args):] )
            closure = f(**fields)
            return ClosureWrapper( closure, repr, fields )
        return wrapper
    return wrap


import itertools
def scan_params( func, output_shape, params, verbose=False ):
    full_shape = tuple(len(param) for param in params) + output_shape
    output = np.zeros(full_shape)
    for t in itertools.product(*(enumerate(i) for i in params)):
        if verbose: print(t)
        indices = tuple(i[0] for i in t)
        values = tuple(i[1] for i in t)
        output[indices] = func(*values)
    return output
    
def med_dev( data, dim ):
    return np.abs( data - np.median(data,dim,keepdims=True) )
    
def max_rel_dev( data, dim ):
    dev = med_dev(data,dim)
    return np.max( dev / np.median(dev,dim,keepdims=True), dim )
    
def outliers( data, dim, m=10 ):
    dev = med_dev(data,dim)
    return dev > m*np.median(dev,dim,keepdims=True)
    
from numpy import ma
def mask_outliers( data, dim, m=10 ):
    mask = outliers( data, dim ,m )
    return ma.masked_array( data, mask )
