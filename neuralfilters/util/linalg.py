# -*- coding: utf-8 -*-

import scipy
import numpy as np

def is_psd( A, tol=0 ):
    try:
        scipy.linalg.cholesky( A + tol*np.eye(A.shape[0]) )
        return True
    except scipy.linalg.LinAlgError:
        return False