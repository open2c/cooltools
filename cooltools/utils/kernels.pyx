from operator import add
from cooler.tools import split

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _matvec_sparse_symmetric(
        np.ndarray[np.double_t, ndim=1] y, 
        np.ndarray[np.int_t, ndim=1] bin1, 
        np.ndarray[np.int_t, ndim=1] bin2, 
        np.ndarray[np.double_t, ndim=1] values,
        np.ndarray[np.double_t, ndim=1] x):
    """
    Perform matrix vector product A * x for a sparse real 
    symmetric matrix A.
    
    bin1, bin2 : 1D array
        sparse representation of A
    x : 1D array
        input vector
    y : 1D array
        output vector (values are added to this)
    
    """
    cdef int n = bin1.shape[0]
    cdef int ptr, i, j
    cdef double Aij
    
    for ptr in range(n):
        i, j, Aij = bin1[ptr], bin2[ptr], values[ptr]
        y[i] += (Aij * x[j])
        if j > i:
            y[j] += (Aij * x[i])


def _matvec_product(x, chunk):
    pixels = chunk['pixels']
    i = pixels['bin1_id']
    j = pixels['bin2_id']
    v = pixels['count'].astype(float)
    y = np.zeros(len(chunk['bins']['chrom']))
    _matvec_sparse_symmetric(y, i, j, v, x)
    return y


class MatVec(object):
    def __init__(self, clr, chunksize, map):
        self.clr = clr
        self.chunksize = chunksize
        self.map = map
    
    def __call__(self, x, mask):
        n = len(mask)
        
        x_full = np.zeros(n)
        x_full[mask] = x
        
        y_full = (
            split(self.clr, map=self.map, chunksize=self.chunksize)
                .pipe(_matvec_product, x_full)
                .reduce(add, np.zeros(n))
        )
        y = y_full[mask]
        
        return y
