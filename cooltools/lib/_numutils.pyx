from operator import add
from cooler.tools import split
import cython
import numpy as np

cimport numpy as np
from cpython cimport bool


ctypedef unsigned long ulong
ctypedef unsigned int uint
ctypedef unsigned short ushort
ctypedef unsigned char uchar

cdef extern from "stdlib.h":
    long c_libc_random "random"()
    double c_libc_drandom "drand48"()


def logbins(lo, hi, ratio=0, N=0, version=2):
    """Make bins with edges evenly spaced in log-space.

    Parameters
    ----------
    lo, hi : int
        The span of the bins.
    ratio : float
        The target ratio between the upper and the lower edge of each bin.
        Either ratio or N must be specified.
    N : int
        The target number of bins. The resulting number of bins is not guaranteed.
        Either ratio or N must be specified.

    version : int
        version=1 is a legacy mirnylib version

        version=2 fixes ordering of bin gaps for some values, making sure
        that bin gaps are always increasing (2/2/2020)

    """
    lo = int(lo)
    hi = int(hi)
    if ratio != 0:
        if N != 0:
            raise ValueError("Please specify N or ratio")
        N = int(np.log(hi / lo) / np.log(ratio))
    elif N == 0:
        raise ValueError("Please specify N or ratio")
    data10 = np.logspace(np.log10(lo), np.log10(hi), N)
    data10 = np.array(np.rint(data10), dtype=int)
    data10 = np.sort(np.unique(data10))

    if version > 1:
        data10 = np.cumsum(np.r_[data10[0], np.sort(np.diff(data10))])

    assert data10[0] == lo
    assert data10[-1] == hi

    return data10


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def observed_over_expected(matrix, mask=None):
    "Calculates observedOverExpected of any contact map, over regions where mask==1"

    cdef int N = matrix.shape[0]

    cdef np.ndarray[np.double_t, ndim = 2] data = np.array(
        matrix, dtype = np.double, order = "C")
    cdef np.ndarray[np.double_t, ndim = 2] datamask
    cdef bool has_mask = mask is not None
    if has_mask:
        datamask = np.array(mask==1, dtype = np.double, order = "C")

    cdef i, j, offset, lo, hi, n_pixels
    cdef double sum_pixels, mean_pixel
    n_pixels = 0
    sum_pixels = 0

    bins = [0] + list(logbins(1, N, 1.03))
    cdef int n_bins = len(bins)
    for lo, hi in zip(bins[:n_bins-1], bins[1:]):
        sum_pixels = 0
        n_pixels = 0
        for offset in range(lo, hi):
            for j in range(0, N-offset):
                if not has_mask or datamask[offset+j, j]==1:
                    sum_pixels += data[offset+j, j]
                    n_pixels += 1
        #print start, end, count
        mean_pixel = sum_pixels / n_pixels
        if mean_pixel != 0:
            for offset in range(lo, hi):
                for j in range(0, N-offset):
                    if not has_mask or datamask[offset+j, j]==1:
                        data[offset + j, j] /= mean_pixel
                        if offset > 0:
                            data[j, offset+j] /= mean_pixel

    return data


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def iterative_correction_symmetric(
    x, max_iter=1000, ignore_diags = 0, tolerance=1e-5):
    """The main method for correcting DS and SS read data.
    By default does iterative correction, but can perform an M-time correction

    Parameters
    ----------

    x : np.ndarray
        A symmetric matrix to correct.
    max_iter : int
        The maximal number of iterations to take.
    ignore_diags : int
        The number of diagonals to ignore during iterative correction.
    tolerance : float
        If less or equal to zero, will perform max_iter iterations.

    Returns
    -------
    _x : np.ndarray
        Corrected matrix
    totalBias : np.ndarray
        Vector with corrections biases
    report : (bool, int)
        A tuple that reports convergence
        status and used number of iterations
    """
    cdef int N = len(x)

    x = np.array(x, np.double, order = 'C')
    cdef np.ndarray[np.double_t, ndim = 2] _x = x
    cdef np.ndarray[np.double_t, ndim = 1] s
    cdef np.ndarray[np.double_t, ndim = 1] s0
    cdef np.ndarray[np.double_t, ndim = 1] totalBias = np.ones(N, np.double)

    cdef int i, j, iternum
    cdef bool converged = False

    for iternum in range(max_iter):
        s0 = np.sum(_x, axis = 1)

        mask = (s0 == 0)

        s = s0.copy()
        for diag_idx in range(ignore_diags):   #excluding the diagonal
            if diag_idx == 0:
                s -= np.diagonal(_x)
            else:
                dia = np.array(np.diagonal(_x, diag_idx))
                s[diag_idx:] = s[diag_idx:] - dia
                s[:len(s) - diag_idx] = s[:len(s) - diag_idx] - dia

        s = s / np.mean(s[s0 != 0])
        s[s0==0] = 1
        s -= 1
        s *= 0.8
        s += 1
        totalBias *= s

        for i in range(N):
            for j in range(N):
                _x[i, j] = _x[i, j] / (s[i] * s[j])

        if (tolerance > 0) and (np.abs(s - 1).max() < tolerance):
            converged=True
            break

    corr = totalBias[s0!=0].mean()  #mean correction factor
    x  = x * corr * corr #renormalizing everything
    totalBias /= corr
    report = (converged, iternum)

    return x, totalBias, report


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
def fake_cis(
        np.ndarray[np.double_t, ndim=2] data,
        np.ndarray[np.uint8_t, ndim=2] mask
):
    """
    Replace target pixels (e.g. cis areas) of a contact matrix with decoy data
    sampled from source pixels (e.g. trans areas) in the same row or column.

    Operation is done in-place.

    Parameters
    ----------
    data : array of double (n, n)
        Contact matrix.
    mask : array of uint8 (n, n)
        Ternary mask. 0 = source pixels to sample from, 1 = target pixels to
        fill, 2 = invalid pixels to ignore.

    """
    cdef int N = data.shape[0]
    cdef int i, j, row_or_col, src_i, src_j

    for i in range(N):
        for j in range(i, N):

            if mask[i, j] == 1:  # valid target pixel?
                while True:
                    row_or_col = c_libc_random() % 2
                    if row_or_col == 0:
                        src_i = i
                    else:
                        src_i = j
                    src_j = c_libc_random() % N

                    if mask[src_i, src_j] == 0:  # valid source pixel?
                        data[i, j] = data[j, i] = data[src_i, src_j]
                        break


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
