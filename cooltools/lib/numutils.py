import warnings
from scipy.linalg import toeplitz
import scipy.sparse.linalg
import scipy.interpolate
from scipy.ndimage import zoom, gaussian_filter1d
import numpy as np
import numba
import cooler
from functools import partial

from ._numutils import (
    iterative_correction_symmetric as _iterative_correction_symmetric,
    observed_over_expected as _observed_over_expected,
    fake_cis,
    logbins,
    MatVec,
)


def get_diag(arr, i=0):
    """Get the i-th diagonal of a matrix.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy
    """
    return arr.ravel()[
        max(i, -arr.shape[1] * i) : max(0, (arr.shape[1] - i))
        * arr.shape[1] : arr.shape[1]
        + 1
    ]


def set_diag(arr, x, i=0, copy=False):
    """
    Rewrite the i-th diagonal of a matrix with a value or an array of values.
    Supports 2D arrays, square or rectangular. In-place by default.

    Parameters
    ----------
    arr : 2-D array
        Array whose diagonal is to be filled.
    x : scalar or 1-D vector of correct length
        Values to be written on the diagonal.
    i : int, optional
        Which diagonal to write to. Default is 0.
        Main diagonal is 0; upper diagonals are positive and
        lower diagonals are negative.
    copy : bool, optional
        Return a copy. Diagonal is written in-place if false.
        Default is False.

    Returns
    -------
    Array with diagonal filled.

    Notes
    -----
    Similar to numpy.fill_diagonal, but allows for kth diagonals as well.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy

    """
    if copy:
        arr = arr.copy()
    start = max(i, -arr.shape[1] * i)
    stop = max(0, (arr.shape[1] - i)) * arr.shape[1]
    step = arr.shape[1] + 1
    arr.flat[start:stop:step] = x
    return arr


def fill_diag(arr, x, i=0, copy=True):
    """Identical to set_diag, but returns a copy by default"""
    return set_diag(arr, x, i, copy)


def fill_na(arr, value=0, copy=True):
    """Replaces np.nan entries in an array with the provided value.

    Parameters
    ----------
    arr : np.array

    value : float

    copy : bool, optional
        If True, creates a copy of x, otherwise replaces values in-place.
        By default, True.

    """
    if copy:
        arr = arr.copy()
    arr[np.isnan(arr)] = value
    return arr


def dist_to_mask(mask, side="min"):
    """
    Calculate the distance to the nearest True element of an array.

    Parameters
    ----------
    mask : iterable of bool
        A boolean array.

    side : str
        The side . Accepted values are:
        'left' : calculate the distance to the nearest True value on the left
        'right' : calculate the distance to the nearest True value on the right
        'min' : calculate the distance to the closest True value
        'max' : calculate the distance to the furthest of the two neighbouring True values

    Returns
    -------
    dist: array of int

    Notes
    -----
    The solution is borrowed from https://stackoverflow.com/questions/18196811/cumsum-reset-at-nan

    """
    if side not in ["left", "right", "min", "max"]:
        raise ValueError("side can be `left`, `right`, `min` or `max`")
    if side == "min":
        return np.minimum(
            dist_to_mask(mask, side="left"), dist_to_mask(mask, side="right")
        )
    if side == "max":
        return np.maximum(
            dist_to_mask(mask, side="left"), dist_to_mask(mask, side="right")
        )

    mask = np.asarray(mask)
    if side == "right":
        mask = mask[::-1]

    d = np.diff(np.r_[0.0, np.cumsum(~mask)[mask]])
    v = mask.astype(int).copy()
    v[mask] = d
    dist = (~mask).cumsum() - np.cumsum(v)

    return dist[::-1] if side == "right" else dist


def get_finite(arr):
    """
    Select only finite elements of an array.
    """
    return arr[np.isfinite(arr)]


def fill_inf(arr, pos_value=0, neg_value=0, copy=True):
    """Replaces positive and negative infinity entries in an array with the
    provided values.

    Parameters
    ----------
    arr : np.array

    pos_value : float
        Fill value for np.inf

    neg_value : float
        Fill value for -np.inf

    copy : bool, optional
        If True, creates a copy of x, otherwise replaces values in-place.
        By default, True.

    """
    if copy:
        arr = arr.copy()
    arr[np.isposinf(arr)] = pos_value
    arr[np.isneginf(arr)] = neg_value
    return arr


def fill_nainf(arr, value=0, copy=True):
    """Replaces np.nan and np.inf entries in an array with the provided value.

    Parameters
    ----------
    arr : np.array

    value : float

    copy : bool, optional
        If True, creates a copy of x, otherwise replaces values in-place.
        By default, True.

    Notes
    -----
    Differs from np.nan_to_num in that it replaces np.inf with the same
    number as np.nan.
    """
    if copy:
        arr = arr.copy()
    arr[~np.isfinite(arr)] = value
    return arr


def interp_nan(a_init, pad_zeros=True, method="linear", verbose=False):
    """Linearly interpolate to fill NaN rows and columns in a matrix.
    Also interpolates NaNs in 1D arrays.

    Parameters
    ----------
    a_init : np.array

    pad_zeros : bool, optional
        If True, pads the matrix with zeros to fill NaNs at the edges.
        By default, True.

    method : str, optional
        For 2D: "linear", "nearest", or "splinef2d"
        For 1D: "linear", "nearest", "zero", "slinear", "quadratic", "cubic"

    Returns
    -------
    array with NaNs linearly interpolated

    Notes
    -----
    1D case adapted from: https://stackoverflow.com/a/39592604
    2D case assumes that entire rows or columns are masked & edges to be
    NaN-free, but is much faster than griddata implementation.

    """
    shape = np.shape(a_init)
    if pad_zeros:
        a = np.zeros(tuple(s + 2 for s in shape))
        a[tuple(slice(1, -1) for _ in shape)] = a_init
    else:
        a = np.array(a_init)
    if len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
        a = a.ravel()

    isnan = np.isnan(a)
    if np.sum(isnan) == 0:
        if verbose:
            print("no nans to interpolate")
        return a_init

    if a.ndim == 2:
        if verbose:
            print("interpolating 2D matrix")
        if np.any(isnan[:, 0] | isnan[:, -1]) or np.any(isnan[0, :] | isnan[-1, :]):
            raise ValueError("Edges must not have NaNs")
        # Rows/cols to be considered fully null may have non-NaN diagonals
        # so we'll take the maximum NaN count to identify them
        n_nans_by_row = np.sum(isnan, axis=1)
        n_nans_by_col = np.sum(isnan, axis=0)
        i_inds = np.where(n_nans_by_row < np.max(n_nans_by_row))[0]
        j_inds = np.where(n_nans_by_col < np.max(n_nans_by_col))[0]
        if np.sum(isnan[np.ix_(i_inds, j_inds)]) > 0:
            raise AssertionError("Found additional NaNs")
        interpolator = partial(
            scipy.interpolate.interpn,
            (i_inds, j_inds),
            a[np.ix_(i_inds, j_inds)],
            method=method,
            bounds_error=False,
        )
    else:
        if verbose:
            print("interpolating 1D vector")
        inds = np.arange(len(a))
        interpolator = scipy.interpolate.interp1d(
            inds[~isnan], a[~isnan], kind=method, bounds_error=False
        )

    loc = np.where(isnan)
    a[loc] = interpolator(loc)

    if pad_zeros:
        a = a[tuple(slice(1, -1) for _ in shape)]

    return a


def slice_sorted(arr, lo, hi):
    """Get the subset of a sorted array with values >=lo and <hi.
    A faster version of arr[(arr>=lo) & (arr<hi)]
    """
    return arr[np.searchsorted(arr, lo) : np.searchsorted(arr, hi)]


def MAD(arr, axis=None, has_nans=False):
    """Calculate the Median Absolute Deviation from the median.

    Parameters
    ----------

    arr : np.ndarray
        Input data.

    axis : int
        The axis along which to calculate MAD.

    has_nans : bool
        If True, use the slower NaN-aware method to calculate medians.
    """

    if has_nans:
        return np.nanmedian(np.abs(arr - np.nanmedian(arr, axis)), axis)
    else:
        return np.median(np.abs(arr - np.median(arr, axis)), axis)


def COMED(xs, ys, has_nans=False):
    """Calculate the comedian - the robust median-based counterpart of
    Pearson's r.

    ::

        comedian = median((xs-median(xs))*(ys-median(ys))) / MAD(xs) / MAD(ys)

    Parameters
    ----------
    has_nans : bool
        if True, mask (x,y) pairs with at least one NaN

    Notes
    -----
    Citations: "On MAD and comedians" by Michael Falk (1997),
    "Robust Estimation of the Correlation Coefficient: An Attempt of Survey"
    by Georgy Shevlyakov and Pavel Smirnov (2011)

    """

    if has_nans:
        mask = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

    med_x = np.median(xs)
    med_y = np.median(ys)
    comedian = np.median((xs - med_x) * (ys - med_y)) / MAD(xs) / MAD(ys)

    return comedian


def normalize_score(arr, norm="z", axis=None, has_nans=True):
    """Normalize an array by subtracting the first moment and dividing the
    residual by the second.

    Parameters
    ----------

    arr : np.ndarray
        Input data.

    norm : str
        The type of normalization.
        'z' - report z-scores,
        norm_arr = (arr - mean(arr)) / std(arr)

        'mad' - report deviations from the median in units of MAD
        (Median Absolute Deviation from the median),
        norm_arr = (arr - median(arr)) / MAD(arr)

        'madz' - report robust z-scores, i.e. estimate the mean as
        the median and the standard error as MAD / 0.67499,
        norm_arr = (arr - median(arr)) / MAD(arr) * 0.67499

    axis : int
        The axis along which to calculate the normalization parameters.

    has_nans : bool
        If True, use slower NaN-aware methods to calculate the
        normalization parameters.

    """

    norm_arr = np.copy(arr)
    norm = norm.lower()

    if norm == "z":
        if has_nans:
            norm_arr -= np.nanmean(norm_arr, axis=axis)
            norm_arr /= np.nanstd(norm_arr, axis=axis)
        else:
            norm_arr -= np.mean(norm_arr, axis=axis)
            norm_arr /= np.std(norm_arr, axis=axis)

    elif norm == "mad" or norm == "madz":
        if has_nans:
            norm_arr -= np.nanmedian(norm_arr, axis=axis)
        else:
            norm_arr -= np.median(norm_arr, axis=axis)
        norm_arr /= MAD(norm_arr, axis=axis, has_nans=has_nans)
        if norm == "madz":
            norm_arr *= 0.67449
    else:
        raise ValueError("Unknown norm type: {}".format(norm))

    return norm_arr


def stochastic_sd(arr, n=10000, seed=0):
    """Estimate the standard deviation of an array by considering only the
    subset of its elements.

    Parameters
    ----------
    n : int
        The number of elements to consider. If the array contains fewer elements,
        use all.

    seed : int
        The seed for the random number generator.
    """
    arr = np.asarray(arr)
    if arr.size < n:
        return np.sqrt(arr.var())
    else:
        return np.sqrt(
            np.random.RandomState(seed).choice(arr.flat, n, replace=True).var()
        )


def is_symmetric(mat):
    """
    Check if a matrix is symmetric.
    """

    maxDiff = np.abs(mat - mat.T).max()
    return maxDiff < stochastic_sd(mat) * 1e-7 + 1e-5


def get_eig(mat, n=3, mask_zero_rows=False, subtract_mean=False, divide_by_mean=False):
    """Perform an eigenvector decomposition.

    Parameters
    ----------
    mat : np.ndarray
        A square matrix, must not contain nans, infs or zero rows.

    n : int
        The number of eigenvectors to return. Output is backfilled with NaNs
        when n exceeds the size of the input matrix.

    mask_zero_rows : bool
        If True, mask empty rows/columns before eigenvector decomposition.
        Works only with symmetric matrices.

    subtract_mean : bool
        If True, subtract the mean from the matrix.

    divide_by_mean : bool
        If True, divide the matrix by its mean.

    Returns
    -------
    eigvecs : np.ndarray
        An array of eigenvectors (in rows), sorted by a decreasing absolute
        eigenvalue.

    eigvals : np.ndarray
        An array of sorted eigenvalues.

    """
    symmetric = is_symmetric(mat)
    if (
        symmetric
        and np.sum(np.sum(np.abs(mat), axis=0) == 0) > 0
        and not mask_zero_rows
    ):
        warnings.warn(
            "The matrix contains empty rows/columns and is symmetric. "
            "Mask the empty rows with remove_zeros=True"
        )
    # size of the input matrix
    n_rows = mat.shape[0]
    if n > n_rows:
        warnings.warn(
            "Number n of requested eigenvalues is larger than the matrix size."
        )

    if mask_zero_rows:
        if not is_symmetric(mat):
            raise ValueError("The input matrix must be symmetric!")

        mask = np.sum(np.abs(mat), axis=0) != 0
        mat_collapsed = mat[mask, :][:, mask]
        eigvecs_collapsed, eigvals = get_eig(
            mat_collapsed,
            n=n,
            mask_zero_rows=False,
            subtract_mean=subtract_mean,
            divide_by_mean=divide_by_mean,
        )
        eigvecs = np.full((n, n_rows), np.nan)
        for i in range(n):
            eigvecs[i][mask] = eigvecs_collapsed[i]

        return eigvecs, eigvals
    else:
        mat = mat.astype(np.float64, copy=True)  # make a copy, ensure float

        # prepare NaN-filled arrays for output in case we requested
        # more eigenvalues that can be computed (eigs/eigsh k limits)
        eigvals = np.full(n, np.nan)
        eigvecs = np.full((n, n_rows), np.nan)

        mean = np.mean(mat)

        if subtract_mean:
            mat -= mean
        if divide_by_mean:
            mat /= mean

        if symmetric:
            # adjust requested number of eigvals for "eigsh"
            _n = n if n < n_rows else (n_rows - 1)
            _eigvals, _eigvecs = scipy.sparse.linalg.eigsh(mat, _n)
        else:
            # adjust requested number of eigvals for "eigs"
            _n = n if n < (n_rows - 1) else (n_rows - 2)
            _eigvals, _eigvecs = scipy.sparse.linalg.eigs(mat, _n)

        # reorder according to eigvals and copy into output arrays
        order = np.argsort(-np.abs(_eigvals))
        eigvals[:_n] = _eigvals[order]
        eigvecs[:_n, :] = _eigvecs.T[order]

        return eigvecs, eigvals


@numba.njit
def _logbins_numba(lo, hi, ratio=0, N=0, prepend_zero=False):
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

    """
    lo = int(lo)
    hi = int(hi)
    if ratio != 0:
        if N != 0:
            raise ValueError("Please specify N or ratio")
        N = np.log(hi / lo) / np.log(ratio)
    elif N == 0:
        raise ValueError("Please specify N or ratio")
    data10 = 10 ** np.linspace(np.log10(lo), np.log10(hi), int(N))
    data10 = np.rint(data10)
    data10_int = np.sort(np.unique(data10)).astype(np.int_)
    assert data10_int[0] == lo
    assert data10_int[-1] == hi
    if prepend_zero:
        data10_int = np.concatenate((np.array([0]), data10_int))
    return data10_int


@numba.njit
def observed_over_expected(
    matrix, mask=np.empty(shape=(0), dtype=np.bool_), dist_bin_edge_ratio=1.03
):
    """
    Normalize the contact matrix for distance-dependent contact decay.
    The diagonals of the matrix, corresponding to contacts between loci pairs
    with a fixed distance, are grouped into exponentially growing bins of
    distances; the diagonals from each bin are normalized by their average value.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D symmetric matrix of contact frequencies.
    mask : np.ndarray
        A 1D or 2D mask of valid data.
        If 1D, it is interpreted as a mask of "good" bins.
        If 2D, it is interpreted as a mask of "good" pixels.
    dist_bin_edge_ratio : float
        The ratio of the largest and the shortest distance in each distance bin.

    Returns
    -------
    OE : np.ndarray
        The diagonal-normalized matrix of contact frequencies.
    dist_bins : np.ndarray
        The edges of the distance bins used to calculate average
        distance-dependent contact frequency.
    sum_pixels : np.ndarray
        The sum of contact frequencies in each distance bin.
    n_pixels : np.ndarray
        The total number of valid pixels in each distance bin.

    """
    N = matrix.shape[0]

    mask2d = np.empty(shape=(0, 0), dtype=np.bool_)
    if mask.ndim == 1:
        if mask.size > 0:
            mask2d = mask.reshape((1, -1)) * mask.reshape((-1, 1))
    elif mask.ndim == 2:
        # Numba expects mask to be a 1d array, so we need to hint
        # that it is now a 2d array
        mask2d = mask.reshape((int(np.sqrt(mask.size)), int(np.sqrt(mask.size))))
    else:
        raise ValueError("The mask must be either 1D or 2D.")

    data = np.copy(matrix).astype(np.float64)

    has_mask = mask2d.size > 0
    dist_bins = _logbins_numba(1, N, dist_bin_edge_ratio)
    dist_bins = np.concatenate((np.array([0]), dist_bins))
    n_pixels_arr = np.zeros_like(dist_bins[1:])
    sum_pixels_arr = np.zeros_like(dist_bins[1:], dtype=np.float64)

    bin_idx, n_pixels, sum_pixels = 0, 0, 0

    for bin_idx, lo, hi in zip(
        range(len(dist_bins) - 1), dist_bins[:-1], dist_bins[1:]
    ):
        sum_pixels = 0
        n_pixels = 0
        for offset in range(lo, hi):
            for j in range(0, N - offset):
                if not has_mask or mask2d[offset + j, j]:
                    sum_pixels += data[offset + j, j]
                    n_pixels += 1

        n_pixels_arr[bin_idx] = n_pixels
        sum_pixels_arr[bin_idx] = sum_pixels

        if n_pixels == 0:
            continue
        mean_pixel = sum_pixels / n_pixels
        if mean_pixel == 0:
            continue

        for offset in range(lo, hi):
            for j in range(0, N - offset):
                if not has_mask or mask2d[offset + j, j]:

                    data[offset + j, j] /= mean_pixel
                    if offset > 0:
                        data[j, offset + j] /= mean_pixel

    return data, dist_bins, sum_pixels_arr, n_pixels_arr


@numba.jit  # (nopython=True)
def iterative_correction_symmetric(
    x, max_iter=1000, ignore_diags=0, tol=1e-5, verbose=False
):
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
    tol : float
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
    N = len(x)

    _x = x.copy()
    if ignore_diags > 0:
        for d in range(0, ignore_diags):
            for j in range(0, N - d):
                _x[j, j + d] = 0
                _x[j + d, j] = 0
    totalBias = np.ones(N, np.double)

    converged = False

    iternum = 0
    mask = np.sum(_x, axis=1) == 0
    for iternum in range(max_iter):
        s = np.sum(_x, axis=1)

        mask = s == 0

        s = s / np.mean(s[~mask])
        s[mask] = 1
        s -= 1
        s *= 0.8
        s += 1
        totalBias *= s

        for i in range(N):
            for j in range(N):
                _x[i, j] /= s[i] * s[j]

        crit = np.var(s)  # np.abs(s - 1).max()
        if verbose:
            print(crit)

        if (tol > 0) and (crit < tol):
            converged = True
            break

    corr = totalBias[~mask].mean()  # mean correction factor
    _x = _x * corr * corr  # renormalizing everything
    totalBias /= corr
    report = (converged, iternum)

    return _x, totalBias, report


@numba.jit  # (nopython=True)
def iterative_correction_asymmetric(x, max_iter=1000, tol=1e-5, verbose=False):
    """Adapted from iterative_correction_symmetric

    Parameters
    ----------
    x : np.ndarray
        An asymmetric matrix to correct.
    max_iter : int
        The maximal number of iterations to take.
    ignore_diags : int
        The number of diagonals to ignore during iterative correction.
    tol : float
        If less or equal to zero, will perform max_iter iterations.

    Returns
    -------
    _x : np.ndarray
        Corrected matrix
    totalBias : np.ndarray
        Vector with corrections biases for columns
    totalBias2 : np.ndarray
        Vector with corrections biases for rows
    report : (bool, int)
        A tuple that reports convergence
        status and used number of iterations
    """
    N2, N = x.shape
    _x = x.copy()
    totalBias = np.ones(N, np.double)
    totalBias2 = np.ones(N2, np.double)
    iternum = 0
    mask = np.sum(_x, axis=0) == 0
    mask2 = np.sum(_x, axis=1) == 0

    for iternum in range(max_iter):
        s = np.sum(_x, axis=0)
        mask = s == 0
        s = s / np.mean(s[~mask])
        s[mask] = 1.0
        s -= 1.0
        s *= 0.8
        s += 1.0
        totalBias *= s

        s2 = np.sum(_x, axis=1)
        mask2 = s2 == 0
        s2 = s2 / np.mean(s2[~mask2])
        s2[mask2] = 1.0
        s2 -= 1.0
        s2 *= 0.8
        s2 += 1.0
        totalBias2 *= s2

        # _x = _x / s[:, None]  / s[None,:]
        # an explicit cycle is 2x faster here
        for i in range(N2):
            for j in range(N):
                _x[i, j] /= s[j] * s2[i]

        crit = np.var(s)  # np.abs(s - 1).max()
        crit2 = np.var(s2)  # np.abs(s - 1).max()
        if verbose:
            print(crit)

        if (tol > 0) and (crit < tol) and (crit2 < tol):
            converged = True
            break

    corr = totalBias[~mask].mean()  # mean correction factor
    corr2 = totalBias2[~mask2].mean()  # mean correction factor
    _x = _x * corr * corr2  # renormalizing everything
    totalBias /= corr
    totalBias2 /= corr2
    report = (converged, iternum)

    return _x, totalBias, totalBias2, report


class LazyToeplitz(cooler.core._IndexingMixin):
    """
    A Toeplitz matrix can be represented with one row and one column.
    This lazy toeplitz object supports slice querying to construct dense
    matrices on the fly.

    """

    def __init__(self, c, r=None):
        if r is None:
            r = c
        elif c[0] != r[0]:
            raise ValueError("First element of `c` and `r` should match")
        self._c = c
        self._r = r

    @property
    def shape(self):
        return (len(self._c), len(self._r))

    def __getitem__(self, key):
        slc0, slc1 = self._unpack_index(key)
        i0, i1 = self._process_slice(slc0, self.shape[0])
        j0, j1 = self._process_slice(slc1, self.shape[1])
        C, R = self._c, self._r

        # symmetric query
        if (i0 == j0) and (i1 == j1):
            c = C[0 : (i1 - i0)]
            r = R[0 : (j1 - j0)]

        # asymmetric query
        else:
            transpose = False
            # tril
            if j0 < i0 or (i0 == j0 and i1 < j1):
                # tranpose the matrix, query,
                # then transpose the result
                i0, i1, j0, j1 = j0, j1, i0, i1
                C, R = R, C
                transpose = True

            c = np.r_[R[(j0 - i0) : max(0, j0 - i1) : -1], C[0 : max(0, i1 - j0)]]
            r = R[(j0 - i0) : (j1 - i0)]

            if transpose:
                c, r = r, c

        return toeplitz(c, r)


def get_kernel(w, p, ktype):
    """
    Return typical kernels given size parameteres w, p,and kernel type.

    Parameters
    ----------
    w : int
        Outer kernel size (actually half of it).
    p : int
        Inner kernel size (half of it).
    ktype : str
        Name of the kernel type, could be one of the following: 'donut',
        'vertical', 'horizontal', 'lowleft', 'upright'.

    Returns
    -------
    kernel : ndarray
        A square matrix of int type filled with 1 and 0, according to the
        kernel type.

    """
    width = 2 * w + 1
    kernel = np.ones((width, width), dtype=np.int64)
    # mesh grid:
    y, x = np.ogrid[-w : w + 1, -w : w + 1]

    if ktype == "donut":
        # mask inner pXp square:
        mask = (((-p) <= x) & (x <= p)) & (((-p) <= y) & (y <= p))
        # mask vertical and horizontal
        # lines of width 1 pixel:
        mask += (x == 0) | (y == 0)
        # they are all 0:
        kernel[mask] = 0
    elif ktype == "vertical":
        # mask outside of vertical line
        # of width 3:
        mask = ((-1 > x) | (x > 1)) & ((y >= -w))
        # mask inner pXp square:
        mask += ((-p <= x) & (x <= p)) & ((-p <= y) & (y <= p))
        # kernel masked:
        kernel[mask] = 0
    elif ktype == "horizontal":
        # mask outside of horizontal line
        # of width 3:
        mask = ((-1 > y) | (y > 1)) & ((x >= -w))
        # mask inner pXp square:
        mask += ((-p <= x) & (x <= p)) & ((-p <= y) & (y <= p))
        # kernel masked:
        kernel[mask] = 0
    # ACHTUNG!!! UPRIGHT AND LOWLEFT ARE SWITCHED ...
    # IT SEEMS FOR UNKNOWN REASON THAT IT SHOULD
    # BE THAT WAY ...
    # OR IT'S A MISTAKE IN hIccups AS WELL ...
    elif ktype == "upright":
        # mask inner pXp square:
        mask = ((x >= -p)) & ((y <= p))
        mask += x >= 0
        mask += y <= 0
        # kernel masked:
        kernel[mask] = 0
    elif ktype == "lowleft":
        # mask inner pXp square:
        mask = ((x >= -p)) & ((y <= p))
        mask += x >= 0
        mask += y <= 0
        # reflect that mask to
        # make it upper-right:
        mask = mask[::-1, ::-1]
        # kernel masked:
        kernel[mask] = 0
    else:
        raise ValueError("Kernel-type {} has not been implemented yet".format(ktype))
    return kernel


def coarsen(reduction, x, axes, trim_excess=False):
    """
    Coarsen an array by applying reduction to fixed size neighborhoods.
    Adapted from `dask.array.coarsen` to work on regular numpy arrays.

    Parameters
    ----------
    reduction : function
        Function like np.sum, np.mean, etc...
    x : np.ndarray
        Array to be coarsened
    axes : dict
        Mapping of axis to coarsening factor
    trim_excess : bool, optional
        Remove excess elements. Default is False.

    Examples
    --------
    Provide dictionary of scale per dimension

    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> coarsen(np.sum, x, {0: 2})
    array([ 3,  7, 11])

    >>> coarsen(np.max, x, {0: 3})
    array([3, 6])

    >>> x = np.arange(24).reshape((4, 6))
    >>> x
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23]])

    >>> coarsen(np.min, x, {0: 2, 1: 3})
    array([[ 0,  3],
           [12, 15]])

    See also
    --------
    dask.array.coarsen

    """
    # Insert singleton dimensions if they don't exist already
    for i in range(x.ndim):
        if i not in axes:
            axes[i] = 1

    if trim_excess:
        ind = tuple(
            slice(0, -(d % axes[i])) if d % axes[i] else slice(None, None)
            for i, d in enumerate(x.shape)
        )
        x = x[ind]

    # (10, 10) -> (5, 2, 5, 2)
    newdims = [(x.shape[i] // axes[i], axes[i]) for i in range(x.ndim)]
    newshape = tuple(np.concatenate(newdims))
    reduction_axes = tuple(range(1, x.ndim * 2, 2))
    return reduction(x.reshape(newshape), axis=reduction_axes)


def smooth(y, box_pts):
    try:
        from astropy.convolution import convolve
    except ImportError:
        raise ImportError("The astropy module is required to use this function")
    box = np.ones(box_pts) / box_pts
    # also: None, fill, wrap, extend
    y_smooth = convolve(y, box, boundary="extend")
    return y_smooth


def infer_mask2D(mat):
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("matix must be symmetric!")
    fill_na(mat, value=0, copy=False)
    sum0 = np.sum(mat, axis=0) > 0
    mask = sum0[:, None] * sum0[None, :]
    return mask


def remove_good_singletons(mat, mask=None, returnMask=False):
    mat = mat.copy()
    if mask is None:
        mask = infer_mask2D(mat)
    badBins = np.sum(mask, axis=0) == 0
    goodBins = badBins == 0
    good_singletons = (goodBins * smooth(goodBins, 3)) == 1 / 3
    mat[good_singletons, :] = np.nan
    mat[:, good_singletons] = np.nan
    mask[good_singletons, :] = 0
    mask[:, good_singletons] = 0
    if returnMask:
        return mat, mask
    else:
        return mat


def interpolate_bad_singletons(
    mat, mask=None, fillDiagonal=True, returnMask=False, secondPass=True, verbose=False
):
    """Interpolate singleton missing bins for visualization

    Examples
    --------
    >>> ax = plt.subplot(121)
    >>> maxval =  np.log(np.nanmean(np.diag(mat,3))*2 )
    >>> plt.matshow(np.log(mat)), vmax=maxval, fignum=False)
    >>> plt.set_cmap('fall');
    >>> plt.subplot(122, sharex=ax, sharey=ax)
    >>> plt.matshow(
    ...     np.log(interpolate_bad_singletons(remove_good_singletons(mat))),
    ...     vmax=maxval,
    ...     fignum=False
    ... )
    >>> plt.set_cmap('fall');
    >>> plt.show()
    """
    mat = mat.copy()
    if mask is None:
        mask = infer_mask2D(mat)
    antimask = ~mask
    badBins = np.sum(mask, axis=0) == 0
    singletons = (badBins * smooth(badBins == 0, 3)) > 1 / 3
    bb_minus_singletons = (badBins.astype("int8") - singletons.astype("int8")).astype(
        "bool"
    )

    mat[antimask] = np.nan
    locs = np.zeros(np.shape(mat))
    locs[singletons, :] = 1
    locs[:, singletons] = 1
    locs[bb_minus_singletons, :] = 0
    locs[:, bb_minus_singletons] = 0
    locs = np.nonzero(locs)  # np.isnan(mat))
    interpvals = np.zeros(np.shape(mat))
    if verbose:
        print("initial pass to interpolate:", len(locs[0]))
    for loc in zip(locs[0], locs[1]):
        i, j = loc
        if loc[0] > loc[1]:
            if (
                (loc[0] > 0)
                and (loc[1] > 0)
                and (loc[0] < len(mat) - 1)
                and (loc[1] < len(mat) - 1)
            ):
                interpvals[i, j] = np.nanmean([mat[i - 1, j - 1], mat[i + 1, j + 1]])
            elif loc[0] == 0:
                interpvals[i, j] = np.nanmean([mat[i, j - 1], mat[i, j + 1]])
            elif loc[1] == 0:
                interpvals[i, j] = np.nanmean([mat[i - 1, j], mat[i + 1, j]])
            elif loc[0] == (len(mat) - 1):
                interpvals[i, j] = np.nanmean([mat[i, j - 1], mat[i, j + 1]])
            elif loc[1] == (len(mat) - 1):
                interpvals[i, j] = np.nanmean([mat[i - 1, j], mat[i + 1, j]])
    interpvals = interpvals + interpvals.T
    mat[locs] = interpvals[locs]
    mask[locs] = 1

    if secondPass:
        locs = np.nonzero(np.isnan(interpvals))  # np.isnan(mat))
        interpvals2 = np.zeros(np.shape(mat))
        if verbose:
            print("still remaining: ", len(locs[0]))
        for loc in zip(locs[0], locs[1]):
            i, j = loc
            if loc[0] > loc[1]:
                if (
                    (loc[0] > 0)
                    and (loc[1] > 0)
                    and (loc[0] < len(mat) - 1)
                    and (loc[1] < len(mat) - 1)
                ):
                    interpvals2[i, j] = np.nanmean(
                        [mat[i - 1, j - 1], mat[i + 1, j + 1]]
                    )
                elif loc[0] == 0:
                    interpvals2[i, j] = np.nanmean([mat[i, j - 1], mat[i, j + 1]])
                elif loc[1] == 0:
                    interpvals2[i, j] = np.nanmean([mat[i - 1, j], mat[i + 1, j]])
                elif loc[0] == (len(mat) - 1):
                    interpvals2[i, j] = np.nanmean([mat[i, j - 1], mat[i, j + 1]])
                elif loc[1] == (len(mat) - 1):
                    interpvals2[i, j] = np.nanmean([mat[i - 1, j], mat[i + 1, j]])
        interpvals2 = interpvals2 + interpvals2.T
        mat[locs] = interpvals2[locs]
        mask[locs] = 1

    if fillDiagonal:
        for i in range(-1, 2):
            set_diag(mat, np.nan, i=i, copy=False)
        for i in range(-1, 2):
            set_diag(mask, 0, i=i, copy=False)

    if returnMask:
        return mat, mask
    else:
        return mat


def zoom_array(
    in_array,
    final_shape,
    same_sum=False,
    zoom_function=partial(zoom, order=1),
    **zoom_kwargs
):
    """Rescale an array or image.

    Normally, one can use scipy.ndimage.zoom to do array/image rescaling.
    However, scipy.ndimage.zoom does not coarsegrain images well. It basically
    takes nearest neighbor, rather than averaging all the pixels, when
    coarsegraining arrays. This increases noise. Photoshop doesn't do that, and
    performs some smart interpolation-averaging instead.

    If you were to coarsegrain an array by an integer factor, e.g. 100x100 ->
    25x25, you just need to do block-averaging, that's easy, and it reduces
    noise. But what if you want to coarsegrain 100x100 -> 30x30?

    Then my friend you are in trouble. But this function will help you. This
    function will blow up your 100x100 array to a 120x120 array using
    scipy.ndimage zoom Then it will coarsegrain a 120x120 array by
    block-averaging in 4x4 chunks.

    It will do it independently for each dimension, so if you want a 100x100
    array to become a 60x120 array, it will blow up the first and the second
    dimension to 120, and then block-average only the first dimension.

    (Copied from mirnylib.numutils)

    Parameters
    ----------
    in_array : ndarray
        n-dimensional numpy array (1D also works)
    final_shape : shape tuple
        resulting shape of an array
    same_sum : bool, optional
        Preserve a sum of the array, rather than values. By default, values
        are preserved
    zoom_function : callable
        By default, scipy.ndimage.zoom with order=1. You can plug your own.
    **zoom_kwargs :
        Options to pass to zoomFunction.

    Returns
    -------
    rescaled : ndarray
        Rescaled version of in_array

    """
    in_array = np.asarray(in_array, dtype=np.double)
    in_shape = in_array.shape
    assert len(in_shape) == len(final_shape)
    mults = []  # multipliers for the final coarsegraining
    for i in range(len(in_shape)):
        if final_shape[i] < in_shape[i]:
            mults.append(int(np.ceil(in_shape[i] / final_shape[i])))
        else:
            mults.append(1)
    # shape to which to blow up
    temp_shape = tuple([i * j for i, j in zip(final_shape, mults)])

    # stupid zoom doesn't accept the final shape. Carefully crafting the
    # multipliers to make sure that it will work.
    zoom_multipliers = np.array(temp_shape) / np.array(in_shape) + 0.0000001
    assert zoom_multipliers.min() >= 1

    # applying scipy.ndimage.zoom
    rescaled = zoom_function(in_array, zoom_multipliers, **zoom_kwargs)

    for ind, mult in enumerate(mults):
        if mult != 1:
            sh = list(rescaled.shape)
            assert sh[ind] % mult == 0
            newshape = sh[:ind] + [sh[ind] // mult, mult] + sh[ind + 1 :]
            rescaled.shape = newshape
            rescaled = np.mean(rescaled, axis=ind + 1)
    assert rescaled.shape == final_shape

    if same_sum:
        extra_size = np.prod(final_shape) / np.prod(in_shape)
        rescaled /= extra_size
    return rescaled


def adaptive_coarsegrain(ar, countar, cutoff=5, max_levels=8, min_shape=8):
    """
    Adaptively coarsegrain a Hi-C matrix based on local neighborhood pooling
    of counts.

    Parameters
    ----------
    ar : array_like, shape (n, n)
        A square Hi-C matrix to coarsegrain. Usually this would be a balanced
        matrix.

    countar : array_like, shape (n, n)
        The raw count matrix for the same area. Has to be the same shape as the
        Hi-C matrix.

    cutoff : float, optional
        A minimum number of raw counts per pixel required to stop 2x2 pooling.
        Larger cutoff values would lead to a more coarse-grained, but smoother
        map. 3 is a good default value for display purposes, could be lowered
        to 1 or 2 to make the map less pixelated. Setting it to 1 will only
        ensure there are no zeros in the map.

    max_levels : int, optional
        How many levels of coarsening to perform. It is safe to keep this
        number large as very coarsened map will have large counts and no
        substitutions would be made at coarser levels.
    min_shape : int, optional
        Stop coarsegraining when coarsegrained array shape is less than that.

    Returns
    -------
    Smoothed array, shape (n, n)

    Notes
    -----
    The algorithm works as follows:

    First, it pads an array with NaNs to the nearest power of two. Second, it
    coarsens the array in powers of two until the size is less than minshape.

    Third, it starts with the most coarsened array, and goes one level up.
    It looks at all 4 pixels that make each pixel in the second-to-last
    coarsened array. If the raw counts for any valid (non-NaN) pixel are less
    than ``cutoff``, it replaces the values of the valid (4 or less) pixels
    with the NaN-aware average. It is then applied to the next
    (less coarsened) level until it reaches the original resolution.

    In the resulting matrix, there are guaranteed to be no zeros, unless very
    large zero-only areas were provided such that zeros were produced
    ``max_levels`` times when coarsening.

    Examples
    --------
    >>> c = cooler.Cooler("/path/to/some/cooler/at/about/2000bp/resolution")

    >>> # sample region of about 6000x6000
    >>> mat = c.matrix(balance=True).fetch("chr1:10000000-22000000")
    >>> mat_raw = c.matrix(balance=False).fetch("chr1:10000000-22000000")
    >>> mat_cg = adaptive_coarsegrain(mat, mat_raw)

    >>> plt.figure(figsize=(16,7))
    >>> ax = plt.subplot(121)
    >>> plt.imshow(np.log(mat), vmax=-3)
    >>> plt.colorbar()
    >>> plt.subplot(122, sharex=ax, sharey=ax)
    >>> plt.imshow(np.log(mat_cg), vmax=-3)
    >>> plt.colorbar()

    """

    def _coarsen(ar, operation=np.sum):
        """Coarsegrains an array by a factor of 2"""
        M = ar.shape[0] // 2
        newar = np.reshape(ar, (M, 2, M, 2))
        cg = operation(newar, axis=1)
        cg = operation(cg, axis=2)
        return cg

    def _expand(ar, counts=None):
        """
        Performs an inverse of nancoarsen
        """
        N = ar.shape[0] * 2
        newar = np.zeros((N, N))
        newar[::2, ::2] = ar
        newar[1::2, ::2] = ar
        newar[::2, 1::2] = ar
        newar[1::2, 1::2] = ar
        return newar

    # defining arrays, making sure they are floats
    ar = np.asarray(ar, float)
    countar = np.asarray(countar, float)

    # TODO: change this to the nearest shape correctly counting the smallest
    # shape the algorithm will reach
    Norig = ar.shape[0]
    Nlog = np.log2(Norig)
    if not np.allclose(Nlog, np.rint(Nlog)):
        newN = np.int(2 ** np.ceil(Nlog))  # next power-of-two sized matrix
        newar = np.empty((newN, newN), dtype=float)  # fitting things in there
        newar[:] = np.nan
        newcountar = np.zeros((newN, newN), dtype=float)
        newar[:Norig, :Norig] = ar
        newcountar[:Norig, :Norig] = countar
        ar = newar
        countar = newcountar

    armask = np.isfinite(ar)  # mask of "valid" elements
    countar[~armask] = 0
    ar[~armask] = 0

    assert np.isfinite(countar).all()
    assert countar.shape == ar.shape

    # We will be working with three arrays.
    ar_cg = [ar]  # actual Hi-C data
    countar_cg = [countar]  # counts contributing to Hi-C data (raw Hi-C reads)
    armask_cg = [armask]  # mask of "valid" pixels of the heatmap

    # 1. Forward pass: coarsegrain all 3 arrays
    for i in range(max_levels):
        if countar_cg[-1].shape[0] > min_shape:
            countar_cg.append(_coarsen(countar_cg[-1]))
            armask_cg.append(_coarsen(armask_cg[-1]))
            ar_cg.append(_coarsen(ar_cg[-1]))

    # Get the most coarsegrained array
    ar_cur = ar_cg.pop()
    countar_cur = countar_cg.pop()
    armask_cur = armask_cg.pop()

    # 2. Reverse pass: replace values starting with most coarsegrained array
    # We have 4 pixels that were coarsegrained to one pixel.
    # Let V be the array of values (ar), and C be the array of counts of
    # valid pixels. Then the coarsegrained values and valid pixel counts
    # are:
    # V_{cg} = V_{0,0} + V_{0,1} + V_{1,0} + V_{1,1}
    # C_{cg} = C_{0,0} + C_{0,1} + C_{1,0} + C_{1,1}
    # The average value at the coarser level is V_{cg} / C_{cg}
    # The average value at the finer level is V_{0,0} / C_{0,0}, etc.
    #
    # We would replace 4 values with the average if counts for either of the
    # 4 values are less than cutoff. To this end, we perform nanmin of raw
    # Hi-C counts in each 4 pixels
    # Because if counts are 0 due to this pixel being invalid - it's fine.
    # But if they are 0 in a valid pixel - we replace this pixel.
    # If we decide to replace the current 2x2 square with coarsegrained
    # values, we need to make it produce the same average value
    # To this end, we would replace V_{0,0} with V_{cg} * C_{0,0} / C_{cg} and
    # so on.
    for i in range(len(countar_cg)):
        ar_next = ar_cg.pop()
        countar_next = countar_cg.pop()
        armask_next = armask_cg.pop()

        # obtain current "average" value by dividing sum by the # of valid pixels
        val_cur = ar_cur / armask_cur
        # expand it so that it is the same shape as the previous level
        val_exp = _expand(val_cur)
        # create array of substitutions: multiply average value by counts
        addar_exp = val_exp * armask_next

        # make a copy of the raw Hi-C array at current level
        countar_next_mask = np.array(countar_next)
        countar_next_mask[armask_next == 0] = np.nan  # fill nans
        countar_exp = _expand(_coarsen(countar_next, operation=np.nanmin))

        curmask = countar_exp < cutoff  # replacement mask
        ar_next[curmask] = addar_exp[curmask]  # procedure of replacement
        ar_next[armask_next == 0] = 0  # now setting zeros at invalid pixels

        # prepare for the next level
        ar_cur = ar_next
        countar_cur = countar_next
        armask_cur = armask_next

    ar_next[armask_next == 0] = np.nan
    ar_next = ar_next[:Norig, :Norig]

    return ar_next


def robust_gauss_filter(
    ar, sigma=2, functon=gaussian_filter1d, kwargs=None
):
    """
    Implements an edge-handling mode for gaussian filter that basically ignores
    the edge, and also handles NaNs.

    Parameters
    ----------
    ar : array-like
        Input array
    sigma : float
        sigma to be passed to the filter
    function : callable
        Filter to use. Default is gauusian_filter1d
    kwargs : dict
        Additional args to pass to the filter. Default:None

    Notes
    -----
    Available edge-handling modes in ndimage.filters attempt to somehow
    "extrapolate" the edge value and then  apply the filter (see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve.html).
     That's likely because convolve uses fast fourier transform, which requires
    the kernel to be constant.  Here we design a better edge-handling for the
    gaussian smoothing.

    In a gaussian-filtered array, a pixel away from the edge is a mean of nearby
    pixels with gaussian weights.  With this mode, pixels near start/end are
    also a mean of nearby pixels with gaussian weights. That's it.  If we
    encounter NANs, we also simply ignore them, following the same definition:
    mean of nearby valid pixels.  Yes, it rases the weights for the first/last
    pixels, because now only a part of the whole gaussian is being used  (up to
    1/2 for the first/last pixel and large sigma). But it preserves the "mean of
    nearby pixels" definition.  It is different from padding with zeros (it
    would drag the first pixel down to be more like zero).  It is also different
    from "nearest" - that gives too much weight to the first/last pixel.

    To achieve this smoothing, we preform regular gaussian smoothing using
    mode="constant"  (pad with zeros). Then we takes an array of valid pixels
    and smooth it the same way.  This calculates how many "average valid pixels"
    contributed to each point of a smoothed array.  Dividing one by the other
    achieves the desired result.
    """
    if kwargs is None:
        kwargs = {}
    ar = np.asarray(ar, dtype=float)
    mask = np.isfinite(ar)
    ar[~mask] = 0
    a = functon(ar, sigma=sigma, mode="constant", **kwargs)
    b = functon(1.0 * mask, sigma=sigma, mode="constant", **kwargs)
    return a / b


def weighted_groupby_mean(df, group_by, weigh_by, mode="mean"):
    """
    Weighted mean, std, and std in log space for a dataframe.groupby

    Parameters
    ----------
    df : dataframe
        Dataframe to groupby
    group_by : str or list
        Columns to group by
    weight_by : str
        Column to use as weights
    mode : "mean", "std" or "logstd"
        Do the weighted mean, the weighted standard deviaton,
        or the weighted std in log-space from the mean-log value
        (useful for P(s) etc.)
    """
    if type(group_by) == str:
        group_by = [group_by]
    gr = df.groupby(group_by)
    if mode == "mean":

        def wstd(x):
            return np.average(x, weights=df.loc[x.index, weigh_by])

        wm = wstd
    elif mode == "std":

        def wstd(x):
            wm = np.average(x, weights=df.loc[x.index, weigh_by])
            dev = x - wm
            res = np.sqrt(np.average(dev ** 2, weights=df.loc[x.index, weigh_by]))
            return res

        wm = wstd
    elif mode == "logstd":

        def wstd(x):
            x = np.log(x)
            wm = np.average(x, weights=df.loc[x.index, weigh_by])
            dev = x - wm
            res = np.sqrt(np.average(dev ** 2, weights=df.loc[x.index, weigh_by]))
            return np.exp(res)

        wm = wstd
    else:
        raise NotImplementedError

    f = {}
    for i in df.columns:
        if i in group_by:
            continue
        elif i == weigh_by:
            f[i] = ["sum"]
        else:
            f[i] = [wm]
    agg = gr.agg(f)
    agg.columns = [i[0] for i in agg.columns]
    return agg


def persistent_log_bins(end=10, bins_per_order_magnitude=10):
    """
    Creates most nicely looking log-spaced integer bins starting at 1, with the
    defined number of bins per order of magnitude.

    Parameters
    ----------
    end : number (int recommended) log10 of the last value. It is safe to put a
    large value here and select your range of bins later.

    bins_per_order_magnitude : int >0 how many bins per order of magnitude

    Notes
    -----
    This is not a replacement for logbins, and it has a different purpose.

    Difference between this and logbins
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Logbins creates bins from lo to hi, spaced logarithmically with an
    appriximate ratio. Logbins  makes sure that the last bin is large  (i.e.
    hi/ratio ... hi), and will not allow the last  bin to be much less than
    ratio. It would slightly adjust the ratio to achieve that.  As a result, by
    construciton, logbins bins are different for different lo or hi.

    This function is designed to create exactly the same bins that only depend
    on one parameter,  bins_per_order_magnitude. The goal is to make things
    calculated for different datasets/organisms/etc. comparable. For example, if
    these bins are used, it would allow us to divide P(s) for two different
    organisms by each other because it was calculated for the same bins.

    The price you pay for such versatility is that the last bin can be much less
    than others in real application.  For example, if you have 10 bins per order
    of magnitude (ratio of 1.25), but your data ends at 10500, then  the only
    points in the last bin would be 10000..10500, 1/5 of what could be. This may
    make the last point noisy.

    The main part is done using np.logspace and rounding to the nearest integer,
    followed by unique.  The gaps are then re-sorted to ensure that gaps are
    strictly increasing. The re-sorting of  gaps was essential, and produced
    better results than manual adjustment.

    Alternatives that produce irregular bins
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Using np.unique(np.logspace(a,b,N,dtype=int)) can be sub-optimal  For
    example, np.unique(np.logspace(0,1,11,dtype=int)) = [ 1,  2,  3,  5,  6,  7,
    10] Note the gaps jump from 1 to 2 back to 1

    Similarly using np.unique(np.rint(np.logspace..)) can be suboptimal
    np.unique(np.array(np.rint(np.logspace(0,1,9)),dtype=int))  = [ 1,  2,  3,
    4,  6,  7, 10]

    for bins_per_order_of_magnitude=16, 10 is not in bins. Other than that, 10,
    100, 1000, etc. are always included.
    """
    if end > 50:
        raise ValueError("End is a log10(max_value), not the max_value itself")
    bin_float = np.logspace(0, end, end * bins_per_order_magnitude + 1)
    bin_int = np.array(np.rint(bin_float), dtype=int)  # rounding to the nearest int
    bins = np.unique(bin_int)  # unique bins
    bins = np.cumsum(
        np.sort(np.r_[1, np.diff(bins)])
    )  # re-ordering gaps (important step)
    return bins
