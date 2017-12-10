import warnings

import numpy as np
import scipy.sparse.linalg

import numba 

from ._numutils_cy import  iterative_correction_symmetric

def get_diag(mat, i=0):
    '''Get the i-th diagonal of a matrix.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy
    '''
    return mat.ravel()[
        max(i,-mat.shape[1]*i)
        :max(0,(mat.shape[1]-i))*mat.shape[1]
        :mat.shape[1]+1]


def set_diag(mat, x, i=0):
    '''Rewrite in place the i-th diagonal of a matrix with a value or an array
    of values.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy'''
    mat.flat[
        max(i,-mat.shape[1]*i)
        :max(0,(mat.shape[1]-i))*mat.shape[1]
        :mat.shape[1]+1
        ] = x


def fill_nainf(arr, value=0, copy=True):
    '''Replaces np.nan and np.inf entries in an array with the provided value.

    Parameters
    ----------

    arr : np.array

    value : float

    copy : bool, optional
        If True, creates a copy of x, otherwise replaces values in-place. 
        By default, True.

    .. note:: differs from np.nan_to_num in that it replaces np.inf with the same
    number as np.nan.
    '''
    if copy:
        arr = arr.copy()
    arr[~np.isfinite(arr)] = value
    return arr


def slice_sorted(arr, lo, hi):
    '''Get the subset of a sorted array with values >=lo and <hi.
    A faster version of arr[(arr>=lo) & (arr<hi)]
    '''
    return arr[np.searchsorted(arr, lo)
               :np.searchsorted(arr, hi)]

def MAD(arr, axis=None, has_nans=False):
    '''Calculate the Median Absolute Deviation from the median.
    
    Parameters
    ----------
    
    arr : np.ndarray
        Input data.
    
    axis : int
        The axis along which to calculate MAD.
    
    has_nans : bool 
        If True, use the slower NaN-aware method to calculate medians.
    '''
    
    if has_nans:
        return np.nanmedian(np.abs(arr - np.nanmedian(arr, axis)), axis)
    else:
        return np.median(np.abs(arr - np.median(arr, axis)), axis)
        

def stochastic_sd(arr, n=10000, seed=0):
    '''Estimate the standard deviation of an array by considering only the 
    subset of its elements.
    
    Parameters
    ----------
    n : int
        The number of elements to consider. If the array contains fewer elements,
        use all.

    seed : int
        The seed for the random number generator.
    '''
    arr = np.asarray(arr)
    if arr.size < n: 
        return np.sqrt(arr.var())
    else:
        return np.sqrt(
            np.random.RandomState(seed).choice(arr.flat, n, replace=True).var())


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
        The number of eigenvectors to return.

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

    eigvecs : np.ndarray
        An array of sorted eigenvalues.

    """
    symmetric = is_symmetric(mat)
    if (symmetric 
        and np.sum(np.sum(np.abs(mat), axis=0) == 0) > 0 
        and not mask_zero_rows
        ):
        warnings.warn(
            "The matrix contains empty rows/columns and is symmetric. "
            "Mask the empty rows with remove_zeros=True")

    if mask_zero_rows:
        if not is_symmetric(mat):
            raise ValueError('The input matrix must be symmetric!')

        zero_rows_mask = np.sum(np.abs(mat), axis=0) == 0
        mat_collapsed = mat[~zero_rows_mask]
        mat_collapsed = mat_collapsed[:, ~zero_rows_mask]
        eigvecs_collapsed, eigvals = get_eig(
            mat_collapsed, n=n, mask_zero_rows=False, 
            subtract_mean=subtract_mean, divide_by_mean=divide_by_mean)

        n_rows = mat.shape[0]
        eigvecs = np.zeros((n, n_rows), dtype=float)
        for i in range(n):
            eigvecs[i][~zero_rows_mask] = eigvecs_collapsed[i]

        return eigvecs, eigvals

    mat = mat.astype(np.float, copy=True) # make a copy, ensure float
    mean = np.mean(mat)

    if subtract_mean: 
        mat -= mean
    if divide_by_mean:
        mat /= mean
    if symmetric:
        [eigvals, eigvecs] = scipy.sparse.linalg.eigsh(mat, n)
    else:
        [eigvals, eigvecs] = scipy.sparse.linalg.eigs(mat, n)
    order = np.argsort(-np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs.T[order]

    return eigvecs, eigvals 


def logbins(lo, hi, ratio=0, N=0, prepend_zero=False):
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
    data10 = np.logspace(np.log10(lo), np.log10(hi), N)                          
    data10 = np.array(np.rint(data10), dtype=int)                                
    data10 = np.sort(np.unique(data10))
    assert data10[0] == lo
    assert data10[-1] == hi                                           
    if prepend_zero:
        data10 = np.r_[0, data10]
    return data10                                                         

@numba.jit
def observed_over_expected(
        matrix, 
        mask=np.empty(shape=(0), dtype=np.bool),
        dist_bin_edge_ratio=1.03):
    '''
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

    '''
                                                                                 
    N = matrix.shape[0]                                                 
    mask2d = np.empty(shape=(0,0), dtype=np.bool)
    if (mask.ndim == 1) and (mask.size > 0):
        mask2d = mask[:,None] * mask[None, :]
    elif mask.ndim == 2:
        mask2d = mask
    else:
        raise ValueError('The mask must be either 1D or 2D.')
                                                                                 
    data = np.array(matrix, dtype = np.double, order = "C")
    
    has_mask = mask2d.size>0
    dist_bins = np.r_[0, np.array(logbins(1, N, dist_bin_edge_ratio))]
    n_pixels_arr = np.zeros_like(dist_bins[1:])
    sum_pixels_arr = np.zeros_like(dist_bins[1:])

    bin_idx, n_pixels, sum_pixels = 0, 0, 0

    for bin_idx, lo, hi in zip(range(len(dist_bins)-1), 
                               dist_bins[:-1], 
                               dist_bins[1:]):
        sum_pixels = 0                                                                   
        n_pixels = 0                                                                
        for offset in range(lo, hi):                                             
            for j in range(0, N-offset):                                         
                if not has_mask or mask2d[offset+j, j]:
                    sum_pixels += data[offset+j, j]
                    n_pixels += 1

        n_pixels_arr[bin_idx] = n_pixels
        sum_pixels_arr[bin_idx] = sum_pixels

        if n_pixels == 0:
            continue
        mean_pixel = sum_pixels / n_pixels                                       
        if mean_pixel == 0:                                                          
            continue

        for offset in range(lo, hi):                                         
            for j in range(0, N-offset):
                if not has_mask or mask2d[offset+j, j]:

                    data[offset + j, j] /= mean_pixel                        
                    if offset > 0:                                           
                        data[j, offset+j] /= mean_pixel                      

    return data, dist_bins, sum_pixels_arr, n_pixels_arr


@numba.jit #(nopython=True)
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
                                                                                 
    """                                                                          
    N = len(x)                                                          
                                                                                 
    _x = x.copy()
    if ignore_diags>0:
        for d in range(0, ignore_diags):
#            set_diag(_x, 0, d) # explicit cycles are easier to jit
             for j in range(0, N-d):
                 _x[j, j+d] = 0
                 _x[j+d, j] = 0
    totalBias = np.ones(N, np.double)     
                                                                                 
    converged = False                                                  
             
    iternum = 0
    mask = np.sum(_x, axis=1)==0
    for iternum in range(max_iter):                                              
        s = np.sum(_x, axis = 1)                                                
                                                                                 
        mask = (s == 0)                                                         
                                                                                           
        s = s / np.mean(s[~mask])                                              
        s[mask] = 1                                                             
        s -= 1                                                                   
        s *= 0.8                                                                 
        s += 1                                                                   
        totalBias *= s                                                           

        
        #_x = _x / s[:, None]  / s[None,:]
        # an explicit cycle is 2x faster here
        for i in range(N):
            for j in range(N):
                _x[i,j] /= s[i] * s[j]
            
        if (tolerance > 0) and (np.abs(s - 1).max() < tolerance):                
            converged=True                                                       
            break                                                                
                                                                                 
    corr = totalBias[~mask].mean()  #mean correction factor                      
    x = x * corr * corr #renormalizing everything                               
    totalBias /= corr                                                            
    report = {'converged':converged, 'iternum':iternum}                          
                                                                                 
    return x, totalBias, report
