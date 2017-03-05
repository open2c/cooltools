def get_diag(mat, i=0):
    '''Get the i-th diagonal of a matrix.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy'''
    return mat.ravel()[
        max(i,-mat.shape[1]*i)
        :max(0,(mat.shape[1]-i))*mat.shape[1]
        :mat.shape[1]+1]

def set_diag(mat, x, i=0):
    '''Rewrite in place the i-th diagonal of a matrix with a value or 
    an array of values.
    This solution was borrowed from
    http://stackoverflow.com/questions/9958577/changing-the-values-of-the-diagonal-of-a-matrix-in-numpy'''
    mat.flat[
        max(i,-mat.shape[1]*i)
        :max(0,(mat.shape[1]-i))*mat.shape[1]
        :mat.shape[1]+1
        ] = x


def slice_sorted(arr, lo, hi):
    '''Get the subset of a sorted array with values >=lo and <hi.
    A faster version of arr[(arr>=lo) & (arr<hi)]
    '''
    return arr[np.searchsorted(arr, lo)
               :np.searchsorted(arr, hi)]
