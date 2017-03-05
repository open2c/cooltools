import warnings
import numpy as np
from .num import peaks, numutils

def insul_diamond(mat, window=10, min_dist_bins=3):
    """
    Calculates the insulation score of a Hi-C interaction matrix.

    Parameters
    ----------
    mat : numpy.array
        A dense square matrix of Hi-C interaction frequencies. 
        May contain nans, e.g. in rows/columns excluded from the analysis.
    
    window : int
        The width of the window to calculate the insulation score.

    min_dist_bins :
        If >0, the interactions at separations < `min_dist_bins` are ignored
        when calculating the insulation score.
    
    """
    if (min_dist_bins):
        mat = mat.copy()
        for i in range(-min_dist_bins+1, min_dist_bins):
            set_diag(mat, np.nan, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N = mat.shape[0]
        bad_bins = (np.nansum(mat, axis=0) == 0)
        score = np.nan * np.ones(N)
        norm = np.zeros(N)
        for i in range(0, N):
            lo = max(0, i-window)
            hi = min(i+window, N)
            if (bad_bins[lo:i].sum() <= window//2) or (bad_bins[i:hi].sum() <= window//2):
                # nanmean of interactions to reduce the effect of bad bins
                score[i] = np.nanmean(mat[lo:i, i:hi])
                norm[i] = (~np.isnan(mat[lo:i, i:hi])).sum()
        score = score / np.where(norm>0, norm, 1)
        score /= np.nanmedian(score)
    return score

