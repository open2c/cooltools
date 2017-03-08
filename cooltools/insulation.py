import warnings
import numpy as np
import pandas as pd
from .num import peaks, numutils

def insul_diamond(mat, window=10, ignore_diags=2):
    """
    Calculates the insulation score of a Hi-C interaction matrix.

    Parameters
    ----------
    mat : numpy.array
        A dense square matrix of Hi-C interaction frequencies. 
        May contain nans, e.g. in rows/columns excluded from the analysis.
    
    window : int
        The width of the window to calculate the insulation score.

    ignore_diags : int
        If > 0, the interactions at separations <= `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals 
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.
    
    """
    if (ignore_diags):
        mat = mat.copy()
        for i in range(-ignore_diags, ignore_diags+1):
            numutils.set_diag(mat, np.nan, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N = mat.shape[0]
        score = np.nan * np.ones(N)
        for i in range(0, N):
            lo = max(0, i-window)
            hi = min(i+window, N)
            # nanmean of interactions to reduce the effect of bad bins
            score[i] = np.nanmean(mat[lo:i, i:hi])
        score /= np.nanmedian(score)
    return score



def find_insulating_boundaries(
    c,
    window_bp = 100000,
    min_dist_bad_bin = 2, 
):
    '''Calculate the diamond insulation scores and call insulating boundaries.

    Parameters
    ----------
    c : cooler.Cooler
        A cooler with balanced Hi-C data.
    window_bp : int
        The size of the sliding diamond window used to calculate the insulation
        score.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin. Do not calculate insulation
        scores for bins having a bad bin closer than this distance.

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins and 
        the insulating boundary strengths.
    '''

    bin_size = c.info['bin-size']
    ignore_diags = c._load_attrs('/bins/weight')['ignore_diags']
    window_bins = window_bp // bin_size
    
    if (window_bp % bin_size !=0):
        raise Exception(
            'The window size ({}) has to be a multiple of the bin size {}'.format(
                window_bp, bin_size))
        
    ins_chrom_tables = []
    for chrom in c.chroms()[:]['name']:
        ins_chrom = c.bins().fetch(chrom)[['chrom', 'start', 'end']]
        is_bad_bin = np.isnan(c.bins().fetch(chrom)['weight'].values)

        m=c.matrix(balance=True).fetch(chrom)

        with warnings.catch_warnings():                      
            warnings.simplefilter("ignore", RuntimeWarning)  
            ins_track = insul_diamond(
                m, window_bins, ignore_diags)
            ins_track[ins_track==0] = np.nan
            ins_track = np.log2(ins_track)
        
        bad_bin_neighbor = np.zeros_like(is_bad_bin)
        for i in range(0, min_dist_bad_bin):
            if i == 0:
                bad_bin_neighbor = bad_bin_neighbor | is_bad_bin
            else:
                bad_bin_neighbor = bad_bin_neighbor | np.r_[[True]*i, is_bad_bin[:-i]]
                bad_bin_neighbor = bad_bin_neighbor | np.r_[is_bad_bin[i:], [True]*i]            

        ins_track[bad_bin_neighbor] = np.nan
        ins_chrom['bad_bin_masked'] = bad_bin_neighbor
        
        ins_track[~np.isfinite(ins_track)] = np.nan
        
        ins_chrom['log2_insulation_score_{}'.format(window_bp)] = ins_track

        poss, proms = peaks.find_peak_prominence(-ins_track)
        ins_prom_track = np.zeros_like(ins_track) * np.nan
        ins_prom_track[poss] = proms
        ins_chrom['boundary_strength_{}'.format(window_bp)] = ins_prom_track
        ins_chrom['boundary_strength_{}'.format(window_bp)] = ins_prom_track

        ins_chrom_tables.append(ins_chrom)

    ins_table = pd.concat(ins_chrom_tables)
    return ins_table

