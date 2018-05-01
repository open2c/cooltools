import warnings
import numpy as np
import pandas as pd
from .utils import peaks, numutils


def insul_diamond(pixels, bins,
        window=10, ignore_diags=2, balanced=True, norm_by_median=True):
    """
    Calculates the insulation score of a Hi-C interaction matrix.

    Parameters
    ----------
    pixels : pandas.DataFrame
        A table of Hi-C interactions. Must follow the Cooler columnar format: 
        bin1_id, bin2_id, count, balanced (optional)).
    
    bins : pandas.DataFrame
        A table of bins, is used to determine the span of the matrix 
        and the locations of bad bins.

    window : int
        The width (in bins) of the diamond window to calculate the insulation score.

    ignore_diags : int
        If > 0, the interactions at separations < `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals 
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.

    norm_by_median : bool
        If True, normalize the insulation score by its NaN-median.
    
    
    """

    lo_bin_id = bins.index.min()
    hi_bin_id = bins.index.max() + 1
    N = hi_bin_id - lo_bin_id
    sum_pixels = np.zeros(N)
    n_pixels = np.zeros(N)

    bad_bin_mask = bins.weight.isnull().values if balanced else np.zeros(N, dtype=bool)

    diag_pixels = pixels[pixels.bin2_id - pixels.bin1_id <= (window - 1) * 2]
    if balanced: 
        diag_pixels = diag_pixels[~diag_pixels.balanced.isnull()]

    i = diag_pixels.bin1_id.values - lo_bin_id
    j = diag_pixels.bin2_id.values - lo_bin_id
    val = diag_pixels.balanced.values if balanced else diag_pixels['count'].values

    for i_shift in range(0, window):
        for j_shift in range(0, window):
            if i_shift+j_shift < ignore_diags:
                continue

            mask = (i+i_shift == j-j_shift) & (i + i_shift < N ) & (j - j_shift >= 0 ) 

            sum_pixels += np.bincount(i[mask] + i_shift, val[mask], minlength=N)

            loc_bad_bin_mask = np.zeros(N, dtype=bool)
            if i_shift == 0:
                loc_bad_bin_mask |= bad_bin_mask
            else:
                loc_bad_bin_mask[i_shift:] |= bad_bin_mask[:-i_shift]
            if j_shift == 0:
                loc_bad_bin_mask |= bad_bin_mask
            else:
                loc_bad_bin_mask[:-j_shift] |= bad_bin_mask[j_shift:]

            n_pixels[i_shift:(-j_shift if j_shift else None)] += (
                 1 - loc_bad_bin_mask[i_shift:(-j_shift if j_shift else None)])


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        score = sum_pixels / n_pixels

        if norm_by_median:
            score /= np.nanmedian(score)

    return score

def find_insulating_boundaries(
    clr,
    window_bp=100000,
    min_dist_bad_bin=2, 
    balance='weight',
    ignore_diags=None,
    chromosomes=None,
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
    ignore_diags : int
        The number of diagonals to ignore. If None, equals the number of 
        diagonals ignored during IC balancing.

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins and 
        the insulating boundary strengths.
    '''
    if chromosomes is None:
        chromosomes = clr.chromnames

    bin_size = clr.info['bin-size']
    ignore_diags = (ignore_diags 
        if ignore_diags is not None 
        else clr._load_attrs(clr.root.rstrip('/')+'/bins/weight')['ignore_diags'] )
    window_bins = window_bp // bin_size
    
    if (window_bp % bin_size !=0):
        raise Exception(
            'The window size ({}) has to be a multiple of the bin size {}'.format(
                window_bp, bin_size))
        
    ins_chrom_tables = []
    for chrom in chromosomes:
        chrom_bins = clr.bins().fetch(chrom)
        chrom_pixels = clr.matrix(as_pixels=True, balance=balance).fetch(chrom)

        with warnings.catch_warnings():                      
            warnings.simplefilter("ignore", RuntimeWarning)  
            ins_track = insul_diamond(chrom_pixels, chrom_bins, 
                window=window_bins, ignore_diags=ignore_diags)
            ins_track[ins_track==0] = np.nan
            ins_track = np.log2(ins_track)
        
        is_bad_bin = np.isnan(chrom_bins['weight'].values)
        bad_bin_neighbor = np.zeros_like(is_bad_bin)
        for i in range(0, min_dist_bad_bin):
            if i == 0:
                bad_bin_neighbor = bad_bin_neighbor | is_bad_bin
            else:
                bad_bin_neighbor = bad_bin_neighbor | np.r_[[True]*i, is_bad_bin[:-i]]
                bad_bin_neighbor = bad_bin_neighbor | np.r_[is_bad_bin[i:], [True]*i]            

        ins_chrom = chrom_bins[['chrom', 'start', 'end']].copy()
        ins_track[bad_bin_neighbor] = np.nan
        ins_chrom['bad_bin_masked'] = bad_bin_neighbor
        
        ins_track[~np.isfinite(ins_track)] = np.nan
        
        ins_chrom['log2_insulation_score_{}'.format(window_bp)] = ins_track

        poss, proms = peaks.find_peak_prominence(-ins_track)
        ins_prom_track = np.zeros_like(ins_track) * np.nan
        ins_prom_track[poss] = proms
        ins_chrom['boundary_strength_{}'.format(window_bp)] = ins_prom_track

        ins_chrom_tables.append(ins_chrom)

    ins_table = pd.concat(ins_chrom_tables)
    return ins_table


def _insul_diamond_dense(mat, window=10, ignore_diags=2, norm_by_median=True):
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
        If > 0, the interactions at separations < `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals 
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.

    norm_by_median : bool
        If True, normalize the insulation score by its NaN-median.
    
    """
    if (ignore_diags):
        mat = mat.copy()
        for i in range(-ignore_diags+1, ignore_diags):
            numutils.set_diag(mat, np.nan, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N = mat.shape[0]
        score = np.nan * np.ones(N)
        for i in range(0, N):
            lo = max(0, i+1-window)
            hi = min(i+window, N)
            # nanmean of interactions to reduce the effect of bad bins
            score[i] = np.nanmean(mat[lo:i+1, i:hi])
        if norm_by_median:
            score /= np.nanmedian(score)
    return score

def _find_insulating_boundaries_dense(
    clr,
    window_bp=100000,
    balance='weight',
    min_dist_bad_bin=2, 
    ignore_diags=None,
    chromosomes=None,
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
    ignore_diags : int
        The number of diagonals to ignore. If None, equals the number of 
        diagonals ignored during IC balancing.

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins and 
        the insulating boundary strengths.
    '''
    if chromosomes is None:
        chromosomes = clr.chromnames

    bin_size = clr.info['bin-size']
    ignore_diags = (ignore_diags 
        if ignore_diags is not None 
        else clr._load_attrs(clr.root.rstrip('/')+'/bins/weight')['ignore_diags'] )
    window_bins = window_bp // bin_size
    
    if (window_bp % bin_size !=0):
        raise Exception(
            'The window size ({}) has to be a multiple of the bin size {}'.format(
                window_bp, bin_size))
        
    ins_chrom_tables = []
    for chrom in chromosomes:
        ins_chrom = clr.bins().fetch(chrom)[['chrom', 'start', 'end']]
        is_bad_bin = np.isnan(clr.bins().fetch(chrom)['weight'].values)

        m = clr.matrix(balance=balance).fetch(chrom)

        with warnings.catch_warnings():                      
            warnings.simplefilter("ignore", RuntimeWarning)  
            ins_track = _insul_diamond_dense(
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


