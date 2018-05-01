"""
Saddle plot code.

Authors
~~~~~~~
* Anton Goloborodko
* Nezar Abdennur

"""
from scipy.linalg import toeplitz
from cytoolz import merge
import numpy as np
import pandas as pd
from .utils import numutils


def make_histbins(signal, n, by_percentile=False, prange=None):
    if prange is None:
        prange = (0, 100)
    n_edges = n + 1
    if by_percentile:
        # exclude outliers from the percentile range
        perc_edges = np.linspace(prange[0], prange[1], n_edges)
        # make equal-sized bins in rank-space
        binedges = np.nanpercentile(signal, perc_edges)
    else:
        # exclude outliers from the value range
        lo = np.nanpercentile(signal, prange[0])
        hi = np.nanpercentile(signal, prange[1])
        # make equal-sized bins in signal-space
        binedges = np.linspace(lo, hi, n_edges)
    return binedges
    

def digitize_track(binedges, track, chromosomes=None):
    """
    Digitize per-chromosome genomic tracks.
    
    encoding: 
    * 0 : left outlier
    * n + 1 : right outlier
    * -1 : missing data
    * 1..n : bucket id
    
    """
    if not isinstance(track, tuple):
        raise ValueError(
            "``track`` should be a tuple of (dataframe, column_name)")
    track, name = track
    
    # subset and re-order chromosome groups
    if chromosomes is not None:
        grouped = track.groupby('chrom')
        track = pd.concat(grouped.get_group(chrom) for chrom in chromosomes)
    
    # histogram the signal
    digitized = track.copy()
    digitized[name+'.d'] = np.digitize(track[name].values, binedges, right=False)
    mask = track[name].isnull()
    digitized.loc[mask, name+'.d'] = -1
    x = digitized[name + '.d'].values.copy()
    x = x[(x > 0) & (x < len(binedges) + 1)]
    counts = np.bincount(x, minlength=len(binedges) + 1)
    return digitized, counts


# def digitize_track(
#         histbins,
#         track,
#         prange=None,
#         chromosomes=None,
#         by_percentile=False):
#     """
#     Digitize per-chromosome genomic tracks.
    
#     Parameters
#     ----------
#     get_track : function
#         A function returning a genomic track given a chromosomal name/index.
#     get_mask : function
#         A function returning a binary mask of valid genomic bins given a 
#         chromosomal name/index.
#     chromosomes : list or iterator
#         A list of names/indices of all chromosomes.
#     bins : int or list or numpy.ndarray
#         If list or numpy.ndarray, `bins` must contain the bin edges for 
#         digitization.
#         If int, then the specified number of bins will be generated 
#         automatically from the genome-wide range of the track values.
#     prange : pair of floats
#         The percentile of the genome-wide range of the track values used to 
#         generate bins. E.g., if `prange`=(2. 98) the lower bin would 
#         start at the 2-nd percentile and the upper bin would end at the 98-th 
#         percentile of the genome-wide signal.
#         Use to prevent the extreme track values from exploding the bin range.
#         Is ignored if `bins` is a list or a numpy.ndarray.
#     by_percentile : bool
#         If true then the automatically generated bins will contain an equal 
#         number of genomic bins genome-wide (i.e. track values are binned 
#         according to their percentile). Otherwise, bins edges are spaced equally. 
#         Is ignored if `bins` is a list or a numpy.ndarray.
        
#     Returns
#     -------
#     digitized : dict
#         A dictionary of the digitized track, split by chromosome.
#         The value of -1 corresponds to the masked genomic bins, the values of 0 
#         and the number of bin-edges (bins+1) correspond to the values lying below
#         and above the bin range limits, correspondingly.
#         See https://docs.scipy.org/doc/numpy/reference/generated/numpy.digitize.html
#         for reference.
#     binedges : numpy.ndarrayxxxx
#         The edges of bins used to digitize the track.
#     """
#     if not isinstance(track, tuple):
#         raise ValueError(
#             "``track`` should be a tuple of (dataframe, column_name)")
#     track, name = track
    
#     # subset and re-order chromosome groups
#     if chromosomes is not None:
#         grouped = track.groupby('chrom')
#         track = pd.concat(grouped.get_group(chrom) for chrom in chromosomes)
    
#     # set the histogram bucket edges
#     signal = track[name].dropna().values
#     if hasattr(histbins, '__len__'):
#         binedges = histbins
#     else:
#         if prange is None:
#             prange = (0, 100)

#         # n bins have n+1 edges
#         n_edges = histbins + 1
#         if by_percentile:
#             # exclude outliers from the percentile range
#             perc_edges = np.linspace(prange[0], prange[1], n_edges)
#             # make equal-sized bins in rank-space
#             binedges = np.nanpercentile(signal, perc_edges)
#         else:
#             # exclude outliers from the value range
#             lo = np.nanpercentile(signal, prange[0])
#             hi = np.nanpercentile(signal, prange[1])
#             # make equal-sized bins in signal-space
#             binedges = np.linspace(lo, hi, n_edges)

#     # histogram the signal
#     digitized = track.copy()
#     digitized[name] = np.digitize(track[name].values, binedges, right=False)
#     mask = track[name].isnull()
#     digitized.loc[mask, name] = -1

#     return binedges, digitized


def make_cis_obsexp_fetcher(clr, expected):
    """
    Construct a function that returns intra-chromosomal OBS/EXP.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : DataFrame 
        Diagonal summary statistics for each chromosome.
    name : str
        Name of data column in ``expected`` to use.

    Returns
    -------
    getexpected(chrom, _). 2nd arg is ignored.

    """
    expected, name = expected
    expected = {k: x.values for k, x in expected.groupby('chrom')[name]}
    return lambda chrom, _: (
            clr.matrix().fetch(chrom) / 
                toeplitz(expected[chrom])
        )


def make_trans_obsexp_fetcher(clr, expected):
    """
    Construct a function that returns OBS/EXP for any pair of chromosomes.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : DataFrame or scalar
        Average trans values. If a scalar, it is assumed to be a global trans 
        expected value. If a dataframe, it must have a MultiIndex with 'chrom1'
        and 'chrom2' and must also have a column labeled ``name``.
    name : str
        Name of data column in ``expected`` if it is a data frame.

    Returns
    -----
    getexpected(chrom1, chrom2)
    
    """
    expected, name = expected
    expected = {k: x.values for k, x in 
                   expected.groupby(['chrom1', 'chrom2'])[name]}

    def _fetch_trans_exp(chrom1, chrom2):
        # Handle chrom flipping
        if (chrom1, chrom2) in expected.keys():
            return expected[chrom1, chrom2]
        elif (chrom2, chrom1) in expected.keys():
            return expected[chrom2, chrom1]
        # .loc is the right way to get [chrom1,chrom2] value from MultiIndex df:
        # https://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-indexing-with-hierarchical-index
        else:
            raise KeyError(
                "trans-exp index is missing a pair of chromosomes: "
                "{}, {}".format(chrom1,chrom2))

    if np.isscalar(expected):
        return lambda chrom1, chrom2: (
            clr.matrix().fetch(chrom1, chrom2) / expected)
    else:
        if name is None:
            raise ValueError("Name of data column not provided.")
        return lambda chrom1, chrom2: (
                clr.matrix().fetch(chrom1, chrom2) / 
                    _fetch_trans_exp(chrom1, chrom2))


def make_saddle(
        get_matrix,
        digitized,
        contact_type,
        chromosomes=None,
        verbose=False):
    """
    Make a matrix of average interaction probabilities between genomic bin pairs
    as a function of a specified genomic track. The provided genomic track must
    be pre-binned (i.e. digitized).
    
    Parameters
    ----------
    get_matrix : function
        A function returning an matrix of interaction between two chromosomes 
        given their names/indicies.
    get_digitized : function
        A function returning a track of the digitized target genomic track given
        a chromosomal name/index.
    chromosomes : list or iterator
        A list of names/indices of all chromosomes.    
    contact_type : str
        If 'cis' then only cis interactions are used to build the matrix.
        If 'trans', only trans interactions are used.
    verbose : bool
        If True then reports progress.

    Returns
    -------
    interaction_sum : 2D array
        The matrix of summed interaction probability between two genomic bins 
        given their values of the provided genomic track.
    interaction_count : 2D array
        The matrix of the number of genomic bin pairs that contributed to the 
        corresponding pixel of ``interaction_sum``.

    """
    if contact_type not in ['cis', 'trans']:
        raise ValueError("The allowed values for the contact_type "
                         "argument are 'cis' or 'trans'.")

    digitized, name = digitized

    # n_bins here includes 2 open bins
    # for values <lo and >hi.
    n_bins = digitized[name].values.max() + 1        
    interaction_sum   = np.zeros((n_bins, n_bins))
    interaction_count = np.zeros((n_bins, n_bins))

    digitized = {k: x.values for k, x in digitized.groupby('chrom')[name]}
    if chromosomes is None:
        chromosomes = list(digitized.keys())
    
    for k, chrom1 in enumerate(chromosomes):
        for chrom2 in chromosomes[k:]:
            if (((contact_type == 'trans') and (chrom1 == chrom2)) or 
                ((contact_type == 'cis') and (chrom1 != chrom2))):
                continue
                
            matrix = get_matrix(chrom1, chrom2)
            for d in [-2, -1, 0, 1, 2]:
                numutils.set_diag(matrix, np.nan, d)

            if verbose:
                print('chromosomes {} vs {}'.format(chrom1, chrom2))
                
            for i in range(n_bins):
                row_mask = (digitized[chrom1] == i)
                for j in range(n_bins):
                    col_mask = (digitized[chrom2] == j)
                    data = matrix[row_mask, :][:, col_mask]
                    data = data[np.isfinite(data)]
                    interaction_sum[i, j]   += np.sum(data)
                    interaction_count[i, j] += float(len(data))

    interaction_sum   += interaction_sum.T
    interaction_count += interaction_count.T
    
    return interaction_sum, interaction_count


def saddleplot(binedges,
               counts,
               saddledata,
               color,
               cbar_label=None,
               fig_kws=None,
               heatmap_kws=None, 
               margin_kws=None,
               fig=None):
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt
    from cytoolz import merge

    # signal histogram
    n_bins = len(binedges) - 1
    lo, hi = binedges[0], binedges[-1]
#     track, name = track
#     signal = track[name].values
#     signal_hist, _ = np.histogram(
#         signal[~np.isnan(signal)], 
#         binedges, 
#         range=(lo, hi))
    signal_hist = counts[1:-1]
    
    # Layout
    gs = GridSpec(
        nrows=3, 
        ncols=3, 
        width_ratios=[0.2, 1, 0.1], 
        height_ratios=[0.2, 1, 0.1],
        wspace=0.05,
        hspace=0.05,
    )
        
    # Figure
    if fig is None:
        fig_kws_default = dict(figsize=(5, 5))
        fig_kws = merge(
            fig_kws_default,
            fig_kws if fig_kws is not None else {}

        )
        fig = plt.figure(**fig_kws)

    # Heatmap
    ax = ax1 = plt.subplot(gs[4])
    heatmap_kws_default = dict(
        aspect='auto', 
        cmap='coolwarm', 
        interpolation='none',
        rasterized=True,
        extent=[lo, hi, hi, lo],
        vmin=-1,
        vmax=1)
    heatmap_kws = merge(
        heatmap_kws_default,
        heatmap_kws if heatmap_kws is not None else {}, 
    )
    img = ax.imshow(saddledata[1:-1, 1:-1], **heatmap_kws)
    vmin = heatmap_kws['vmin']
    vmax = heatmap_kws['vmax']
    plt.gca().yaxis.set_visible(False)

    # Margins
    margin_kws_default = dict(
        edgecolor='k',
        facecolor=color,
        linewidth=1
    )
    margin_kws = merge(
        margin_kws_default,
        margin_kws if margin_kws is not None else {},
    )
    # left margin hist
    plt.subplot(gs[3], sharey=ax1)
    plt.barh(binedges[:-1], 
             height=np.diff(binedges), 
             width=signal_hist,
             align='edge',
             **margin_kws)
    plt.xlim(plt.xlim()[1], plt.xlim()[0])  # fliplr
    plt.ylim(hi, lo)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    # top margin hist
    plt.subplot(gs[1], sharex=ax1)
    plt.bar(left=binedges[:-1], 
            width=np.diff(binedges), 
            height=signal_hist,
            align='edge',
             **margin_kws)
    plt.xlim(lo, hi)
    #plt.ylim(plt.ylim())  # correct
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    
    # Colorbar
    plt.subplot(gs[5])
    cb = plt.colorbar(
        img, 
        fraction=0.8, 
        label=cbar_label)
    if vmin is not None and vmax is not None:
        cb.set_ticks(np.arange(vmin, vmax + 0.001, 0.5))

    # extra settings
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(hi, lo)
    plt.grid(False)
    plt.axis('off')


# TODO:
# add flag to calculate compartment strength ...
#########################################
# strength ?!?!?!?!
#########################################
def get_compartment_strength(saddledata, fraction):
    """
    Naive compartment strength calculator.
    """
    # assuming square shape:
    n_bins,n_bins = saddledata.shape
    # fraction must be < 0.5:
    if fraction >= 0.5:
        raise ValueError(
            "Fraction for compartment strength calculations must be <0.5.")
    # # of bins to take for strenght computation:
    bins_for_strength = int(n_bins*fraction)
    # intra- (BB):
    intra_BB = saddledata[0:bins_for_strength,\
                        0:bins_for_strength]
    # intra- (AA):
    intra_AA = saddledata[n_bins-bins_for_strength:n_bins,\
                        n_bins-bins_for_strength:n_bins]
    intra = np.concatenate((intra_AA, intra_BB), axis=0)
    intra_median = np.median(intra)
    # inter- (BA):
    inter_BA = saddledata[0:bins_for_strength,\
                        n_bins-bins_for_strength:n_bins]
    # inter- (AB):
    inter_AB = saddledata[n_bins-bins_for_strength:n_bins,\
                        0:bins_for_strength]
    inter = np.concatenate((inter_BA, inter_AB), axis=0)
    inter_median = np.median(inter)
    # returning intra-/inter- ratrio as a
    # measure of compartment strength:
    return intra_median / inter_median
