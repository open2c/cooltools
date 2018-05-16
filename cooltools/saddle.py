from itertools import combinations
from functools import partial
from scipy.linalg import toeplitz
from cytoolz import merge
import numpy as np
import pandas as pd
from .lib import numutils


def ecdf(x, v, side='left'):
    """
    Return array `x`'s empirical CDF value(s) at the points in `v`.
    This is based on the :func:`statsmodels.distributions.ECDF` step function. 
    This is the inverse of `quantile`.
    
    """
    x = np.asarray(x)
    ind = np.searchsorted(np.sort(x), v, side=side) - 1
    y = np.linspace(1./len(x), 1., len(x))
    return y[ind]


def quantile(x, q, **kwargs):
    """
    Return the values of the quantile cut points specified by fractions `q` of 
    a sequence of data given by `x`.
    
    """
    x = np.asarray(x)
    p = np.asarray(q) * 100
    return np.nanpercentile(x, p, **kwargs)


def digitize_track(binedges, track, chromosomes=None):
    """
    Digitize genomic signal tracks into integers between `1` and `n`.

    Parameters
    ----------
    binedges : 1D array (length n + 1)
        Bin edges for quantization of signal. For `n` bins, there are `n + 1`
        edges. See encoding details in Notes.
    track : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the value column.
    chromosomes : sequence of chromosome names
        List of chromosomes to include.
    
    Returns
    -------
    digitized : DataFrame
        New bedGraph-like dataframe with value column and an additional
        digitized value column with name suffixed by '.d'
    hist : 1D array (length n + 2)
        Histogram of digitized signal values. Its length is `n + 2` because 
        the first and last elements correspond to outliers. See notes.

    Notes
    -----
    The digital encoding is as follows:

    - `1..n` <-> values assigned to histogram bins
    - `0` <-> left outlier values
    - `n+1` <-> right outlier values
    - `-1` <-> missing data (NaNs)
    
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
    hist = np.bincount(x, minlength=len(binedges) + 1)
    return digitized, hist


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


def _accumulate(S, C, getmatrix, digitized, chrom1, chrom2, verbose):
    n_bins = S.shape[0]
    matrix = getmatrix(chrom1, chrom2)

    if chrom1 == chrom2:
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
            S[i, j] += np.sum(data)
            C[i, j] += float(len(data))


def make_saddle(getmatrix, binedges, digitized, contact_type, chromosomes=None, 
                trim_outliers=False, verbose=False):
    """
    Make a matrix of average interaction probabilities between genomic bin pairs
    as a function of a specified genomic track. The provided genomic track must
    be pre-quantized as integers (i.e. digitized).
    
    Parameters
    ----------
    getmatrix : function
        A function returning a matrix of interaction between two chromosomes 
        given their names/indicies.
    binedges : 1D array (length n + 1)
        Bin edges of the digitized signal. For `n` bins, there are `n + 1`
        edges. See :func:`digitize_track`.
    digitized : tuple of (DataFrame, str)
        BedGraph-like dataframe of digitized signal along with the name of
        the digitized value column.
    contact_type : str
        If 'cis' then only cis interactions are used to build the matrix.
        If 'trans', only trans interactions are used.
    chromosomes : sequence of str, optional
        A list of names/indices of all chromosomes to use.
    trim_outliers : bool, optional
        Remove first and last row and column from the output matrix.
    verbose : bool, optional
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
    digitized, name = digitized

    # n_bins here includes 2 open bins
    # for values <lo and >hi.
    n_bins = len(binedges) + 1
    interaction_sum   = np.zeros((n_bins, n_bins))
    interaction_count = np.zeros((n_bins, n_bins))

    digitized = {k: x.values for k, x in digitized.groupby('chrom')[name]}
    if chromosomes is None:
        chromosomes = list(digitized.keys())
    
    if contact_type == 'cis':
        supports = list(zip(chromosomes, chromosomes))
    elif contact_type == 'trans':
        supports = list(combinations(chromosomes, 2))
    else:
        raise ValueError("The allowed values for the contact_type "
                         "argument are 'cis' or 'trans'.")

    for chrom1, chrom2 in supports:
        _accumulate(interaction_sum, interaction_count, getmatrix, digitized,
                    chrom1, chrom2, verbose)

    interaction_sum   += interaction_sum.T
    interaction_count += interaction_count.T
    
    if trim_outliers:
        interaction_sum = interaction_sum[1:-1, 1:-1]
        interaction_count = interaction_count[1:-1, 1:-1]

    return interaction_sum, interaction_count


def saddleplot(binedges, counts, saddledata, cmap='coolwarm', vmin=-1, vmax=1,
               color=None, title=None, xlabel=None, ylabel=None, clabel=None, 
               fig=None, fig_kws=None, heatmap_kws=None, margin_kws=None, 
               cbar_kws=None, subplot_spec=None):
    """
    Generate a saddle plot.

    Parameters
    ----------
    binedges : 1D array-like
        For `n` bins, there should be `n + 1` bin edges
    counts : 1D array-like
        Signal track histogram produced by `digitize_track`. It will include
        2 flanking elements for outlier values, thus the length should be 
        `n + 2`. 
    saddledata : 2D array-like
        Saddle matrix produced by `make_saddle`. It will include 2 flanking
        rows/columns for outlier signal values, thus the shape should be
        `(n+2, n+2)`.
    cmap : str or matplotlib colormap
        Colormap to use for plotting the saddle heatmap
    vmin, vmax : float
        Value limits for coloring the saddle heatmap
    color : matplotlib color value
        Face color for margin bar plots
    fig : matplotlib Figure, optional
        Specified figure to plot on. A new figure is created if none is 
        provided.
    fig_kws : dict, optional
        Passed on to `plt.Figure()`
    heatmap_kws : dict, optional
        Passed on to `ax.imshow()`
    margin_kws : dict, optional
        Passed on to `ax.bar()` and `ax.barh()`
    cbar_kws : dict, optional
        Passed on to `plt.colorbar()`
    subplot_spec : GridSpec object
        Specify a subregion of a figure to using a GridSpec.

    Returns
    -------
    Dictionary of axes objects.

    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import matplotlib.pyplot as plt
    from cytoolz import merge

    n_edges = len(binedges)
    n_bins = n_edges - 1
    lo, hi = binedges[0], binedges[-1]

    # Histogram and saddledata are flanked by outlier bins
    n = saddledata.shape[0]
    X, Y = np.meshgrid(binedges, binedges)
    C = saddledata
    hist = counts
    if (n - n_bins) == 2:
        C = C[1:-1, 1:-1]
        hist = hist[1:-1]

    # Layout
    if subplot_spec is not None:
        GridSpec = partial(GridSpecFromSubplotSpec, subplot_spec=subplot_spec)
    grid = {}
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
    grid['ax_heatmap'] = ax = plt.subplot(gs[4])
    heatmap_kws_default = dict( 
        cmap='coolwarm', 
        rasterized=True,
        vmin=vmin,
        vmax=vmax)
    heatmap_kws = merge(
        heatmap_kws_default,
        heatmap_kws if heatmap_kws is not None else {})
    img = ax.pcolormesh(X, Y, C, **heatmap_kws)
    vmin = heatmap_kws['vmin']
    vmax = heatmap_kws['vmax']
    plt.gca().yaxis.set_visible(False)

    # Margins
    margin_kws_default = dict(
        edgecolor='k',
        facecolor=color,
        linewidth=1)
    margin_kws = merge(
        margin_kws_default,
        margin_kws if margin_kws is not None else {})
    # left margin hist
    grid['ax_margin_y'] = plt.subplot(gs[3], sharey=grid['ax_heatmap'])
    plt.barh(binedges[:-1], 
             height=np.diff(binedges), 
             width=hist,
             align='edge',
             **margin_kws)
    plt.xlim(plt.xlim()[1], plt.xlim()[0])  # fliplr
    plt.ylim(hi, lo)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    # top margin hist
    grid['ax_margin_x'] = plt.subplot(gs[1], sharex=grid['ax_heatmap'])
    plt.bar(left=binedges[:-1], 
            width=np.diff(binedges), 
            height=hist,
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
    grid['ax_cbar'] = plt.subplot(gs[5])
    cbar_kws_default = dict(
        fraction=0.8,
        label=clabel or '')
    cbar_kws = merge(
        cbar_kws_default,
        cbar_kws if cbar_kws is not None else {})
    grid['cbar'] = cb = plt.colorbar(img, **cbar_kws)
    if vmin is not None and vmax is not None:
        # cb.set_ticks(np.arange(vmin, vmax + 0.001, 0.5))
        # # do linspace between vmin and vmax of 5 segments and trunc to 1 decimal:
        decimal = 10
        nsegments = 5
        cd_ticks = np.trunc(np.linspace(vmin, vmax, nsegments)*decimal)/decimal
        cb.set_ticks( cd_ticks )

    # extra settings
    grid['ax_heatmap'].set_xlim(lo, hi)
    grid['ax_heatmap'].set_ylim(hi, lo)
    plt.grid(False)
    plt.axis('off')
    if title is not None:
        grid['ax_margin_x'].set_title(title)
    if xlabel is not None:
        grid['ax_heatmap'].set_xlabel(xlabel)
    if ylabel is not None:
        grid['ax_margin_y'].set_ylabel(ylabel)

    return grid


def saddle_strength(S, C):
    """
    Parameters
    ----------
    S, C : 2D arrays, square, same shape
        Saddle sums and counts, respectively
        
    Returns
    -------
    1D array
    Ratios of cumulative corner interaction scores, where the saddle data is 
    grouped over the AA+BB corners and AB+BA corners with increasing extent.
    
    """
    m, n = S.shape
    if m != n:
        raise ValueError("`saddledata` should be square.")

    ratios = np.zeros(n)
    for k in range(1, n):
        intra_sum = S[0:k, 0:k].sum() + S[n-k:n, n-k:n].sum() 
        intra_count = C[0:k, 0:k].sum() + C[n-k:n, n-k:n].sum()
        intra = intra_sum / intra_count
        
        inter_sum = S[0:k, n-k:n].sum() + S[n-k:n, 0:k].sum()
        inter_count =  C[0:k, n-k:n].sum() + C[n-k:n, 0:k].sum()
        inter = inter_sum / inter_count
        
        ratios[k] = intra / inter
    return ratios
