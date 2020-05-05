from itertools import combinations
from functools import partial
from scipy.linalg import toeplitz
from cytoolz import merge
import numpy as np
import pandas as pd
from .lib import numutils

import bioframe


def ecdf(x, v, side="left"):
    """
    Return array `x`'s empirical CDF value(s) at the points in `v`.
    This is based on the :func:`statsmodels.distributions.ECDF` step function.
    This is the inverse of `quantile`.

    """
    x = np.asarray(x)
    ind = np.searchsorted(np.sort(x), v, side=side) - 1
    y = np.linspace(1.0 / len(x), 1.0, len(x))
    return y[ind]


def quantile(x, q, **kwargs):
    """
    Return the values of the quantile cut points specified by fractions `q` of
    a sequence of data given by `x`.

    """
    x = np.asarray(x)
    p = np.asarray(q) * 100
    return np.nanpercentile(x, p, **kwargs)


def mask_bad_bins(track, bintable):
    """
    Mask (set to NaN) values in track where bin is masked in bintable.

    Parameters
    ----------
    track : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the value column.
    bintable : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the weight column.

    Returns
    -------
    track : DataFrame
        New bedGraph-like dataframe with bad bins masked in the value column
    """
    track, name = track

    bintable, weight_name = bintable

    track = pd.merge(
        track[["chrom", "start", "end", name]], bintable, on=["chrom", "start", "end"]
    )
    track.loc[~np.isfinite(track[weight_name]), name] = np.nan
    track = track[["chrom", "start", "end", name]]

    return track


def digitize_track(binedges, track, regions=None):
    """
    Digitize genomic signal tracks into integers between `1` and `n`.

    Parameters
    ----------
    binedges : 1D array (length n + 1)
        Bin edges for quantization of signal. For `n` bins, there are `n + 1`
        edges. See encoding details in Notes.
    track : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the value column.
    regions: sequence of str or tuples
        List of genomic regions to include. Each can be a chromosome, a
        UCSC-style genomic region string or a tuple.

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
        raise ValueError("``track`` should be a tuple of (dataframe, column_name)")
    track, name = track

    # subset and re-order chromosome groups
    if regions is not None:
        regions = [bioframe.parse_region(reg) for reg in regions]
        grouped = track.groupby("chrom")
        track = pd.concat(
            bioframe.bedslice(grouped, chrom, st, end) for (chrom, st, end) in regions
        )

    # histogram the signal
    digitized = track.copy()
    digitized[name + ".d"] = np.digitize(track[name].values, binedges, right=False)
    mask = track[name].isnull()
    digitized.loc[mask, name + ".d"] = -1
    x = digitized[name + ".d"].values.copy()
    x = x[(x > 0) & (x < len(binedges) + 1)]
    hist = np.bincount(x, minlength=len(binedges) + 1)
    return digitized, hist


def make_cis_obsexp_fetcher(clr, expected, weight_name="weight"):
    """
    Construct a function that returns intra-chromosomal OBS/EXP.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : tuple of (DataFrame, str)
        Diagonal summary statistics for each chromosome, and name of the column
        to use
    weight_name : str
        Name of the column in the clr.bins to use as balancing weights

    Returns
    -------
    getexpected(chrom, _). 2nd arg is ignored.

    """
    expected, name = expected
    expected = {k: x.values for k, x in expected.groupby("chrom")[name]}

    def _fetch_cis_oe(reg1, reg2):
        obs_mat = clr.matrix(balance=weight_name).fetch(reg1)
        exp_mat = toeplitz(expected[reg1[0]][: obs_mat.shape[0]])
        return obs_mat / exp_mat

    return _fetch_cis_oe


def make_trans_obsexp_fetcher(clr, expected, weight_name="weight"):
    """
    Construct a function that returns OBS/EXP for any pair of chromosomes.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : (DataFrame, name) or scalar
        Average trans values. If a scalar, it is assumed to be a global trans
        expected value. If a tuple of (dataframe, name), the dataframe must
        have a MultiIndex with 'chrom1' and 'chrom2' and must also have a column
        labeled ``name``.
    weight_name : str
        Name of the column in the clr.bins to use as balancing weights

    Returns
    -----
    getexpected(reg1, reg2)

    """

    if np.isscalar(expected):
        return lambda reg1, reg2: (
            clr.matrix(balance=weight_name).fetch(reg1, reg2) / expected
        )

    elif isinstance(expected, (tuple, list)):
        expected, name = expected

        if not name:
            raise ValueError("Name of data column not provided.")

        expected = {
            k: x.values for k, x in expected.groupby(["chrom1", "chrom2"])[name]
        }

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
                    "{}, {}".format(chrom1, chrom2)
                )

        def _fetch_trans_oe(reg1, reg2):
            reg1 = bioframe.parse_region(reg1)
            reg2 = bioframe.parse_region(reg2)

            return clr.matrix(balance=weight_name).fetch(reg1, reg2) / _fetch_trans_exp(
                reg1[0], reg2[0]
            )

        return _fetch_trans_oe

    else:
        raise ValueError("Unknown type of expected")


def _accumulate(S, C, getmatrix, digitized, reg1, reg2, min_diag, max_diag, verbose):
    n_bins = S.shape[0]
    matrix = getmatrix(reg1, reg2)

    if reg1[0] == reg2[0]:
        for d in np.arange(-min_diag + 1, min_diag):
            numutils.set_diag(matrix, np.nan, d)
        if max_diag >= 0:
            for d in np.append(
                np.arange(-matrix.shape[0], -max_diag),
                np.arange(max_diag + 1, matrix.shape[0]),
            ):
                numutils.set_diag(matrix, np.nan, d)

    if verbose:
        print("regions {} vs {}".format(reg1, reg2))

    for i in range(n_bins):
        row_mask = digitized[reg1] == i
        for j in range(n_bins):
            col_mask = digitized[reg2] == j
            data = matrix[row_mask, :][:, col_mask]
            data = data[np.isfinite(data)]
            S[i, j] += np.sum(data)
            C[i, j] += float(len(data))


def make_saddle(
    getmatrix,
    binedges,
    digitized,
    contact_type,
    regions=None,
    min_diag=3,
    max_diag=-1,
    trim_outliers=False,
    verbose=False,
):
    """
    Make a matrix of average interaction probabilities between genomic bin
    pairs as a function of a specified genomic track. The provided genomic
    track must be pre-quantized as integers (i.e. digitized).

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
    regions : sequence of str or tuple, optional
        A list of genomic regions to use. Each can be a chromosome, a
        UCSC-style genomic region string or a tuple.
    min_diag : int
        Smallest diagonal to include in computation. Ignored with
        contact_type=trans.
    max_diag : int
        Biggest diagonal to include in computation. Ignored with
        contact_type=trans.
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
    digitized_df, name = digitized

    if regions is None:
        regions = [
            (chrom, df.start.min(), df.end.max())
            for chrom, df in digitized_df.groupby("chrom")
        ]
    else:
        regions = [bioframe.parse_region(reg) for reg in regions]

    digitized_tracks = {
        reg: bioframe.bedslice(digitized_df.groupby("chrom"), reg[0], reg[1], reg[2])[
            name
        ]
        for reg in regions
    }

    if contact_type == "cis":
        supports = list(zip(regions, regions))
    elif contact_type == "trans":
        supports = list(combinations(regions, 2))
    else:
        raise ValueError(
            "The allowed values for the contact_type " "argument are 'cis' or 'trans'."
        )

    # n_bins here includes 2 open bins
    # for values <lo and >hi.
    n_bins = len(binedges) + 1
    interaction_sum = np.zeros((n_bins, n_bins))
    interaction_count = np.zeros((n_bins, n_bins))

    for reg1, reg2 in supports:
        _accumulate(
            interaction_sum,
            interaction_count,
            getmatrix,
            digitized_tracks,
            reg1,
            reg2,
            min_diag,
            max_diag,
            verbose,
        )

    interaction_sum += interaction_sum.T
    interaction_count += interaction_count.T

    if trim_outliers:
        interaction_sum = interaction_sum[1:-1, 1:-1]
        interaction_count = interaction_count[1:-1, 1:-1]

    return interaction_sum, interaction_count


def saddleplot(
    binedges,
    counts,
    saddledata,
    cmap="coolwarm",
    scale="log",
    vmin=0.5,
    vmax=2,
    color=None,
    title=None,
    xlabel=None,
    ylabel=None,
    clabel=None,
    fig=None,
    fig_kws=None,
    heatmap_kws=None,
    margin_kws=None,
    cbar_kws=None,
    subplot_spec=None,
):
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
    scale : str
        Color scaling to use for plotting the saddle heatmap: log or linear
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
    from matplotlib.colors import Normalize, LogNorm
    from matplotlib import ticker
    import matplotlib.pyplot as plt

    class MinOneMaxFormatter(ticker.LogFormatter):
        def set_locs(self, locs=None):
            self._sublabels = set([vmin % 10 * 10, vmax % 10, 1])

        def __call__(self, x, pos=None):
            if x not in [vmin, 1, vmax]:
                return ""
            else:
                return "{x:g}".format(x=x)

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
        fig_kws = merge(fig_kws_default, fig_kws if fig_kws is not None else {})
        fig = plt.figure(**fig_kws)

    # Heatmap
    if scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif scale == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        raise ValueError("Only linear and log color scaling is supported")

    grid["ax_heatmap"] = ax = plt.subplot(gs[4])
    heatmap_kws_default = dict(cmap="coolwarm", rasterized=True)
    heatmap_kws = merge(
        heatmap_kws_default, heatmap_kws if heatmap_kws is not None else {}
    )
    img = ax.pcolormesh(X, Y, C, norm=norm, **heatmap_kws)
    plt.gca().yaxis.set_visible(False)

    # Margins
    margin_kws_default = dict(edgecolor="k", facecolor=color, linewidth=1)
    margin_kws = merge(margin_kws_default, margin_kws if margin_kws is not None else {})
    # left margin hist
    grid["ax_margin_y"] = plt.subplot(gs[3], sharey=grid["ax_heatmap"])
    plt.barh(
        binedges[:-1], height=np.diff(binedges), width=hist, align="edge", **margin_kws
    )
    plt.xlim(plt.xlim()[1], plt.xlim()[0])  # fliplr
    plt.ylim(hi, lo)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    # top margin hist
    grid["ax_margin_x"] = plt.subplot(gs[1], sharex=grid["ax_heatmap"])
    plt.bar(
        binedges[:-1], width=np.diff(binedges), height=hist, align="edge", **margin_kws
    )
    plt.xlim(lo, hi)
    # plt.ylim(plt.ylim())  # correct
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)

    # Colorbar
    grid["ax_cbar"] = plt.subplot(gs[5])
    cbar_kws_default = dict(fraction=0.8, label=clabel or "")
    cbar_kws = merge(cbar_kws_default, cbar_kws if cbar_kws is not None else {})
    if scale == "linear" and vmin is not None and vmax is not None:
        grid["cbar"] = cb = plt.colorbar(img, **cbar_kws)
        # cb.set_ticks(np.arange(vmin, vmax + 0.001, 0.5))
        # # do linspace between vmin and vmax of 5 segments and trunc to 1 decimal:
        decimal = 10
        nsegments = 5
        cd_ticks = np.trunc(np.linspace(vmin, vmax, nsegments) * decimal) / decimal
        cb.set_ticks(cd_ticks)
    else:
        grid["cbar"] = cb = plt.colorbar(img, format=MinOneMaxFormatter(), **cbar_kws)
        cb.ax.yaxis.set_minor_formatter(MinOneMaxFormatter())

    # extra settings
    grid["ax_heatmap"].set_xlim(lo, hi)
    grid["ax_heatmap"].set_ylim(hi, lo)
    plt.grid(False)
    plt.axis("off")
    if title is not None:
        grid["ax_margin_x"].set_title(title)
    if xlabel is not None:
        grid["ax_heatmap"].set_xlabel(xlabel)
    if ylabel is not None:
        grid["ax_margin_y"].set_ylabel(ylabel)

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
        intra_sum = np.nansum(S[0:k, 0:k]) + np.nansum(S[n - k : n, n - k : n])
        intra_count = np.nansum(C[0:k, 0:k]) + np.nansum(C[n - k : n, n - k : n])
        intra = intra_sum / intra_count

        inter_sum = np.nansum(S[0:k, n - k : n]) + np.nansum(S[n - k : n, 0:k])
        inter_count = np.nansum(C[0:k, n - k : n]) + np.nansum(C[n - k : n, 0:k])
        inter = inter_sum / inter_count

        ratios[k] = intra / inter
    return ratios
