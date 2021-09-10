from itertools import combinations
from functools import partial
from scipy.linalg import toeplitz
from cytoolz import merge
import numpy as np
import pandas as pd
from .lib import numutils
import warnings

import bioframe


def _make_cooler_view(view_df, clr):
    try:
        if not bioframe.is_viewframe(view_df, raise_errors=True):
            raise ValueError("view_df is not a valid viewframe.")
    except Exception as e:  # AssertionError or ValueError, see https://github.com/gfudenberg/bioframe/blob/main/bioframe/core/checks.py#L177
        warnings.warn(
            "view_df has to be a proper viewframe from next release",
            DeprecationWarning,
            stacklevel=2,
        )
        view_df = bioframe.make_viewframe(view_df)
    if not bioframe.is_contained(view_df, bioframe.make_viewframe(clr.chromsizes)):
        raise ValueError("View table is out of the bounds of chromosomes in cooler.")
    return view_df


def _view_from_track(track_df):
    bioframe.core.checks._verify_columns(track_df, ["chrom", "start", "end"])
    return bioframe.make_viewframe(
        [
            (chrom, df.start.min(), df.end.max())
            for chrom, df in track_df.groupby("chrom")
        ]
    )


def _ecdf(x, v, side="left"):
    """
    Return array `x`'s empirical CDF value(s) at the points in `v`.
    This is based on the :func:`statsmodels.distributions.ECDF` step function.
    This is the inverse of `quantile`.

    """
    x = np.asarray(x)
    ind = np.searchsorted(np.sort(x), v, side=side) - 1
    y = np.linspace(1.0 / len(x), 1.0, len(x))
    return y[ind]


def _quantile(x, q, **kwargs):
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

    Currently used in `cli.get_saddle()`. TODO: determine if this should be moved to cooltools.core.

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
    # TODO: update to new track format

    track, name = track

    bintable, clr_weight_name = bintable

    track = pd.merge(
        track[["chrom", "start", "end", name]], bintable, on=["chrom", "start", "end"]
    )
    track.loc[~np.isfinite(track[clr_weight_name]), name] = np.nan
    track = track[["chrom", "start", "end", name]]

    return track


def _make_cis_obsexp_fetcher(
    clr,
    expected,
    view_df,
    clr_weight_name="weight",
    expected_value_col="balanced.avg",
    view_name_col="name",
):
    """
    Construct a function that returns intra-chromosomal OBS/EXP for symmetrical regions
    defined in view_df.

    Used in `get_saddle()`.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : DataFrame
        Diagonal summary statistics for each chromosome.
    view_df: viewframe
        Viewframe with genomic regions.
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights.
    expected_value_col : str
        Name of the column in expected used for normalizing.
    view_name_col : str
        Name of column in view_df with region names.

    Returns
    -------
    getexpected(reg, _). 2nd arg is ignored.

    """
    expected = {k: x.values for k, x in expected.groupby("region")[expected_value_col]}
    view_df = view_df.set_index(view_name_col)

    def _fetch_cis_oe(reg1, reg2):
        reg1_coords = tuple(view_df.loc[reg1])
        # reg2_coords = tuple(view_df.loc[reg2])
        obs_mat = clr.matrix(balance=clr_weight_name).fetch(reg1_coords)
        exp_mat = toeplitz(expected[reg1][: obs_mat.shape[0]])
        return obs_mat / exp_mat

    return _fetch_cis_oe


def _make_trans_obsexp_fetcher(
    clr,
    expected,
    view_df,
    clr_weight_name="weight",
    expected_value_col="balanced.avg",
    view_name_col="name",
):

    """
    Construct a function that returns OBS/EXP for any pair of chromosomes.

    Used in `get_saddle()`.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : DataFrame or scalar
        Average trans values. If a scalar, it is assumed to be a global trans
        expected value. If a tuple of (dataframe, name), the dataframe must
        have a MultiIndex with 'region1' and 'region2' and must also have a column
        labeled ``name``, with the values of expected.
    view_df: viewframe
        Viewframe with genomic regions.
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights
    expected_value_col : str
        Name of the column in expected used for normalizing.
    view_name_col : str
        Name of column in view_df with region names.

    Returns
    -----
    getexpected(reg1, reg2)

    """

    if np.isscalar(expected):
        return lambda reg1, reg2: (
            clr.matrix(balance=clr_weight_name).fetch(reg1, reg2) / expected
        )

    elif type(expected) is pd.core.frame.DataFrame:

        expected = {
            k: x.values
            for k, x in expected.groupby(["region1", "region2"])[expected_value_col]
        }

        def _fetch_trans_exp(region1, region2):
            # Handle region flipping
            if (region1, region2) in expected.keys():
                return expected[region1, region2]
            elif (region2, region1) in expected.keys():
                return expected[region2, region1]
            # .loc is the right way to get [region1,region2] value from MultiIndex df:
            # https://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-indexing-with-hierarchical-index
            else:
                raise KeyError(
                    "trans-exp index is missing a pair of chromosomes: "
                    "{}, {}".format(region1, region2)
                )

        def _fetch_trans_oe(reg1, reg2):
            return clr.matrix(balance=clr_weight_name).fetch(
                reg1, reg2
            ) / _fetch_trans_exp(reg1, reg2)

        return _fetch_trans_oe

    else:
        raise ValueError("Unknown type of expected")


def _accumulate(
    S, C, getmatrix, digitized, reg1, reg2, min_diag=3, max_diag=-1, verbose=False
):
    """
    Helper function to aggregate across region pairs.
    If regions are identical, also masks returned matrices below min_diag and above max_diag.

    Used in `get_saddle()`.
    """

    n_bins = S.shape[0]
    matrix = getmatrix(reg1, reg2)

    if reg1 == reg2:
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


def _make_binedges(
    track_values, n_bins, quantiles=False, range_=None, qrange=(0.0, 1.0)
):
    """
    helper function for get_digitized
    """
    if quantiles:
        if range_ is not None:
            qlo, qhi = _ecdf(track_values, range_)
        elif len(qrange):
            qlo, qhi = qrange
        else:
            qlo, qhi = 0.0, 1.0
        q_edges = np.linspace(qlo, qhi, n_bins + 1)
        binedges = _quantile(track_values, q_edges)
        return binedges, qlo, qhi
    else:
        if range_ is not None:
            lo, hi = range_
        elif len(qrange):
            lo, hi = _quantile(track_values, qrange)
        else:
            lo, hi = np.nanmin(track_values), np.nanmax(track_values)
        binedges = np.linspace(lo, hi, n_bins + 1)
        return binedges, lo, hi


def get_digitized(
    track,
    n_bins,
    quantiles=False,
    range_=None,
    qrange=(0.0, 1.0),
    digitized_suffix=".d",
):
    """
    Digitize genomic signal tracks into integers between `1` and `n`.



    Parameters
    ----------
    track : 4-column DataFrame
        bedGraph-like dataframe with columns understood as (chrom,start,end,value).

    n_bins : int
        number of bins for signal quantization.

    quantiles : bool
        Whether to digitize by quantiles.

    range_ : tuple
        Low and high values used for binning genome-wide track values, e.g.
        if `range`=(-0.05, 0.05), `n-bins` equidistant bins would be generated.

    qrange : tuple
        The fraction of the genome-wide range of the track values used to
        generate bins. E.g., if `qrange`=(0.02, 0.98) the lower bin would
        start at the 2nd percentile and the upper bin would end at the 98th
        percentile of the genome-wide signal. Use to exclude extreme track
        values for making saddles.

    digitized_suffix : str
        suffix to append to fourth column name

    Returns
    -------
    digitized : DataFrame
        New bedGraph-like dataframe with value column and an additional
        digitized value column with name suffixed by '.d'
        The digized column is returned as a categorical.
    binedges : 1D array (length n + 1)
        Bin edges used in quantization of track. For `n` bins, there are `n + 1`
        edges. See encoding details in Notes.

    Notes
    -----
    The digital encoding is as follows:

    - `1..n` <-> values assigned to histogram bins
    - `0` <-> left outlier values
    - `n+1` <-> right outlier values
    - `-1` <-> missing data (NaNs)

    """

    ###TODO add input check for track, qrange, range_

    digitized = track.copy()
    track_value_col = track.columns[3]

    track_values = track[track_value_col]
    track_values.loc[track_values.isnull()] = np.nan
    track_values = track_values.values

    digitized_col = track_value_col + digitized_suffix

    binedges, lo, hi = _make_binedges(
        track_values, n_bins, quantiles=quantiles, range_=range_, qrange=qrange
    )

    digitized[digitized_col] = np.digitize(track_values, binedges, right=False)

    mask = track[track_value_col].isnull()
    digitized.loc[mask, digitized_col] = -1

    digitized_cats = pd.CategoricalDtype(
        categories=np.arange(-1, n_bins + 2), ordered=True
    )
    digitized = digitized.astype({digitized_col: digitized_cats})

    # return a 4-column digitized track
    digitized = digitized[list(track.columns[:3]) + [digitized_col]]
    return digitized, binedges


def get_saddle(
    clr,
    expected,
    digitized_track,
    contact_type,
    view_df=None,
    clr_weight_name="weight",
    expected_value_col="balanced.avg",
    view_name_col="name",
    min_diag=3,
    max_diag=-1,
    trim_outliers=False,
    verbose=False,
):
    """
    Get a matrix of average interactions between genomic bin
    pairs as a function of a specified genomic track.

    The provided genomic track must a dataframe with a categorical
    column, as generated by `get_digitized()`.

    Parameters
    ----------
    clr : cooler.Cooler
        Observed matrix.
    expected : DataFrame in expected format
        Diagonal summary statistics for each chromosome, and name of the column
        with the values of expected to use.
    contact_type : str
        If 'cis' then only cis interactions are used to build the matrix.
        If 'trans', only trans interactions are used.
    digitized_track : DataFrame with digitized value column
        A track, i.e. BedGraph-like dataframe, of digitized signal.
        The value column specifies a category for every position in the track.
        Generated by get_digitzed() from track.
    view_df: viewframe
        Viewframe with genomic regions. If none, generate from track chromosomes.
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights.
    expected_value_col : str
        Name of the column in expected used for normalizing.
    view_name_col : str
        Name of column in view_df with region names.
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

    ### TODO add input validation for: track, expeced,
    if type(digitized_track.dtypes[3]) is not pd.core.dtypes.dtypes.CategoricalDtype:
        raise ValueError(
            "a digitized track, where the value column is a"
            + "pandas categorical must be provided as input. see get_digitized()."
        )
    digitized_col = digitized_track.columns[3]
    cats = digitized_track[digitized_col].dtype.categories.values
    n_bins = len(cats[cats > -1]) - 2

    if view_df is None:
        view_df = _view_from_track(digitized_track)
    else:
        view_df = _make_cooler_view(view_df, clr)

    digitized_tracks = {}
    for num, reg in view_df.iterrows():
        digitized_reg = bioframe.select(digitized_track, reg)
        digitized_tracks[reg[view_name_col]] = digitized_reg[digitized_col]

    ### set "cis" or "trans" for supports (regions to iterate over) and matrix fetcher
    if contact_type == "cis":
        supports = list(zip(view_df[view_name_col], view_df[view_name_col]))
        if not bioframe.is_cataloged(
            expected, view_df, df_view_col="region", view_name_col=view_name_col
        ):
            raise ValueError("Region names in expected are not cataloged in view_df.")
        getmatrix = _make_cis_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
    elif contact_type == "trans":
        supports = list(combinations(view_df[view_name_col], 2))
        getmatrix = _make_trans_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
    else:
        raise ValueError("Allowed values for contact_type are 'cis' or 'trans'.")

    # n_bins here includes 2 open bins for values <lo and >hi.
    interaction_sum = np.zeros((n_bins + 2, n_bins + 2))
    interaction_count = np.zeros((n_bins + 2, n_bins + 2))

    for reg1, reg2 in supports:
        _accumulate(
            interaction_sum,
            interaction_count,
            getmatrix,
            digitized_tracks,
            reg1,
            reg2,
            min_diag=min_diag,
            max_diag=max_diag,
            verbose=verbose,
        )

    interaction_sum += interaction_sum.T
    interaction_count += interaction_count.T

    if trim_outliers:
        interaction_sum = interaction_sum[1:-1, 1:-1]
        interaction_count = interaction_count[1:-1, 1:-1]

    return interaction_sum, interaction_count


def saddleplot(
    track,
    saddledata,
    n_bins,
    quantiles=False,
    range_=None,
    qrange=(0.0, 1.0),
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
    track : pd.DataFrame
        See get_digitized() for details.
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

    track_value_col = track.columns[3]
    track_values = track[track_value_col].values

    digitized_track, binedges = get_digitized(
        track, n_bins, quantiles=quantiles, range_=range_, qrange=qrange
    )
    x = digitized_track[digitized_track.columns[3]].values.astype(int).copy()
    x = x[(x > 0) & (x < len(binedges) + 1)]
    hist = np.bincount(x, minlength=len(binedges) + 1)

    if quantiles:
        binedges, lo, hi = _make_binedges(
            track_values, n_bins, quantiles=quantiles, range_=range_, qrange=qrange
        )
        binedges = np.linspace(lo, hi, n_bins + 1)

    # Histogram and saddledata are flanked by outlier bins
    n = saddledata.shape[0]
    X, Y = np.meshgrid(binedges, binedges)
    C = saddledata
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
