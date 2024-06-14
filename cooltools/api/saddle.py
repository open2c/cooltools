from itertools import combinations
from functools import partial
from scipy.linalg import toeplitz
import numpy as np
import pandas as pd
from ..lib import numutils
from ..lib.checks import (
    is_compatible_viewframe,
    is_valid_expected,
    is_cooler_balanced,
    is_track,
)
from ..lib.common import view_from_track, align_track_with_cooler

import warnings
import bioframe


def _merge_dict(a, b):
    return {**a, **b}


def _ecdf(x, v, side="left"):
    """mask_bad_bins
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
        Diagonal summary statistics for a number of regions.
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
    expected = {
        k: x.values
        for k, x in expected.groupby(["region1", "region2"])[expected_value_col]
    }
    view_df = view_df.set_index(view_name_col)

    def _fetch_cis_oe(reg1, reg2):
        reg1_coords = tuple(view_df.loc[reg1])
        reg2_coords = tuple(view_df.loc[reg2])
        obs_mat = clr.matrix(balance=clr_weight_name).fetch(reg1_coords, reg2_coords)
        exp_mat = toeplitz(expected[reg1, reg2][: obs_mat.shape[0]])
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

    view_df = view_df.set_index(view_name_col)

    if np.isscalar(expected):

        def _fetch_trans_oe(reg1, reg2):
            reg1_coords = tuple(view_df.loc[reg1])
            reg2_coords = tuple(view_df.loc[reg2])
            obs_mat = clr.matrix(balance=clr_weight_name).fetch(
                reg1_coords, reg2_coords
            )
            return obs_mat / expected

        return _fetch_trans_oe

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
            reg1_coords = tuple(view_df.loc[reg1])
            reg2_coords = tuple(view_df.loc[reg2])
            obs_mat = clr.matrix(balance=clr_weight_name).fetch(
                reg1_coords, reg2_coords
            )
            exp = _fetch_trans_exp(reg1, reg2)
            return obs_mat / exp

        return _fetch_trans_oe

    else:
        raise ValueError("Unknown type of expected")


def _accumulate_trans(
    S, C, getmatrix, digitized, reg1, reg2, max_dist=1, verbose=False
):
    """
    Helper function to aggregate across region pairs for trans-data
    S and C - are 3D
    """

    n_bins = S.shape[-1]
    matrix = getmatrix(reg1, reg2)

    if verbose:
        print("regions {} vs {}".format(reg1, reg2))

    for i in range(n_bins):
        row_mask = digitized[reg1] == i
        for j in range(n_bins):
            col_mask = digitized[reg2] == j
            data = matrix[row_mask, :][:, col_mask]
            is_finite_mask = np.isfinite(data)
            data = data[is_finite_mask]
            S[max_dist-1, i, j] += np.sum(data)
            C[max_dist-1, i, j] += float(len(data))


def _accumulate_cis(
    S, C, getmatrix, digitized, reg1, reg2, max_dist, verbose=False
):
    """
    Helper function to aggregate across region pairs for cis-data.
    Do it separately for every diagonal.
    """

    if reg1 != reg2:
        raise ValueError("this is special version of accumulate for cis data only ...")

    n_bins = S.shape[-1]
    matrix = getmatrix(reg1, reg2)

    if verbose:
        print("regions {} vs {}".format(reg1, reg2))

    # Toeplitz matrix with distance for every pixel ...
    _dist_vec = np.arange(matrix.shape[0])
    dist_mat = np.abs(_dist_vec[None,:] - _dist_vec[:,None])

    for i in range(n_bins):
        row_mask = digitized[reg1] == i
        for j in range(n_bins):
            col_mask = digitized[reg2] == j
            data = matrix[row_mask, :][:, col_mask]
            dist = dist_mat[row_mask, :][:, col_mask]
            is_finite_mask = np.isfinite(data)
            data = data[is_finite_mask]
            dist = dist[is_finite_mask]
            # S unrolled by distances - inefficient memory access - isn't it ?
            S[:, i, j] += np.bincount(dist, weights=data, minlength=max_dist)
            # C unrolled by distances
            C[:, i, j] += np.bincount(dist, minlength=max_dist).astype(float)


def _make_binedges(track_values, n_bins, vrange=None, qrange=None):
    """
    Helper function to make bins for `get_digitized()`.

    Nakes binedges in real space from value limits provided by vrange,
    or in quantile space from quantile limits provided by qrange.

    """

    if qrange is not None and vrange is not None:
        raise ValueError("only one of vrange or qrange can be supplied")

    elif vrange is not None:
        lo, hi = vrange
        if lo > hi:
            raise ValueError("vrange does not satisfy vrange[0]<vrange[1]")
        binedges = np.linspace(lo, hi, n_bins + 1)
        return binedges, lo, hi

    elif qrange is not None:
        qlo, qhi = qrange
        if qlo < 0.0 or qhi > 1.0:
            raise ValueError("qrange must specify quantiles in (0.0,1.0)")
        if qlo > qhi:
            raise ValueError("qrange does not satisfy qrange[0]<qrange[1]")
        q_edges = np.linspace(qlo, qhi, n_bins + 1)
        binedges = _quantile(track_values, q_edges)
        return binedges, qlo, qhi

    else:
        raise ValueError("either vrange or qrange must be supplied")


def digitize(
    track,
    n_bins,
    vrange=None,
    qrange=None,
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
    vrange : tuple
        Low and high values used for binning track values.
        E.g. if `vrange`=(-0.05, 0.05), equal width bins would be generated
        between the value -0.05 and 0.05.
    qrange : tuple
        Low and high values for quantile binning track values.
        E.g., if `qrange`=(0.02, 0.98) the lower bin would
        start at the 2nd percentile and the upper bin would end at the 98th
        percentile of the track value range.
        Low must be 0.0 or more, high must be 1.0 or less.
    digitized_suffix : str
        suffix to append to the track value name in the fourth column.

    Returns
    -------
    digitized : DataFrame
        New track dataframe (bedGraph-like) with
        digitized value column with name suffixed by '.d'
        The digized column is returned as a categorical.
    binedges : 1D array (length n + 1)
        Bin edges used in quantization of track. For `n` bins, there are `n + 1`
        edges. See encoding details in Notes.

    Notes
    -----
    The digital encoding is as follows:

    - `1..n` <-> values assigned to bins defined by vrange or qrange
    - `0` <-> left outlier values
    - `n+1` <-> right outlier values
    - `-1` <-> missing data (NaNs)

    """

    if type(n_bins) is not int:
        raise ValueError("n_bins must be provided as an int")
    is_track(track, raise_errors=True)

    digitized = track.copy()
    track_value_col = track.columns[3]
    digitized_col = track_value_col + digitized_suffix

    track_values = track[track_value_col].copy()
    track_values = track_values.astype({track_value_col: np.float64}).values

    binedges, lo, hi = _make_binedges(
        track_values, n_bins, vrange=vrange, qrange=qrange
    )
    digitized[digitized_col] = np.digitize(track_values, binedges, right=False)
    # re-assign values equal to the max value to bin n
    digitized.loc[
        digitized[track_value_col] == np.max(binedges), track_value_col
    ] = n_bins

    mask = track[track_value_col].isnull()
    digitized.loc[mask, digitized_col] = -1

    digitized_cats = pd.CategoricalDtype(
        categories=np.arange(-1, n_bins + 2), ordered=True
    )
    digitized = digitized.astype({digitized_col: digitized_cats})

    # return a 4-column digitized track
    digitized = digitized[list(track.columns[:3]) + [digitized_col]]
    return digitized, binedges


def saddle(
    clr,
    expected,
    track,
    contact_type,
    n_bins,
    vrange=None,
    qrange=None,
    view_df=None,
    clr_weight_name="weight",
    expected_value_col="balanced.avg",
    view_name_col="name",
    min_diag=3,
    max_diag=-1,
    trim_outliers=False,
    verbose=False,
    drop_track_na=False,
):
    """
    Get a matrix of average interactions between genomic bin
    pairs as a function of a specified genomic track.

    The provided genomic track is either:
    (a) digitized inside this function by passing 'n_bins', and one of 'v_range' or 'q_range'
    (b) passed as a pre-digitized track with a categorical value column as generated by `get_digitized()`.

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
    track : DataFrame
        A track, i.e. BedGraph-like dataframe, which is digitized with
        the options n_bins, vrange and qrange. Can optionally be passed
        as a pre-digitized dataFrame with a categorical value column,
        as generated by get_digitzied(), also passing n_bins as None.
    n_bins : int or None
        number of bins for signal quantization. If None, then track must
        be passed as a pre-digitized track.
    vrange : tuple
        Low and high values used for binning track values.
        See get_digitized().
    qrange : tuple
        Low and high values for quantile binning track values.
        Low must be 0.0 or more, high must be 1.0 or less.
        Only one of vrange or qrange can be passed. See get_digitzed().
    view_df: viewframe
        Viewframe with genomic regions. If none, generate from track chromosomes.
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights.
        Using raw unbalanced data is not supported for saddles.
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
    drop_track_na : bool, optional
        If True then drops NaNs in input track (as if they were missing),
        If False then counts NaNs as present in dataframe.
        In general, this only adds check form chromosomes that have all missing values, but does not affect the results.
    Returns
    -------
    interaction_sum : 2D array
        The matrix of summed interaction probability between two genomic bins
        given their values of the provided genomic track.
    interaction_count : 2D array
        The matrix of the number of genomic bin pairs that contributed to the
        corresponding pixel of ``interaction_sum``.
    """

    if type(n_bins) is int:
        # perform digitization
        track = align_track_with_cooler(
            track,
            clr,
            view_df=view_df,
            clr_weight_name=clr_weight_name,
            mask_clr_bad_bins=True,
            drop_track_na=drop_track_na,  # this adds check for chromosomes that have all missing values
        )
        digitized_track, binedges = digitize(
            track.iloc[:, :4],
            n_bins,
            vrange=vrange,
            qrange=qrange,
            digitized_suffix=".d",
        )
        digitized_col = digitized_track.columns[3]

    elif n_bins is None:
        # assume and test if track is pre-digitized
        digitized_track = track
        digitized_col = digitized_track.columns[3]
        is_track(track.astype({digitized_col: "float"}), raise_errors=True)
        if (
            type(digitized_track.dtypes[3])
            is not pd.core.dtypes.dtypes.CategoricalDtype
        ):
            raise ValueError(
                "when n_bins=None, saddle assumes the track has been "
                + "pre-digitized and the value column is a "
                + "pandas categorical. See get_digitized()."
            )
        cats = digitized_track[digitized_col].dtype.categories.values
        # cats has two additional categories, 0 and n_bins+1, for values
        # falling outside range, as well as -1 for NAs.
        n_bins = len(cats[cats > -1]) - 2
    else:
        raise ValueError("n_bins must be provided as int or None")

    if view_df is None:
        view_df = view_from_track(digitized_track)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,  # just in case
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # make sure provided expected is compatible
    try:
        _ = is_valid_expected(
            expected,
            contact_type,
            view_df,
            verify_cooler=clr,
            expected_value_cols=[
                expected_value_col,
            ],
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided expected is not compatible") from e

    # check if cooler is balanced
    if clr_weight_name:
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e

    digitized_tracks = {}
    for num, reg in view_df.iterrows():
        digitized_reg = bioframe.select(digitized_track, reg)
        digitized_tracks[reg[view_name_col]] = digitized_reg[digitized_col]

    # set "cis" or "trans" for supports (regions to iterate over) and matrix fetcher
    if contact_type == "cis":
        # precalculate max_dist using provided expected:
        max_dist = expected["dist"].max() + 1
        # redefine max_diag if it is -1:
        max_diag = None if max_diag < 0 else max_diag
        # only symmetric intra-chromosomal regions :
        supports = list(zip(view_df[view_name_col], view_df[view_name_col]))
        getmatrix = _make_cis_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
        # n_bins here includes 2 open bins for values <lo and >hi.
        interaction_sum = np.zeros((max_dist, n_bins + 2, n_bins + 2))
        interaction_count = np.zeros((max_dist, n_bins + 2, n_bins + 2))
        for reg1, reg2 in supports:
            _accumulate_cis(
                interaction_sum,
                interaction_count,
                getmatrix,
                digitized_tracks,
                reg1,
                reg2,
                max_dist,
                verbose=verbose
            )
        # symmetrise by adding transpose "saddle"
        # transpose 2nd and 3rd coords by leaving the 1st alone
        interaction_sum += interaction_sum.transpose(0,2,1)
        interaction_count += interaction_count.transpose(0,2,1)
        if trim_outliers:
            interaction_sum = interaction_sum[:, 1:-1, 1:-1]
            interaction_count = interaction_count[:, 1:-1, 1:-1]
        # in order to stay compatible - return aggregated sums and counts for now:
        return (
            np.nansum(interaction_sum[min_diag:max_diag], axis=0),
            np.nansum(interaction_count[min_diag:max_diag], axis=0)
        )
    elif contact_type == "trans":
        # precalculate max_dist using provided expected:
        max_dist = 1
        # asymmetric inter-chromosomal regions :
        view_df_index = view_df.set_index(view_name_col)
        supports = []
        for i1, i2 in combinations(view_df[view_name_col], 2):
            if view_df_index.at[i1, "chrom"] != view_df_index.at[i2, "chrom"]:
                supports.append((i1, i2))
        getmatrix = _make_trans_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
        # n_bins here includes 2 open bins for values <lo and >hi.
        interaction_sum = np.zeros((max_dist, n_bins + 2, n_bins + 2))
        interaction_count = np.zeros((max_dist, n_bins + 2, n_bins + 2))
        for reg1, reg2 in supports:
            _accumulate_trans(
                interaction_sum,
                interaction_count,
                getmatrix,
                digitized_tracks,
                reg1,
                reg2,
                max_dist,
                verbose=verbose,
            )
        # symmetrise by adding transpose "saddle"
        # transpose 2nd and 3rd coords by leaving the 1st alone
        interaction_sum += interaction_sum.transpose(0,2,1)
        interaction_count += interaction_count.transpose(0,2,1)
        if trim_outliers:
            interaction_sum = interaction_sum[:, 1:-1, 1:-1]
            interaction_count = interaction_count[:, 1:-1, 1:-1]
        # in order to stay compatible - return aggregated sums and counts for now:
        return (
            np.nansum(interaction_sum, axis=0),
            np.nansum(interaction_count, axis=0)
        )
    else:
        raise ValueError("Allowed values for contact_type are 'cis' or 'trans'.")


def saddle_stack(
    clr,
    expected,
    track,
    contact_type,
    n_bins,
    vrange=None,
    qrange=None,
    view_df=None,
    clr_weight_name="weight",
    expected_value_col="balanced.avg",
    view_name_col="name",
    trim_outliers=False,
    verbose=False,
    drop_track_na=False,
):
    """
    Get a matrix of average interactions between genomic bin
    pairs as a function of a specified genomic track.

    The provided genomic track is either:
    (a) digitized inside this function by passing 'n_bins', and one of 'v_range' or 'q_range'
    (b) passed as a pre-digitized track with a categorical value column as generated by `get_digitized()`.

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
    track : DataFrame
        A track, i.e. BedGraph-like dataframe, which is digitized with
        the options n_bins, vrange and qrange. Can optionally be passed
        as a pre-digitized dataFrame with a categorical value column,
        as generated by get_digitzied(), also passing n_bins as None.
    n_bins : int or None
        number of bins for signal quantization. If None, then track must
        be passed as a pre-digitized track.
    vrange : tuple
        Low and high values used for binning track values.
        See get_digitized().
    qrange : tuple
        Low and high values for quantile binning track values.
        Low must be 0.0 or more, high must be 1.0 or less.
        Only one of vrange or qrange can be passed. See get_digitzed().
    view_df: viewframe
        Viewframe with genomic regions. If none, generate from track chromosomes.
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights.
        Using raw unbalanced data is not supported for saddles.
    expected_value_col : str
        Name of the column in expected used for normalizing.
    view_name_col : str
        Name of column in view_df with region names.
    trim_outliers : bool, optional
        Remove first and last row and column from the output matrix.
    verbose : bool, optional
        If True then reports progress.
    drop_track_na : bool, optional
        If True then drops NaNs in input track (as if they were missing),
        If False then counts NaNs as present in dataframe.
        In general, this only adds check form chromosomes that have all missing values, but does not affect the results.
    Returns
    -------
    interaction_sum : 3D array
        The matrix of summed interaction probability between two genomic bins
        given their values of the provided genomic track. The first dimension
        corresponds to the index of the diagonal where the data was collected from.
    interaction_count : 3D array
        The matrix of the number of genomic bin pairs that contributed to the
        corresponding pixel of ``interaction_sum``. The first dimension
        corresponds to the index of the diagonal where the data was collected from.
    """

    if type(n_bins) is int:
        # perform digitization
        track = align_track_with_cooler(
            track,
            clr,
            view_df=view_df,
            clr_weight_name=clr_weight_name,
            mask_clr_bad_bins=True,
            drop_track_na=drop_track_na,  # this adds check for chromosomes that have all missing values
        )
        digitized_track, binedges = digitize(
            track.iloc[:, :4],
            n_bins,
            vrange=vrange,
            qrange=qrange,
            digitized_suffix=".d",
        )
        digitized_col = digitized_track.columns[3]

    elif n_bins is None:
        # assume and test if track is pre-digitized
        digitized_track = track
        digitized_col = digitized_track.columns[3]
        is_track(track.astype({digitized_col: "float"}), raise_errors=True)
        if (
            type(digitized_track.dtypes[3])
            is not pd.core.dtypes.dtypes.CategoricalDtype
        ):
            raise ValueError(
                """when n_bins=None, saddle assumes the track has been
                pre-digitized and the value column is a
                pandas categorical. See get_digitized()."""
            )
        cats = digitized_track[digitized_col].dtype.categories.values
        # cats has two additional categories, 0 and n_bins+1, for values
        # falling outside range, as well as -1 for NAs.
        n_bins = len(cats[cats > -1]) - 2
    else:
        raise ValueError("n_bins must be provided as int or None")

    if view_df is None:
        view_df = view_from_track(digitized_track)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,  # just in case
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # make sure provided expected is compatible
    try:
        _ = is_valid_expected(
            expected,
            contact_type,
            view_df,
            verify_cooler=clr,
            expected_value_cols=[
                expected_value_col,
            ],
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided expected is not compatible") from e

    # check if cooler is balanced
    if clr_weight_name:
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e

    digitized_tracks = {}
    for num, reg in view_df.iterrows():
        digitized_reg = bioframe.select(digitized_track, reg)
        digitized_tracks[reg[view_name_col]] = digitized_reg[digitized_col]

    # set "cis" or "trans" for supports (regions to iterate over) and matrix fetcher
    if contact_type == "cis":
        # precalculate max_dist using provided expected:
        max_dist = expected["dist"].max() + 1
        # only symmetric intra-chromosomal regions :
        supports = list(zip(view_df[view_name_col], view_df[view_name_col]))
        getmatrix = _make_cis_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
        _acc = partial(_accumulate_cis,
            getmatrix=getmatrix,
            digitized=digitized_tracks,
            max_dist=max_dist,
            verbose=verbose,
        )
    elif contact_type == "trans":
        # fake max_dist for trans - 1:
        max_dist = 1
        # asymmetric inter-chromosomal regions :
        supports = list(combinations(view_df[view_name_col], 2))
        supports = [
            i
            for i in supports
            if (
                view_df["chrom"].loc[view_df[view_name_col] == i[0]].values
                != view_df["chrom"].loc[view_df[view_name_col] == i[1]].values
            )
        ]
        getmatrix = _make_trans_obsexp_fetcher(
            clr,
            expected,
            view_df,
            view_name_col=view_name_col,
            expected_value_col=expected_value_col,
            clr_weight_name=clr_weight_name,
        )
        _acc = partial(_accumulate_trans,
            getmatrix=getmatrix,
            digitized=digitized_tracks,
            max_dist=max_dist,
            verbose=verbose,
        )
    else:
        raise ValueError("Allowed values for contact_type are 'cis' or 'trans'.")

    # n_bins here includes 2 open bins for values <lo and >hi.
    _sum = np.zeros((max_dist, n_bins + 2, n_bins + 2))
    _count = np.zeros((max_dist, n_bins + 2, n_bins + 2))

    for reg1, reg2 in supports:
        _acc( _sum, _count, reg1=reg1, reg2=reg2 )

    # symmetrise by adding transpose "saddle"
    # transpose 2nd and 3rd coords by leaving the 1st alone
    _sum += _sum.transpose(0,2,1)
    _count += _count.transpose(0,2,1)
    if trim_outliers:
        _sum = _sum[:, 1:-1, 1:-1]
        _count = _count[:, 1:-1, 1:-1]
    return _sum, _count


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
