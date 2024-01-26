"""
This module enables construction of observed over expected pixels tables and
storing them inside a cooler.

It includes 3 main functions.
expected_full - is a convenience function that calculates cis and trans-expected
    and "stitches" them togeter. Such a stitched expected that "covers"
    entire Hi-C heatmap can be easily merged with the pixel table.
expected_full_fast - generated the same output as `expected_full` but ~2x faster.
    Efficiency is achieved through calculating cis and trans expected in one
    pass of the pixel table. Post-processing is not fully implemented yet.
obs_over_exp - is a function that merges pre-calculated full expected with the pixel
    table in pd.DataFrame or dask.DataFrame formats.
obs_over_exp_generator - is a function/generator(lazy iterator) that wraps
    `obs_over_exp` function and yields chunks of observed/expected pixel table.
    Such a "stream" can be used in cooler.create as a "pixels" argument to write
    obs/exp cooler-file.

It also includes 3 helper functions (used in `expected_full_fast`):
make_pairwise_expected_table - a function that creates an empty table for the
    full expected with all the right sizes and number of valid pixels pre-filled.
    Combines functionality of `make_diag_tables` and `make_block_tables` from the API.
sum_pairwise - a function that calculates the full pixel summary for all pairwise
    combinations of the regions in the view, and for each genomic separation for
    cis-combinations of regions. In a nutshell - it calls `make_pairwise_expected_table`
    to generate empty table for expected, and it gets filled by applying `_sum_` to the
    pixels table.
_sum_ - a function that does the actual summing of pixel values grouped by regions
    and genomic separations - can work on a chunk of pixel table.
"""
import time
import logging

import numpy as np
import pandas as pd
import multiprocess as mp

import cooler
from cooler.util import partition

from itertools import (
    tee,
    chain,
    combinations_with_replacement
)

from functools import reduce, partial

from cooltools import (
    expected_cis,
    expected_trans
)
from cooltools.lib.common import (
    assign_supports,
    make_cooler_view
)

from cooltools.lib import (
    is_compatible_viewframe,
    is_cooler_balanced
)

from cooltools.api.expected import make_block_table, make_diag_tables

from cooltools.lib.schemas import (
    diag_expected_dtypes,
    block_expected_dtypes
)

from cooltools.sandbox import expected_smoothing

# common expected_df column names, take from schemas
_REGION1_NAME = list(diag_expected_dtypes)[0]
_REGION2_NAME = list(diag_expected_dtypes)[1]
_DIST_NAME = list(diag_expected_dtypes)[2]
_NUM_VALID_NAME = list(diag_expected_dtypes)[3]

TRANS_DIST_VALUE = -1  # special value for the "genomic distance" for the trans data

logging.basicConfig(level=logging.INFO)


def expected_full(
        clr,
        view_df=None,
        smooth_cis=False,
        aggregate_smoothed=False,
        smooth_sigma=0.1,
        aggregate_trans=False,
        expected_column_name="expected",
        ignore_diags=2,
        clr_weight_name='weight',
        chunksize=10_000_000,
        nproc=4,
    ):
    """
    Generate a DataFrame with expected for *all* 2D regions
    tiling entire heatmap in clr.
    Such 2D regions are defined as all pairwise combinations
    of the regions in view_df. Average distance decay is calculated
    for every cis-region (e.g. inter- and intra-arms), and
    a "simple" average over each block is caculated for trans-
    regions.

    When sub-chromosomal view is provided, trans averages
    can be aggregated back to the level of full chromosomes.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        expected is calculated for all pairwise combinations of regions
        in view_df. Distance dependent expected is calculated for cis
        regions, and block-level average is calculated for trans regions.
    smooth_cis: bool
        Apply smoothing to cis-expected. Will be stored in an additional column
    aggregate_smoothed: bool
        When smoothing cis expected, average over all regions, ignored without smoothing.
    smooth_sigma: float
        Control smoothing with the standard deviation of the smoothing Gaussian kernel.
        Ignored without smoothing.
    aggregate_trans : bool
        Aggregate trans-expected at the inter-chromosomal level.
    expected_column_name : str
        Name of the column where to store combined expected
    ignore_diags : int, optional
        Number of intial diagonals to exclude for calculation of distance dependent
        expected.
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    chunksize : int, optional
        Size of pixel table chunks to process
    nproc : int, optional
        How many processes to use for calculation
    Returns
    -------
    expected_df: pd.DataFrame
        cis and trans expected combined together
    """

    # contacs vs distance - i.e. intra/cis expected
    time_start = time.perf_counter()
    cvd = expected_cis(
        clr,
        view_df=view_df,
        intra_only=False,  # get cvd for all 2D regions
        smooth=smooth_cis,
        smooth_sigma=smooth_sigma,
        aggregate_smoothed=aggregate_smoothed,
        clr_weight_name=clr_weight_name,
        ignore_diags=ignore_diags,
        chunksize=chunksize,
        nproc=nproc,
    )
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"Done calculating cis expected in {time_elapsed:.3f} sec ...")

    # contacts per block - i.e. inter/trans expected
    time_start = time.perf_counter()
    cpb = expected_trans(
        clr,
        view_df=view_df,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
        nproc=nproc,
    )
    # pretend that they also have a "dist" to make them mergeable with cvd
    cpb["dist"] = TRANS_DIST_VALUE
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"Done calculating trans expected in {time_elapsed:.3f} sec ...")

    # annotate expected_df with the region index and chromosomes
    view_label = view_df \
                .reset_index() \
                .rename(columns={"index":"r"}) \
                .set_index("name")

    # which expected column to use, based on requested "modifications":
    cis_expected_name = "balanced.avg" if clr_weight_name else "count.avg"
    if smooth_cis:
        cis_expected_name = f"{cis_expected_name}.smoothed"
        if aggregate_smoothed:
            cis_expected_name = f"{cis_expected_name}.agg"
    # copy to the prescribed column for the final output:
    cvd[expected_column_name] = cvd[cis_expected_name].copy()

    # aggregate trans if requested and deide which trans-expected column to use:
    trans_expected_name = "balanced.avg" if clr_weight_name else "count.avg"
    if aggregate_trans:
        trans_expected_name = f"{trans_expected_name}.agg"
        additive_cols = ["n_valid","count.sum"]
        if clr_weight_name:
            additive_cols.append("balanced.sum")
        # groupby chrom1, chrom2 and aggregate additive fields (sums and n_valid):
        _cpb_agg = cpb.groupby(
            [
                view_label["chrom"].loc[cpb["region1"]].to_numpy(),  # chrom1
                view_label["chrom"].loc[cpb["region2"]].to_numpy(),  # chrom2
            ]
        )[additive_cols].transform("sum")
        # recalculate aggregated averages:
        cpb["count.avg.agg"] = _cpb_agg["count.sum"]/_cpb_agg["n_valid"]
        if clr_weight_name:
            cpb["balanced.avg.agg"] = _cpb_agg["balanced.sum"]/_cpb_agg["n_valid"]
    # copy to the prescribed column for the final output:
    cpb[expected_column_name] = cpb[trans_expected_name].copy()

    # concatenate cvd and cpb (cis and trans):
    expected_df = pd.concat([cvd, cpb], ignore_index=True)

    # add r1 r2 labels to the final dataframe for obs/exp merging
    expected_df["r1"] = view_label["r"].loc[expected_df["region1"]].to_numpy()
    expected_df["r2"] = view_label["r"].loc[expected_df["region2"]].to_numpy()

    # and return joined cis/trans expected in the same format
    logging.info(f"Returning combined expected DataFrame.")
    # consider purging unneccessary columns here
    return expected_df


def make_pairwise_expected_table(clr, view_df, clr_weight_name):
    """
    create a DataFrame for accumulating expected summaries (blocks and diagonal ones)
    it also contains "n_valid" column for dividing summaries by.
    """

    # create pairwise combinations of regions from view_df
    cis_combs, trans_combs = tee(
        combinations_with_replacement(view_df.itertuples(), 2)
    )
    # filter cis
    cis_combs = ((r1, r2) for r1, r2 in cis_combs if (r1.chrom == r2.chrom))
    # filter trans
    trans_combs = ((r1, r2) for r1, r2 in trans_combs if (r1.chrom != r2.chrom))

    # cis dtables ...
    # unzip regions1 regions2 defining the blocks for summary collection
    regions1, regions2 = zip(*cis_combs)
    regions1 = pd.DataFrame(regions1).drop(columns=["Index"])
    regions2 = pd.DataFrame(regions2).drop(columns=["Index"])
    # prepare dtables: for every region a table with diagonals, number of valid pixels, etc
    dtables = make_diag_tables(clr, regions1, regions2, clr_weight_name=clr_weight_name)

    # trans btables ...
    # unzip regions1 regions2 defining the blocks for summary collection
    regions1, regions2 = zip(*trans_combs)
    regions1 = pd.DataFrame(regions1).drop(columns=["Index"])
    regions2 = pd.DataFrame(regions2).drop(columns=["Index"])
    btables = make_block_table(clr, regions1, regions2, clr_weight_name=clr_weight_name)

    # rearrange dtables and btables to prepare their concatenation:
    _tables = []
    for _r1, chrom1, name1 in view_df[["chrom","name"]].itertuples():
        for _r2, chrom2, name2 in view_df[["chrom","name"]].itertuples():
            # upper triangle
            if (_r2 >= _r1):
                if (chrom1==chrom2):
                    df = dtables[(name1, name2)].reset_index()
                    df.insert(0, "r2", _r2)
                    df.insert(0, "r1", _r1)
                    _tables.append(df.set_index(["r1", "r2", _DIST_NAME]))
                if (chrom1 != chrom2):
                    df = pd.DataFrame(btables[(name1, name2)], index=[0])
                    df.insert(0, _DIST_NAME, TRANS_DIST_VALUE)  # special trans-value for distance
                    df.insert(0, "r2", _r2)
                    df.insert(0, "r1", _r1)
                    _tables.append(df.set_index(["r1", "r2", _DIST_NAME]))

    # return all concatenated DataFrame for cis and trans blocks:
    return pd.concat( _tables )


def _sum_(clr, fields, transforms, clr_weight_name, regions, span):
    """
    calculates summaries for every pixel block defined by pairwise
    combinations of the specified regions: calculates per-diagonal
    sums for intra-chromosomal blocks and overall sums for inter-
    chromosomal blocks.

    pixels in the blocks are labeled with regions' serial numbers,
    and groupby-s are used grouping.

    Trans-values have a special value for their diagonal/dist -1

    Return:
    dictionary of DataFrames with diagonal sums and overall sums
    for the "fields", indexed with the (i,j) combinations of serial
    numbers of the regions.
    """
    lo, hi = span
    bins = clr.bins()[:]
    bins["r"] = assign_supports(bins, regions)  # astype float
    # subset a group of pixels abd annotate it with with the region-labeled bins
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)

    # pre-filter unannotated pixels and masked out by balancing weights
    if clr_weight_name is None:
        pixels = pixels.dropna(subset=["r1", "r2"])
    else:
        pixels = pixels.dropna(
            subset=["r1", "r2", clr_weight_name + "1", clr_weight_name + "2"]
        )

    # cast to int, as there are no more NaNs among r1/r2
    pixels = pixels.astype({"r1":int, "r2":int})

    # create a cis-mask, trans-mask
    cis_mask = pixels["chrom1"] == pixels["chrom2"]

    # initialize _DIST_NAME as 0 for all pixels
    # consider using -1 as a special value to distinguish trans data easily ...
    pixels.loc[:, _DIST_NAME] = TRANS_DIST_VALUE
    # calculate actual genomic _DIST_NAME for cis-pixels:
    pixels.loc[cis_mask, _DIST_NAME] = pixels.loc[cis_mask, "bin2_id"] - pixels.loc[cis_mask, "bin1_id"]
    # apply requested transforms, e.g. balancing:
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    # perform aggregation by r1, r2 and _DIST_NAME
    _blocks = pixels.groupby(["r1", "r2", _DIST_NAME])

    # calculate summaries and add ".sum" suffix to field column-names
    return _blocks[fields].sum().add_suffix(".sum")


def sum_pairwise(
    clr,
    view_df,
    transforms={},
    clr_weight_name="weight",
    ignore_diags=2,
    chunksize=10_000_000,
    map=map,
):
    """

    Intra-chromosomal diagonal summary statistics for asymmetric blocks of
    contact matrix defined as pairwise combinations of regions in "view_df.

    Note
    ----
    This is a special case of asymmetric diagonal summary statistic that is
    efficient and covers the most important practical case of inter-chromosomal
    arms "expected" calculation.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        view_df of regions for intra-chromosomal diagonal summation, has to
        be sorted according to the order of chromosomes in cooler.
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    clr_weight_name : str
        name of the balancing weight vector used to count "bad"
        pixels per diagonal. Set to `None` not to mask
        "bad" pixels (raw data only).
    chunksize : int, optional
        Size of pixel table chunks to process
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    Dataframe of diagonal statistics for all intra-chromosomal blocks defined as
    pairwise combinations of regions in the view

    """
    spans = partition(0, len(clr.pixels()), chunksize)
    fields = list(chain(["count"], transforms))
    # names of summary results
    summary_fields = [f"{field}.sum" for field in fields]

    # check viewframe
    try:
        _ = is_compatible_viewframe(
            view_df,
            clr,
            check_sorting=True,  # required for pairwise combinations
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided view_df is not valid") from e

    # prepare an expected table: for every pairwise region combination a table with diagonals,
    # number of valid pixels, etc. Inter-chromosomal entries all have dist=0 value.
    exp_table = make_pairwise_expected_table(clr, view_df, clr_weight_name=clr_weight_name)

    # initialize columns to store summary results in exp_table
    for agg_name in summary_fields:
        exp_table[agg_name] = 0

    # apply _diagsum_pairwise to chunks of pixels
    job = partial(
        _sum_, clr, fields, transforms, clr_weight_name, view_df.to_numpy()
    )
    results = map(job, spans)

    # accumulate every chunk of summary results to exp_table
    result_df = reduce(lambda df1,df2: df1.add(df2, fill_value=0), results, exp_table)

    # following can be done easily, when _DIST_NAME has a special value for trans ...
    if ignore_diags:
        for _d in range(ignore_diags):
            # extract fist "ignore_diags" from DataFrame and fill them with NaNs
            _idx = result_df.xs(_d, level=_DIST_NAME, drop_level=False).index
            result_df.loc[_idx, summary_fields] = np.nan

    # # returning a pd.DataFrame for API consistency:
    result_df.reset_index(level=_DIST_NAME, inplace=True)
    # region1 for the final table
    result_df.insert(0, _REGION1_NAME, view_df.loc[result_df.index.get_level_values("r1"), "name"].to_numpy())
    # region2 for the final table
    result_df.insert(1, _REGION2_NAME, view_df.loc[result_df.index.get_level_values("r2"), "name"].to_numpy())
    # drop r1/r2 region labels
    result_df.reset_index(level=["r1", "r2"], drop=True, inplace=True)

    return result_df


def expected_full_fast(
        clr,
        view_df=None,
        smooth_cis=False,
        aggregate_cis=False,
        smooth_sigma=0.1,
        aggregate_trans=False,
        expected_column_name="expected",
        ignore_diags=2,
        clr_weight_name='weight',
        chunksize=10_000_000,
        nproc=4,
    ):
    """
    Generate a DataFrame with expected for *all* 2D regions
    tiling entire heatmap in clr.
    Such 2D regions are defined as all pairwise combinations
    of the regions in view_df. Average distance decay is calculated
    for every cis-region (e.g. inter- and intra-arms), and
    a "simple" average over each block is caculated for trans-
    regions.

    When sub-chromosomal view is provided, trans averages
    can be aggregated back to the level of full chromosomes.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        expected is calculated for all pairwise combinations of regions
        in view_df. Distance dependent expected is calculated for cis
        regions, and block-level average is calculated for trans regions.
    smooth_cis: bool
        Apply smoothing to cis-expected. Will be stored in an additional column
    aggregate_smoothed: bool
        When smoothing cis expected, average over all regions, ignored without smoothing.
    smooth_sigma: float
        Control smoothing with the standard deviation of the smoothing Gaussian kernel.
        Ignored without smoothing.
    aggregate_trans : bool
        Aggregate trans-expected at the inter-chromosomal level.
    expected_column_name : str
        Name of the column where to store combined expected
    ignore_diags : int, optional
        Number of intial diagonals to exclude for calculation of distance dependent
        expected.
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    chunksize : int, optional
        Size of pixel table chunks to process
    nproc : int, optional
        How many processes to use for calculation
    Returns
    -------
    expected_df: pd.DataFrame
        cis and trans expected combined together
    """

    if view_df is None:
            # Generate chromosome-wide viewframe from clr.chromsizes:
            view_df = make_cooler_view(clr)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                # (?) must be sorted for asymmetric case
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # define transforms - balanced and raw ('count') for now
    cols = {
        "dist": _DIST_NAME,
        "n_pixels": _NUM_VALID_NAME,
        "smooth_suffix": ".smooth",
    }
    if clr_weight_name is None:
        # no transforms
        transforms = {}
        cols["n_contacts"] = "count.sum"
        cols["contact_freq"] = "count.avg"
    elif is_cooler_balanced(clr, clr_weight_name):
        # define balanced data transform:
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
        cols["n_contacts"] = "balanced.sum"
        cols["contact_freq"] = "balanced.avg"
    else:
        raise ValueError(
            "cooler is not balanced, or"
            f"balancing weight {clr_weight_name} is not available in the cooler."
        )

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly ans start the timer
    time_start = time.perf_counter()
    try:
        result = sum_pairwise(
            clr,
            view_df=view_df,
            transforms=transforms,
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
            map=map_,
        )
    finally:
        if nproc > 1:
            pool.close()

    # calculate actual averages by dividing sum by n_valid:
    result[cols["contact_freq"]] = result[cols["n_contacts"]].divide(
        result[cols["n_pixels"]]
    )

    # annotate result with the region index and chromosomes
    view_label = view_df \
                .reset_index() \
                .rename(columns={"index":"r"}) \
                .set_index("name")

    # add r1 r2 labels to the final dataframe for obs/exp merging
    result["r1"] = view_label.loc[result["region1"], "r"].to_numpy()
    result["r2"] = view_label.loc[result["region2"], "r"].to_numpy()

    # initialize empty column with the final name of expected
    result.insert(loc=len(result.columns), column=expected_column_name, value=np.nan)

    # annotate result with chromosomes in case chrom-level aggregation is requested
    if "chrom" in [aggregate_trans, aggregate_cis]:
        result["chrom1"] = view_label.loc[result["region1"], "chrom"].to_numpy()
        result["chrom2"] = view_label.loc[result["region2"], "chrom"].to_numpy()

    if aggregate_cis:
        if aggregate_cis == "chrom":
            grp_columns = ["chrom1", "chrom2"]  # chrom-level aggregation
        elif aggregate_cis == "genome":
            grp_columns = None  # genome-wide level aggregation
        else:
            raise ValueError("aggregate_cis could be only chrom, genome or False")
    else:
        grp_columns = [_REGION1_NAME, _REGION2_NAME]  # no aggregation, keep as in view_df

    # prepare cis/trans masks for smoothing, aggregation and/or simply copying values
    # to the final column:
    _cis_mask = result[_DIST_NAME] != TRANS_DIST_VALUE
    _trans_mask = result[_DIST_NAME] == TRANS_DIST_VALUE

    # additive columns for aggregation
    additive_cols = [ cols["n_pixels"], cols["n_contacts"] ]

    # additional smoothing and aggregating options would add columns only, not replace them
    if smooth_cis:
        # smooth and aggregate
        _smooth_df = expected_smoothing.agg_smooth_cvd(
            result.loc[_cis_mask],
            groupby=grp_columns,
            sigma_log10=smooth_sigma,
            cols=cols,
        )
        _smooth_col_name = cols["contact_freq"] + cols["smooth_suffix"]
        # add smoothed columns to the result
        result = result.merge(
            _smooth_df[[ _DIST_NAME, _smooth_col_name ]],
            on=[*(grp_columns or []), _DIST_NAME],
            how="left",
        )
        # add the results to the expected column
        result.loc[_cis_mask, expected_column_name] = result.loc[_cis_mask, _smooth_col_name]
    elif aggregate_cis:
        # aggregate only if requested:
        _agg_df = result.loc[_cis_mask] \
            .groupby([*(grp_columns or []), _DIST_NAME], observed=True)[additive_cols] \
            .transform("sum") \
            .add_suffix(".agg")
        # calculate new average
        _agg_df[expected_column_name] = _agg_df[f"""{cols["n_contacts"]}.agg"""].divide(
            _agg_df[f"""{cols["n_pixels"]}.agg"""]
        )
        # add aggregated result to the result df
        result.loc[_cis_mask, expected_column_name] = _agg_df[expected_column_name]
    else:
        # just copy unchanged result to expected_column_name
        result.loc[_cis_mask, expected_column_name] = result.loc[_cis_mask, cols["contact_freq"]]

    # aggregate trans on requested level and copy result into final expected column:
    if aggregate_trans:
        if aggregate_trans == "chrom":
            # groupby chromosomes and sum up additive values:
            _trans_agg_df = result.loc[_trans_mask] \
                .groupby(["chrom1", "chrom2"], observed=True)[additive_cols] \
                .transform("sum") \
                .add_suffix(".agg")
        elif aggregate_trans == "genome":
            # genome-wide transform :
            _trans_df = result.loc[_trans_mask]
            _trans_agg_df = result.loc[_trans_mask, additive_cols] \
                .transform("sum") \
                .add_suffix(".agg")
        else:
            raise ValueError("aggregate_trans could be only chrom, genome or False")
        # complete aggregation by recalculating new average:
        _trans_agg_df[expected_column_name] = _trans_agg_df[f"""{cols["n_contacts"]}.agg"""].divide(
            _trans_agg_df[f"""{cols["n_pixels"]}.agg"""]
        )
        # and adding aggregated result to the result df:
        result.loc[_trans_mask, expected_column_name] = _trans_agg_df[expected_column_name]
    else:
        # just copy unchanged result to expected_column_name
        result.loc[_trans_mask, expected_column_name] = result.loc[_trans_mask, cols["contact_freq"]]

    # time is up
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"Done calculating full expected {time_elapsed:.3f} sec ...")

    return result


def obs_over_exp(
    pixels,
    bins,
    expected_full,
    view_column_name="r",
    expected_column_name="expected",
    clr_weight_name='weight',
    oe_column_name="oe",
):
    """
    A function that returns pixel table with observed over expected.
    It takes a DataFrame of pixels (complete or a chunk, in pandas or dask formats),
    and merges it with the `expected_full` DataFrame, in order to assign appropriate
    expected for each pixels. This assignment is done according to the pixels' location,
    specifically - which combination of regions in the view and genomic distance for
    cis-pixels.

    Parameters
    ----------
    pixels: pd.DataFrame | dask.DataFrame
        DataFrame of pixels
    bins : pd.DataFrame
        A bin table with a view column annotation.
    expected_full : pd.DataFrame
        DataFrame expected for all regions in the view used for annotation of bins.
    view_column_name : str
        Name of the column with the view annotations in `bins` and `expected_full`
    expected_column_name : str
        Name of the column with the expected values in `expected_full`
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    oe_column_name : str
        Name of the column to store observed over expected in.

    Returns
    -------
    pixels_oe : pd.DataFrame | dask.DataFrame
        DataFrame of pixels with observed/expected
    """

    # use balanced data, when clr_weight is provided - otherwise raw counts
    if clr_weight_name:
        observed_column_name = "balanced"
        weight_col1 = f"{clr_weight_name}1"
        weight_col2 = f"{clr_weight_name}2"
    else:
        observed_column_name = "count"

    # names of the view
    view_col1 = f"{view_column_name}1"
    view_col2 = f"{view_column_name}2"

    # labeling with the view-labels view_column_name1/view_column_name2:
    pixels_oe = cooler.annotate(pixels, bins, replace=False)

    # calculate balanced pixel values and drop NAs for bad bins
    # as well as pixels_oe not covered by the view (i.e. with view_column annotation as NA)
    if clr_weight_name:
        pixels_oe[observed_column_name] = pixels_oe["count"] * pixels_oe[weight_col1] * pixels_oe[weight_col2]
        pixels_oe = pixels_oe.dropna( subset=[view_col1, view_col1, weight_col1, weight_col2] )
    else:
        pixels_oe = pixels_oe.dropna( subset=[view_col1, view_col1] )

    # cast to int, as there are no more NaNs among view_column_name1/view_column_name2
    pixels_oe = pixels_oe.astype({view_col1 : int, view_col1 : int})

    # initialize distance for all values, and then correct for trans
    pixels_oe["dist"] = pixels_oe["bin2_id"] - pixels_oe["bin1_id"]
    cis_mask = pixels_oe["chrom1"] == pixels_oe["chrom2"]
    # use dask-compatible where notation, instead of loc/iloc assignment
    pixels_oe["dist"] = pixels_oe["dist"].where(cis_mask, TRANS_DIST_VALUE)

    # merge pixels_oe with the expected_full - to assign appropriate expected to each pixel
    # dask-compatible notation instead of pd.merge
    pixels_oe = pixels_oe.merge(
        expected_full[[view_col1, view_col2, "dist", expected_column_name]],
        how="left",
        on=[view_col1, view_col2, "dist"],
    )

    # observed over expected = observed / expected
    pixels_oe[oe_column_name] = pixels_oe[observed_column_name] / pixels_oe[expected_column_name]

    return pixels_oe


def obs_over_exp_generator(
    clr,
    expected_full,
    view_df=None,
    expected_column_name="expected",
    clr_weight_name='weight',
    oe_column_name="count",
    chunksize = 1_000_000
):
    """
    Generator yielding chunks of pixels with
    pre-caluclated observed over expected.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    expected_full : pd.DataFrame
        DataFrame expected for all pairwise combinations of regions in view_df
    view_df : viewframe
        viewframe of regions that were used to calculate expected_full
    expected_column_name : str
        Name of the column with the combined expected
    oe_column_name : str
        Name of the column to store observed over expected
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    chunksize : int, optional
        Size of pixel table chunks to process and output

    Yields
    ------
    pixel_df: pd.DataFrame
        chunks of pixels with observed over expected
    """
    # use the same view that was used to calculate full expected
    if view_df is None:
        view_array = make_cooler_view(clr).to_numpy()
    else:
        view_array = view_df.to_numpy()

    # extract and pre-process cooler bintable
    view_column_name = "r"
    bins_view = clr.bins()[:]
    bins_view[view_column_name] = assign_supports(bins_view, view_array)  # astype float

    # define chunks of pixels to work on
    spans = partition(0, len(clr.pixels()), chunksize)
    for span in spans:
        lo, hi = span
        # logging.info(f"Calculating observed over expected for pixels [{lo}:{hi}]")
        oe_chunk = obs_over_exp(
            clr.pixels()[lo:hi],
            bins_view,
            expected_full,
            view_column_name=view_column_name,
            expected_column_name=expected_column_name,
            clr_weight_name=clr_weight_name,
            oe_column_name=oe_column_name,
        )

        # yield pixel-table-like chunks of observed over expected
        yield oe_chunk[["bin1_id", "bin2_id", oe_column_name]]
