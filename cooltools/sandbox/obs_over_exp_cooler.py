"""
This module enables construction of observed over expected pixels tables and
storing them inside a cooler.

It includes 2 functions.
expected_full - is a convenience function that calculates cis and trans-expected
    and "stitches" them togeter. Such a stitched expected that "covers"
    entire Hi-C heatmap can be easily merged with the pixel table.
obs_over_exp_generator - is a function/generator(lazy iterator) that merges
    pre-calculated full expected with the pixel table in clr and yields chunks
    of observed/expected pixel table. Such a "stream" can be used in cooler.create
    as a "pixels" argument to write obs/exp cooler-file.
"""
import time
import logging

import numpy as np
import pandas as pd
import multiprocess as mp

import cooler
from cooler.tools import partition

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
    make_cooler_view,
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


# common expected_df column names, take from schemas
_REGION1 = list(diag_expected_dtypes)[0]
_REGION2 = list(diag_expected_dtypes)[1]
_DIST = list(diag_expected_dtypes)[2]
_NUM_VALID = list(diag_expected_dtypes)[3]

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
    # pretend that they also have a "dist"
    # to make them mergeable with cvd
    cpb["dist"] = -1
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
                    _tables.append(df.set_index(["r1", "r2", _DIST]))
                if (chrom1 != chrom2):
                    df = pd.DataFrame(btables[(name1, name2)], index=[0])
                    df.insert(0, _DIST, -1)  # special trans-value for distance
                    df.insert(0, "r2", _r2)
                    df.insert(0, "r1", _r1)
                    _tables.append(df.set_index(["r1", "r2", _DIST]))

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

    # initialize _DIST as 0 for all pixels
    # consider using -1 as a special value to distinguish trans data easily ...
    pixels.loc[:, _DIST] = -1
    # calculate actual genomic _DIST for cis-pixels:
    pixels.loc[cis_mask, _DIST] = pixels.loc[cis_mask, "bin2_id"] - pixels.loc[cis_mask, "bin1_id"]
    # apply requested transforms, e.g. balancing:
    for field, t in transforms.items():
        pixels[field] = t(pixels)
    
    # perform aggregation by r1, r2 and _DIST
    _blocks = pixels.groupby(["r1", "r2", _DIST])
    
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

    # following can be done easily, when _DIST has a special value for trans ...
    if ignore_diags:
        for _d in range(ignore_diags):
            # extract fist "ignore_diags" from DataFrame and fill them with NaNs
            _idx = result_df.xs(_d, level=_DIST, drop_level=False).index
            result_df.loc[_idx, summary_fields] = np.nan

    # # returning a pd.DataFrame for API consistency:
    result_df.reset_index(level=_DIST, inplace=True)
    # region1 for the final table
    result_df.insert(0, _REGION1, view_df.loc[result_df.index.get_level_values("r1"), "name"].to_numpy())
    # region2 for the final table
    result_df.insert(1, _REGION2, view_df.loc[result_df.index.get_level_values("r2"), "name"].to_numpy())
    # drop r1/r2 region labels
    result_df.reset_index(level=["r1", "r2"], drop=True, inplace=True)

    return result_df


def expected_full_fast(
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
    if clr_weight_name is None:
        # no transforms
        transforms = {}
    elif is_cooler_balanced(clr, clr_weight_name):
        # define balanced data transform:
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
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

    # using try-clause to close mp.Pool properly
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
    for key in chain(["count"], transforms):
        result[f"{key}.avg"] = result[f"{key}.sum"] / result[_NUM_VALID]

    # # additional smoothing and aggregating options would add columns only, not replace them
    # if smooth:
    #     result_smooth = expected_smoothing.agg_smooth_cvd(
    #         result,
    #         sigma_log10=smooth_sigma,
    #     )
    #     # add smoothed columns to the result (only balanced for now)
    #     result = result.merge(
    #         result_smooth[["balanced.avg.smoothed", _DIST]],
    #         on=[_REGION1, _REGION2, _DIST],
    #         how="left",
    #     )
    #     if aggregate_smoothed:
    #         result_smooth_agg = expected_smoothing.agg_smooth_cvd(
    #             result,
    #             groupby=None,
    #             sigma_log10=smooth_sigma,
    #         ).rename(columns={"balanced.avg.smoothed": "balanced.avg.smoothed.agg"})
    #         # add smoothed columns to the result
    #         result = result.merge(
    #             result_smooth_agg[["balanced.avg.smoothed.agg", _DIST]],
    #             on=[
    #                 _DIST,
    #             ],
    #             how="left",
    #         )

    return result


    # # annotate expected_df with the region index and chromosomes
    # view_label = view_df \
    #             .reset_index() \
    #             .rename(columns={"index":"r"}) \
    #             .set_index("name")

    # # which expected column to use, based on requested "modifications":
    # cis_expected_name = "balanced.avg" if clr_weight_name else "count.avg"
    # if smooth_cis:
    #     cis_expected_name = f"{cis_expected_name}.smoothed"
    #     if aggregate_smoothed:
    #         cis_expected_name = f"{cis_expected_name}.agg"
    # # copy to the prescribed column for the final output:
    # cvd[expected_column_name] = cvd[cis_expected_name].copy()

    # # aggregate trans if requested and deide which trans-expected column to use:
    # trans_expected_name = "balanced.avg" if clr_weight_name else "count.avg"
    # if aggregate_trans:
    #     trans_expected_name = f"{trans_expected_name}.agg"
    #     additive_cols = ["n_valid","count.sum"]
    #     if clr_weight_name:
    #         additive_cols.append("balanced.sum")
    #     # groupby chrom1, chrom2 and aggregate additive fields (sums and n_valid):
    #     _cpb_agg = cpb.groupby(
    #         [
    #             view_label["chrom"].loc[cpb["region1"]].to_numpy(),  # chrom1
    #             view_label["chrom"].loc[cpb["region2"]].to_numpy(),  # chrom2
    #         ]
    #     )[additive_cols].transform("sum")
    #     # recalculate aggregated averages:
    #     cpb["count.avg.agg"] = _cpb_agg["count.sum"]/_cpb_agg["n_valid"]
    #     if clr_weight_name:
    #         cpb["balanced.avg.agg"] = _cpb_agg["balanced.sum"]/_cpb_agg["n_valid"]
    # # copy to the prescribed column for the final output:
    # cpb[expected_column_name] = cpb[trans_expected_name].copy()

    # # concatenate cvd and cpb (cis and trans):
    # expected_df = pd.concat([cvd, cpb], ignore_index=True)

    # # add r1 r2 labels to the final dataframe for obs/exp merging
    # expected_df["r1"] = view_label["r"].loc[expected_df["region1"]].to_numpy()
    # expected_df["r2"] = view_label["r"].loc[expected_df["region2"]].to_numpy()

    # # and return joined cis/trans expected in the same format
    # logging.info(f"Returning combined expected DataFrame.")
    # # consider purging unneccessary columns here
    # return expected_df


def obs_over_exp_generator(
        clr,
        expected_full,
        view_df=None,
        expected_column_name="expected",
        oe_column_name="count",  # how to store obs/exp
        clr_weight_name='weight',
        chunksize = 1_000_000,
        # todo: consider yielding cis-only, trans-only
        # todo: consider yielding fully annotated chunks
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
    bins = clr.bins()[:]
    bins["r"] = assign_supports(bins, view_array)  # astype float
    # todo: try the same trick for the future version of "expected_full"
    # where expected_full will be calculated in one pass, instead of
    # separate expected_cis and expected_trans calls

    # use balanced data, when clr_weight is provided - otherwise raw counts
    if clr_weight_name:
        observed_column_name = "balanced"
        weight_col1 = f"{clr_weight_name}1"
        weight_col2 = f"{clr_weight_name}2"
    else:
        observed_column_name = "count"

    # define chunks of pixels to work on
    spans = partition(0, len(clr.pixels()), chunksize)
    for span in spans:
        lo, hi = span
        pixels = clr.pixels()[lo:hi]
        # full re-annotation may not be needed - to be optimized
        # labeling with the regions-labels r1/r2 is happening here:
        pixels = cooler.annotate(pixels, bins, replace=False)

        logging.info(f"Calculating observed over expected for pixels [{lo}:{hi}]")

        # calculate balanced pixel values
        if clr_weight_name:
            pixels[observed_column_name] = pixels["count"] * pixels[weight_col1] * pixels[weight_col2]

        # consider dropping NAs if view_df covers only part of the genome and for "bad" bins
        if clr_weight_name:
            pixels = pixels.dropna( subset=["r1", "r2", weight_col1, weight_col2] )
        else:
            pixels = pixels.dropna( subset=["r1", "r2"] )

        # cast to int, as there are no more NaNs among r1/r2
        pixels = pixels.astype({"r1":int, "r2":int})

        # trans pixels will have "feature"-dist of 0
        pixels["dist"] = 0
        # cis pixels will have "feature"-dist "bind2_id - bin1_id"
        cis_mask = (pixels["chrom1"] == pixels["chrom2"])
        pixels.loc[cis_mask,"dist"] = pixels.loc[cis_mask,"bin2_id"] - pixels.loc[cis_mask,"bin1_id"]

        # merge observed (i.e. all_pixels) with the expected
        pixels = pd.merge(
            pixels,
            expected_full[["r1","r2","dist",expected_column_name]],
            how="left",
            on=["r1","r2","dist"],
        )

        pixels[oe_column_name] = pixels[observed_column_name] / pixels[expected_column_name]

        # yield pixel-table-like chunks of observed over expected
        yield pixels[["bin1_id", "bin2_id", oe_column_name]]
