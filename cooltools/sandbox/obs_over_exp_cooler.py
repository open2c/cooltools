"""
This module enables construction of observed over expected pixels tables and
storing them inside a cooler.

It includes 2 functions.
expected_full - is a convenience function that calculates cis and trans-expected
    and "stitches" them togeter. Such a stitched expected that "covers"
    entire Hi-C heatmap can be easily merged with the pixel table.
obs_over_exp - is a function that merges pre-calculated full expected with the pixel
    table in pd.DataFrame or dask.DataFrame formats.
obs_over_exp_generator - is a function/generator(lazy iterator) that wraps
    `obs_over_exp` function and yields chunks of observed/expected pixel table.
    Such a "stream" can be used in cooler.create as a "pixels" argument to write
    obs/exp cooler-file.
"""
import time
import logging

import numpy as np
import pandas as pd

import cooler
from cooler.tools import partition

from cooltools import (
    expected_cis,
    expected_trans
)
from cooltools.lib.common import (
    assign_supports,
    make_cooler_view
)


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
    cpb["dist"] = 0
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

    # trans pixels_oe will have "feature"-dist of 0 (or some other special value)
    pixels_oe["dist"] = 0

    # cis pixels_oe will have "feature"-dist "bind2_id - bin1_id"
    # dask-compatible notation using `where`, as assign to iloc/loc isn't supported
    cis_mask = (pixels_oe["chrom1"] == pixels_oe["chrom2"])
    pixels_oe["dist"] = pixels_oe["dist"].where(
        ~cis_mask,
        pixels_oe.loc[cis_mask, "bin2_id"] - pixels_oe.loc[cis_mask, "bin1_id"]
    )

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
