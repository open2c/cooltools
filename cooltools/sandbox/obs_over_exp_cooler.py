"""
This module enables construction of observed over expected pixels tables and
storing them inside a cooler.

It includes 2 functions.
expected_full - is a convenience function that calculates cis and trans-expected
    and "stitches" them in a certain way. Such a stitched expected that "covers"
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
        cis_view_df=None,
        trans_view_df=None,
        ignore_diags=2,
        clr_weight_name='weight',
        chunksize=10_000_000,
        nproc=4,
    ):
    """
    generate a DataFrame with expected for 2D regions
    tiling entire heatmap in clr.
    For example, distance-decay-like expected for all
    chromosomes and block-average-like expected for all
    inter-chromosomal pairs.

    Intra-chromosomal and inter-chromosomal regions of
    the heatmap can be partitioned differently. But it
    is assumed here that intra-partitioning is strict
    subpartitioning of the inter-. E.g. one case use
    chromosome arms in cis, and full chromosomes in trans.

    Parameters
    ----------
    clr
    cis_view_df
    trans_view_df
    ignore_diags
    clr_weight_name
    chunksize
    nproc

    Returns
    -------
    combined_expected: pd.DataFrame
        cis and trans expected combined together
    """

    # contacs vs distance - i.e. intra/cis expected
    time_start = time.perf_counter()
    cvd = expected_cis(
        clr,
        view_df=cis_view_df,
        intra_only=False,  # get cvd for all 2D regions
        smooth=False,  # ignore that for now
        clr_weight_name=clr_weight_name,
        ignore_diags=ignore_diags,
        chunksize=chunksize,
        nproc=nproc,
    )
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"Done calculating cis expected in {time_elapsed:.3f} sec ...")

    # contacts per block - i.e. inter/trans expected
    if trans_view_df is not None:
        raise NotImplementedError("only full-chromosome trans-expected is supported, set trans_view_df to None")
    time_start = time.perf_counter()
    cpb = expected_trans(
        clr,
        view_df=trans_view_df,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
        nproc=nproc,
    )
    # pretend that they also have a "dist"
    # to make them mergeable with cvd
    cpb["dist"] = 0
    time_elapsed = time.perf_counter() - time_start
    logging.info(f"Done calculating trans expected in {time_elapsed:.3f} sec ...")

    # cis view -> DataFrame with numerical labels for regions
    cis_label = cis_view_df \
                .reset_index() \
                .rename(columns={"index":"r"})[["r","name"]]
    # assign r1 and r2 region-labels to cvd
    cvd = pd.merge(
        cvd,
        cis_label.rename(columns={"name":"region1"}),
        on="region1",
    )
    cvd = pd.merge(
        cvd,
        cis_label.rename(columns={"name":"region2"}),
        on="region2",
        suffixes=("1","2"),
    )

    # align trans-expected to the same regions that are in cis_regions
    # e.g. if cvd is by-arm, then for each pairs of chroms in cpb, report
    # arm-by-arm trans expected with the same average value
    #
    # splitting trans-view into cis-view
    # cis view -> DataFrame with numerical labels for trans-regions
    trans_label = cis_view_df \
                .reset_index() \
                .rename(columns={"index":"r"})[["r","chrom"]]
    cpb = pd.merge(
        cpb,
        trans_label.rename(columns={"chrom":"region1"}),
        on="region1"
    )
    cpb = pd.merge(
        cpb,
        trans_label.rename(columns={"chrom":"region2"}),
        on="region2",
        suffixes=("1","2"),
    )

    logging.info(f"Returning combined expected DataFrame.")

    # returned joined cis/trans expected in the same format
    # this could be large for higher resolution data
    return pd.concat([cvd, cpb], ignore_index=True)


def obs_over_exp_generator(
        clr,
        expected_full,
        cis_view_df=None,
        clr_weight_name='weight',
        oe_column_name="count",  # how to store obs/exp
        chunksize = 1_000_000,
    ):
    """
    Generator yielding chunks of pixels with
    pre-caluclated observed over expected.
    """

    spans = partition(0, len(clr.pixels()), chunksize)
    bins = clr.bins()[:]
    # use the same view that was used to calculate full expected
    if cis_view_df is None:
        view_array = make_cooler_view(clr).to_numpy()
    else:
        view_array = cis_view_df.to_numpy()

    for span in spans:
        lo, hi = span
        pixels = clr.pixels()[lo:hi]
        # full re-annotation may not be needed - to be optimized
        pixels = cooler.annotate(pixels, bins, replace=False)

        logging.info(f"Calculating observed over expected for pixels [{lo}:{hi}]")

        # calculate balanced pixel values
        pixels["balanced"] = pixels["count"] * pixels[f"{clr_weight_name}1"] * pixels[f"{clr_weight_name}2"]

        # assign cis_view_df to pixels - i.e. cis 2D regions
        pixels["r1"] = assign_supports(pixels, view_array, suffix="1")
        pixels["r2"] = assign_supports(pixels, view_array, suffix="2")
        # cast to int
        pixels = pixels.astype({"r1":int, "r2":int})
        # consider dropping NAs if view_df covers part of the genome only
        # and for the masked bins :
        pixels = pixels.dropna( subset=["r1", "r2", f"{clr_weight_name}1", f"{clr_weight_name}2"] )

        # trans pixels will have "feature"-dist of 0
        pixels["dist"] = 0
        # cis pixels will have "feature"-dist "bind2_id - bin1_id"
        cis_mask = (pixels["chrom1"] == pixels["chrom2"])
        pixels.loc[cis_mask,"dist"] = pixels.loc[cis_mask,"bin2_id"] - pixels.loc[cis_mask,"bin1_id"]

        # merge observed (i.e. all_pixels) with the expected
        pixels = pd.merge(
            pixels,
            expected_full[["r1","r2","dist","balanced.avg"]],
            how="left",
            on=["r1","r2","dist"],
        )

        pixels[oe_column_name] = pixels["balanced"] / pixels["balanced.avg"]

        # yield reannotated pieces ...
        yield pixels[["bin1_id", "bin2_id", oe_column_name]]
