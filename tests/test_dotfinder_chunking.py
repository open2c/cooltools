# create a test for square_tiling and get_adjusted_expected_tile_some_nans

import numpy as np
import pandas as pd

import os.path as op

from cooltools.lib.numutils import LazyToeplitz
from cooltools.api.dotfinder import (
    tile_square_matrix,
    get_adjusted_expected_tile_some_nans,
)

# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, "data", "dotfinder_mock_inputs.npz")
mock_result = op.join(testdir, "data", "dotfinder_mock_res.csv.gz")

# load mock results:
mock_res = pd.read_csv(mock_result).rename(columns={"row": "bin1_id", "col": "bin2_id"})

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_raw, M_ice, E_ice and v_ice are supposed
# to be there ...
mock_M_raw = arrays_loaded["mock_M_raw"]
mock_M_ice = arrays_loaded["mock_M_ice"]
mock_E_ice = arrays_loaded["mock_E_ice"]
mock_v_ice = arrays_loaded["mock_v_ice"]

# 1D expected extracted for tiling-tests:
mock_exp = LazyToeplitz(mock_E_ice[0, :])

# we need kernel_half_width-edge for tiling procedures:
kernel_half_width = 3
# just a simple donut kernel for testing:
kernel = np.array(
    [
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1],
    ]
)
# mock bin size 20kb
b = 20000
# 2MB aroud diagonal to look at:
band = 2e6

# start, stop for tiling procedures:
start, stop = 0, len(mock_M_raw)


#################################################################
# test 'get_adjusted_expected_tile_some_nans' without chunking :
#################################################################
def test_adjusted_expected_tile_some_nans():
    print("Running tile some nans la_exp test")
    # first, generate that locally-adjusted expected:
    res = get_adjusted_expected_tile_some_nans(
        origin_ij=(0, 0),
        observed=mock_M_raw,  # should be RAW ...
        expected=mock_E_ice,
        bal_weights=mock_v_ice,
        kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
    )
    # mock results are supposedly for
    # the contacts closer than the band
    # nucleotides to each other...
    # we want to retire that functional
    # from the 'get_adjusted_expected_tile_some_nans'
    # so we should complement that selection
    # here:
    nnans = 1  # no nans tolerated
    band_idx = int(band / b)
    is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
    does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
    # so, selecting inside band and nNaNs compliant results:
    res = res[is_inside_band & does_comply_nans].reset_index(drop=True)
    #
    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert res[["bin1_id", "bin2_id"]].equals(mock_res[["bin1_id", "bin2_id"]])
    # compare floating point part separately:
    assert np.allclose(
        res["la_exp." + "donut" + ".value"], mock_res["la_expected"], equal_nan=True
    )
    #
    # try recovering more NaNs ...
    # (allowing 1 here per pixel's footprint)...
    res = get_adjusted_expected_tile_some_nans(
        origin_ij=(0, 0),
        observed=mock_M_raw,  # should be RAW...
        expected=mock_E_ice,
        bal_weights=mock_v_ice,
        kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
    )
    # post-factum filtering resides
    # outside of get_la_exp now:
    nnans = 2  # just 1 nan tolerated
    band_idx = int(band / b)
    is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
    does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
    # so, selecting inside band and comply nNaNs results only:
    res = res[is_inside_band & does_comply_nans].reset_index(drop=True)

    # now we can only guess the size:
    assert len(res) > len(mock_res)


def test_adjusted_expected_tile_some_nans_and_square_tiling():
    print("Running tile some nans la_exp test + square tiling")
    # first, generate that locally-adjusted expected:
    nnans = 1
    band_idx = int(band / b)
    res_ij = []
    for tilei, tilej in tile_square_matrix(
        stop - start, start, tile_size=40, pad=kernel_half_width
    ):
        # define origin:
        origin = (tilei[0], tilej[0])
        # RAW observed matrix slice:
        observed = mock_M_raw[slice(*tilei), slice(*tilej)]
        # trying new expected function:
        expected = mock_exp[slice(*tilei), slice(*tilej)]
        # for diagonal chuynking/tiling tilei==tilej:
        ice_weight_i = mock_v_ice[slice(*tilei)]
        ice_weight_j = mock_v_ice[slice(*tilej)]
        # that's the main working function from dotfinder:
        res = get_adjusted_expected_tile_some_nans(
            origin_ij=origin,
            observed=observed,
            expected=expected,
            bal_weights=(ice_weight_i, ice_weight_j),
            kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
        )
        is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
        # new style, selecting good guys:
        does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
        # so, select inside band and nNaNs compliant results and append:
        res_ij.append(res[is_inside_band & does_comply_nans])

    # drop dups (from overlaping tiles), sort and reset index:
    res_df = (
        pd.concat(res_ij, ignore_index=True)
        .drop_duplicates()
        .sort_values(by=["bin1_id", "bin2_id"])
        .reset_index(drop=True)
    )

    # prepare mock_data for comparison:
    # apparently sorting is needed in this case:
    mock_res_sorted = (
        mock_res.drop_duplicates()
        .sort_values(by=["bin1_id", "bin2_id"])
        .reset_index(drop=True)
    )

    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert res_df[["bin1_id", "bin2_id"]].equals(
        mock_res_sorted[["bin1_id", "bin2_id"]]
    )
    # compare floating point part separately:
    assert np.allclose(
        res_df["la_exp." + "donut" + ".value"],
        mock_res_sorted["la_expected"],
        equal_nan=True,
    )


def test_adjusted_expected_tile_some_nans_and_square_tiling_diag_band():
    print("Running tile some nans la_exp test + square tiling + diag band")
    # # Essentially, testing this function:
    # def chrom_chunk_generator_s(chroms, kernel_half_width, band):
    #     for chrom in chroms:
    #         chr_start, chr_stop = the_c.extent(chrom)
    #         for tilei, tilej in square_matrix_tiling(chr_start, chr_stop, step, kernel_half_width):
    #             # check if a given tile intersects with
    #             # with the diagonal band of interest ...
    #             diag_from = tilej[0] - tilei[1]
    #             diag_to   = tilej[1] - tilei[0]
    #             #
    #             band_from = 0
    #             band_to   = band
    #             # we are using this >2w trick to exclude
    #             # tiles from the lower triangle from calculations ...
    #             if (min(band_to,diag_to) - max(band_from,diag_from)) > 2*kernel_half_width:
    #                 yield chrom, tilei, tilej
    # first, generate that locally-adjusted expected:
    nnans = 1
    band_idx = int(band / b)
    res_ij = []
    for tilei, tilej in tile_square_matrix(
        stop - start, start, tile_size=40, pad=kernel_half_width
    ):
        # check if a given tile intersects with
        # with the diagonal band of interest ...
        diag_from = tilej[0] - tilei[1]
        diag_to = tilej[1] - tilei[0]
        #
        band_from = 0
        band_to = band_idx
        # we are using this >2w trick to exclude
        # tiles from the lower triangle from calculations ...
        if (min(band_to, diag_to) - max(band_from, diag_from)) > 2 * kernel_half_width:
            # define origin:
            origin = (tilei[0], tilej[0])
            # RAW observed matrix slice:
            observed = mock_M_raw[slice(*tilei), slice(*tilej)]
            # trying new expected function:
            expected = mock_exp[slice(*tilei), slice(*tilej)]
            # for diagonal chuynking/tiling tilei==tilej:
            ice_weight_i = mock_v_ice[slice(*tilei)]
            ice_weight_j = mock_v_ice[slice(*tilej)]
            # that's the main working function from dotfinder:
            res = get_adjusted_expected_tile_some_nans(
                origin_ij=origin,
                observed=observed,
                expected=expected,
                bal_weights=(ice_weight_i, ice_weight_j),
                kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
            )
            is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
            # new style, selecting good guys:
            does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
            # so, select inside band and nNaNs compliant results and append:
            res_ij.append(res[is_inside_band & does_comply_nans])

    # sort and reset index, there shouldn't be any duplicates now:
    res_df = (
        pd.concat(res_ij, ignore_index=True)
        .drop_duplicates()
        .sort_values(by=["bin1_id", "bin2_id"])
        .reset_index(drop=True)
    )

    # prepare mock_data for comparison:
    # apparently sorting is needed in this case:
    mock_res_sorted = (
        mock_res.drop_duplicates()
        .sort_values(by=["bin1_id", "bin2_id"])
        .reset_index(drop=True)
    )

    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert res_df[["bin1_id", "bin2_id"]].equals(
        mock_res_sorted[["bin1_id", "bin2_id"]]
    )
    # compare floating point part separately:
    assert np.allclose(
        res_df["la_exp." + "donut" + ".value"],
        mock_res_sorted["la_expected"],
        equal_nan=True,
    )
