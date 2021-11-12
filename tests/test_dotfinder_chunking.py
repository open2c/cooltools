# create a test for the chunking versions of 'get_adjusted_expected_tile_some_nans':

import numpy as np
import pandas as pd

import os.path as op

from cooltools.api import dotfinder
from cooltools.lib.numutils import LazyToeplitz


# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, "data", "dotfinder_mock_inputs.npz")
mock_result = op.join(testdir, "data", "dotfinder_mock_res.csv.gz")


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

# we need w-edge for tiling procedures:
w = 3
# p = 1
# kernel type: 'donut'
# # just a simple donut kernel for testing:
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
# bin size:
# mock data were extracted
# from 20kb matrix:
b = 20000
# 2MB aroud diagonal to look at:
band = 2e6
# 1MB aroud diagonal to look at:
band_1 = 1e6

# start, stop for tiling procedures:
start, stop = 0, len(mock_M_raw)

# load mock results:
mock_res = pd.read_csv(mock_result).rename(columns={"row": "bin1_id", "col": "bin2_id"})


def test_adjusted_expected_tile_some_nans_and_square_tiling():
    print("Running tile some nans la_exp test + square tiling")
    # first, generate that locally-adjusted expected:
    nnans = 1
    band_idx = int(band / b)
    res_df = pd.DataFrame([])
    for tilei, tilej in dotfinder.square_matrix_tiling(
        start, stop, step=40, edge=w, square=False
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
        res = dotfinder.get_adjusted_expected_tile_some_nans(
            origin=origin,
            observed=observed,
            expected=expected,
            bal_weights=(ice_weight_i, ice_weight_j),
            kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
            # nan_threshold=1,
            verbose=False,
        )
        is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
        # new style, selecting good guys:
        does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
        # so, select inside band and nNaNs compliant results and append:
        res_df = res_df.append(
            res[is_inside_band & does_comply_nans], ignore_index=True
        )

    # drop dups (from overlaping tiles), sort and reset index:
    res_df = (
        res_df.drop_duplicates()
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
    assert np.isclose(
        res_df["la_exp." + "donut" + ".value"],
        mock_res_sorted["la_expected"],
        equal_nan=True,
    ).all()


def test_adjusted_expected_tile_some_nans_and_square_tiling_diag_band():
    print("Running tile some nans la_exp test + square tiling + diag band")
    # # Essentially, testing this function:
    # def chrom_chunk_generator_s(chroms, w, band):
    #     for chrom in chroms:
    #         chr_start, chr_stop = the_c.extent(chrom)
    #         for tilei, tilej in square_matrix_tiling(chr_start, chr_stop, step, w):
    #             # check if a given tile intersects with
    #             # with the diagonal band of interest ...
    #             diag_from = tilej[0] - tilei[1]
    #             diag_to   = tilej[1] - tilei[0]
    #             #
    #             band_from = 0
    #             band_to   = band
    #             # we are using this >2w trick to exclude
    #             # tiles from the lower triangle from calculations ...
    #             if (min(band_to,diag_to) - max(band_from,diag_from)) > 2*w:
    #                 yield chrom, tilei, tilej
    # first, generate that locally-adjusted expected:
    nnans = 1
    band_idx = int(band / b)
    res_df = pd.DataFrame([])
    for tilei, tilej in dotfinder.square_matrix_tiling(
        start, stop, step=40, edge=w, square=False
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
        if (min(band_to, diag_to) - max(band_from, diag_from)) > 2 * w:
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
            res = dotfinder.get_adjusted_expected_tile_some_nans(
                origin=origin,
                observed=observed,
                expected=expected,
                bal_weights=(ice_weight_i, ice_weight_j),
                kernels={"donut": kernel, "footprint": np.ones_like(kernel)},
                # nan_threshold=1,
                verbose=False,
            )
            is_inside_band = res["bin1_id"] > (res["bin2_id"] - band_idx)
            # new style, selecting good guys:
            does_comply_nans = res["la_exp." + "footprint" + ".nnans"] < nnans
            # so, select inside band and nNaNs compliant results and append:
            res_df = res_df.append(
                res[is_inside_band & does_comply_nans], ignore_index=True
            )

    # sort and reset index, there shouldn't be any duplicates now:
    res_df = (
        res_df.drop_duplicates()
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
    assert np.isclose(
        res_df["la_exp." + "donut" + ".value"],
        mock_res_sorted["la_expected"],
        equal_nan=True,
    ).all()
