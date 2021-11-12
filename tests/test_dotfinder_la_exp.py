# create a test for get_locally_adjusted_expected using mock dense matrices ...
# testing la_exp functions on a single piece of matrix
# no chunking applied here ...

import numpy as np
import pandas as pd

import os.path as op

# try importing stuff from dotfinder:
from cooltools.api.dotfinder import get_adjusted_expected_tile_some_nans

# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, "data", "dotfinder_mock_inputs.npz")
mock_result = op.join(testdir, "data", "dotfinder_mock_res.csv.gz")

# load mock results:
mock_res = pd.read_csv(mock_result)
mock_res = mock_res.rename(columns={"row": "bin1_id", "col": "bin2_id"})

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_raw, M_ice, E_ice and v_ice are supposed
# to be there ...
mock_M_raw = arrays_loaded["mock_M_raw"]
mock_M_ice = arrays_loaded["mock_M_ice"]
mock_E_ice = arrays_loaded["mock_E_ice"]
mock_v_ice = arrays_loaded["mock_v_ice"]


# we need w-edge for tiling procedures:
# w = 3
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


#################################################################
# main test function for new 'get_adjusted_expected_tile_some_nans':
#################################################################
def test_adjusted_expected_tile_some_nans():
    print("Running tile some nans la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN =
    res = get_adjusted_expected_tile_some_nans(
        origin=(0, 0),
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
    assert np.isclose(
        res["la_exp." + "donut" + ".value"], mock_res["la_expected"], equal_nan=True
    ).all()
    #
    # try recovering more NaNs ...
    # (allowing 1 here per pixel's footprint)...
    res = get_adjusted_expected_tile_some_nans(
        origin=(0, 0),
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
