# create a test for the chunking versions of 'get_adjusted_expected_tile_some_nans':

import numpy as np
import pandas as pd

import os.path as op

# just follow mirnylab practises:
from nose.tools import assert_raises

# let's try running tests 
# without installing loopify:
import sys
sys.path.append("../")


# try importing stuff from loopify:
from loopify import get_adjusted_expected_tile_some_nans, \
                    diagonal_matrix_tiling, \
                    square_matrix_tiling, \
                    tile_of_expected

# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, 'data', 'mock_inputs.npz')
mock_result = op.join(testdir, 'data', 'mock_res.csv.gz')

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_raw, M_ice, E_ice and v_ice are supposed 
# to be there ...
mock_M_raw = arrays_loaded['mock_M_raw']
mock_M_ice = arrays_loaded['mock_M_ice']
mock_E_ice = arrays_loaded['mock_E_ice']
mock_v_ice = arrays_loaded['mock_v_ice']

# 1D expected extracted for tiling-tests:
mock_exp = mock_E_ice[0,:]
get_mock_exp = lambda start,stop,shift: mock_exp[start+shift:stop+shift]



# we need w-edge for tiling procedures:
w = 3
# p = 1
# kernel = get_kernel(w,p,ktype='donut')
# # just a simple donut kernel for testing:
kernel = np.array([[1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1]])
# bin size:
# mock data were extracted
# from 20kb matrix:
b=20000
# 2MB aroud diagonal to look at:
band=2e+6
# 1MB aroud diagonal to look at:
band_1=1e+6

# start, stop for tiling procedures:
start, stop = 0, len(mock_M_raw)

# load mock results:
mock_res = pd.read_csv(mock_result)


def test_adjusted_expected_tile_some_nans_and_diag_tiling():
    print("Running tile some nans la_exp test + diag tiling")
    # first, generate that locally-adjusted expected:
    band_1_idx = int(band_1/b)
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res_list = []
    for tile in diagonal_matrix_tiling(start, stop, w, band = band_1_idx):
        # let's keep i,j-part explicit here:
        tilei, tilej = tile, tile
        # define origin:
        origin = (tilei[0], tilej[0])
        # RAW observed matrix slice:
        observed = mock_M_raw[slice(*tilei),slice(*tilej)]
        # trying new expected function:
        expected = tile_of_expected(start, tilei, tilej, get_mock_exp)
        # for diagonal chuynking/tiling tilei==tilej:
        ice_weight = mock_v_ice[slice(*tilei)]
        # that's the main working function from loopify:
        res = get_adjusted_expected_tile_some_nans(origin = origin,
                                                 observed = observed,
                                                 expected = expected,
                                                 bal_weight = ice_weight,
                                                 kernels = {"donut":kernel,},
                                                 nan_threshold=1,
                                                 verbose=False)
        is_inside_band = res["row"] > (res["col"]-band_1_idx)
        # so, selecting inside band results only:
        res = res[is_inside_band].reset_index(drop=True)
        res_list.append(res)

    # concat bunch of DFs:
    res_df = pd.concat(res_list, ignore_index=True)

    # drop dups (from overlaping tiles) and reset index:
    res_df = res_df.drop_duplicates().reset_index(drop=True)

    # get a subset of mock results (inside 1Mb band):
    is_inside_band_1 = (mock_res["row"]>(mock_res["col"]-band_1_idx))
    mock_res_1 = mock_res[is_inside_band_1].reset_index(drop=True)

    # apparently sorting is needed in this case:
    res_df = res_df.sort_values(by=['row','col']).reset_index(drop=True)
    mock_res_1 = mock_res_1.sort_values(by=['row','col']).reset_index(drop=True)


    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res_df[['row','col']].equals(
            mock_res_1[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res_df["la_exp."+"donut"+".value"],
            mock_res_1['la_expected'],
            equal_nan=True).all()
        )




def test_adjusted_expected_tile_some_nans_and_square_tiling():
    print("Running tile some nans la_exp test + square tiling")
    # first, generate that locally-adjusted expected:
    band_idx = int(band/b)
    res_list = []
    for tilei, tilej in square_matrix_tiling(start, stop, tile_size=40, edge=w, square=False):
        # define origin:
        origin = (tilei[0], tilej[0])
        # RAW observed matrix slice:
        observed = mock_M_raw[slice(*tilei),slice(*tilej)]
        # trying new expected function:
        expected = tile_of_expected(start, tilei, tilej, get_mock_exp)
        # for diagonal chuynking/tiling tilei==tilej:
        ice_weight_i = mock_v_ice[slice(*tilei)]
        ice_weight_j = mock_v_ice[slice(*tilej)]
        # that's the main working function from loopify:
        res = get_adjusted_expected_tile_some_nans(origin = origin,
                                                 observed = observed,
                                                 expected = expected,
                                                 bal_weight = (ice_weight_i, ice_weight_j),
                                                 kernels = {"donut":kernel,},
                                                 nan_threshold=1,
                                                 verbose=False)
        is_inside_band = res["row"] > (res["col"]-band_idx)
        # so, selecting inside band results only:
        res = res[is_inside_band].reset_index(drop=True)
        res_list.append(res)

    # concat bunch of DFs:
    res_df = pd.concat(res_list, ignore_index=True)

    # drop dups (from overlaping tiles) and reset index:
    res_df = res_df.drop_duplicates().reset_index(drop=True)

    # apparently sorting is needed in this case:
    res_df = res_df.sort_values(by=['row','col']).reset_index(drop=True)
    mock_res_sorted = mock_res.sort_values(by=['row','col']).reset_index(drop=True)

    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res_df[['row','col']].equals(
            mock_res_sorted[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res_df["la_exp."+"donut"+".value"],
            mock_res_sorted['la_expected'],
            equal_nan=True).all()
        )





