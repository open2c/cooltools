# create a test for get_locally_adjusted_expected using mock dense matrices ...
# testing la_exp functions on a single piece of matrix
# no chunking applied here ...

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
from loopify import get_adjusted_expected_tile_some_nans

# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, 'data', 'mock_inputs.npz')
mock_result = op.join(testdir, 'data', 'mock_res.csv.gz')

# load mock results:
mock_res = pd.read_csv(mock_result)

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_raw, M_ice, E_ice and v_ice are supposed 
# to be there ...
mock_M_raw = arrays_loaded['mock_M_raw']
mock_M_ice = arrays_loaded['mock_M_ice']
mock_E_ice = arrays_loaded['mock_E_ice']
mock_v_ice = arrays_loaded['mock_v_ice']


# we need w-edge for tiling procedures:
# w = 3
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



#################################################################
# main test function for new 'get_adjusted_expected_tile_some_nans':
#################################################################
def test_adjusted_expected_tile_some_nans():
    print("Running tile some nans la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res = get_adjusted_expected_tile_some_nans(
                         origin=(0,0),
                         observed=mock_M_raw, # should be RAW ...
                         expected=mock_E_ice,
                         bal_weight=mock_v_ice,
                         kernels={"donut":kernel,
                                 "footprint":np.ones_like(kernel)})
    ###################
    # nan_threshold = 1
    ###################
    # mock results are supposedly for
    # the contacts closer than the band
    # nucleotides to each other...
    # we want to retire that functional 
    # from the 'get_adjusted_expected_tile_some_nans'
    # so we should complement that selection
    # here:
    band_idx = int(band/b)
    is_inside_band = (res["row"]>(res["col"]-band_idx))
    ###################################
    # to add footprint NaN counting ...
    ###################################
    # so, selecting inside band results only:
    res = res[is_inside_band].reset_index(drop=True)
    # 
    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res[['row','col']].equals(
            mock_res[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res["la_exp."+"donut"+".value"],
            mock_res['la_expected'],
            equal_nan=True).all()
        )
    ####################################
    # BEFORE UPDATE:
    # assert (
    #     np.isclose(
    #         res[["expected",'observed']],
    #         mock_res[['la_expected','observed']],
    #         equal_nan=True).all()
    #     )
    ####################################
    #
    # try recovering NaNs ...
    #
    #
    res = get_adjusted_expected_tile_some_nans(
                     origin=(0,0),
                     observed=mock_M_raw, # should be RAW...
                     expected=mock_E_ice,
                     bal_weight=mock_v_ice,
                     kernels={"donut":kernel,
                             "footprint":np.ones_like(kernel)})
    ###################
    # nan_threshold = 2
    ###################
    # neccessary to exclude contacts outside 
    # the diagonal band, after that functional
    # was retired from 'get_adjusted_expected_tile_some_nans':
    band_idx = int(band/b)
    is_inside_band = (res["row"]>(res["col"]-band_idx))
    ###################################
    # to add footprint NaN counting ...
    ###################################
    # so, selecting inside band results only:
    res = res[is_inside_band].reset_index(drop=True)

    # now we can only guess the size:
    assert (res['row'].size > mock_res['row'].size)





