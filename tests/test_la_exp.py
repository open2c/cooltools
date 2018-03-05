# create a test for get_locally_adjusted_expected using mock dense matrices ...

import numpy as np
import pandas as pd

import os.path as op

# just follow mirnylab practises:
from nose.tools import assert_raises

# let's try running tests 
# without installing loopify:
import sys
sys.path.append("../")

# try importing stuff from old_loopify:
from old_loopify import get_adjusted_expected,\
                    get_adjusted_expected_tile,\
                    get_adjusted_expected_some_nans

# try importing stuff from loopify:
from loopify import get_adjusted_expected_tile_some_nans

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



# load mock results:
mock_res = pd.read_csv(mock_result)



def test_adjusted_expected_reference():
    print("Running reference la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res = get_adjusted_expected(observed=mock_M_ice,
                         expected=mock_E_ice,
                         ice_weight=mock_v_ice,
                         kernel=kernel,
                         b=b,
                         return_type="sparse")
    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res[['row','col']].equals(
            mock_res[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res[['expected','la_expected','observed']],
            mock_res[['expected','la_expected','observed']],
            equal_nan=True).all()
        )





def test_adjusted_expected_tile():
    print("Running tile la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res = get_adjusted_expected_tile(
                         origin=(0,0),
                         observed=mock_M_ice,
                         expected=mock_E_ice,
                         ice_weight=mock_v_ice,
                         kernels=(kernel,),
                         b=b,
                         band=2e+6)
    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res[['row','col']].equals(
            mock_res[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res[['expected','observed']],
            mock_res[['la_expected','observed']],
            equal_nan=True).all()
        )




def test_adjusted_expected_tile_some_nans():
    print("Running tile some nans la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res = get_adjusted_expected_tile_some_nans(
                         origin=(0,0),
                         observed=mock_M_raw, # should be RAW ...
                         expected=mock_E_ice,
                         bal_weight=mock_v_ice,
                         kernels={"donut":kernel,},
                         nan_threshold=1)
    # mock results are supposedly for
    # the contacts closer than the band
    # nucleotides to each other...
    # we want to retire that functional 
    # from the 'get_adjusted_expected_tile_some_nans'
    # so we should complement that selection
    # here:
    band_idx = int(band/b)
    is_inside_band = (res["row"]>(res["col"]-band_idx))
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
                     kernels={"donut":kernel,},
                     nan_threshold=2)
    # neccessary to exclude contacts outside 
    # the diagonal band, after that functional
    # was retired from 'get_adjusted_expected_tile_some_nans':
    band_idx = int(band/b)
    is_inside_band = (res["row"]>(res["col"]-band_idx))
    # so, selecting inside band results only:
    res = res[is_inside_band].reset_index(drop=True)

    # now we can only guess the size:
    assert (res['row'].size > mock_res['row'].size)





def test_adjusted_expected_some_nans():
    print("Running some nans la_exp test")
    # first, generate that locally-adjusted expected:
    # Ed_raw, mask_ndx, Cobs, Cexp, NN = 
    res = get_adjusted_expected_some_nans(
                         observed=mock_M_ice,
                         expected=mock_E_ice,
                         ice_weight=mock_v_ice,
                         kernels=(kernel,),
                         b=b,
                         band=2e+6,
                         nan_threshold=1)
    # ACTUAL TESTS:
    # integer part of DataFrame must equals exactly:
    assert (
        res[['row','col']].equals(
            mock_res[['row','col']])
        )
    # compare floating point part separately:
    assert (
        np.isclose(
            res[['expected','observed']],
            mock_res[['la_expected','observed']],
            equal_nan=True).all()
        )
    #
    # try recovering NaNs ...
    #
    #
    res = get_adjusted_expected_some_nans(
                     observed=mock_M_ice,
                     expected=mock_E_ice,
                     ice_weight=mock_v_ice,
                     kernels=(kernel,),
                     b=b,
                     band=2e+6,
                     nan_threshold=2)
    # now we can only guess the size:
    assert (res['row'].size > mock_res['row'].size)


