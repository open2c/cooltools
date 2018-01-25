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

# try importing stuff from loopify:
from loopify import get_adjusted_expected
# ,
                    # get_adjusted_expected_tile,
                    # get_adjusted_expected_tile_some_nans,
                    # get_adjusted_expected_some_nans


testdir = op.realpath(op.dirname(__file__))


mock_input = op.join(testdir, 'data', 'mock_inputs.npz')
mock_result = op.join(testdir, 'data', 'mock_res.csv.gz')

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_ice,E_ice and v_ice are supposed 
# to be there ...
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


# load mock results:
mock_res = pd.read_csv(mock_result)



def test_adjusted_expected():
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























