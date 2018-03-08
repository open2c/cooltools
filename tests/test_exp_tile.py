###################
# trying out the square tiling thing
# in concert with the toeplitz of expected ...

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
from loopify import tile_of_expected

# adjust the path for data:
testdir = op.realpath(op.dirname(__file__))

# mock input data location:
mock_input = op.join(testdir, 'data', 'mock_inputs.npz')

# load bunch of array from a numpy npz container:
arrays_loaded = np.load(mock_input)
# snippets of M_raw, M_ice, E_ice and v_ice are supposed 
# to be there ...
mock_E_ice = arrays_loaded['mock_E_ice']


# 1D expected extracted for tiling-tests:
mock_exp = mock_E_ice[0,:]
get_mock_exp = lambda start,stop,shift: mock_exp[start+shift:stop+shift]


# start, stop for tiling procedures:
start = 0


def test_tile_exp_upper():
    # ####################
    # requested tile is in the upper
    # triangle of the matrix region
    # ####################
    i0, i1 = 10, 90
    j0, j1 = 130, 190
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()


def test_tile_exp_lower():
    # ####################
    # requested tile is in the lower
    # triangle of the matrix region
    # ####################
    i0, i1 = 150, 200
    j0, j1 = 5, 100
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()


def test_tile_exp_diag_square():
    # ####################
    # requested tile is on the diag
    # square tile!
    # ####################
    i0, i1 = 50, 100
    j0, j1 = 50, 100
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()



def test_tile_exp_diag_rectangle():
    # ####################
    # requested tile is on the diag
    # rectangular tile!
    # ####################
    i0, i1 = 50, 100
    j0, j1 = 50, 80
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()



def test_tile_exp_crossed_horizontal():
    # ####################
    # requested tile is crossed by diag
    # rectangular tile!
    # ####################
    i0, i1 = 50, 180
    j0, j1 = 60, 120
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()


def test_tile_exp_crossed_vertical():
    # ####################
    # requested tile is crossed by diag
    # rectangular tile!
    # ####################
    i0, i1 = 50, 180
    j0, j1 = 40, 180
    # 
    E_tile_ref = mock_E_ice[i0:i1,j0:j1]
    # now, let's try to reconstruct it
    # from diag-indexed vector using
    expected = tile_of_expected(start, (i0,i1), (j0,j1), get_mock_exp)
    ##########################
    assert np.isclose(E_tile_ref, expected, equal_nan=True).all()







