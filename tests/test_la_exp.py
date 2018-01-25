# create a test for get_locally_adjusted_expected using mock dense matrices ...

import numpy as np
import pandas as pd

# just follow mirnylab practises:
from nose import assert_raise


# load bunch of array from a numpy npz container:
arrs_loaded = np.load('./data/mock_inputs.npz')
# snippets of M_ice,E_ice and v_ice are supposed 
# to be there ...
mock_M_ice = arrs_loaded['mock_M_ice']
mock_E_ice = arrs_loaded['mock_E_ice']
mock_v_ice = arrs_loaded['mock_v_ice']






# kernels=(kernel,)
b=the_c.info['bin-size']


# first, generate that locally-adjusted expected:
# Ed_raw, mask_ndx, Cobs, Cexp, NN = 
res = get_adjusted_expected(observed=mock_M_ice,
                     expected=mock_E_ice,
                     ice_weight=mock_v_ice,
                     kernel=kernel,
                     b=b,
                     return_type="sparse")




w = 3
p = 1
kernel = get_kernel(w,p,ktype='donut')