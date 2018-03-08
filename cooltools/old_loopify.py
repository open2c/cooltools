# collection of legacy get_la_exp functions:


from scipy.linalg import toeplitz
from scipy.ndimage import convolve
import numpy as np
import pandas as pd
 

###########################
# Calculate couple more columns for filtering/sorting ...
###########################
_cmp_masks = lambda M_superset,M_subset: (0 > M_superset.astype(np.int) -
                                                 M_subset.astype(np.int)).any()




##########################################################################################


###############################################################
# get_locally_adjusted_expected_tile_reference_implementation
###############################################################
# well-tested but old  - it here merely
# as a reference.
def get_adjusted_expected(observed, expected, ice_weight, kernel, b, return_type="dense"):
    '''
    get_adjusted_expected, get the expected adjusted
    to the local background using local-kernel convolution.

    ... it comes down to comparison of pixel intensity
    in observed: M_ice,
    and a locally-modified expected:
                              DONUT[i,j](M_ice)
    Ed_ice[i,j] = E_ice[i,j]*------------------
                              DONUT[i,j](E_ice)
    where DONUT[i,j](M) is a sum of the pixel intensities
    in the donut-shaped (or other shaped) vicinity
    of the pixel of interest (i,j).
    
    However, comparison between balanced/scaled values
    that are 0..1 using Poisson statistics is 
    intractable, and thus both M_ice and Ed_ice, has to be
    rescaled back to raw-count values, i.e. de-iced:
    de-iced M_ice is the same as M_raw, while de-icing Ed:
                              DONUT[i,j](M_ice)
    Ed_raw[i,j] = E_raw[i,j]* -----------------
                              DONUT[i,j](E_ice)
    , where
                     1       1                 
    E_raw[i,j] = --------*-------- * E_ice[i,j]
                 v_ice[i] v_ice[j]             
    
    Parameters
    ----------
    observed : numpy.ndarray
        square symmetrical dense-matrix
        that contains iced observed M_ice
    expected : numpy.ndarray
        square symmetrical dense-matrix
        that contains expected, calculated
        based on iced observed: E_ice.
    ice_weight : numpy.ndarray
        1D vector used to turn raw observed
        into iced observed.
    kernels : list of numpy.arrays
        list of kernels/masks to perform
        convolution of the heatmap. Kernels
        represent the local environment to 
        find prominent peaks against.
        Peak considered significant is it
        passes stat-tests for all kernels
        from the list. Use just one for now.
        ########
        # KERNELS ARE FLIPPED - BEWARE!!!
        ########
    b : int
        bin-size in nucleotides (bases).
    
    Returns
    -------
    Exp_raw : numpy.ndarray
        ndarray that stores a 2D matrix of
        the adjusted expected values Ed_raw.
    mask : numpy.ndarray
        ndarray that stores a 2D matrix of
        type np.bool. Every True element of
        the mask should be excluded from the
        downstream analysis.
    KM : numpy.ndarray
        array that stores a 2D matrix with
        the product of convolution of
        observed with the kernel 
    KE : numpy.ndarray
        array that stores a 2D matrix with
        the product of convolution of
        expected with the kernel 
    NumNans : numpy.ndarray
        array that stores a 2D matrix with
        the number of np.nan elements that
        are surrounding each pixel.

    or a sparsified table:
        columns including: row, col, expected(E_raw),
        locally adjusted expected (Ed_raw)
    '''

    # DEAL WITH BIN_SIZE AND MATRIX DIMENSIONS ...
    # bin size has to be extracted:
    # later it will be replaced with
    # the sliding window-style processing,
    # maybe using dask ...
    # DECIDE HOW DO WE GET EXPECTED - 
    # matrix VS vector ?! ...

    # let's extract full matrices and ice_vector:
    M_ice = observed
    E_ice = expected
    v_ice = ice_weight
    # can deal only with 
    # 1 kernel"
    assert isinstance(kernel, np.ndarray)
    kernel = kernel
    kernel_footprint = np.ones_like(kernel)

    # Now we have all the ingredients ready 
    # and we can go ahead and try to execute 
    # actuall loop-calling algorithm
    # as explained in Rao et al 2014:

    # deiced E_ice: element-wise division of E_ice[i,j] and
    # v_ice[i]*v_ice[j]:
    E_raw = np.divide(E_ice, np.outer(v_ice,v_ice))

    # now let's calculate kernels ...
    # extract heatmap parameters:
    s, s = M_ice.shape
    # extract donut parameters: 
    w, w = kernel.shape
    # size must be odd: pixel (i,j) of interest in the middle
    # and equal amount of pixels are to left,right, up and down
    # of it: 
    assert w%2 != 0
    w = int((w-1)/2)
    #
    #
    # a matrix filled with the donut-sums based on the ICEed matrix
    KM = convolve(M_ice,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # a matrix filled with the donut-sums based on the expected matrix
    KE = convolve(E_ice,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # idea for calculating how many NaNs are there around
    # a pixel, is to take the input matrix, do np.isnan(matrix)
    # and convolve it with the np.ones_like(around_pixel) kernel:
    NN = convolve(np.isnan(M_ice).astype(np.int),
                  kernel_footprint,
                  mode='constant',
                  cval=0.0,
                  origin=0)


    print("kernels convolved with observed and expected ...")

    # details like, boundaries, kernel-footprints with too many NaNs, etc, 
    # are to be addressed later ...
    #
    # now finally, E_raw*(KM/KE), as the 
    # locally-adjusted expected with raw counts as values:
    Ed_raw = np.multiply(E_raw, np.divide(KM, KE))
    # ###################
    # some trivial and not so trivial masking 
    # is needed: exclude contacts >2Mb apart,
    # pixels from NaNs-rich regions, etc.
    #####################
    # mask out CDFs for NaN Ed_raw
    # (would be NaN anyhow?!)
    mask_Ed = np.isnan(Ed_raw)
    # mask out CDFs for NaN M_ice
    # (would be NaN anyhow?!)
    mask_M = np.isnan(M_ice)
    # mask out CDFs for pixels
    # with too many NaNs (#NaN>=1) around:
    mask_NN = (NN >= 1)
    # 
    # simple tests mask_Ed vs mask_M:
    if not _cmp_masks(mask_Ed,mask_M):
        print("mask_Ed include all elements of mask_M (expected)")
    else:
        print("ATTENTION! mask_M has elements absent from mask_Ed")
    # simple tests mask_Ed vs mask_M:
    if not _cmp_masks(mask_Ed,mask_NN):
        print("mask_Ed include all elements of mask_NN (expected)")
    else:
        print("ATTENTION! mask_NN has elements absent from mask_Ed")
    if (mask_Ed == mask_NN).all():
        print("In fact mask_Ed==mask_NN (expected for NN>=1)")
    else:
        print("But mask_Ed!=mask_NN (expected e.g. for NN>=2)")
    # if (np.isnan(pvals) == mask_Ed).all():
    #     print("Also isnan(pvals)==mask_Ed (sort of expected)")
    # else:
    #     print("HOWEVER: isnan(pvals)!=mask_Ed (investigate ...)")
    # keep this test here, while we are learning:
    print("If all test yield as expected, masking is practically useless ...")
    ###########################
    # mask out boundaries:
    # kernel-half width from 
    # every side of the heatmap.
    mask_bnd = np.ones_like(M_ice, dtype=np.bool)
    mask_bnd[w:-w, w:-w] = False

    # masking everyhting further than 2Mb away:
    # index of 2Mb in current bin-sizes:
    idx_2Mb = int(2e+6/b)
    assert s > idx_2Mb
    # everything outside 2Mb will become NaN
    mask_2Mb = np.concatenate(
                    [np.zeros(idx_2Mb),
                    np.ones(s-idx_2Mb)]
                             ).astype(np.bool)
    # In other words:
    # mask_2Mb = np.where(np.arange(s)<idx_2Mb,False,True)

    # 2D mask unrolled:
    # masking everyhting outside 2Mb
    mask_2Mb = toeplitz(mask_2Mb)

    # combine all filters/masks:
    # mask_Ed || mask_M || mask_NN || mask_bnd || mask_2Mb ...
    mask_ndx = np.any(
                      (mask_Ed,
                        mask_M,
                        mask_NN,
                        mask_bnd,
                        mask_2Mb),
                      axis=0)
    # same as:
    # np.logical_or(mask_Ed, mask_M, mask_NN, mask_bnd, mask_2Mb)
    # but logical_or can work only with a pair of arguments.

    if return_type == "dense":
        # any nonzero element in `mask_ndx` 
        # must be masked out for the downstream
        # analysis:
        return Ed_raw, mask_ndx, KM, KE, NN
    elif return_type == "sparse":
        # return sparsified data frame
        # with the same format as the
        # new tiled get_loc_adj_exp funcs.
        # any nonzero element in `mask_ndx` 
        # must be masked out from `pvals`:
        # so, we'll just take the negated
        # elements of the combined mask:
        i, j = np.nonzero(~mask_ndx)
        # take the upper triangle with 2Mb close to diag:
        upper = (i<j)
        i = i[upper]
        j = j[upper]
        # pack it into DataFrame:
        peaks_df = pd.DataFrame({"row": i,
                                 "col": j,
                                 "expected": E_raw[i,j],
                                 "la_expected": Ed_raw[i,j],
                                 "observed": observed[i,j],
                                })
        # return:
        return peaks_df



##########################################################################################



def get_adjusted_expected_tile(origin,
                               observed,
                               expected,
                               ice_weight,
                               kernels,
                               b,
                               band=2e+6):
    """
    get_adjusted_expected_slice is calculating
    locally-adjusted expected for smaller slices
    of the full intra-chromosomal interaction
    matrix.

    Each slice is characterized by the coordinate
    of the top-left corner and size.

    * * * * * * * * * * *  0-th slice
    *       *           *
    *       *           *
    *   * * * * *       *  i-th slice
    *   *   *   *       *
    * * * * *   *       *
    *   *       *       *
    *   * * * * *       *
    *                   *
    *              ...  *  ...
    *                   *
    *                   *
    * * * * * * * * * * *
    """
    # extract origin coordinate of this tile:
    io, jo = origin
    # let's extract full matrices and ice_vector:
    M_ice = observed
    E_ice = expected
    v_ice = ice_weight
    kernel, = kernels

    # deiced E_ice: element-wise division of E_ice[i,j] and
    # v_ice[i]*v_ice[j]:
    E_raw = np.divide(E_ice, np.outer(v_ice,v_ice))

    s, s = M_ice.shape
    # extract donut parameters: 
    w, w = kernel.shape
    # size must be odd: pixel (i,j) of interest in the middle
    # and equal amount of pixels are to left,right, up and down
    # of it: 
    assert w%2 != 0
    w = int((w-1)/2)
    #
    #
    # a matrix filled with the donut-sums based on the ICEed matrix
    KM = convolve(M_ice,
                  kernel,
                  mode='constant',
                  # try to mask out border right away:
                  cval=np.nan,
                  # # before it was:
                  # cval=0.0,
                  origin=0)
    # a matrix filled with the donut-sums based on the expected matrix
    KE = convolve(E_ice,
                  kernel,
                  mode='constant',
                  # try to mask out border right away:
                  cval=np.nan,
                  # # before it was:
                  # cval=0.0,
                  origin=0)
    # idea for calculating how many NaNs are there around
    # a pixel, is to take the input matrix, do np.isnan(matrix)
    # and convolve it with the np.ones_like(around_pixel) kernel:
    NN = convolve(np.isnan(M_ice).astype(np.int),
                  np.ones_like(kernel),
                  mode='constant',
                  cval=0.0,
                  origin=0)


    print("kernels convolved with observed and expected ...")

    # now finally, E_raw*(KM/KE), as the 
    # locally-adjusted expected with raw counts as values:
    Ed_raw = np.multiply(E_raw, np.divide(KM, KE))
    # mask out CDFs for NaN Ed_raw
    # (would be NaN anyhow?!)
    mask_Ed = np.isnan(Ed_raw)
    # mask out CDFs for pixels
    # with too many NaNs (#NaN>=1) around:
    mask_NN = NN >= 1
    # masking everyhting further than 2Mb away
    # is easier to do on indices.
    band_idx = int(band/b)
    assert s > band_idx

    # combine all filters/masks:
    # mask_Ed || mask_NN
    mask_ndx = np.logical_or(mask_Ed, mask_NN)
    # any nonzero element in `mask_ndx` 
    # must be masked out from `pvals`:
    # so, we'll just take the negated
    # elements of the combined mask:
    i, j = np.nonzero(~mask_ndx)
    # take the upper triangle with 2Mb close to diag:
    upper_band = (i<j) & (i>j-band_idx)
    i = i[upper_band]
    j = j[upper_band]
    # pack it into DataFrame:
    peaks_df = pd.DataFrame({"row": i+io,
                             "col": j+jo,
                             "expected": Ed_raw[i,j],
                             "observed": observed[i,j],
                            })

    return peaks_df




##########################################################################################


# no tiling support:
def get_adjusted_expected_some_nans(observed,
                                    expected,
                                    ice_weight,
                                    kernels,
                                    b,
                                    band=2e+6,
                                    nan_threshold=2):
    '''
    Same as 'get_adjusted_expected'
    ...
    ...
    but able to tolerate some NaNs in 
    the kernel area.

    The idea is to fill NaNs with 0s,
    convolve M_ice and E_ice like that.
    Then figure out how many NaNs were
    near each pixel using convolution
    of NaN-masks of the input matrices.

    We have to define a list of common
    NaNs, shared between M_ice and E_ice
    and use it both for assigning 0.0 and
    for calculating how many NaNs are
    around each pixel. If we don't do this
    number of zeroed-NaNs might become
    different between KM/KE. Thus, such
    ratio would be "unfair".

    We could have extrapolated NaNs
    like they do in Astro-convolve
    but I think zeroeing-out is fine,
    since we are dealing with the ratio
    of convolutions KM/KE.

    After that, just skip initial 
    convolution results whenever there are
    too many NaN either in M_ice or E_ice
    to begin with.

    '''

    # DEAL WITH BIN_SIZE AND MATRIX DIMENSIONS ...
    # bin size has to be extracted:
    # later it will be replaced with
    # the sliding window-style processing,
    # maybe using dask ...
    # DECIDE HOW DO WE GET EXPECTED - 
    # matrix VS vector ?! ...

    # let's extract full matrices and ice_vector:
    M_ice = np.copy(observed)
    E_ice = np.copy(expected)
    v_ice = ice_weight
    kernel, = kernels

    # deiced E_ice: element-wise division of E_ice[i,j] and
    # v_ice[i]*v_ice[j]:
    E_raw = np.divide(E_ice, np.outer(v_ice,v_ice))

    s, s = M_ice.shape
    w, w = kernel.shape
    # size must be odd: pixel (i,j) of interest in the middle
    # and equal amount of pixels are to left,right, up and down
    # of it: 
    assert w%2 != 0
    w = int((w-1)/2)
    #
    # let's calculate a matrix of common np.nans
    # as a logical_or:
    N_ice = np.logical_or(np.isnan(M_ice),
                          np.isnan(E_ice))
    # fill in common nan-s 
    # with zeroes:
    M_ice[N_ice] = 0.0
    E_ice[N_ice] = 0.0
    # think about usinf copyto and where functions later:
    # https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another
    # 
    # Perform Convolutions:
    # a matrix filled with the donut-sums based on the ICEed matrix
    KM = convolve(M_ice,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # a matrix filled with the donut-sums based on the expected matrix
    KE = convolve(E_ice,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # idea for calculating how many NaNs are there around
    # a pixel, is to take the matrix, of shared NaNs
    # (shared between M_ice and E_ice) and convolve it
    # with the np.ones_like(around_pixel) kernel:
    NN = convolve(N_ice.astype(np.int),
                  np.ones_like(kernel),
                  mode='constant',
                  # make it look, like there are 
                  # a lot of NaNs beyond
                  # the boundary ...
                  cval=1,
                  origin=0)
    # ####################################
    # we'll use cval=0 in case of real data
    # and we'll use cval=1 for NN mask
    # thus border issue becomes a
    # "too many NaNs"-issue as well.
    # ####################################
    print("kernels convolved with observed and expected ...")
    # 
    # now finally, E_raw*(KM/KE), as the 
    # locally-adjusted expected with raw counts as values:
    Ed_raw = np.multiply(E_raw, np.divide(KM, KE))

    # addressing details:
    # boundaries, kernel-footprints with too many NaNs, etc:
    ######################################################
    # mask out CDFs for NaN in Ed_raw
    # originating from NaNs in E_raw
    # and zero-values in KE, as there 
    # should be no NaNs in KM (or KE) anymore.
    ######################################
    # TODO: SHOULD we worry about np.isfinite ?!
    ######################################
    mask_Ed = np.isnan(Ed_raw)
    # mask out CDFs for pixels
    # with too many NaNs around
    # (#NaN>=threshold) each pixel:
    mask_NN = (NN >= nan_threshold)
    # this way boundary pixels are masked
    # closed to diagonal pixels are masked
    # pixels for which M_ice itself is NaN
    # can be left alone, as they would 
    # be masked during Poisson test comparison.

    # masking everyhting further than 2Mb away
    # is easier to do on indices.
    band_idx = int(band/b)
    assert s > band_idx

    # combine all filters/masks:
    # mask_Ed || mask_NN
    mask_ndx = np.logical_or(mask_Ed,
                             mask_NN)
    # any nonzero element in `mask_ndx` 
    # must be masked out from `pvals`:
    # so, we'll just take the negated
    # elements of the combined mask:
    i, j = np.where(~mask_ndx)
    # take the upper triangle with 2Mb close to diag:
    upper_band = np.logical_and((i<j),
                                (i>j-band_idx))
    # peek the reduced list of pixels:
    i = i[upper_band]
    j = j[upper_band]
    # pack it into DataFrame:
    peaks_df = pd.DataFrame({"row": i,
                             "col": j,
                             "expected": Ed_raw[i,j],
                             "observed": observed[i,j],
                            })
    # INSTEAD OF:
    # return Ed_raw, mask_ndx, KM, KE, NN
    # DO:
    return peaks_df


























