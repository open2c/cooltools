

####################
# wrap up loop-call
####################

from scipy.linalg import toeplitz
from scipy.ndimage import convolve
from scipy.stats import poisson
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd

from sklearn.cluster import Birch


###########################
# Calculate couple more columns for filtering/sorting ...
###########################
get_slice = lambda row,col,window: (
                        slice(row-window, row+window+1),
                        slice(col-window, col+window+1))




def call_dots_matrix(matrices, vectors, kernels, b):
    '''
    it comes down to comparison of pixel intensity
    in observed: M_ice,
    and a locally-modified expected:
                              DONUT[i,j](M_ice)
    Ed_ice[i,j] = E_ice[i,j]*------------------
                              DONUT[i,j](E_ice)
    where DONUT[i0,j](M) is a sum of the pixel intensities
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
    so, at the end of the day,
    simply compare M_raw and Ed_raw ...
    
    Parameters
    ----------
    matrices : tuple-like
        square symmetrical dense-matrices
        that contain iced, raw and expected
        heatmaps in numpy.array format:
        M_ice, M_raw, E_ice to be precise.
    vectors : tuple-like
        1D vectors needed to run dot-caller
        such as icing and probably expected
        in a numpy.array format.
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
    peaks : pandas.DataFrame
        data frame with 2D genomic regions that
        have been called by loop-caller (BEDPE-like).
    
        '''

    # # make sure everything is compatible:
    # try:
    #     # let's extract full matrices and ice_vector:
    #     M_ice, M_raw, E_ice = matrices
    # except ValueError as e:
    #     print("\"matrices\" has to have M_ice,M_raw,E_ice",
    #           "packed into tuple or list: {}".format(e))
    #     raise
    # except TypeError as e:
    #     print("\"matrices\" has to have M_ice,M_raw,E_ice",
    #           "packed into tuple or list: {}".format(e))
    #     raise
    # else:
    #     assert isinstance(M_ice, np.ndarray)
    #     assert isinstance(M_raw, np.ndarray)
    #     assert isinstance(E_ice, np.ndarray)
    #     print("\"matrices\" unpacked succesfully")


    # DEAL WITH BIN_SIZE AND MATRIX DIMENSHIONS ...
    # bin size has to be extracted:
    # later it will be replaced with
    # the sliding window-style processing,
    # maybe using dask ...
    # DECIDE HOW DO WE GET EXPECTED - 
    # matrix VS vector ?! ...

    # let's extract full matrices and ice_vector:
    M_ice, M_raw, E_ice = matrices
    v_ice, = vectors
    # we'll allow the DONUT-mask, i.e. kernel(s) to
    # be provided as a parameter:
    kernel, = kernels
    # assume there will be a bunch of kernels to use
    # at some point:
    # but use just one for now (DONUT):

    # Now we have all the ingredients ready 
    # and we can go ahead and try to execute 
    # actuall loop-calling algorithm
    # as explained in Rao et al 2014:

    # v_deice = 1/v_ice
    v_deice = np.reciprocal(v_ice)
    # deiced E_ice: element-wise multiplication of E_ice[i,j] and
    # v_deice[i]*v_deice[j]:
    E_raw = np.multiply(E_ice, np.outer(v_deice,v_deice))

    # now let's calculate DONUTS ...
    # to be replaced later by:
    #
    # for kernel in kernels:
    #     call_peaks()
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
                  np.ones_like(kernel),
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

    # M_raw and Ed_raw can be compared element-wise now

    ##################
    # Enhancement:
    # no need to perform Poisson tests
    # for all pixels ...
    # so, maybe we'd filter out the data first
    # and do Poisson after ?!
    ##################
    #
    # moreover take into account that
    # Poisson CDF will use int values anyways ...
    #
    ####################
    print("Poisson testing to be done for all pixels ...")
    # element-wise Poisson test:
    p_cdf = poisson.cdf( M_raw.flatten(),
                        Ed_raw.flatten())
    # keep it as CDF and reshape back:
    # preserving correspondnce between pixels in 
    # M_raw/ice and 
    p_cdf = p_cdf.reshape(M_raw.shape)
    print("Poisson testing is complete ...")

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
    # with too many NaNs around:
    mask_NN = (NN<kernel.size)
    ####
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
    # 2D mask unrolled:
    # masking everyhting outside 2Mb
    mask_2Mb = toeplitz(mask_2Mb)

    # combine all filters/masks:
    # mask_Ed || mask_M || mask_NN || mask_bnd || mask_2Mb ...
    mask_ndx = np.any((mask_Ed,
                       mask_M,
                       mask_NN,
                       mask_bnd,
                       mask_2Mb))

    # any nonzero element in `mask_ndx` 
    # must be masked out from `p_cdf`:
    # so, we'll just take the negated
    # elements of the combined mask:
    i_ndx, j_ndx = np.nonzero(~mask_ndx)
    # take the upper triangle only:
    up_ndx = (i_ndx < j_ndx)
    i_ndx = i_ndx[up_ndx]
    j_ndx = j_ndx[up_ndx]
    # now melt `p_cdf` into tidy-data df:
    peaks_df = pd.DataFrame({"row": i_ndx,
                             "col": j_ndx,
                             "CDF": p_cdf[i_ndx,j_ndx],
                             "NN_around": NN[i_ndx,j_ndx],
                            })
    #
    peaks_df['pval'] = 1.0 - peaks_df['CDF']
    # count finite values in a vicinity of a pixel ...
    # to see if "NN<kernel.size"-mask worked:
    get_finite_vicinity = lambda ser: np.isfinite(
                                          np.log2(M_ice)[get_slice(
                                                            ser['row'],
                                                            ser['col']
                                              )] ).sum()
    # calculate that vicinity enrichment:
    peaks_df['fin'] = peaks_df[['row','col']].apply(get_finite_vicinity,
                                                    axis=1)



    print("Final filtering of CDF/pvalue data is complete")

    ################
    # Determine p-value threshold here:
    ################

    ## BH FDR procedure:
    # http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    #
    # reject null-hypothesis only for p-values that are $<\alpha*\frac{i}{N}$,
    # where $\alpha=0.1$ is a typical FDR, $N$ - is a total number of observations,
    # and $i$ is an index of a p-values sorted by ascending.
    pvals       = peaks_df['pval'].as_matrix()
    sortind     = np.argsort(pvals)
    pvals       = np.take(pvals, sortind)
    alpha,n_obs = 0.1, pvals.size
    ecdffactor  = np.arange(1,n_obs+1)/float(n_obs)
    # for which observations to reject null-hypothesis ...
    reject_null = pvals <= alpha*ecdffactor

    pval_max_reject = pvals[reject_null].max()
    pval_min_null   = pvals[~reject_null].min()

    if reject_null.any():
        print("Some significant peaks have been detected!\n"
            "pval border is between {:.4f} and {:.4f}".format(
                                                pval_max_reject,
                                                pval_min_null))
    # now we have to create ndarray reject_
    # that stores rej_null values in the order of
    # original pvals array ... 
    reject_ = np.empty_like(reject_null)
    reject_[sortind] = reject_null
    peaks_df['rej_null'] = reject_


    # # some reasonble sorting:
    # ptest_df.sort_values(by=['pval','fin'],
    #                      ascending=[True,False])[['row',
    #                                               'col',
    #                                               'pval',
    #                                               'fin']]

    # ptest_df_to_clust ...

    return peaks_df

#     ###############
#     # clustering starts here:
#     # http://scikit-learn.org/stable/modules/clustering.html
#     # picked Birch, as the most appropriate here:
#     ###############

#     threshold_based_on_bin_size = 2
#     brc = Birch(n_clusters=None,threshold=threshold_based_on_bin_size)
#     clust_info = brc.fit_predict(ptest_df_to_clust)


# ##################
# # TO BE CONTINUED ....

# mmmm =brc.fit_transform(df_to_clust)

# rrrr = brc.fit(df_to_clust)

# rrrr.subcluster_centers_.astype(np.int)


# df_to_clust['clust'] = ccccc

# sorted_clusters = df_to_clust.sort_values(by=['clust',],ascending=[True,])






































def call_dots_matrix(cooler, chrom, expected, kernels, clustering_threshold=40000):
    '''Description
    
    Parameters
    ----------
    cooler : cooler-file
    chrom : str
        chromosome name to perform loop-calling on
    expected : pandas.DataFrame
        data frame with the `balanced.avg` column
        that contains expected Hi-C signal
    kernels : list of numpy.arrays
        list of kernels/masks to perform
        convolution of the heatmap. Kernels
        represent the local environment to 
        find prominent peaks against.
        Peak considered significant is it
        passes stat-tests for all kernels
        from the list. Use just one for now.
    clustering_threshold : int
        thershold for clustering neighboring
        peaks, in kbases. Should be ~40-60kb.
    
    Returns
    -------
    peaks : pandas.DataFrame
        data frame with 2D genomic regions that
        have been called by loop-caller (BEDPE-like).
    
        '''
    # assuming cooler, chrom and expected as input ...
    ice_v_name = "weight"
    exp_v_name = "balanced.avg"

    # make sure everything is compatible:
    # chrom is a legit one:
    assert chrom in cooler.chromnames
    # cooler is balanced:
    assert ice_v_name in cooler.bins().columns

    # let's extract full matrices and ice_vector:
    # it's inefficient from memory consumption
    # perspective, but it's good for
    # small chromosomes @10+ kb
    M_ice = cooler.matrix().fetch(chrom)
    M_raw = cooler.matrix(balance=False).fetch(chrom)
    ice_v = cooler.bins().fetch(chrom)[ice_v_name].as_matrix()

    print("Hi-C matrices extracted ...")

    # bin size has to be extracted:
    bin_size = cooler.info['bin-size']
    M_dim = M_ice.shape[0]

    # later it will be replaced with
    # the sliding window-style processing,
    # maybe using dask ...


    # there is no concrete format for storing expected
    # so we'll use the return of the compute_expected
    # as a reference:
    assert 'chrom'    in expected.index.names
    assert 'diag'     in expected.index.names
    assert exp_v_name in expected.columns
    # extract 1D expected vector:
    exp_v = expected.loc[chrom][exp_v_name].as_matrix()

    # We'll reconstruct a 2D expected matrix from 
    # the `exp_v` for ease of coding for now, 
    # but overall it is another major inefficiency
    # of the code (memory-footprint etc).
    M_exp = toeplitz(exp_v)

    print("Hi-C expected reconstructed ...")

    # Now we have all the ingredients ready 
    # and we can go ahead and try to execute 
    # actuall loop-calling algorithm
    # as explained in Rao et al 2014:


    # it comes down to comparison of pixel intensity
    # in observed: M_ice,
    # and a locally-modified expected:
    #                     DONUT[i,j](M_ice)
    # D[i,j] = M_exp[i,j]*-----------------
    #                     DONUT[i,j](M_exp)
    # where DONUT[i,j](M) is a sum of the pixel intensities
    # in the donut-shaped (or other shaped) vicinity
    # of the pixel of interest (i,j).
    #
    # However, comparison between balanced/scaled values
    # that are 0..1 using Poisson statistics is 
    # intractable, and thus both M_ice and D, has to be
    # rescaled back to raw-count values, i.e. de-iced:
    # de-iced M_ice is the same as M_raw, while de-icing D:
    #                  1       1                   DONUT[i,j](M_ice)
    # D_raw[i,j] = --------*-------- * M_exp[i,j]* -----------------
    #              ice_v[i] ice_v[j]               DONUT[i,j](M_exp)
    #
    #
    # so, at the end of the day, simply compare M_raw and D_raw ...

    # deice_v = 1/ice_v
    deice_v = np.reciprocal(ice_v)

    # deiced M_exp: element-wise multiplication of M_exp[i,j] and
    # deice_v[i]*deice_v[j]:
    deice_M_exp = np.multiply(M_exp,
                            np.outer(deice_v,deice_v))

    # now let's calculate DONUTS ...
    # we'll allow the DONUT-mask, i.e. kernel(s) to
    # be provided as a parameter:
    # assume there will be a bunch of kernels to use
    # at some point:
    assert isinstance(kernels,list)
    # but use just one for now (DONUT):
    kernel, = kernels
    #
    #to be replaced later by:
    #
    # for kernel in kernels:
    #     call_peaks()
    #
    kernel_width = kernel.shape[0]
    # size must be odd: pixel (i,j) of interest in the middle
    # and equal amount of pixels are to left,right, up and down
    # of it: 
    assert kernel_width%2 != 0
    kernel_half_width = int((kernel_width-1)/2)
    #
    #
    # a matrix filled with the donut-sums based on the ICEed matrix
    conv_M_ice = convolve(M_ice,
                          kernel,
                          mode='constant',
                          cval=0.0,
                          origin=0)
    # a matrix filled with the donut-sums based on the expected matrix
    conv_M_exp = convolve(M_exp,
                          kernel,
                          mode='constant',
                          cval=0.0,
                          origin=0)

    print("kernels convolved with observed and expected ...")

    #
    # details like, boundaries, kernel-footprints with too many NaNs, etc, 
    # are to be detailed later ...
    #
    #
    # now finally, deice_M_exp * (conv_M_ice/conv_M_exp), as the 
    # locally-adjusted expected with raw counts as values:
    M_exp_local = np.multiply(deice_M_exp,
                        np.divide(conv_M_ice, conv_M_exp))


    #
    # M_raw and M_exp_local can be compared element-wise now
    #

    print("kernels convolved with observed and expected ...")

    ##################
    # Enhancement:
    # no need to perform Poisson tests
    # for all pixels ...
    # so, maybe we'd filter out the data first
    # and do Poisson after ?!
    ##################

    print("Poisson testing to be done for all pixels ...")
    # element-wise Poisson test:
    p_cdf = poisson.cdf(M_raw.flatten(),
                        M_exp_local.flatten())

    # keep it as CDF and reshape back:
    # preserving correspondnce between pixels in 
    # M_raw/ice and 
    p_cdf = p_cdf.reshape(M_raw.shape)
    print("Poisson testing is complete ...")

    # some trivial and not so trivial masking 
    # is needed: exclude contacts >2Mb apart,
    # pixels from NaNs-rich regions, etc.
    #####################
    # TODO:
    # check if these 2 steps do anyhting at all...
    ######################
    p_cdf[np.isnan(M_exp_local)] = np.nan
    p_cdf[np.isnan(M_ice)] = np.nan

    # masking everyhting further than 2Mb away:
    # index of 2Mb in current bin-sizes:
    idx_2Mb = int(2000000.0/bin_size)
    assert M_dim > idx_2Mb

    mask_2Mb = np.zeros(M_dim,dtype=np.bool)
    mask_2Mb[:idx_2Mb] = True
    # 2D mask unrolled:
    mask_2Mb = toeplitz(mask_2Mb)
    # masking everyhting outside 2Mb
    p_cdf[~mask_2Mb] = np.nan

    #####
    # filter out boundaries:
    #####
    mask_bnd = np.zeros_like(M_ice, dtype=np.bool)
    mask_bnd[kernel_half_width:-kernel_half_width,
             kernel_half_width:-kernel_half_width] = True
    p_cdf[~mask_bnd] = np.nan
    #
    # p_cdf is quite sparse at this point
    #

    # borrowed from:
    # https://stackoverflow.com/questions/14374791/what-is-the-fastest-way-to-initialise-a-scipy-sparse-matrix-with-numpy-nan
    #########################
    # get rid of NaNs in COO ...
    ########################
    nontrivial_idx = np.nonzero(~np.isnan(p_cdf))
    p_cdf_coo = coo_matrix((p_cdf[nontrivial_idx],nontrivial_idx),
                           shape=p_cdf.shape)
    # now to DataFrame:
    p_cdf_df = pd.DataFrame({"row" :p_cdf_coo.row,
                             "col" :p_cdf_coo.col,
                             "CDF" :p_cdf_coo.data})
    print("initial filtering (<2Mb) of CDFs complete ...\
           and a matrix with CDFs is turned into COO df.")
    #####################
    # ACHTUNG!!! ptest <-> 1.0 - ptest
    # we might be using 1-pval at some point
    # maybe keep 1-pval till the very last minute ...
    #####################
    p_cdf_df['pval'] = 1.0 - p_cdf_df['CDF']

    #######
    # TO BE CONTINUED ...

    ###########################
    # Calculate couple more columns for filtering/sorting ...
    ###########################
    get_slice = lambda row,col: (slice(row-kernel_half_width,
                                       row+kernel_half_width+1),
                                 slice(col-kernel_half_width,
                                       col+kernel_half_width+1))
    # get_slice2 = lambda row,col: (slice(col-kernel_half_width, col+kernel_half_width+1),
    #                               slice(row-kernel_half_width, row+kernel_half_width+1))

    # count finite values in a vicinity of a pixel ...
    get_finite_vicinity = lambda ser: np.isfinite(
                                            np.log2( # do we need this log2 here ???
                                                M_ice[
                                                    get_slice(ser['row'],
                                                              ser['col'])
                                                      ]
                                                    )
                                                  ).sum()
    # calculate that vicinity enrichment:
    p_cdf_df['fin'] = p_cdf_df[['row','col']].apply(get_finite_vicinity,
                                                    axis=1)
    # taking only upper triangle of the matrix:
    p_cdf_df = p_cdf_df[p_cdf_df['row'] > p_cdf_df['col']]

    print("Final filtering of CDF/pvalue data is complete")

    ################
    # ACHTUNG: play multiple hyp. correction games here:
    ################

    # some reasonble sorting:
    ptest_df.sort_values(by=['pval','fin'],
                         ascending=[True,False])[['row',
                                                  'col',
                                                  'pval',
                                                  'fin']]

    # ptest_df_to_clust ...

    return ptest_df

#     ###############
#     # clustering starts here:
#     # http://scikit-learn.org/stable/modules/clustering.html
#     # picked Birch, as the most appropriate here:
#     ###############

#     threshold_based_on_bin_size = 2
#     brc = Birch(n_clusters=None,threshold=threshold_based_on_bin_size)
#     clust_info = brc.fit_predict(ptest_df_to_clust)


# ##################
# # TO BE CONTINUED ....

# mmmm =brc.fit_transform(df_to_clust)

# rrrr = brc.fit(df_to_clust)

# rrrr.subcluster_centers_.astype(np.int)


# df_to_clust['clust'] = ccccc

# sorted_clusters = df_to_clust.sort_values(by=['clust',],ascending=[True,])






















