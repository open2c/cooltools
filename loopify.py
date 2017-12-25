

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
_get_slice = lambda row,col,window: (
                        slice(row-window, row+window+1),
                        slice(col-window, col+window+1))


def _multiple_test_BH(pvals,alpha=0.1):
    # prepare:
    pvals = np.asarray(pvals)
    n_obs = pvals.size
    # sort p-values ...
    sortind     = np.argsort(pvals)
    pvals_sort       = pvals[sortind]
    # the alpha*i/N_obs (empirical CDF):
    ecdffactor  = np.arange(1,n_obs+1)/float(n_obs)
    # for which observations to reject null-hypothesis ...
    reject_null = (pvals_sort <= alpha*ecdffactor)

    # let's extract border-line significant P-value:
    pval_max_reject_null = pvals[ reject_null].max()
    pval_min_accept_null = pvals[~reject_null].min()

    if reject_null.any():
        print("Some significant peaks have been detected!\n"
            "pval border is between {:.4f} and {:.4f}".format(
                                            pval_max_reject_null,
                                            pval_min_accept_null))
    # now we have to create ndarray reject_
    # that stores rej_null values in the order of
    # original pvals array ... 
    reject_ = np.empty_like(reject_null)
    reject_[sortind] = reject_null
    # return the reject_ status list and pval-range:
    return reject_, (pval_max_reject_null, pval_min_accept_null)





def _clust_2D_pixels(pixels,threshold_cluster=2):
    # clustering object prepare:
    brc = Birch(n_clusters=None,threshold=threshold_cluster)
    # cluster selected pixels ...
    brc.fit(pixels)
    brc.predict(pixels)
    # array of labels assigned to each pixel
    # after clustering: brc.labels_
    # array of (tuples?) with X,Y coordinates 
    # for centroids of corresponding clusters:
    # brc.subcluster_centers_
    uniq_labels, uniq_counts = np.unique(brc.labels_,
                                        # return_inverse=True,
                                        return_counts=True)

    # small message:
    print("Clustering is completed:"+
          "there are {} clusters detected".format(uniq_counts.size)+
          "mean size {:.3f}+/-{:.3f}".format(uniq_counts.mean(),
                                             uniq_counts.std())+
          "labels and centroids to be reported.")

    # # repeat centroids coordinates
    # # as many times as there are pixels
    # # in each cluster:
    centroids = np.repeat(brc.subcluster_centers_,
                          uniq_counts,
                          axis=0)
    c_sizes   = np.repeat(uniq_counts,
                          uniq_counts)
    # let's return column names as well for convenience:
    column_names = ['c_label','c_size','c_row','c_col']
    #
    return np.column_stack((brc.labels_,c_sizes,centroids)), column_names
    ##########################################################
    # sub_centers = pd.DataFrame(brc.subcluster_centers_,
    #                            columns=['c_col','c_row'])
    ##########################################################
    # # p_cdf_df_signif.merge?
    # peaks_merged = p_cdf_df_signif.merge(sub_centers,
    #                                      how='inner',
    #                                      left_on='clust',
    #                                      right_index=True,
    #                                      sort=True)
    ##########################################################




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


    # DEAL WITH BIN_SIZE AND MATRIX DIMENSIONS ...
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
    # In other words:
    # mask_2Mb = np.where(np.arange(s)<idx_2Mb,False,True)

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
                                          _np.log2(M_ice)[get_slice(
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
    peaks_df['rej_null'], pv_range = _multiple_test_BH(peaks_df['pval'],alpha=0.1)
    # 


    # 
    # Next step is clustering of the data:
    ###############
    # clustering starts here:
    # http://scikit-learn.org/stable/modules/clustering.html
    # picked Birch, as the most appropriate here:
    ###############
    pixels_to_clust = peaks_df[['col','row']][peaks_df['rej_null']]

    cluster_radius = 35000
    threshold_cluster = round(cluster_radius/float(b))
    # cluster em' using the threshold:
    clustered_pix,pix_columns = _clust_2D_pixels(
                                    pixels_to_clust.values,
                                    threshold_cluster=threshold_cluster)
    # pack into DataFrame ...
    peaks_clust = pd.DataFrame(
                        clustered_pix,
                        index=pixels_to_clust.index,
                        columns=pix_columns)
    # and merge (index-wise) with the main DataFrame:
    peaks_df=peaks_df.merge(
                        peaks_clust,
                        how='left',
                        left_index=True,
                        right_index=True)


    return peaks_df






















