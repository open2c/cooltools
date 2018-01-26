

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


_cmp_masks = lambda M_superset,M_subset: (0 > M_superset.astype(np.int) -
                                                 M_subset.astype(np.int)).any()


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




def multiple_test_BH(pvals,alpha=0.1):
    '''
    take an array of N p-values, sort then
    in ascending order p1<p2<p3<...<pN,
    and find a threshold p-value, pi
    for which pi < alpha*i/N, and pi+1 is
    already pi+1 >= alpha*(i+1)/N.
    Peaks corresponding to p-values
    p1<p2<...pi are considered significant.

    Parameters
    ----------
    pvals : array-like
        array of p-values to use for
        multiple hypothesis testing
    alpha : float
        rate of false discovery (FDR)

    
    Returns
    -------
    reject_ : numpy.ndarray
        array of type np.bool storing
        status of a pixel: significant (True)
        non-significant (False)
    pvals_threshold: tuple
        pval_max_reject_null, pval_min_accept_null

    Notes
    -----
    - Mostly follows the statsmodels implementation:
    http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    - Using alpha=0.02 it is possible to achieve
    called dots similar to pre-update status
    
    '''
    # 
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
    pval_max_reject_null = pvals_sort[ reject_null].max()
    pval_min_accept_null = pvals_sort[~reject_null].min()

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





def clust_2D_pixels(pixels_df,threshold_cluster=2):
    '''
    Group significant pixels by proximity
    using Birch clustering.

    Parameters
    ----------
    pixels_df : pandas.DataFrame
        a DataFrame of 2 columns, with left-one
        being the column index of a pixel and
        the right-one being row index of a pixel
    threshold_cluster : int
        clustering radius for Birch clustering
        derived from ~40kb radius of clustering
        and bin size.

    
    Returns
    -------
    peak_tmp : pandas.DataFrame
        DataFrame with c_row,c_col,c_label,c_size - 
        columns. row/col are coordinates of centroids,
        label and sizes are unique pixel-cluster labels
        and their corresponding sizes.


    Notes
    -----
    TODO: figure out Birch clustering
    CFNodes etc, check if there might
    be some empty subclusters.
    
    '''
    pixels  = pixels_df.values
    pix_idx = pixels_df.index
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
    uniq_labels, inverse_idx, uniq_counts = np.unique(
                                                brc.labels_,
                                                return_inverse=True,
                                                return_counts=True)
    # cluster sizes taken to match labels:
    clust_sizes = uniq_counts[inverse_idx]
    ####################
    # After discovering a bug ...
    # bug (or misunderstanding, rather):
    # uniq_labels is a subset of brc.subcluster_labels_
    # TODO: dive deeper into Birch ...
    ####################
    # repeat centroids coordinates
    # as many times as there are pixels
    # in each cluster:
    # IN OTHER WORDS (after bug fix):
    # take centroids corresponding to labels:
    centroids = np.take(brc.subcluster_centers_,
                        brc.labels_,
                        axis=0)

    # small message:
    print("Clustering is completed:\n"+
          "there are {} clusters detected\n".format(uniq_counts.size)+
          "mean size {:.6f}+/-{:.6f}\n".format(uniq_counts.mean(),
                                             uniq_counts.std())+
          "labels and centroids to be reported.")

    # let's create output DataFrame
    peak_tmp = pd.DataFrame(
                        centroids,
                        index=pix_idx,
                        columns=['c_row','c_col'])
    # add labels:
    peak_tmp['c_label'] = brc.labels_.astype(np.int)
    # add cluster sizes:
    peak_tmp['c_size'] = clust_sizes.astype(np.int)
    

    return peak_tmp



############################################################
# 
# we need to make this work for slices
# of the intra-chromosomal Hi-C heatmaps
# 
############################################################
def generate_intra_chrom_chunks(matrix, slice_size):
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
    
    # 
    # c.matrix(balance=False)[1000:1005, 1000:1005]
    # 
    raise NotImplementedError("To be implemented")


########################################################################
# this should be a main function to get locally adjusted expected
########################################################################
def get_adjusted_expected_tile_some_nans(origin,
                                         observed,
                                         expected,
                                         ice_weight,
                                         kernels,
                                         b,
                                         band=2e+6,
                                         nan_threshold=2):
    """
    get_adjusted_expected_tile_some_nans
    combines get_adjusted_expected_tile and
    get_adjusted_expected_some_nans ...

    """
    # extract origin coordinate of this tile:
    io, jo = origin
    # let's extract full matrices and ice_vector:
    M_ice = np.copy(observed)
    E_ice = np.copy(expected)
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
                  # we have to use kernel
                  # all made of 1s, since
                  # even 0 * nan == nan
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
    # mask_Ed = np.isfinite(Ed_raw) ?
    ######################################
    mask_Ed = ~np.isfinite(Ed_raw)
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

    # ########################
    # Sparsify Ed_raw using 
    # derived masks ...
    # ########################
    # combine all filters/masks:
    # mask_Ed || mask_NN
    mask_ndx = np.logical_or(mask_Ed,
                             mask_NN)
    # any nonzero element in `mask_ndx` 
    # must be masked out from `pvals`:
    # so, we'll just take the negated
    # elements of the combined mask:
    i, j = np.nonzero(~mask_ndx)
    # take the upper triangle with 2Mb close to diag:
    upper_band = np.logical_and((i<j),
                                (i>j-band_idx))
    # peek the reduced list of pixels
    i = i[upper_band]
    j = j[upper_band]
    # pack it into DataFrame:
    peaks_df = pd.DataFrame({"row": i+io,
                             "col": j+jo,
                             "expected": Ed_raw[i,j],
                             "observed": observed[i,j],
                            })
    # return sparsified DF:
    return peaks_df






# to be deprecated or heavily modified
# since we are switching to sparsified 
# output from get_adjusted_expected - sort
# of functions ...
def compare_observed_expected(observed, expected, mask):
    '''
    simply takes obs and some exp, 
    compares them using Poisson test,
    generating p-values for each pair,
    and then turns everyhting into a
    DataFrame after some masking.
             
    
    Parameters
    ----------
    observed : numpy.ndarray
        square symmetrical dense-matrix
        that contains raw observed heatmap
    expected : numpy.ndarray
        square symmetrical dense-matrix
        that contains deiced expected
        of some sort.
    mask : numpy.ndarray
        square symmetrical dense-matrix
        of type np.bool. Each True element
        will be masked out of the downstream
        analysis and the resulting DataFrame.

    Returns
    -------
    peaks : pandas.DataFrame
        DataFrame with the pixels that
        remained after masking out with their
        coordinates, obs, exp andp-values
        provided. Columns are:
        row, col, pval, expected, observed
    '''
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
    # element-wise Poisson test (Poisson.cdf(obs,exp)):
    # 1.0-CDF = p-value
    pvals = 1.0 - poisson.cdf( observed, expected)
    # all set ...
    # maybe add global pval here as well
    # we'd need that at least for lab meeting.
    print("Poisson testing is complete ...")

    # any nonzero element in `mask_ndx` 
    # must be masked out from `pvals`:
    # so, we'll just take the negated
    # elements of the combined mask:
    i, j = np.nonzero(~mask)
    # take the upper triangle only:
    upper = i<j
    i = i[upper]
    j = j[upper]
    # now melt `pvals` into tidy-data df:
    peaks_df = pd.DataFrame({"row": i,
                             "col": j,
                             "pval": pvals[i,j],
                             "expected": expected[i,j],
                             "observed": observed[i,j],
                            })

    print("Final filtering of CDF/pvalue data is complete")
    #
    #
    return peaks_df




# to be modified due to API changed ...
def call_dots_matrix(matrices, vectors, kernels, b, alpha, clust_radius):
    '''
    description
    
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
    alpha : float
        rate of false discovery (FDR)
    clust_radius : int
        approximate size of a loop (dot)
        irrespective of the resolution
        should be around 20-40 kb.
    
    Returns
    -------
    peaks : pandas.DataFrame
        data frame with 2D genomic regions that
        have been called by loop-caller (BEDPE-like).
    
    '''

    M_ice, M_raw, E_ice = matrices
    v_ice, = vectors
    # GLOBAL expected:
    # deiced E_ice: element-wise division of E_ice[i,j] and
    # v_ice[i]*v_ice[j]:
    E_raw = np.multiply(E_ice, np.outer(v_deice,v_deice))

    # first, generate that locally-adjusted expected:
    Ed_raw, mask_ndx, NN = get_adjusted_expected(observed=M_ice,
                                                 expected=E_ice,
                                                 ice_weight=v_ice,
                                                 kernels=kernels,
                                                 b=b)


    peaks_local = compare_observed_expected(observed=M_raw,
                                            expected=Ed_raw,
                                            mask=mask_ndx)

    peaks_global = compare_observed_expected(observed=M_raw,
                                             expected=E_raw,
                                             mask=mask_ndx)

    # merge local and global calculations 
    peaks = pd.merge(
        left=peaks_local,
        right=peaks_global,
        how='outer', # should be exactly the same: mask_ndx
        on=['row','col'],
        suffixes=('_local', '_global'),
        validate="one_to_one"
        )

    # add information about # of NaNs surrounding each pixel:
    peaks['nans_around'] = NN[peaks['row'],peaks['col']]

    ###############################
    #
    # following analysis is based on 
    # locally-adjusted p-values only ...
    #
    ################################

    ################
    # Determine p-value threshold here:
    ################

    ## BH FDR procedure:
    # http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    #
    # reject null-hypothesis only for p-values that are $<\alpha*\frac{i}{N}$,
    # where $\alpha=0.1$ is a typical FDR, $N$ - is a total number of observations,
    # and $i$ is an index of a p-values sorted by ascending.
    peaks['rej_null'], pv_range = multiple_test_BH(
                                                peaks['pval_local'],
                                                alpha=alpha)


    # 
    # Next step is clustering of the data:
    ###############
    # clustering starts here:
    # http://scikit-learn.org/stable/modules/clustering.html
    # picked Birch, as the most appropriate here:
    ###############
    pixels_to_clust = peaks[['col','row']][peaks['rej_null']]

    threshold_cluster = round(clust_radius/float(b))
    # cluster em' using the threshold:
    peaks_clust = clust_2D_pixels(
                        pixels_to_clust,
                        threshold_cluster=threshold_cluster)
    # and merge (index-wise) with the main DataFrame:
    return peaks.merge(
                    peaks_clust,
                    how='left',
                    left_index=True,
                    right_index=True)



##############
# let it be
##############

def main():
    pass

if __name__ == '__main__':
    main()

