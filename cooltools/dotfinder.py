"""
Collection of functions needed for dot-calling

"""
from scipy.linalg import toeplitz
from scipy.ndimage import convolve
from scipy.stats import poisson
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from sklearn.cluster import Birch


def get_qvals(pvals):
    '''
    B&H FDR control procedure: sort a given array of N p-values, determine their
    rank i and then for each p-value compute a corres- ponding q-value, which is
    a minimum FDR at which that given p-value would pass a BH-FDR test.

    Parameters
    ----------
    pvals : array-like
        array of p-values to use for multiple hypothesis testing
    
    Returns
    -------
    qvals : numpy.ndarray
        array of q-values

    Notes
    -----
    - BH-FDR reminder: given an array of N p-values, sort it in ascending order 
    p[1]<p[2]<p[3]<...<p[N], and find a threshold p-value, p[j] for which 
    p[j] < FDR*j/N, and p[j+1] is already p[j+1] >= FDR*(j+1)/N. Peaks 
    corresponding to p-values p[1]<p[2]<...p[j] are considered significant.

    - Mostly follows the statsmodels implementation:
    http://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html
    
    - Using alpha=0.02 it is possible to achieve called dots similar to 
    pre-update status alpha is meant to be a q-value threshold: "qvals <= alpha"
    
    '''
    # NOTE: This is tested and is capable of reproducing 
    # `multiple_test_BH` results
    pvals = np.asarray(pvals)
    n_obs = pvals.size
    # determine rank of p-values (1-indexed):
    porder = np.argsort(pvals)
    prank  = np.empty_like(porder)
    prank[porder] = np.arange(1, n_obs+1)
    # q-value = p-value*N_obs/i(rank of a p-value) ...
    qvals = np.true_divide(n_obs*pvals, prank)
    # return the qvals sorted as the initial pvals:
    return qvals


def clust_2D_pixels(pixels_df,
                    threshold_cluster=2,
                    bin1_id_name='bin1_id',
                    bin2_id_name='bin2_id',
                    verbose=True):
    '''
    Group significant pixels by proximity using Birch clustering. We use
    "n_clusters=None", which implies no AgglomerativeClustering, and thus 
    simply reporting "blobs" of pixels of radii <="threshold_cluster" along 
    with corresponding blob-centroids as well.

    Parameters
    ----------
    pixels_df : pandas.DataFrame
        a DataFrame with pixel coordinates that must have at least 2 columns
        named 'bin1_id' and 'bin2_id', where first is pixels's row and the
        second is pixel's column index.
    threshold_cluster : int
        clustering radius for Birch clustering derived from ~40kb radius of
        clustering and bin size.
    bin1_id_name : str
        Name of the 1st coordinate (row index) in 'pixel_df', by default
        'bin1_id'. 'start1/end1' could be usefull as well.
    bin2_id_name : str
        Name of the 2nd coordinate (column index) in 'pixel_df', by default
        'bin2_id'. 'start2/end2' could be usefull as well.
    verbose : bool
        Print verbose clustering summary report defaults is True.
    
    Returns
    -------
    peak_tmp : pandas.DataFrame
        DataFrame with c_row,c_col,c_label,c_size - columns. row/col are
        coordinates of centroids, label and sizes are unique pixel-cluster
        labels and their corresponding sizes.

    '''

    # col (bin2) must precede row (bin1):
    pixels     = pixels_df[[bin1_id_name, bin2_id_name]].values
    pixel_idxs = pixels_df.index

    # perform BIRCH clustering of pixels:
    # "n_clusters=None" implies using BIRCH without AgglomerativeClustering, 
    # thus simply reporting "blobs" of pixels of radius "threshold_cluster" 
    # along with blob-centroids as well:
    brc = Birch(n_clusters=None,
                threshold=threshold_cluster,
                # branching_factor=50, (it's default)
                compute_labels=True)
    brc.fit(pixels)
    # # following is redundant,
    # # as it's done here:
    # # https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/cluster/birch.py#L638
    # clustered_labels = brc.predict(pixels)

    # labels of nearest centroid, assigned to each pixel,
    # BEWARE: labels might not be continuous, i.e.,
    # "np.unique(clustered_labels)" isn't same as "brc.subcluster_labels_", because:
    # https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/cluster/birch.py#L576
    clustered_labels    = brc.labels_
    # centroid coordinates ( <= len(clustered_labels)):
    clustered_centroids = brc.subcluster_centers_
    # count unique labels and get their continuous indices
    uniq_labels, inverse_idx, uniq_counts = np.unique(
                                                clustered_labels,
                                                return_inverse=True,
                                                return_counts=True)
    # cluster sizes taken to match labels:
    cluster_sizes       = uniq_counts[inverse_idx]
    # take centroids corresponding to labels (as many as needed):
    centroids_per_pixel = np.take(clustered_centroids,
                                  clustered_labels,
                                  axis=0)

    if verbose:
        # prepare clustering summary report:
        msg = "Clustering is completed:\n" + \
              "{} clusters detected\n".format(uniq_counts.size) + \
              "{:.2f}+/-{:.2f} mean size\n".format(uniq_counts.mean(),
                                                 uniq_counts.std())
        print(msg)

    # create output DataFrame
    centroids_n_labels_df = pd.DataFrame(
                                centroids_per_pixel,
                                index=pixel_idxs,
                                columns=['c'+bin1_id_name,'c'+bin2_id_name])
    # add labels per pixel:
    centroids_n_labels_df['c_label'] = clustered_labels.astype(np.int)
    # add cluster sizes:
    centroids_n_labels_df['c_size'] = cluster_sizes.astype(np.int)
    

    return centroids_n_labels_df


def diagonal_matrix_tiling(start, stop, bandwidth, edge=0, verbose=False):
    """
    Generate a stream of tiling coordinates that guarantee to cover a diagonal
    area of the matrix of size 'bandwidth'. cis-signal only!

    Parameters
    ----------
    start : int
        Starting position of the matrix slice to be tiled (inclusive, bins,
        0-based).
    stop : int
        End position of the matrix slice to be tiled (exclusive, bins, 0-based).
    edge : int
        Small edge around each tile to be included in the yielded coordinates.
    bandwidth : int
        Diagonal tiling is guaranteed to cover a diagonal are of width 
        'bandwidth'.

    Yields
    ------
    Pairs of indices for every chunk use those indices [cstart:cstop) to fetch 
    chunks from the cooler-object:
    >>> clr.matrix()[cstart:cstop, cstart:cstop]

    Notes
    -----
    Specify a [start,stop) region of a matrix you want to tile diagonally.
    Each slice is characterized by the coordinate of the top-left corner and
    size.

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
    
    These slices are supposed to cover up a diagonal band of size 'bandwidth'. 

    """
    size = stop - start
    tiles = size // bandwidth + bool(size % bandwidth)
    
    # matrix parameters before chunking:
    if verbose:
        print("matrix of size {}X{} to be split so that\n".format(size,size)+
         "  diagonal region of size {} would be completely\n".format(bandwidth)+
         "  covered by the tiling, additionally keeping\n"+
         "  a small 'edge' of size w={}, to allow for\n".format(edge)+
         "  meaningfull convolution around boundaries.\n"+
         "  Resulting number of tiles is {}".format(tiles-1)+
         "  Non-edge case size of each tile is {}X{}".format(2*(bandwidth+edge), 
                                                             2*(bandwidth+edge)))

    # actual number of tiles is tiles-1
    # by doing range(1, tiles) we are making sure we are processing the 
    # upper-left chunk only once:
    for t in range(1, tiles):
        # l = max(0,M*t-M)
        # r = min(L,M*t+M)
        lw = max(0    , bandwidth*(t-1) - edge)
        rw = min(size , bandwidth*(t+1) + edge)
        # don't forget about the 'start' origin:
        yield lw+start, rw+start


def square_matrix_tiling(start, stop, step, edge, square=False, verbose=False):
    """
    Generate a stream of tiling coordinates that guarantee to cover an entire 
    matrix. cis-signal only!

    Parameters
    ----------
    start : int
        Starting position of the matrix slice to be tiled (inclusive, bins,
        0-based).
    stop : int
        End position of the matrix slice to be tiled (exclusive, bins, 0-based).
    step : int
        Requested size of the tiles. Boundary tiles may or may not be of 
        'step', see 'square'.
    edge : int
        Small edge around each tile to be included in the yielded coordinates.
    square : bool, optional

    Yields
    ------
    Pairs of indices for every chunk use those indices [cstart:cstop) to fetch 
    chunks from the cooler-object:
    >>> clr.matrix()[cstart:cstop, cstart:cstop]

    Notes
    -----
    Each slice is characterized by the coordinate
    of the top-left corner and size.

    * * * * * * * * * * * * *
    *       *       *       *
    *       *       *  ...  *
    *       *       *       *
    * * * * * * * * *       *
    *       *               *
    *       *    ...        *
    *       *               *
    * * * * *               *
    *                       *
    *                       *
    *                       *
    * * * * * * * * * * * * *

    Square parameter determines behavior
    of the tiling function, when the
    size of the matrix is not an exact
    multiple of the 'step':

    square = False
    * * * * * * * * * * *
    *       *       *   *
    *       *       *   *
    *       *       *   *
    * * * * * * * * * * *
    *       *       *   *
    *       *       *   *
    *       *       *   *
    * * * * * * * * * * *
    *       *       *   *
    *       *       *   *
    * * * * * * * * * * *
    WARNING: be carefull with extracting
    expected in this case, as it is 
    not trivial at all !!!

    square = True
    * * * * * * * * * * *
    *       *   *   *   *
    *       *   *   *   *
    *       *   *   *   *
    * * * * * * * * * * *
    *       *   *   *   *
    *       *   *   *   *
    * * * * * * * * * * *
    * * * * * * * * * * *
    *       *   *   *   *
    *       *   *   *   *
    * * * * * * * * * * *

    """
    size = stop - start
    tiles = size // step + bool(size % step)

    if verbose:
        print("matrix of size {}X{} to be splitted\n".format(size,size)+
         "  into square tiles of size {}.\n".format(step)+
         "  A small 'edge' of size w={} is added, to allow for\n".format(edge)+
         "  meaningfull convolution around boundaries.\n"+
         "  Resulting number of tiles is {}".format(tiles*tiles))

    for tx in range(tiles):
        for ty in range(tiles):

            lwx = max(0,    step*tx - edge)
            rwx = min(size, step*(tx+1) + edge)
            if square and (rwx >= size):
                lwx = size - step - edge

            lwy = max(0,    step*ty - edge)
            rwy = min(size, step*(ty+1) + edge)
            if square and (rwy >= size):
                lwy = size - step - edge

            yield (lwx+start,rwx+start), (lwy+start,rwy+start)


def _convolve_and_count_nans(O_bal,E_bal,E_raw,N_bal,kernel):
    """
    Dense versions of a bunch of matrices needed for convolution and 
    calculation of number of NaNs in a vicinity of each pixel. And a kernel to 
    be provided of course.
    
    """
    # a matrix filled with the kernel-weighted sums
    # based on a balanced observed matrix:
    KO = convolve(O_bal,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # a matrix filled with the kernel-weighted sums
    # based on a balanced expected matrix:
    KE = convolve(E_bal,
                  kernel,
                  mode='constant',
                  cval=0.0,
                  origin=0)
    # get number of NaNs in a vicinity of every
    # pixel (kernel's nonzero footprint)
    # based on the NaN-matrix N_bal.
    # N_bal is shared NaNs between O_bal E_bal,
    # is it redundant ? 
    NN = convolve(N_bal.astype(np.int),
                  # we have to use kernel's
                  # nonzero footprint:
                  (kernel != 0).astype(np.int),
                  mode='constant',
                  # there are only NaNs 
                  # beyond the boundary:
                  cval=1,
                  origin=0)
    ######################################
    # using cval=0 for actual data and
    # cval=1 for NaNs matrix reduces 
    # "boundary issue" to the "number of
    # NaNs"-issue
    # ####################################

    # now finally, E_raw*(KO/KE), as the 
    # locally-adjusted expected with raw counts as values:
    Ek_raw = np.multiply(E_raw, np.divide(KO, KE))
    # return locally adjusted expected and number of NaNs
    # in the form of dense matrices:
    return Ek_raw, NN


########################################################################
# this should be a MAIN function to get locally adjusted expected
# Die Hauptfunktion
########################################################################
def get_adjusted_expected_tile_some_nans(origin,
                                         observed,
                                         expected,
                                         bal_weights,
                                         kernels,
                                         balance_factor=None,
                                         verbose=False):
    """
    'get_adjusted_expected_tile_some_nans', get locally adjusted
    expected for a collection of local-filters (kernels).

    Such locally adjusted expected, 'Ek' for a given kernel,
    can serve as a baseline for deciding whether a given
    pixel is enriched enough to call it a feature (dot-loop,
    flare, etc.) in a downstream analysis.

    For every pixel of interest [i,j], locally adjusted
    expected is a product of a global expected in that
    pixel E_bal[i,j] and an enrichment of local environ-
    ment of the pixel, described with a given kernel:
                              KERNEL[i,j](O_bal)
    Ek_bal[i,j] = E_bal[i,j]* ------------------
                              KERNEL[i,j](E_bal)
    where KERNEL[i,j](X) is a result of convolution
    between the kernel and a slice of matrix X centered
    around (i,j). See link below for details:
    https://en.wikipedia.org/wiki/Kernel_(image_processing)

    Returned values for observed and all expecteds
    are rescaled back to raw-counts, for the sake of
    downstream statistical analysis, which is using
    Poisson test to decide is a given pixel is enriched.
    (comparison between balanced values using Poisson-
    test is intractable):
                              KERNEL[i,j](O_bal)
    Ek_raw[i,j] = E_raw[i,j]* ------------------ ,
                              KERNEL[i,j](E_bal)
    where E_raw[i,j] is:
          1               1                 
    -------------- * -------------- * E_bal[i,j]
    bal_weights[i]   bal_weights[j]             
    

    Parameters
    ----------
    origin : (int,int) tuple
        tuple of interegers that specify the
        location of an observed matrix slice.
        Measured in bins, not in nucleotides.
    observed : numpy.ndarray
        square symmetrical dense-matrix
        that contains balanced observed O_bal
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        should we switch to RAW here ?
        it would be easy for the output ....
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    expected : numpy.ndarray
        square symmetrical dense-matrix
        that contains expected, calculated
        based on balanced observed: E_bal.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        should we switch to its 1D representation here ?
        good for now, but expected might change later ...
        Tanay's expected, expected with modeled TAD's etc ...
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    bal_weights : numpy.ndarray or (numpy.ndarray, numpy.ndarray)
        1D vector used to turn raw observed
        into balanced observed for a slice of
        a matrix with the origin on the diagonal;
        and a tuple/list of a couple of 1D arrays
        in case it is a slice with an arbitrary 
        origin.
    kernels : dict of (str, numpy.ndarray)
        dictionary of kernels/masks to perform
        convolution of the heatmap. Kernels
        describe the local environment, and
        used to estimate baseline for finding
        enriched/prominent peaks.
        Peak must be enriched with respect to
        all local environments (all kernels),
        to be considered significant.
        Dictionay keys must contain names for
        each kernel.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Beware!: kernels are flipped and 
        only then multiplied to matrix by
        scipy.ndimage.convolve 
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    balance_factor: float
        Multiplicative Balancing factor:
        balanced matrix multiplied by this factor
        sums up to the total number of reads (taking
        symmetry into account) instead of number of
        bins in matrix. defaults to None and skips
        a lowleft KernelObserved factor calculation.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        We need this to test how dynamic donut size
        is affecting peak calling results.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    verbose: bool
        Set to True to print some progress
        messages to stdout.
    
    Returns
    -------
    peaks_df : pandas.DataFrame
        sparsified DataFrame that stores results of
        locally adjusted calculations for every kernel
        for a given slice of input matrix. Multiple
        instences of such 'peaks_df' can be concatena-
        ted and deduplicated for the downstream analysis.
        Reported columns: 
        bin1_id - bin1_id index (row), adjusted to origin
        bin2_id - bin bin2_id index, adjusted to origin
        la_exp - locally adjusted expected (for each kernel)
        la_nan - number of NaNs around (each kernel's footprint)
        exp.raw - global expected, rescaled to raw-counts
        obs.raw - observed values in raw-counts.

    """
    # extract origin coordinate of this tile:
    io, jo = origin
    # let's extract full matrices and ice_vector:
    O_raw = observed # raw observed, no need to copy, no modifications.
    E_bal = np.copy(expected)
    # 'bal_weights': ndarray or a couple of those ...
    if isinstance(bal_weights, np.ndarray):
        v_bal_i = bal_weights
        v_bal_j = bal_weights
    elif isinstance(bal_weights, (tuple,list)):
        v_bal_i,v_bal_j = bal_weights
    else:
        raise ValueError("'bal_weights' must be an numpy.ndarray"
                    "for slices of a matrix with diagonal-origin or"
                    "a tuple/list of a couple of numpy.ndarray-s"
                    "for a slice of matrix with an arbitrary origin.")
    # kernels must be a dict with kernel-names as keys
    # and kernel ndarrays as values.
    if not isinstance(kernels, dict):
        raise ValueError("'kernels' must be a dictionary"
                    "with name-keys and ndarrays-values.")

    # balanced observed, from raw-observed
    # by element-wise multiply:
    O_bal = np.multiply(O_raw, np.outer(v_bal_i,v_bal_j))
    # O_bal is separate from O_raw memory-wise.

    # raw E_bal: element-wise division of E_bal[i,j] and
    # v_bal[i]*v_bal[j]:
    E_raw = np.divide(E_bal, np.outer(v_bal_i,v_bal_j))

    # let's calculate a matrix of common NaNs
    # shared between observed and expected:
    # check if it's redundant ? (is NaNs from O_bal sufficient? )
    N_bal = np.logical_or(np.isnan(O_bal),
                          np.isnan(E_bal))
    # fill in common nan-s with zeroes, preventing
    # NaNs during convolution part of '_convolve_and_count_nans': 
    O_bal[N_bal] = 0.0
    E_bal[N_bal] = 0.0
    # think about usinf copyto and where functions later:
    # https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another
    # 
    # 
    # we are going to accumulate all the results
    # into a DataFrame, keeping NaNs, and other
    # unfiltered results (even the lower triangle for now):
    i,j = np.indices(O_raw.shape)
    # pack it into DataFrame to accumulate results:
    peaks_df = pd.DataFrame({"bin1_id": i.flatten()+io,
                             "bin2_id": j.flatten()+jo})


    for kernel_name, kernel in kernels.items():
        ###############################
        # kernel-specific calculations:
        ###############################
        # kernel paramters such as width etc
        # are taken into account implicitly ...
        ########################################
        #####################
        # unroll _convolve_and_count_nans function back
        # for us to test the dynamic donut criteria ...
        #####################
        # Ek_raw, NN = _convolve_and_count_nans(O_bal,
        #                                     E_bal,
        #                                     E_raw,
        #                                     N_bal,
        #                                     kernel)
        # Dense versions of a bunch of matrices needed for convolution and 
        # calculation of number of NaNs in a vicinity of each pixel. And a kernel to 
        # be provided of course.
        # a matrix filled with the kernel-weighted sums
        # based on a balanced observed matrix:
        KO = convolve(O_bal,
                      kernel,
                      mode='constant',
                      cval=0.0,
                      origin=0)
        # a matrix filled with the kernel-weighted sums
        # based on a balanced expected matrix:
        KE = convolve(E_bal,
                      kernel,
                      mode='constant',
                      cval=0.0,
                      origin=0)
        # get number of NaNs in a vicinity of every
        # pixel (kernel's nonzero footprint)
        # based on the NaN-matrix N_bal.
        # N_bal is shared NaNs between O_bal E_bal,
        NN = convolve(N_bal.astype(np.int),
                      # we have to use kernel's
                      # nonzero footprint:
                      (kernel != 0).astype(np.int),
                      mode='constant',
                      # there are only NaNs 
                      # beyond the boundary:
                      cval=1,
                      origin=0)
        ######################################
        # using cval=0 for actual data and
        # cval=1 for NaNs matrix reduces 
        # "boundary issue" to the "number of
        # NaNs"-issue
        # ####################################
        # now finally, E_raw*(KO/KE), as the 
        # locally-adjusted expected with raw counts as values:
        Ek_raw = np.multiply(E_raw, np.divide(KO, KE))

        # this is the place where we would need to extract
        # some results of convolution and multuplt it by the
        # appropriate factor "cooler._load_attrs(‘bins/weight’)[‘scale’]" ...
        if balance_factor and (kernel_name == "lowleft"):
            peaks_df["factor_balance."+kernel_name+".KerObs"] = KO * balance_factor        
            # KO*balance_factor: to be compared with 16 ...
        if verbose:
            print("Convolution with kernel {} is complete.".format(kernel_name))
        #
        # accumulation into single DataFrame:
        # store locally adjusted expected for each kernel
        # and number of NaNs in the footprint of each kernel
        peaks_df["la_exp."+kernel_name+".value"] = Ek_raw.flatten()
        peaks_df["la_exp."+kernel_name+".nnans"] = NN.flatten()
        # do all the filter/logic/masking etc on the complete DataFrame ...
    #####################################
    # downstream stuff is supposed to be
    # aggregated over all kernels ...
    #####################################
    peaks_df["exp.raw"] = E_raw.flatten()
    peaks_df["obs.raw"] = O_raw.flatten()

    # TO BE REFACTORED/deprecated ...
    # compatibility with legacy API is completely BROKEN
    # post-processing allows us to restore it, see tests,
    # but we pay with the processing speed for it.
    mask_ndx = pd.Series(0, index=peaks_df.index, dtype=np.bool)
    for kernel_name, kernel in kernels.items():
        # accummulating with a vector full of 'False':
        mask_ndx_kernel = ~np.isfinite(peaks_df["la_exp."+kernel_name+".value"])
        mask_ndx = np.logical_or(mask_ndx_kernel,mask_ndx)

    # returning only pixels from upper triangle of a matrix
    # is likely here to stay:
    upper_band = (peaks_df["bin1_id"] < peaks_df["bin2_id"])
    # selecting pixels in relation to diagonal - too far, too
    # close etc, is now shifted to the outside of this function
    # a way to simplify code.

    # return good semi-sparsified DF:
    return peaks_df[~mask_ndx & upper_band].reset_index(drop=True)


def heatmap_tiles_generator_diag(clr, chroms, pad_size, tile_size, band_to_cover):
    """
    A generator yielding heatmap tiles that are needed to cover the requested 
    band_to_cover around diagonal. Each tile is "padded" with pad_size edge to 
    allow proper kernel-convolution of pixels close to boundary.
    
    Parameters
    ----------
    clr : cooler
        Cooler object to use to extract chromosome extents.
    chroms : iterable
        Iterable of chromosomes to process
    pad_size : int
        Size of padding around each tile. Typically the outer size of the 
        kernel.
    tile_size : int
        Size of the heatmap tile.
    band_to_cover : int
        Size of the diagonal band to be covered by the generated tiles. 
        Typically correspond to the max_loci_separation for called dots.
        
    Returns
    -------
    tile : tuple
        Generator of tuples of three, which contain
        chromosome name, row index of the tile,
        column index of the tile (chrom, tilei, tilej).

    """

    for chrom in chroms:
        chr_start, chr_stop = clr.extent(chrom)
        for tilei, tilej in square_matrix_tiling(chr_start,
                                                 chr_stop,
                                                 tile_size,
                                                 pad_size):
            # check if a given tile intersects with 
            # with the diagonal band of interest ...
            diag_from = tilej[0] - tilei[1]
            diag_to   = tilej[1] - tilei[0]
            #
            band_from = 0
            band_to   = band_to_cover
            # we are using this >2*padding trick to exclude
            # tiles from the lower triangle from calculations ...
            if (min(band_to,diag_to) - max(band_from,diag_from)) > 2*pad_size:
                yield chrom, tilei, tilej
