"""
Collection of functions related to dot-calling

"""
from functools import partial, reduce
import multiprocess as mp
import logging

from scipy.linalg import toeplitz
from scipy.ndimage import convolve
from scipy.stats import poisson
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
import cooler

from ..lib.numutils import LazyToeplitz, get_kernel
from ..lib.common import is_cooler_balanced

import bioframe

logging.basicConfig(level=logging.INFO)

# this is to mitigate and parameterize the obs.raw vs count controversy:
observed_count_name = "count"
expected_count_name = "exp.raw"
adjusted_exp_name = lambda kernel_name: f"la_exp.{kernel_name}.value"
nans_inkernel_name = lambda kernel_name: f"la_exp.{kernel_name}.nnans"
bin1_id_name = "bin1_id"
bin2_id_name = "bin2_id"


def recommend_kernel_params(binsize):
    """
    Recommned kernel parameters for the
    standard convolution kernels, 'donut', etc
    same as in Rao et al 2014

    Parameters
    ----------
    binsize : integer
        binsize of the provided cooler

    Returns
    -------
    (w,p) : (integer, integer)
        tuple of the outer and inner kernel sizes
    """
    # kernel parameters depend on the cooler resolution
    # TODO: rename w, p to wid, pix probably, or _w, _p to avoid naming conflicts
    if binsize > 28000:
        # > 30 kb - is probably too much ...
        raise ValueError(
            f"Provided cooler has resolution {binsize} bases,"
            " which is too coarse for analysis."
        )
    elif binsize >= 18000:
        # ~ 20-25 kb:
        w, p = 3, 1
    elif binsize >= 8000:
        # ~ 10 kb
        w, p = 5, 2
    elif binsize >= 4000:
        # ~5 kb
        w, p = 7, 4
    else:
        # < 5 kb - is probably too fine ...
        raise ValueError(
            f"Provided cooler has resolution {binsize} bases,"
            " which is too fine for analysis."
        )
    # return the results:
    return w, p


def annotate_pixels_with_qvalues(
    pixels_df, qvalues, kernels, inplace=False, obs_raw_name=observed_count_name
):
    """
    Add columns with the qvalues to a DataFrame of pixels
    ... detailed but unedited notes ...
    Extract q-values using l-chunks and IntervalIndex.
    we'll do it in an ugly but workign fashion, by simply
    iteration over pairs of obs, la_exp and extracting needed qvals
    one after another
    ...

    Parameters
    ----------
    pixels_df : pandas.DataFrame
        a DataFrame with pixel coordinates that must have at least 2 columns
        named 'bin1_id' and 'bin2_id', where first is pixels's row and the
        second is pixel's column index.
    qvalues : dict of DataFrames
        A dictionary with keys being kernel names and values DataFrames
        storing q-values for each observed count values in each lambda-
        chunk. Colunms are Intervals defined by 'ledges' boundaries.
        Rows corresponding to a range of observed count values.
    kernels : dict
        A dictionary with keys being kernels names and values being ndarrays
        representing those kernels.

    Returns
    -------
    pixels_qvalue_df : pandas.DataFrame
        DataFrame of pixels with additional columns
        storing qvalues corresponding to the observed
        count value of a given pixel, given kernel-type,
        and a lambda-chunk.

    Notes
    -----
    Should be applied to a filtered DF of pixels, otherwise would
    be too resource-hungry.
    """
    if inplace:
        pixels_qvalue_df = pixels_df
    else:
        # let's do it "safe" - using a copy:
        pixels_qvalue_df = pixels_df.copy()
    # attempting to extract q-values using l-chunks and IntervalIndex:
    # we'll do it in an ugly but workign fashion, by simply
    # iteration over pairs of obs, la_exp and extracting needed qvals
    # one after another ...
    for k in kernels:
        pixels_qvalue_df[f"la_exp.{k}.qval"] = [
            qvalues[k].loc[o, e]
            for o, e in pixels_df[[obs_raw_name, f"la_exp.{k}.value"]].itertuples(
                index=False
            )
        ]
    # qvalues : dict
    #   A dictionary with keys being kernel names and values pandas.DataFrame-s
    #   storing q-values: each column corresponds to a lambda-chunk,
    #   while rows correspond to observed pixels values.
    return pixels_qvalue_df


def clust_2D_pixels(
    pixels_df,
    threshold_cluster=2,
    bin1_id_name="bin1_id",
    bin2_id_name="bin2_id",
    clust_label_name="c_label",
    clust_size_name="c_size",
    verbose=True,
):
    """
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
    clust_label_name : str
        Name of the cluster of pixels label. "c_label" by default.
    clust_size_name : str
        Name of the cluster of pixels size. "c_size" by default.
    verbose : bool
        Print verbose clustering summary report defaults is True.

    Returns
    -------
    peak_tmp : pandas.DataFrame
        DataFrame with the following columns:
        [c+bin1_id_name, c+bin2_id_name, clust_label_name, clust_size_name]
        row/col (bin1/bin2) are coordinates of centroids,
        label and sizes are unique pixel-cluster
        labels and their corresponding sizes.
    """

    # col (bin2) must precede row (bin1):
    pixels = pixels_df[[bin1_id_name, bin2_id_name]].values.astype(np.float64)
    # added astype(float) to avoid further issues with clustering, as it
    # turned out start1/start2 genome coordinates could be int32 or int64
    # and int32 is not enough for some operations, i.e., integer overflow.
    pixel_idxs = pixels_df.index

    # perform BIRCH clustering of pixels:
    # "n_clusters=None" implies using BIRCH without AgglomerativeClustering,
    # thus simply reporting "blobs" of pixels of radius "threshold_cluster"
    # along with blob-centroids as well:
    brc = Birch(
        n_clusters=None,
        threshold=threshold_cluster,
        # branching_factor=50, (it's default)
        compute_labels=True,
    )
    brc.fit(pixels)
    # # following is redundant,
    # # as it's done here:
    # # https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/cluster/birch.py#L638
    # clustered_labels = brc.predict(pixels)

    # labels of nearest centroid, assigned to each pixel,
    # BEWARE: labels might not be continuous, i.e.,
    # "np.unique(clustered_labels)" isn't same as "brc.subcluster_labels_", because:
    # https://github.com/scikit-learn/scikit-learn/blob/a24c8b464d094d2c468a16ea9f8bf8d42d949f84/sklearn/cluster/birch.py#L576
    clustered_labels = brc.labels_
    # centroid coordinates ( <= len(clustered_labels)):
    clustered_centroids = brc.subcluster_centers_
    # count unique labels and get their continuous indices
    uniq_labels, inverse_idx, uniq_counts = np.unique(
        clustered_labels, return_inverse=True, return_counts=True
    )
    # cluster sizes taken to match labels:
    cluster_sizes = uniq_counts[inverse_idx]
    # take centroids corresponding to labels (as many as needed):
    centroids_per_pixel = np.take(clustered_centroids, clustered_labels, axis=0)

    if verbose:
        # prepare clustering summary report:
        msg = (
            "Clustering is completed:\n"
            + f"{uniq_counts.size} clusters detected\n"
            + f"{uniq_counts.mean():.2f}+/-{uniq_counts.std():.2f} mean size\n"
        )
        logging.info(msg)

    # create output DataFrame
    centroids_n_labels_df = pd.DataFrame(
        centroids_per_pixel,
        index=pixel_idxs,
        columns=["c" + bin1_id_name, "c" + bin2_id_name],
    )
    # add labels per pixel:
    centroids_n_labels_df[clust_label_name] = clustered_labels.astype(np.int64)
    # add cluster sizes:
    centroids_n_labels_df[clust_size_name] = cluster_sizes.astype(np.int64)

    return centroids_n_labels_df


##################################
# matrix tiling and tiles-generator
##################################

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
        logging.info(
            f"matrix of size {size}X{size} to be splitted\n"
            + f"  into square tiles of size {step}.\n"
            + f"  A small 'edge' of size w={edge} is added, to allow for\n"
            + "  meaningfull convolution around boundaries.\n"
            + f"  Resulting number of tiles is {tiles * tiles}"
        )

    for tx in range(tiles):
        for ty in range(tiles):

            lwx = max(0, step * tx - edge)
            rwx = min(size, step * (tx + 1) + edge)
            if square and (rwx >= size):
                lwx = size - step - edge

            lwy = max(0, step * ty - edge)
            rwy = min(size, step * (ty + 1) + edge)
            if square and (rwy >= size):
                lwy = size - step - edge

            yield (lwx + start, rwx + start), (lwy + start, rwy + start)


def heatmap_tiles_generator_diag(clr, view_df, pad_size, tile_size, band_to_cover):
    """
    A generator yielding heatmap tiles that are needed to cover the requested
    band_to_cover around diagonal. Each tile is "padded" with pad_size edge to
    allow proper kernel-convolution of pixels close to boundary.

    Parameters
    ----------
    clr : cooler
        Cooler object to use to extract chromosome extents.
    view_df : viewframe
        Viewframe with genomic regions to process, chrom, start, end, name.
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
        column index of the tile (region_name, tilei, tilej).

    """

    for chrom, start, end, region_name in view_df.itertuples(index=False):
        region_begin, region_end = clr.extent((chrom, start, end))
        for tilei, tilej in square_matrix_tiling(
            region_begin, region_end, tile_size, pad_size
        ):
            # check if a given tile intersects with
            # with the diagonal band of interest ...
            diag_from = tilej[0] - tilei[1]
            diag_to = tilej[1] - tilei[0]
            #
            band_from = 0
            band_to = band_to_cover
            # we are using this >2*padding trick to exclude
            # tiles from the lower triangle from calculations ...
            if (min(band_to, diag_to) - max(band_from, diag_from)) > 2 * pad_size:
                yield region_name, tilei, tilej


##################################
# kernel-convolution related:
##################################
def _convolve_and_count_nans(O_bal, E_bal, E_raw, N_bal, kernel):
    """
    Dense versions of a bunch of matrices needed for convolution and
    calculation of number of NaNs in a vicinity of each pixel. And a kernel to
    be provided of course.

    """
    # a matrix filled with the kernel-weighted sums
    # based on a balanced observed matrix:
    KO = convolve(O_bal, kernel, mode="constant", cval=0.0, origin=0)
    # a matrix filled with the kernel-weighted sums
    # based on a balanced expected matrix:
    KE = convolve(E_bal, kernel, mode="constant", cval=0.0, origin=0)
    # get number of NaNs in a vicinity of every
    # pixel (kernel's nonzero footprint)
    # based on the NaN-matrix N_bal.
    # N_bal is shared NaNs between O_bal E_bal,
    # is it redundant ?
    NN = convolve(
        N_bal.astype(np.int64),
        # we have to use kernel's
        # nonzero footprint:
        (kernel != 0).astype(np.int64),
        mode="constant",
        # there are only NaNs
        # beyond the boundary:
        cval=1,
        origin=0,
    )
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
# this is the MAIN function to get locally adjusted expected
########################################################################
def get_adjusted_expected_tile_some_nans(
    origin, observed, expected, bal_weights, kernels, balance_factor=None, verbose=False
):
    """
    Get locally adjusted expected for a collection of local-filters (kernels).

    Such locally adjusted expected, 'Ek' for a given kernel,
    can serve as a baseline for deciding whether a given
    pixel is enriched enough to call it a feature (dot-loop,
    flare, etc.) in a downstream analysis.

    For every pixel of interest [i,j], locally adjusted
    expected is a product of a global expected in that
    pixel E_bal[i,j] and an enrichment of local environ-
    ment of the pixel, described with a given kernel:

    ::

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

    ::

                                  KERNEL[i,j](O_bal)
        Ek_raw[i,j] = E_raw[i,j]* ------------------ ,
                                  KERNEL[i,j](E_bal)

    where E_raw[i,j] is:

    ::

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
    expected : numpy.ndarray
        square symmetrical dense-matrix
        that contains expected, calculated
        based on balanced observed: E_bal.
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
        Note, scipy.ndimage.convove flips kernel
        first and only then applies it to matrix.
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
    O_raw = observed  # raw observed, no need to copy, no modifications.
    E_bal = np.copy(expected)
    # 'bal_weights': ndarray or a couple of those ...
    if isinstance(bal_weights, np.ndarray):
        v_bal_i = bal_weights
        v_bal_j = bal_weights
    elif isinstance(bal_weights, (tuple, list)):
        v_bal_i, v_bal_j = bal_weights
    else:
        raise ValueError(
            "'bal_weights' must be an numpy.ndarray"
            "for slices of a matrix with diagonal-origin or"
            "a tuple/list of a couple of numpy.ndarray-s"
            "for a slice of matrix with an arbitrary origin."
        )
    # kernels must be a dict with kernel-names as keys
    # and kernel ndarrays as values.
    if not isinstance(kernels, dict):
        raise ValueError(
            "'kernels' must be a dictionary" "with name-keys and ndarrays-values."
        )

    # balanced observed, from raw-observed
    # by element-wise multiply:
    O_bal = np.multiply(O_raw, np.outer(v_bal_i, v_bal_j))
    # O_bal is separate from O_raw memory-wise.

    # fill lower triangle of O_bal and E_bal with NaNs
    # in order to prevent peak calling from the lower triangle
    # and also to provide fair locally adjusted expected
    # estimation for pixels very close to diagonal, whose
    # "donuts"(kernels) would be crossing the main diagonal.
    # The trickiest thing here would be dealing with the origin: io,jo.
    O_bal[np.tril_indices_from(O_bal, k=(io - jo) - 1)] = np.nan
    E_bal[np.tril_indices_from(E_bal, k=(io - jo) - 1)] = np.nan

    # raw E_bal: element-wise division of E_bal[i,j] and
    # v_bal[i]*v_bal[j]:
    E_raw = np.divide(E_bal, np.outer(v_bal_i, v_bal_j))

    # let's calculate a matrix of common NaNs
    # shared between observed and expected:
    # check if it's redundant ? (is NaNs from O_bal sufficient? )
    N_bal = np.logical_or(np.isnan(O_bal), np.isnan(E_bal))
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
    i, j = np.indices(O_raw.shape)
    # pack it into DataFrame to accumulate results:
    peaks_df = pd.DataFrame({"bin1_id": i.flatten() + io, "bin2_id": j.flatten() + jo})

    with np.errstate(divide="ignore", invalid="ignore"):
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
            KO = convolve(O_bal, kernel, mode="constant", cval=0.0, origin=0)
            # a matrix filled with the kernel-weighted sums
            # based on a balanced expected matrix:
            KE = convolve(E_bal, kernel, mode="constant", cval=0.0, origin=0)
            # get number of NaNs in a vicinity of every
            # pixel (kernel's nonzero footprint)
            # based on the NaN-matrix N_bal.
            # N_bal is shared NaNs between O_bal E_bal,
            NN = convolve(
                N_bal.astype(np.int64),
                # we have to use kernel's
                # nonzero footprint:
                (kernel != 0).astype(np.int64),
                mode="constant",
                # there are only NaNs
                # beyond the boundary:
                cval=1,
                origin=0,
            )
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
                peaks_df[f"factor_balance.{kernel_name}.KerObs"] = (
                    balance_factor * KO.flatten()
                )
                # KO*balance_factor: to be compared with 16 ...
            if verbose:
                logging.info(f"Convolution with kernel {kernel_name} is complete.")
            #
            # accumulation into single DataFrame:
            # store locally adjusted expected for each kernel
            # and number of NaNs in the footprint of each kernel
            peaks_df[f"la_exp.{kernel_name}.value"] = Ek_raw.flatten()
            peaks_df[f"la_exp.{kernel_name}.nnans"] = NN.flatten()
            # do all the filter/logic/masking etc on the complete DataFrame ...
    #####################################
    # downstream stuff is supposed to be
    # aggregated over all kernels ...
    #####################################
    peaks_df["exp.raw"] = E_raw.flatten()
    # obs.raw -> count
    peaks_df["count"] = O_raw.flatten()

    # TO BE REFACTORED/deprecated ...
    # compatibility with legacy API is completely BROKEN
    # post-processing allows us to restore it, see tests,
    # but we pay with the processing speed for it.
    mask_ndx = pd.Series(0, index=peaks_df.index, dtype=np.bool)
    for kernel_name, kernel in kernels.items():
        # accummulating with a vector full of 'False':
        mask_ndx_kernel = ~np.isfinite(peaks_df["la_exp." + kernel_name + ".value"])
        mask_ndx = np.logical_or(mask_ndx_kernel, mask_ndx)

    # returning only pixels from upper triangle of a matrix
    # is likely here to stay:
    upper_band = peaks_df["bin1_id"] < peaks_df["bin2_id"]
    # Consider filling lower triangle of the OBSERVED matrix tile
    # with NaNs, instead of this - we'd need this for a fair
    # consideration of the pixels that are super close to the
    # diagonal and in a case, when the corresponding donut would
    # cross a diagonal line.
    # selecting pixels in relation to diagonal - too far, too
    # close etc, is now shifted to the outside of this function
    # a way to simplify code.

    # return good semi-sparsified DF:
    return peaks_df[~mask_ndx & upper_band].reset_index(drop=True)


##################################
# step-specific dot-calling functions
##################################


def score_tile(
    tile_cij,
    clr,
    cis_exp,
    exp_v_name,
    clr_weight_name,
    kernels,
    nans_tolerated,
    band_to_cover,
    balance_factor,
    verbose,
):
    """
    The main working function that given a tile of a heatmap, applies kernels to
    perform convolution to calculate locally-adjusted expected and then
    calculates a p-value for every meaningfull pixel against these l.a. expected
    values.

    Parameters
    ----------
    tile_cij : tuple
        Tuple of 3: chromosome name, tile span row-wise, tile span column-wise:
        (chrom, tile_i, tile_j), where tile_i = (start_i, end_i), and
        tile_j = (start_j, end_j).
    clr : cooler
        Cooler object to use to extract Hi-C heatmap data.
    cis_exp : pandas.DataFrame
        DataFrame with cis-expected, indexed with 'name' and 'diag'.
    exp_v_name : str
        Name of a value column in expected DataFrame
    clr_weight_name : str
        Name of a value column with balancing weights in a cooler.bins()
        DataFrame. Typically 'weight'.
    kernels : dict
        A dictionary with keys being kernels names and values being ndarrays
        representing those kernels.
    nans_tolerated : int
        Number of NaNs tolerated in a footprint of every kernel.
    band_to_cover : int
        Results would be stored only for pixels connecting loci closer than
        'band_to_cover'.
    balance_factor : float
        Balancing factor to turn sum of balanced matrix back approximately
        to the number of pairs (used for dynamic-donut criteria mostly).
        use None value to disable dynamic-donut criteria calculation.
    verbose : bool
        Enable verbose output.

    Returns
    -------
    res_df : pandas.DataFrame
        results: annotated pixels with calculated locally adjusted expected
        for every kernels, observed, precalculated pvalues, number of NaNs in
        footprint of every kernels, all of that in a form of an annotated
        pixels DataFrame for eligible pixels of a given tile.

    """
    # unpack tile's coordinates
    region_name, tilei, tilej = tile_cij
    origin = (tilei[0], tilej[0])

    # we have to do it for every tile, because
    # region_name is not known apriori (maybe move outside)
    # use .loc[region, region] for symmetric cis regions to conform with expected v1.0
    lazy_exp = LazyToeplitz(cis_exp.loc[region_name, region_name][exp_v_name].values)

    # RAW observed matrix slice:
    observed = clr.matrix(balance=False)[slice(*tilei), slice(*tilej)]
    # expected as a rectangular tile :
    expected = lazy_exp[slice(*tilei), slice(*tilej)]
    # slice of balance_weight for row-span and column-span :
    bal_weight_i = clr.bins()[slice(*tilei)][clr_weight_name].values
    bal_weight_j = clr.bins()[slice(*tilej)][clr_weight_name].values

    # do the convolutions
    result = get_adjusted_expected_tile_some_nans(
        origin=origin,
        observed=observed,
        expected=expected,
        bal_weights=(bal_weight_i, bal_weight_j),
        kernels=kernels,
        balance_factor=balance_factor,
        verbose=verbose,
    )

    # Post-processing filters
    # (1) exclude pixels that connect loci further than 'band_to_cover' apart:
    is_inside_band = result["bin1_id"] > (result["bin2_id"] - band_to_cover)

    # (2) identify pixels that pass number of NaNs compliance test for ALL kernels:
    does_comply_nans = np.all(
        result[[f"la_exp.{k}.nnans" for k in kernels]] < nans_tolerated, axis=1
    )
    # so, selecting inside band and nNaNs compliant results:
    # ( drop dropping index maybe ??? ) ...
    res_df = result[is_inside_band & does_comply_nans].reset_index(drop=True)
    # #######################################################################
    # # the following should be rewritten such that we return
    # # opnly bare minimum number of columns per chunk - i.e. annotating is too heavy
    # # to be performed here ...
    # #
    # # I'll do it here - switch off annotation, it'll break "extraction" and other
    # # downstream stuff - but we'll fix it afterwards ...
    # #
    # ########################################################################
    # ########################################################################
    # # consider retiring Poisson testing from here, in case we
    # # stick with l-chunking or opposite - add histogramming business here(!)
    # ########################################################################
    # # do Poisson tests:
    # get_pval = lambda la_exp : 1.0 - poisson.cdf(res_df["obs.raw"], la_exp)
    # for k in kernels:
    #     res_df["la_exp."+k+".pval"] = get_pval( res_df["la_exp."+k+".value"] )
    # # annotate and return
    # return cooler.annotate(res_df.reset_index(drop=True), clr.bins()[:])
    #
    # so return only bin_ids , observed-raw (rename to counts, by Nezar's suggestion)
    # and a bunch of locally adjusted expected estimates - 1 per kernel -
    # that's the bare minimum ...
    #
    # per Nezar's suggestion
    # rename obs.raw -> count
    # this would break A LOT of downstream stuff - but let it be ...
    #
    return res_df[
        ["bin1_id", "bin2_id", "count"] + [f"la_exp.{k}.value" for k in kernels]
    ].astype(dtype={f"la_exp.{k}.value": "float64" for k in kernels})


def histogram_scored_pixels(
    scored_df, kernels, ledges, verbose, obs_raw_name=observed_count_name
):
    """
    An attempt to implement HiCCUPS-like lambda-chunking
    statistical procedure.
    This function aims at building up a histogram of locally
    adjusted expected scores for groups of characterized
    pixels.
    Such histograms are later used to compute FDR thresholds
    for different "classes" of hypothesis (classified by their
    l.a. expected scores).

    Parameters
    ----------
    scored_df : pd.DataFrame
        A table with the scoring information for a group of pixels.
    kernels : dict
        A dictionary with keys being kernels names and values being ndarrays
        representing those kernels.
    ledges : ndarray
        An ndarray with bin lambda-edges for groupping loc. adj. expecteds,
        i.e., classifying statistical hypothesis into lambda-classes.
        Left-most bin (-inf, 1], and right-most one (value,+inf].
    verbose : bool
        Enable verbose output.
    obs_raw_name : str
        Name of the column/field that carry number of counts per pixel,
        i.e. observed raw counts.

    Returns
    -------
    hists : dict of pandas.DataFrame
        A dictionary of pandas.DataFrame with lambda/observed histogram for
        every kernel-type.


    Notes
    -----
    This is just an attempt to implement HiCCUPS-like lambda-chunking.
    So we'll be returning histograms corresponding to the chunks of
    scored pixels.
    Consider modifying/accumulation a globally defined hists object,
    or probably making something like a Feature2D class later on
    where hists would be a class feature and histogramming_step would be
    a method.


    """

    # lambda-chunking implies different 'pval' calculation
    # procedure with a single Poisson expected for all the
    # hypothesis in a same "class", i.e. with the l.a. expecteds
    # from the same histogram bin.

    ########################
    # implementation ideas:
    ########################
    # same observations/hypothesis needs to be classified according
    # to different l.a. expecteds (i.e. for different kernel-types),
    # which can be done with a pandas groupby, something like that:
    # https://stackoverflow.com/questions/21441259
    #
    # after that we could iterate over groups and do np.bincout on
    # the "observed.raw" column (assuming it's of integer type) ...

    hists = {}
    for k in kernels:
        # verbose:
        if verbose:
            logging.info(f"Building a histogram for kernel-type {k}")
        #  we would need to generate a bunch of these histograms for all of the
        # kernel types:
        # needs to be lambda-binned             : scored_df["la_exp."+k+".value"]
        # needs to be histogrammed in every bin : scored_df["obs.raw"]
        #
        # lambda-bin index for kernel-type "k":
        lbins = pd.cut(scored_df[f"la_exp.{k}.value"], ledges)
        # now for each lambda-bin construct a histogramm of "obs.raw":
        obs_hist = {}
        for lbin, grp_df in scored_df.groupby(lbins):
            # check if obs.raw is integer of spome kind (temporary):
            # obs.raw -> count
            assert np.issubdtype(grp_df[obs_raw_name].dtype, np.integer)
            # perform bincounting ...
            obs_hist[lbin] = pd.Series(np.bincount(grp_df[obs_raw_name]))
            # ACHTUNG! assigning directly to empty DF leads to data loss!
            # turn ndarray in Series for ease of handling, i.e.
            # assign a bunch of columns of different sizes to DataFrame.
            #
            # Consider updating global "hists" later on, or implementing a
            # special class for that. Mind that Python multiprocessing
            # "threads" are processes and thus cannot share/modify a shared
            # memory location - deal with it, maybe dask-something ?!
            #
            # turned out that storing W1xW2 for every "thread" requires a ton
            # of memory - that's why different sizes... @nvictus ?
        # store W1x(<=W2) hist for every kernel-type:
        hists[k] = pd.DataFrame(obs_hist).fillna(0).astype(np.int64)
    # return a dict of DataFrames with a bunch of histograms:
    return hists


def determine_thresholds(kernels, ledges, gw_hist, fdr):
    """
    given a 'gw_hist' histogram of observed counts
    for each lambda-chunk for each kernel-type, and
    also given a FDR, calculate q-values for each observed
    count value in each lambda-chunk for each kernel-type.

    Returns
    -------
    threshold_df : dict
      each threshold_df[k] is a Series indexed by la_exp intervals
      (IntervalIndex) and it is all we need to extract "good" pixels from
      each chunk ...
    qvalues : dict
      A dictionary with keys being kernel names and values pandas.DataFrames
      storing q-values: each column corresponds to a lambda-chunk,
      while rows correspond to observed pixels values.


    """
    rcs_hist = {}
    rcs_Poisson = {}
    qvalues = {}
    threshold_df = {}
    for k in kernels:
        # Reverse cumulative histogram for this kernel.
        # First row contains total # of pixels in each lambda-chunk.
        rcs_hist[k] = gw_hist[k].iloc[::-1].cumsum(axis=0).iloc[::-1]

        # Assign a unit Poisson distribution to each lambda-chunk.
        # The expected value is the upper boundary of the lambda-chunk.
        #   poisson.sf = 1 - poisson.cdf, but more precise
        #   poisson.sf(-1,mu) == 1.0, i.e. is equivalent to the
        #   poisson.pmf(gw_hist[k].index, mu)[::-1].cumsum()[::-1]
        rcs_Poisson[k] = pd.DataFrame()
        for mu, column in zip(ledges[1:-1], gw_hist[k].columns):
            renorm_factors = rcs_hist[k].loc[0, column]
            rcs_Poisson[k][column] = renorm_factors * poisson.sf(
                gw_hist[k].index - 1, mu
            )

        # Determine the threshold by checking the value at which 'fdr_diff'
        # first turns positive. Fill NaNs with an "unreachably" high value.
        fdr_diff = fdr * rcs_hist[k] - rcs_Poisson[k]
        very_high_value = len(rcs_hist[k])
        threshold_df[k] = (
            fdr_diff.where(fdr_diff > 0)
            .apply(lambda col: col.first_valid_index())
            .fillna(very_high_value)
            .astype(np.int64)
        )
        # q-values
        # bear in mind some issues with lots of NaNs and Infs after
        # such a brave operation ...
        qvalues[k] = rcs_Poisson[k] / rcs_hist[k]

    return threshold_df, qvalues


def extract_scored_pixels(
    scored_df, kernels, thresholds, ledges, verbose, obs_raw_name=observed_count_name
):
    """
    An attempt to implement HiCCUPS-like lambda-chunking
    statistical procedure.
    Use FDR thresholds for different "classes" of hypothesis
    (classified by their l.a. expected scores), in order to
    extract pixels that stand out.

    Parameters
    ----------
    scored_df : pd.DataFrame
        A table with the scoring information for a group of pixels.
    kernels : dict
        A dictionary with keys being kernel names and values being ndarrays
        representing those kernels.
    thresholds : dict
        A dictionary with keys being kernel names and values pandas.Series
        indexed with Intervals defined by 'ledges' boundaries and storing FDR
        thresholds for observed values.
    ledges : ndarray
        An ndarray with bin lambda-edges for groupping loc. adj. expecteds,
        i.e., classifying statistical hypothesis into lambda-classes.
        Left-most bin (-inf, 1], and right-most one (value,+inf].
    verbose : bool
        Enable verbose output.
    obs_raw_name : str
        Name of the column/field that carry number of counts per pixel,
        i.e. observed raw counts.

    Returns
    -------
    scored_df_slice : pandas.DataFrame
        Filtered DataFrame of pixels extracted applying FDR thresholds.

    Notes
    -----
    This is just an attempt to implement HiCCUPS-like lambda-chunking.

    """
    comply_fdr_list = np.ones(len(scored_df), dtype=np.bool)

    for k in kernels:
        # using special features of IntervalIndex:
        # http://pandas.pydata.org/pandas-docs/stable/advanced.html#intervalindex
        # i.e. IntervalIndex can be .loc-ed with values that would be
        # corresponded with their Intervals (!!!):
        # obs.raw -> count
        comply_fdr_k = (
            scored_df[obs_raw_name].values
            > thresholds[k].loc[scored_df[f"la_exp.{k}.value"]].values
        )
        # extracting q-values for all of the pixels takes a lot of time
        # we'll do it externally for filtered_pixels only, in order to save
        # time
        #
        # accumulate comply_fdr_k into comply_fdr_list
        # using np.logical_and:
        comply_fdr_list = np.logical_and(comply_fdr_list, comply_fdr_k)
    # return a slice of 'scored_df' that complies FDR thresholds:
    return scored_df[comply_fdr_list]


##################################
# large CLI-helper functions wrapping smaller step-specific ones:
# basically - the dot-calling steps - ONE PASS DOT-CALLING:
##################################


def scoring_step(
    clr,
    expected,
    expected_name,
    clr_weight_name,
    tiles,
    kernels,
    max_nans_tolerated,
    loci_separation_bins,
    output_path,
    nproc,
    verbose,
):
    """
    Calculates locally adjusted expected
    for each pixel in a designated area of
    the heatmap and return it as a big
    single pixel-table (pandas.DataFrame)
    """
    if verbose:
        logging.info(f"Preparing to convolve {len(tiles)} tiles:")

    # check if cooler is balanced
    try:
        _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
    except Exception as e:
        raise ValueError(
            f"provided cooler is not balanced or {clr_weight_name} is missing"
        ) from e

    # add very_verbose to supress output from convolution of every tile
    very_verbose = False
    job = partial(
        score_tile,
        clr=clr,
        cis_exp=expected,
        exp_v_name=expected_name,
        clr_weight_name=clr_weight_name,
        kernels=kernels,
        nans_tolerated=max_nans_tolerated,
        band_to_cover=loci_separation_bins,
        balance_factor=None,
        verbose=very_verbose,
    )

    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.imap
        map_kwargs = dict(chunksize=int(np.ceil(len(tiles) / nproc)))
        if verbose:
            logging.info(
                f"creating a Pool of {nproc} workers to tackle {len(tiles)} tiles"
            )
    else:
        map_ = map
        if verbose:
            logging.info("fallback to serial implementation.")
        map_kwargs = {}
    try:
        # do the work here
        chunks = map_(job, tiles, **map_kwargs)
        # ###########################################
        # # local - in memory copy of the dataframe (danger of RAM overuse)
        # ###########################################
        # reset index is required, otherwise there will be duplicate
        # indices in the output of this function
        return pd.concat(chunks).reset_index(drop=True)
    finally:
        if nproc > 1:
            pool.close()


def clustering_step(
    scores_df,
    expected_regions,
    dots_clustering_radius,
    verbose,
    obs_raw_name=observed_count_name,
):
    """

    This is a new "clustering" step updated for the pixels processed by lambda-
    chunking multiple hypothesis testing.

    This method assumes that 'scores_df' is a DataFrame with all of the pixels
    that needs to be clustered, thus there is no additional 'comply_fdr' column
    and selection of compliant pixels.

    This step is a clustering-only (using Birch from scikit).

    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame that stores filtered pixels that are ready to be
        clustered, no more 'comply_fdr' column dependency.
    expected_regions : iterable
        An iterable of regions to be clustered.
    dots_clustering_radius : int
        Birch-clustering threshold.
    verbose : bool
        Enable verbose output.
    Returns
    -------
    centroids : pandas.DataFrame
        Pixels from 'scores_df' annotated with clustering information.

    Notes
    -----
    'dots_clustering_radius' in Birch clustering algorithm corresponds to a
    double the clustering radius in the "greedy"-clustering used in HiCCUPS
    (to be tested).

    """
    # Annotate regions, if needed:

    scores_df = scores_df.copy()
    if (
        not "region" in scores_df.columns
    ):  # If input scores are not annotated by regions:
        scores_df["region"] = np.where(
            scores_df["chrom1"] == scores_df["chrom2"], scores_df["chrom1"], np.nan
        )
    # using different bin12_id_names since all
    # pixels are annotated at this point.
    pixel_clust_list = []
    for region in expected_regions:
        # Perform clustering for each region separately.
        df = scores_df[((scores_df["region"].astype(str) == str(region)))]
        if not len(df):
            continue

        pixel_clust = clust_2D_pixels(
            df,
            threshold_cluster=dots_clustering_radius,
            bin1_id_name="start1",
            bin2_id_name="start2",
            verbose=verbose,
        )
        pixel_clust_list.append(pixel_clust)
    if verbose:
        logging.info("Clustering is over!")
    # concatenate clustering results ...
    # indexing information persists here ...

    if len(pixel_clust_list) == 0:
        if verbose:
            logging.info("No clusters found! Output will be empty")
        empty_output = pd.DataFrame(
            [],
            columns=list(scores_df.columns)
            + ["region1", "region2", "c_label", "c_size", "cstart1", "cstart2"],
        )
        return empty_output  # Empty dataframe with the same columns as anticipated
    pixel_clust_df = pd.concat(
        pixel_clust_list, ignore_index=False
    )  # Concatenate the clustering results for different regions

    # now merge pixel_clust_df and scores_df DataFrame ...
    # # and merge (index-wise) with the main DataFrame:
    df = pd.merge(
        scores_df, pixel_clust_df, how="left", left_index=True, right_index=True
    )
    # prevents scores_df categorical values (all chroms, including chrM)
    df["region1"] = df["region"].astype(str)
    df["region2"] = df["region"].astype(str)
    # report only centroids with highest Observed:
    chrom_clust_group = df.groupby(["region1", "region2", "c_label"])
    centroids = df.loc[
        chrom_clust_group[obs_raw_name].idxmax()
    ]  # Select the brightest pixel in the cluster
    return centroids


# consider switching step names - "extraction" to "FDR-thresholding"
# and "thresholding" to "enrichment" - for final enrichment selection:
def thresholding_step(centroids, obs_raw_name=observed_count_name):
    # (2)
    # filter by FDR, enrichment etc:
    enrichment_factor_1 = 1.5
    enrichment_factor_2 = 1.75
    enrichment_factor_3 = 2.0
    FDR_orphan_threshold = 0.02
    ######################################################################
    # # Temporarily remove orphans filtering, until q-vals are calculated:
    ######################################################################
    enrichment_fdr_comply = (
        (
            centroids[obs_raw_name]
            > enrichment_factor_2 * centroids["la_exp.lowleft.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_2 * centroids["la_exp.donut.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_1 * centroids["la_exp.vertical.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_1 * centroids["la_exp.horizontal.value"]
        )
        & (
            (
                centroids[obs_raw_name]
                > enrichment_factor_3 * centroids["la_exp.lowleft.value"]
            )
            | (
                centroids[obs_raw_name]
                > enrichment_factor_3 * centroids["la_exp.donut.value"]
            )
        )
        & (
            (centroids["c_size"] > 1)
            | (
                (
                    centroids["la_exp.lowleft.qval"]
                    + centroids["la_exp.donut.qval"]
                    + centroids["la_exp.vertical.qval"]
                    + centroids["la_exp.horizontal.qval"]
                )
                <= FDR_orphan_threshold
            )
        )
    )
    # #
    # enrichment_fdr_comply = (
    #     (centroids[obs_raw_name] > enrichment_factor_2 * centroids["la_exp.lowleft.value"]) &
    #     (centroids[obs_raw_name] > enrichment_factor_2 * centroids["la_exp.donut.value"]) &
    #     (centroids[obs_raw_name] > enrichment_factor_1 * centroids["la_exp.vertical.value"]) &
    #     (centroids[obs_raw_name] > enrichment_factor_1 * centroids["la_exp.horizontal.value"]) &
    #     ( (centroids[obs_raw_name] > enrichment_factor_3 * centroids["la_exp.lowleft.value"])
    #         | (centroids[obs_raw_name] > enrichment_factor_3 * centroids["la_exp.donut.value"]) )
    # )
    # use "enrichment_fdr_comply" to filter out
    # non-satisfying pixels:
    out = centroids[enrichment_fdr_comply]

    # ...
    # to be added to the list of output columns:
    # "factor_balance."+"lowleft"+".KerObs"
    # ...

    # tentaive output columns list:
    columns_for_output = [
        "chrom1",
        "start1",
        "end1",
        "chrom2",
        "start2",
        "end2",
        "cstart1",
        "cstart2",
        "c_label",
        "c_size",
        obs_raw_name,
        # 'exp.raw',
        "la_exp.donut.value",
        "la_exp.vertical.value",
        "la_exp.horizontal.value",
        "la_exp.lowleft.value",
        # "factor_balance.lowleft.KerObs",
        # 'la_exp.upright.value',
        # 'la_exp.upright.qval',
        "la_exp.donut.qval",
        "la_exp.vertical.qval",
        "la_exp.horizontal.qval",
        "la_exp.lowleft.qval",
    ]
    return out[columns_for_output]


##################################
# large CLI-helper functions wrapping smaller step-specific ones:
# basically - the dot-calling steps - ON THE FLY - 2 PASS DOT-CALLING (HiCCUPS-style):
##################################


def scoring_and_histogramming_step(
    clr,
    expected,
    expected_name,
    clr_weight_name,
    tiles,
    kernels,
    ledges,
    max_nans_tolerated,
    loci_separation_bins,
    nproc,
    verbose,
):
    """
    This is a derivative of the 'scoring_step' which is supposed to implement
    the 1st of the lambda-chunking procedure - histogramming.

    Basically we are piping scoring operation together with histogramming into a
    single pipeline of per-chunk operations/transforms.
    """
    if verbose:
        logging.info(f"Preparing to convolve {len(tiles)} tiles:")

    # add very_verbose to supress output from convolution of every tile
    very_verbose = False

    # check if cooler is balanced
    try:
        _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
    except Exception as e:
        raise ValueError(
            f"provided cooler is not balanced or {clr_weight_name} is missing"
        ) from e

    # to score per tile:
    to_score = partial(
        score_tile,
        clr=clr,
        cis_exp=expected,
        exp_v_name=expected_name,
        clr_weight_name=clr_weight_name,
        kernels=kernels,
        nans_tolerated=max_nans_tolerated,
        band_to_cover=loci_separation_bins,
        # do not calculate dynamic-donut criteria
        # for now.
        balance_factor=None,
        verbose=very_verbose,
    )

    # to hist per scored chunk:
    to_hist = partial(
        histogram_scored_pixels, kernels=kernels, ledges=ledges, verbose=very_verbose
    )

    # composing/piping scoring and histogramming
    # together :
    job = lambda tile: to_hist(to_score(tile))

    # copy paste from @nvictus modified 'scoring_step':
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.imap
        map_kwargs = dict(chunksize=int(np.ceil(len(tiles) / nproc)))
        if verbose:
            logging.info(
                f"creating a Pool of {nproc} workers to tackle {len(tiles)} tiles"
            )
    else:
        map_ = map
        if verbose:
            logging.info("fallback to serial implementation.")
        map_kwargs = {}
    try:
        # consider using
        # https://github.com/mirnylab/cooler/blob/9e72ee202b0ac6f9d93fd2444d6f94c524962769/cooler/tools.py#L59
        # here:
        hchunks = map_(job, tiles, **map_kwargs)
        # hchunks TO BE ACCUMULATED
        # hopefully 'hchunks' would stay in memory
        # until we would get a chance to accumulate them:
    finally:
        if nproc > 1:
            pool.close()
    #
    # now we need to combine/sum all of the histograms
    # for different kernels:
    #
    # assuming we know "kernels"
    # this is very ugly, but ok
    # for the draft lambda-chunking
    # lambda version of lambda-chunking:
    def _sum_hists(hx, hy):
        # perform a DataFrame summation
        # for every value of the dictionary:
        hxy = {}
        for k in kernels:
            hxy[k] = hx[k].add(hy[k], fill_value=0).astype(np.int64)
        # returning the sum:
        return hxy

    # ######################################################
    # this approach is tested and at the very least
    # number of pixels in a dump list matches
    # with the .sum().sum() of the histogram
    # both for 10kb and 5kb,
    # thus we should consider this as a reference
    # implementation, albeit not a very efficient one ...
    # ######################################################
    final_hist = reduce(_sum_hists, hchunks)
    # we have to make sure there is nothing in the
    # top bin, i.e., there are no l.a. expecteds > base^(len(ledges)-1)
    for k in kernels:
        last_la_exp_bin = final_hist[k].columns[-1]
        last_la_exp_vals = final_hist[k].iloc[:, -1]
        # checking the top bin:
        if last_la_exp_vals.sum() != 0:
            raise ValueError(
                f"There are la_exp.{k}.value in {last_la_exp_bin}, please check the histogram"
            )
        # drop that last column/bin (last_edge, +inf]:
        final_hist[k] = final_hist[k].drop(columns=last_la_exp_bin)
        # consider dropping all of the columns that have zero .sum()
    # returning filtered histogram
    return final_hist


def scoring_and_extraction_step(
    clr,
    expected,
    expected_name,
    clr_weight_name,
    tiles,
    kernels,
    ledges,
    thresholds,
    max_nans_tolerated,
    balance_factor,
    loci_separation_bins,
    output_path,
    nproc,
    verbose,
    bin1_id_name="bin1_id",
    bin2_id_name="bin2_id",
):
    """
    This is a derivative of the 'scoring_step' which is supposed to implement
    the 2nd of the lambda-chunking procedure - extracting pixels that are FDR
    compliant.

    Basically we are piping scoring operation together with extraction into a
    single pipeline of per-chunk operations/transforms.

    """
    if verbose:
        logging.info(f"Preparing to convolve {len(tiles)} tiles:")

    # add very_verbose to supress output from convolution of every tile
    very_verbose = False

    # check if cooler is balanced
    try:
        _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
    except Exception as e:
        raise ValueError(
            f"provided cooler is not balanced or {clr_weight_name} is missing"
        ) from e

    # to score per tile:
    to_score = partial(
        score_tile,
        clr=clr,
        cis_exp=expected,
        exp_v_name=expected_name,
        clr_weight_name=clr_weight_name,
        kernels=kernels,
        nans_tolerated=max_nans_tolerated,
        band_to_cover=loci_separation_bins,
        balance_factor=balance_factor,
        verbose=very_verbose,
    )

    # to hist per scored chunk:
    to_extract = partial(
        extract_scored_pixels,
        kernels=kernels,
        thresholds=thresholds,
        ledges=ledges,
        verbose=very_verbose,
    )

    # composing/piping scoring and histogramming
    # together :
    job = lambda tile: to_extract(to_score(tile))

    # copy paste from @nvictus modified 'scoring_step':
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.imap
        map_kwargs = dict(chunksize=int(np.ceil(len(tiles) / nproc)))
        if verbose:
            logging.info(
                f"creating a Pool of {nproc} workers to tackle {len(tiles)} tiles"
            )
    else:
        map_ = map
        if verbose:
            logging.info("fallback to serial implementation.")
        map_kwargs = {}
    try:
        # consider using
        # https://github.com/mirnylab/cooler/blob/9e72ee202b0ac6f9d93fd2444d6f94c524962769/cooler/tools.py#L59
        # here:
        filtered_pix_chunks = map_(job, tiles, **map_kwargs)
        significant_pixels = pd.concat(filtered_pix_chunks, ignore_index=True)
        if output_path is not None:
            significant_pixels.to_csv(
                output_path, sep="\t", header=True, index=False, compression=None
            )
    finally:
        if nproc > 1:
            pool.close()
    # there should be no duplicates in the "significant_pixels" DataFrame of pixels:
    significant_pixels_dups = significant_pixels.duplicated()
    if significant_pixels_dups.any():
        raise ValueError(
            f"Duplicated pixels detected during exctraction {significant_pixels[significant_pixels_dups]}"
        )
    # sort the result just in case and drop its index:
    return significant_pixels.sort_values(by=[bin1_id_name, bin2_id_name]).reset_index(
        drop=True
    )
