"""
Collection of functions related to dot-calling

The main user-facing API function is:

.. code-block:: python

    dots(
        clr,
        expected,
        expected_value_col="balanced.avg",
        clr_weight_name="weight",
        view_df=None,
        kernels=None,
        max_loci_separation=10_000_000,
        max_nans_tolerated=1,
        n_lambda_bins=40,
        lambda_bin_fdr=0.1,
        clustering_radius=20_000,
        cluster_filtering=None,
        tile_size=5_000_000,
        nproc=1,
    )

This function implements HiCCUPS-style dot calling, but enables user-specified
modifications at multiple steps. The current implementation makes two passes
over the input data, first to create a histogram of pixel enrichment values, 
and second to extract significantly enriched pixels.

- The function starts with compatibility verifications   
- Recommendation or verification for `kernels` is done next.  
  Custom kernels must satisfy properties including: square shape,
  equal sizes, odd sizes, zeros in the middle, etc. By default,
  HiCCUPS-style kernels are recommended based on the binsize.
- Lambda bins are defined for multiple hypothesis  
  testing separately for different value ranges of the locally adjusted expected.
  Currently, log-binned lambda-bins are hardcoded using a pre-defined
  BASE of 2^(1/3). `n_lambda_bins` controls the total number of bins.
  for the `clr`, `expected` and `view` of interest.
- Genomic regions in the specified `view`(all chromosomes by default)  
  are split into smaller tiles of size `tile_size`.
- `scoring_and_histogramming_step()` is performed independently  
  on the genomic tiles. In this step, locally adjusted expected is
  calculated using convolution kernels for each pixel in the tile.
  All surveyed pixels are histogrammed according to their adjusted 
  expected and raw observed counts. Locally adjusted expected is 
  not stored in memory.
- Chunks of histograms are aggregated together and a modified BH-FDR  
  procedure is applied to the result in `determine_thresholds()`.
  This returns thresholds for statistical significance 
  in each lambda-bin (for observed counts), along with the adjusted
  p-values (q-values).
- Calculated thresholds are used to extract statistically significant  
  pixels in `scoring_and_extraction_step()`. Because locally adjusted
  expected is not stored in memory, it is re-caluclated
  during this step, which makes it computationally intensive.
  Locally adjusted expected values are required in order to apply
  different thresholds of significance depending on the lambda-bin.
- Returned filtered pixels, or 'dots', are significantly enriched  
  relative to their locally adjusted expecteds and thus have potential
  biological interest. Dots are further annotated with their 
  genomic coordinates and q-values (adjusted p-values) for
  all applied kernels.
- All further steps perform optional post-processing on called dots

  - enriched pixels that are within `clustering_radius` of each other  
    are clustered together and the brightest one is selected as the
    representative position of a dot.
  - cluster-representatives along with "singletons" (enriched pixels  
    that are not part of any cluster) can be subjected to further
    empirical enrichment filtering in `cluster_filtering_hiccups()`. This 
    both requires clustered dots exceed prescribed enrichment thresholds 
    relative to their local neighborhoods and that singletons pass an 
    even more stringent q-value threshold.
"""

from functools import partial, reduce
import multiprocess as mp
import logging
import warnings
import time

from scipy.linalg import toeplitz
from scipy.ndimage import convolve
from scipy.stats import poisson
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from sklearn.cluster import Birch
import cooler

from ..lib.numutils import LazyToeplitz, get_kernel
from ..lib.common import assign_regions, make_cooler_view
from ..lib.checks import (
    is_cooler_balanced,
    is_compatible_viewframe,
    is_valid_expected,
)

from bioframe import make_viewframe

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logging.basicConfig(level=logging.INFO)

# this is to mitigate and parameterize the obs.raw vs count controversy:
observed_count_name = "count"
expected_count_name = "exp.raw"
adjusted_exp_name = lambda kernel_name: f"la_exp.{kernel_name}.value"
nans_inkernel_name = lambda kernel_name: f"la_exp.{kernel_name}.nnans"
bin1_id_name = "bin1_id"
bin2_id_name = "bin2_id"
bedpe_required_cols = [
    "chrom1",
    "start1",
    "end1",
    "chrom2",
    "start2",
    "end2",
]


# define basepairs to bins for clarity
def bp_to_bins(basepairs, binsize):
    return int(basepairs / binsize)


def recommend_kernels(binsize):
    """
    Return a recommended set of convolution kernels for dot-calling
    based on the resolution, or binsize, of the input data.

    This function currently recommends the four kernels used in the HiCCUPS method:
    donut, horizontal, vertical, lowerleft. Kernels are recommended for resolutions
    near 5 kb, 10 kb, and 25 kb. Dots are not typically visible at lower resolutions
    (binsize >28kb) and the majority of datasets are too sparse for dot-calling
    at very high resolutions (<4kb). Given this, default kernels are not
    recommended for resolutions outside this range.

    Parameters
    ----------
    binsize : integer
        binsize of the provided cooler

    Returns
    -------
    kernels : {str:ndarray}
        dictionary of convolution kernels as ndarrays, with their
        names as keys.
    """
    # define "donut" parameters w and p based on the resolution of the data
    # w is the outside half-width, and p is the internal half-width.
    if binsize > 28000:
        raise ValueError(
            f"Provided cooler has resolution of {binsize} bases,"
            " which is too coarse for automated kernel recommendation."
            " Provide custom kernels to proceed."
        )
    elif binsize >= 18000:
        w, p = 3, 1
    elif binsize >= 8000:
        w, p = 5, 2
    elif binsize >= 4000:
        w, p = 7, 4
    else:
        raise ValueError(
            f"Provided cooler has resolution {binsize} bases,"
            " which is too fine for automated kernel recommendation. Provide custom kernels"
            " to proceed."
        )
    logging.info(
        f"Using recommended donut-based kernels with w={w}, p={p} for binsize={binsize}"
    )
    # standard set of 4 kernels used in Rao et al 2014
    # 'upright' is a symmetrical inversion of "lowleft", not needed.
    kernel_types = ["donut", "vertical", "horizontal", "lowleft"]

    # generate standard kernels - consider providing custom ones
    kernels = {k: get_kernel(w, p, k) for k in kernel_types}

    return kernels


def is_compatible_kernels(kernels, binsize, max_nans_tolerated):
    """
    TODO implement checks for kernels:
     - matrices are of the same size
     - they should be squared (too restrictive ? maybe pad with 0 as needed)
     - dimensions are odd, to have a center pixel to refer to
     - they can be turned into int 1/0 ones (too restrictive ? allow weighted kernels ?)
     - the central pixel should be zero perhaps (unless weights are allowed 4sure)
     - maybe introduce an upper limit to the size - to avoid crazy long calculations
     - check relative to the binsize maybe ? what's the criteria ?
    """

    # kernels must be a dict with kernel-names as keys
    # and kernel ndarrays as values.
    if not isinstance(kernels, dict):
        raise ValueError(
            "'kernels' must be a dictionary" "with name-keys and ndarrays-values."
        )

    # deduce kernel_width - overall footprint
    kernel_widths = [len(k) for kn, k in kernels.items()]
    # kernels must have the same width for now:
    if min(kernel_widths) != max(kernel_widths):
        raise ValueError(f"all 'kernels' must have the same size, now: {kernel_widths}")
    # now extract their dimensions:
    kernel_width = max(kernel_widths)
    kernel_half_width = (kernel_width - 1) / 2  # former w parameter
    if (kernel_half_width <= 0) or not kernel_half_width.is_integer():
        raise ValueError(
            f"Size of the convolution kernels has to be odd and > 3, currently {kernel_width}"
        )

    # once kernel parameters are setup check max_nans_tolerated
    # to make sure kernel footprints overlaping 1 side with the
    # NaNs filled row/column are not "allowed"
    if not max_nans_tolerated <= kernel_width:
        raise ValueError(
            f"Too many NaNs allowed max_nans_tolerated={max_nans_tolerated}"
        )
    # may lead to scoring the same pixel twice, - i.e. duplicates.

    # return True if everyhting passes
    return True


def annotate_pixels_with_qvalues(pixels_df, qvalues, obs_raw_name=observed_count_name):
    """
    Add columns with the qvalues to a DataFrame of scored pixels

    Parameters
    ----------
    pixels_df : pandas.DataFrame
        a DataFrame with pixel coordinates that must have at least 2 columns
        named 'bin1_id' and 'bin2_id', where first is pixels's row and the
        second is pixel's column index.
    qvalues : dict of DataFrames
        A dictionary with keys being kernel names and values DataFrames
        storing q-values for each observed count values in each lambda-
        bin. Colunms are Intervals defined by 'ledges' boundaries.
        Rows corresponding to a range of observed count values.
    obs_raw_name : str
        Name of the column/field that carry number of counts per pixel,
        i.e. observed raw counts.

    Returns
    -------
    pixels_qvalue_df : pandas.DataFrame
        DataFrame of pixels with additional columns la_exp.{k}.qval,
        storing q-values (adjusted p-values) corresponding to the count
        value of a pixel, its kernel, and a lambda-bin it belongs to.
    """
    # do it "safe" - using a copy:
    pixels_qvalue_df = pixels_df.copy()
    # columns to return
    cols = list(pixels_qvalue_df.columns)
    # will do it efficiently using "melted" qvalues table:
    for k, qval_df in qvalues.items():
        lbins = pd.IntervalIndex(qval_df.columns)
        pixels_qvalue_df["lbins"] = pd.cut(
            pixels_qvalue_df[f"la_exp.{k}.value"], bins=lbins
        )
        pixels_qvalue_df = pixels_qvalue_df.merge(
            # melted qval_df columns: [counts, la_exp.k.value, value]
            qval_df.melt(ignore_index=False).reset_index(),
            left_on=[obs_raw_name, "lbins"],
            right_on=[obs_raw_name, f"la_exp.{k}.value"],
            suffixes=("", "_"),
        )
        qval_col_name = f"la_exp.{k}.qval"
        pixels_qvalue_df = pixels_qvalue_df.rename(columns={"value": qval_col_name})
        cols.append(qval_col_name)

    # return q-values annotated pixels
    return pixels_qvalue_df.loc[:, cols]


def clust_2D_pixels(
    pixels_df,
    threshold_cluster=2,
    bin1_id_name="bin1_id",
    bin2_id_name="bin2_id",
    clust_label_name="c_label",
    clust_size_name="c_size",
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

    # clustering message:
    logging.info(
        f"detected {uniq_counts.size} clusters of {uniq_counts.mean():.2f}+/-{uniq_counts.std():.2f} size"
    )

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


def tile_square_matrix(matrix_size, offset, tile_size, pad=0):
    """
    Generate a stream of coordinates of tiles that cover a matrix of a given size.
    Matrix has to be square, on-digaonal one: e.g. corresponding to a chromosome
    or a chromosomal arm.

    Parameters
    ----------
    matrix_size : int
        Size of a squared matrix
    offset : int
        Offset coordinates of generated tiles by 'offset'
    tile_size : int
        Requested size of the tiles. Tiles near
        the right and botoom edges could be rectangular
        and smaller then 'tile_size'
    pad : int
        Small padding around each tile to be included in the yielded coordinates.

    Yields
    ------
    Pairs of indices/coordinates of every tile: (start_i, end_i), (start_j, end_j)

    Notes
    -----
    Generated tiles coordinates [start_i,end_i) , [start_i,end_i)
    can be used to fetch heatmap tiles from cooler:
    >>> clr.matrix()[start_i:end_i, start_j:end_j]

    'offset' is useful when a given matrix is part of a
    larger matrix (a given chromosome or arm), and thus
    all coordinated needs to be offset to get absolute
    coordinates.

    Tiles are non-overlapping (pad=0), but tiles near
    the right and bottom edges could be rectangular:

    * * * * * * * * *
    *     *     *   *
    *     *     *   *
    * * * * * * *   *
    *     *         *
    *     *  ...    *
    * * * *         *
    *               *
    * * * * * * * * *
    """
    # number of tiles along each axis
    if matrix_size % tile_size:
        num_tiles = matrix_size // tile_size + 1
    else:
        num_tiles = matrix_size // tile_size

    logging.info(
        f" matrix {matrix_size}X{matrix_size} to be split into {num_tiles * num_tiles} tiles of {tile_size}X{tile_size}."
    )
    if pad:
        logging.info(
            f" tiles are padded (width={pad}) to enable convolution near the edges"
        )

    # generate 'num_tiles X num_tiles' tiles
    for ti in range(num_tiles):
        for tj in range(num_tiles):

            start_i = max(0, tile_size * ti - pad)
            start_j = max(0, tile_size * tj - pad)

            end_i = min(matrix_size, tile_size * (ti + 1) + pad)
            end_j = min(matrix_size, tile_size * (tj + 1) + pad)

            yield (start_i + offset, end_i + offset), (start_j + offset, end_j + offset)


def generate_tiles_diag_band(clr, view_df, pad_size, tile_size, band_to_cover):
    """
    A generator yielding corrdinates of heatmap tiles that are needed to cover
    the requested band_to_cover around diagonal. Each tile is "padded" with
    the pad of size 'pad_size' to allow for convolution near the boundary of
    a tile.

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
    tile_coords : tuple
        Generator of tile coordinates, i.e. tuples of three:
        (region_name, tile_span_i, tile_span_j), where 'tile_span_i/j'
        each is a tuple of bin ids (bin_start, bin_end).
    """

    for chrom, start, end, region_name in view_df.itertuples(index=False):
        region_start, region_end = clr.extent((chrom, start, end))
        region_size = region_end - region_start
        for tile_span_i, tile_span_j in tile_square_matrix(
            matrix_size=region_size,
            offset=region_start,
            tile_size=tile_size,
            pad=pad_size,
        ):
            # check if a given tile intersects with
            # with the diagonal band of interest ...
            tile_diag_start = tile_span_j[0] - tile_span_i[1]
            tile_diag_end = tile_span_j[1] - tile_span_i[0]
            # TODO allow more flexible definition of a band to cover
            band_start, band_end = 0, band_to_cover
            # we are using this >2*padding trick to exclude
            # tiles from the lower triangle from calculations ...
            if (
                min(band_end, tile_diag_end) - max(band_start, tile_diag_start)
            ) > 2 * pad_size:
                yield region_name, tile_span_i, tile_span_j


########################################################################
# this is the MAIN function to get locally adjusted expected
########################################################################
def get_adjusted_expected_tile_some_nans(
    origin_ij, observed, expected, bal_weights, kernels
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
    origin_ij : (int,int) tuple
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
        a matrix with the origin_ij on the diagonal;
        and a tuple/list of a couple of 1D arrays
        in case it is a slice with an arbitrary
        origin_ij.
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
        Note, scipy.ndimage.convolve first flips kernel
        and only then applies it to matrix.

    Returns
    -------
    peaks_df : pandas.DataFrame
        DataFrame with the results of locally adjusted calculations
        for every kernel for a given slice of input matrix.

    Notes
    -----

    Reported columns:
        bin1_id - bin1_id index (row), adjusted to tile_start_i
        bin2_id - bin bin2_id index, adjusted to tile_start_j
        la_exp - locally adjusted expected (for each kernel)
        la_nan - number of NaNs around (each kernel's footprint)
        exp.raw - global expected, rescaled to raw-counts
        obs.raw(counts) - observed values in raw-counts.

    Depending on the intial tiling of the interaction matrix,
    concatened `peaks_df` may require "deduplication", as some pixels
    can be evaluated in several tiles (e.g. near the tile edges).
    Default tilitng in the `dots` functions, should avoid this problem.

    """
    # extract origin_ij coordinate of this tile:
    io, jo = origin_ij
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
    # prepare matrix of balancing weights Ci*Cj
    bal_weights_ij = np.outer(v_bal_i, v_bal_j)

    # balanced observed, from raw-observed
    # by element-wise multiply:
    O_bal = np.multiply(O_raw, bal_weights_ij)
    # O_bal is separate from O_raw memory-wise.

    # fill lower triangle of O_bal and E_bal with NaNs
    # in order to prevent peak calling from the lower triangle
    # and also to provide fair locally adjusted expected
    # estimation for pixels very close to diagonal, whose
    # "donuts"(kernels) would be crossing the main diagonal.
    # The trickiest thing here would be dealing with the origin_ij: io,jo.
    O_bal[np.tril_indices_from(O_bal, k=(io - jo) - 1)] = np.nan
    E_bal[np.tril_indices_from(E_bal, k=(io - jo) - 1)] = np.nan

    # raw E_bal: element-wise division of E_bal[i,j] and
    # v_bal[i]*v_bal[j]:
    E_raw = np.divide(E_bal, bal_weights_ij)

    # let's calculate a matrix of common NaNs
    # shared between observed and expected:
    # check if it's redundant ? (is NaNs from O_bal sufficient? )
    N_bal = np.logical_or(np.isnan(O_bal), np.isnan(E_bal))
    # fill in common nan-s with zeroes, preventing
    # NaNs during convolution:
    O_bal[N_bal] = 0.0
    E_bal[N_bal] = 0.0
    # think about usinf copyto and where functions later:
    # https://stackoverflow.com/questions/6431973/how-to-copy-data-from-a-numpy-array-to-another
    #
    # we are going to accumulate all the results
    # into a DataFrame, keeping NaNs, and other
    # unfiltered results (even the lower triangle for now):
    i, j = np.indices(O_raw.shape)
    # pack it into DataFrame to accumulate results:
    peaks_df = pd.DataFrame({"bin1_id": i.ravel() + io, "bin2_id": j.ravel() + jo, "count": O_raw.ravel()})

    with np.errstate(divide="ignore", invalid="ignore"):
        for kernel_name, kernel in kernels.items():
            ###############################
            # kernel-specific calculations:
            ###############################
            # kernel paramters such as width etc
            # are taken into account implicitly ...
            ########################################
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
                (kernel != 0).astype(np.int64),  # non-zero footprint as kernel
                mode="constant",
                cval=1,  # NaNs beyond boundaries
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
            local_adjustment_factor = np.divide(KO, KE)
            Ek_raw = np.multiply(E_raw, local_adjustment_factor)

            logging.debug(
                f"Convolution with kernel {kernel_name} is done for tile @ {io} {jo}."
            )
            # accumulation into single DataFrame:
            # store locally adjusted expected for each kernel
            # and number of NaNs in the footprint of each kernel
            peaks_df[f"la_exp.{kernel_name}.value"] = Ek_raw.ravel()
            peaks_df[f"la_exp.{kernel_name}.nnans"] = NN.ravel()
            # division by KE=0 has to be treated separately:
            peaks_df[f"safe_division.{kernel_name}"] = np.isfinite(local_adjustment_factor.ravel())
            # do all the filter/logic/masking etc on the complete DataFrame ...

    return peaks_df


##################################
# step-specific dot-calling functions
##################################
def score_tile(
    tile_cij,
    clr,
    expected_indexed,
    expected_value_col,
    clr_weight_name,
    kernels,
    max_nans_tolerated,
    band_to_cover,
):
    """
    The main working function that given a tile of a heatmap, applies kernels to
    perform convolution to calculate locally-adjusted expected and then
    calculates a p-value for every meaningfull pixel against these
    locally-adjusted expected (la_exp) values.

    Parameters
    ----------
    tile_cij : tuple
        Tuple of 3: region name, tile span row-wise, tile span column-wise:
        (region, tile_span_i, tile_span_j), where tile_span_i = (start_i, end_i), and
        tile_span_j = (start_j, end_j).
    clr : cooler
        Cooler object to use to extract Hi-C heatmap data.
    expected_indexed : pandas.DataFrame
        DataFrame with cis-expected, indexed with 'region1', 'region2', 'dist'.
    expected_value_col : str
        Name of a value column in expected DataFrame
    clr_weight_name : str
        Name of a value column with balancing weights in a cooler.bins()
        DataFrame. Typically 'weight'.
    kernels : dict
        A dictionary with keys being kernels names and values being ndarrays
        representing those kernels.
    max_nans_tolerated : int
        Number of NaNs tolerated in a footprint of every kernel.
    band_to_cover : int
        Results would be stored only for pixels connecting loci closer than
        'band_to_cover'.

    Returns
    -------
    res_df : pandas.DataFrame
        results: annotated pixels with calculated locally adjusted expected
        for every kernels, observed, precalculated pvalues, number of NaNs in
        footprint of every kernels, all of that in a form of an annotated
        pixels DataFrame for eligible pixels of a given tile.

    """
    # unpack tile's coordinates
    region_name, tile_span_i, tile_span_j = tile_cij
    tile_start_ij = (tile_span_i[0], tile_span_j[0])

    # we have to do it for every tile, because
    # region_name is not known apriori (maybe move outside)
    # use .loc[region, region] for symmetric cis regions to conform with expected v1.0
    lazy_exp = LazyToeplitz(
        expected_indexed.loc[region_name, region_name][expected_value_col].to_numpy()
    )

    # RAW observed matrix slice:
    observed = clr.matrix(balance=False)[slice(*tile_span_i), slice(*tile_span_j)]
    # expected as a rectangular tile :
    expected = lazy_exp[slice(*tile_span_i), slice(*tile_span_j)]
    # slice of balance_weight for row-span and column-span :
    bal_weight_i = clr.bins()[slice(*tile_span_i)][clr_weight_name].to_numpy()
    bal_weight_j = clr.bins()[slice(*tile_span_j)][clr_weight_name].to_numpy()

    # do the convolutions
    result = get_adjusted_expected_tile_some_nans(
        origin_ij=tile_start_ij,
        observed=observed,
        expected=expected,
        bal_weights=(bal_weight_i, bal_weight_j),
        kernels=kernels,
    )

    # Post-processing filters
    # (0) keep only upper-triangle pixels:
    upper_band = result["bin1_id"] < result["bin2_id"]

    # (1) exclude pixels that connect loci further than 'band_to_cover' apart:
    is_inside_band = result["bin1_id"] > (result["bin2_id"] - band_to_cover)

    # (2) identify pixels that pass number of NaNs compliance test for ALL kernels:
    does_comply_nans = np.all(
        result[[f"la_exp.{k}.nnans" for k in kernels]] < max_nans_tolerated, axis=1
    )

    # (3) keep pixels without nan/infinite local adjustment factors for all kernel
    finite_values_only = result[[f"safe_division.{k}" for k in kernels]].all(axis="columns")

    # so, selecting inside band and nNaNs compliant results:
    res_df = result[upper_band & is_inside_band & does_comply_nans & finite_values_only].reset_index(
        drop=True
    )
    #
    # so return only bare minimum: bin_ids , observed-raw counts
    # and locally adjusted expected estimates for every kernel
    return res_df[
        ["bin1_id", "bin2_id", "count"] + [f"la_exp.{k}.value" for k in kernels]
    ].astype(dtype={f"la_exp.{k}.value": "float64" for k in kernels})


def histogram_scored_pixels(
    scored_df, kernels, ledges, obs_raw_name=observed_count_name
):
    """
    An attempt to implement HiCCUPS-like lambda-binning statistical procedure.
    This function aims at building up a histogram of locally adjusted
    expected scores for groups of characterized pixels.

    Such histograms are subsequently used to compute FDR thresholds
    for different "classes" of hypothesis (classified by their
    locally-adjusted expected (la_exp)).

    Parameters
    ----------
    scored_df : pd.DataFrame
        A table with the scoring information for a group of pixels.
    kernels : dict
        A dictionary with keys being kernels names and values being ndarrays
        representing those kernels.
    ledges : ndarray
        An ndarray with bin lambda-edges for groupping locally adjusted
        expecteds, i.e., classifying statistical hypothesis into lambda-bins.
        Left-most bin (-inf, 1], and right-most one (value,+inf].
    obs_raw_name : str
        Name of the column/field that carry number of counts per pixel,
        i.e. observed raw counts.

    Returns
    -------
    hists : dict of pandas.DataFrame
        A dictionary of pandas.DataFrame with lambda/observed 2D histogram for
        every kernel-type.


    Notes
    -----
    returning histograms corresponding to the chunks of scored pixels.
    """

    hists = {}
    for k in kernels:
        #  we would need to generate a bunch of these histograms for all of the
        # kernel types:
        # needs to be lambda-binned             : scored_df["la_exp."+k+".value"]
        # needs to be histogrammed in every bin : scored_df["obs.raw"]
        #
        # lambda-bin index for kernel-type "k":
        lbins = pd.cut(scored_df[f"la_exp.{k}.value"], ledges)
        # group scored_df by counts and lambda-bins to contructs histograms:
        hists[k] = (
            scored_df.groupby([obs_raw_name, lbins], dropna=False, observed=False)[
                f"la_exp.{k}.value"
            ]
            .count()
            .unstack()
            .fillna(0)
            .astype(np.int64)
        )
    # return a dict of DataFrames with a bunch of histograms:
    return hists


def determine_thresholds(gw_hist, fdr):
    """
    given a 'gw_hist' histogram of observed counts
    for each lambda-bin and for each kernel-type, and
    also given a FDR, calculate q-values for each observed
    count value in each lambda-bin for each kernel-type.

    Parameters
    ----------
    gw_hist_kernels : dict
        dictionary {kernel_name : 2D_hist}, where '2D_hist' is a pd.DataFrame
    fdr : float
        False Discovery Rate level

    Returns
    -------
    threshold_df : dict
      each threshold_df[k] is a Series indexed by la_exp intervals
      (IntervalIndex) and it is all we need to extract "good" pixels from
      each chunk ...
    qvalues : dict
      A dictionary with keys being kernel names and values pandas.DataFrames
      storing q-values: each column corresponds to a lambda-bin,
      while rows correspond to observed pixels values.


    """

    # extracting significantly enriched interactions:
    # We have our *null* hypothesis: intensity of a HiC pixel is Poisson-distributed
    # with a certain expected. In this case that would be *locally-adjusted expected*.
    #
    # Thus for dot-calling, we could estimate a *p*-value for every pixel based
    # on its observed intensity and its expected intensity, e.g.:
    # lambda = la_exp["la_exp."+k+".value"]; pvals = 1.0 - poisson.cdf(la_exp["count"], lambda)
    # However this is technically challenging (too many pixels - genome wide) and
    # might not be sensitive enough due to wide dyamic range of interaction counts
    # Instead we use the *lambda*-binning procedure from Rao et al 2014 to tackle
    # both technicall challenges and some issues associated with the wide dynamic range
    # of the expected for the dot-calling (due to distance decay).
    #
    # Some extra points:
    # 1. simple p-value thresholding should be replaced to more "productive" FDR, which is more tractable
    # 2. "unfair" to treat all pixels with the same stat-testing (multiple hypothesis) - too wide range of "expected"
    # 3. (2) is addressed by spliting the pixels in the groups by their localy adjusted expected - lambda-bins
    # 4. upper boundary of each lambda-bin is used as expected for every pixel that belongs to the chunk:
    #                   - for technical/efficiency reasons - test pixels in a chunk all at once
    # for each lambda-bin q-values are calculated in an efficient way:
    # in part, efficiency comes from collapsing identical observed values, i.e. histogramming
    # also upper boundary of each lambda-bin is used as an expected for every pixel in this lambda-bin

    qvalues = {}
    threshold_df = {}
    for k, _hist in gw_hist.items():
        # Reverse cumulative histogram for kernel 'k'.
        rcs_hist = _hist.iloc[::-1].cumsum(axis=0).iloc[::-1]
        # 1st row of 'rcs_hist' contains total pixels-counts in each lambda-bin.
        norm = rcs_hist.iloc[0, :]

        # Assign a unit Poisson distribution to each lambda-bin.
        # The expected value 'lambda' is the upper boundary of each lambda-bin:
        #   poisson.sf = 1 - poisson.cdf, but more precise
        #   poisson.sf(-1, lambda) == 1.0, i.e. is equivalent to the
        #   poisson.pmf(rcs_hist.index, lambda)[::-1].cumsum()[::-1]
        # unit Poisson is a collection of 1-CDF distributions for each l-chunk
        # same dimensions as rcs_hist - matching lchunks and observed values:
        unit_Poisson = pd.DataFrame().reindex_like(rcs_hist)
        for lbin in rcs_hist.columns:
            # Number of occurances in Poisson distribution for which we estimate
            _occurances = rcs_hist.index.to_numpy()
            unit_Poisson[lbin] = poisson.sf(_occurances, lbin.right)
        # normalize unit-Poisson distr for the total pixel counts per lambda-bin
        unit_Poisson = norm * unit_Poisson

        # Determine the threshold by checking the value at which 'fdr_diff'
        # first turns positive. Fill NaNs with a high value, that's out of reach.
        _high_value = rcs_hist.index.max() + 1
        fdr_diff = ((fdr * rcs_hist) - unit_Poisson).cummax()
        # cummax ensures monotonic increase of differences
        threshold_df[k] = (
            fdr_diff.mask(fdr_diff < 0)  # mask negative with nans
            .idxmin()  # index of the first positive difference
            .fillna(_high_value)  # set high threshold if no pixels pass
            .astype(np.int64)
        )
        qvalues[k] = (unit_Poisson / rcs_hist).cummin()
        # run cumulative min, on the array of adjusted p-values
        # to make sure q-values are monotonously decreasing with pvals
        qvalues[k] = qvalues[k].mask(qvalues[k] > 1.0, 1.0)
        # cast categorical index of dtype-interval to proper interval index
        threshold_df[k].index = pd.IntervalIndex(threshold_df[k].index)

    return threshold_df, qvalues


def extract_scored_pixels(scored_df, thresholds, obs_raw_name=observed_count_name):
    """
    Implementation of HiCCUPS-like lambda-binning statistical procedure.
    Use FDR thresholds for different "classes" of hypothesis
    (classified by their locally-adjusted expected (la_exp) scores),
    in order to extract "enriched" pixels.

    Parameters
    ----------
    scored_df : pd.DataFrame
        A table with the scoring information for a group of pixels.
    thresholds : dict
        A dictionary {kernel_name : lambda_thresholds}, where 'lambda_thresholds'
        are pd.Series with FDR thresholds indexed by lambda-bin intervals
    obs_raw_name : str
        Name of the column/field with number of counts per pixel,
        i.e. observed raw counts.

    Returns
    -------
    scored_df_slice : pandas.DataFrame
        Filtered DataFrame of pixels that satisfy thresholds.

    """
    compliant_pixel_masks = []
    for kernel_name, threshold in thresholds.items():
        # locally adjusted expected (lambda) of the scored pixels:
        lambda_of_pixels = scored_df[f"la_exp.{kernel_name}.value"]
        # reconstruct edges of lambda bins from threshold's index:
        ledges_reconstruct = np.r_[threshold.index.left, threshold.index.right[-1]]
        # find indices of lambda-bins where pixels belong
        lbin_idx = pd.cut(lambda_of_pixels, ledges_reconstruct, labels=False)
        # extract threholds for every pixel, based on lambda-bin each of the belongs
        threshold_of_pixels = threshold.iloc[lbin_idx]
        compliant_pixel_masks.append(
            scored_df[obs_raw_name].to_numpy() >= threshold_of_pixels.to_numpy()
        )
    # return pixels from 'scored_df' that satisfy FDR thresholds for all kernels:
    return scored_df[np.all(compliant_pixel_masks, axis=0)]


def clustering_step(
    scored_df,
    dots_clustering_radius,
    assigned_regions_name="region",
    obs_raw_name=observed_count_name,
):
    """
    Group together adjacent significant pixels into clusters after
    the lambda-binning multiple hypothesis testing by iterating over
    assigned regions and calling `clust_2D_pixels`.

    Parameters
    ----------
    scored_df : pandas.DataFrame
        DataFrame with enriched pixels that are ready to be
        clustered and are annotated with their genomic  coordinates.
    dots_clustering_radius : int
        Birch-clustering threshold.
    assigned_regions_name : str | None
        Name of the column in scored_df to use for grouping pixels
        before clustering. When None, full chromosome clustering is done.
    obs_raw_name : str
        name of the column with raw observed pixel counts
    Returns
    -------
    centroids : pandas.DataFrame
        Pixels from 'scored_df' annotated with clustering information.

    Notes
    -----
    'dots_clustering_radius' in Birch clustering algorithm corresponds to a
    double the clustering radius in the "greedy"-clustering used in HiCCUPS

    """
    # make sure provided pixels are annotated with genomic corrdinates and raw counts column is present:
    if not {"chrom1", "chrom2", "start1", "start2", obs_raw_name}.issubset(scored_df):
        raise ValueError("Scored pixels provided for clustering are not annotated")

    scored_df = scored_df.copy()
    if (
        not assigned_regions_name in scored_df.columns
    ):  # If input scores are not annotated by regions:
        logging.warning(
            f"No regions assigned to the scored pixels before clustering, using chromosomes"
        )
        scored_df[assigned_regions_name] = np.where(
            scored_df["chrom1"] == scored_df["chrom2"], scored_df["chrom1"], np.nan
        )

    # cluster within each regions separately and accumulate the result:
    pixel_clust_list = []
    scored_pixels_by_region = scored_df.groupby(assigned_regions_name, observed=True)
    for region, _df in scored_pixels_by_region:
        logging.info(f"clustering enriched pixels in region: {region}")
        # Using genomic corrdinated for clustering, not bin_id
        pixel_clust = clust_2D_pixels(
            _df,
            threshold_cluster=dots_clustering_radius,
            bin1_id_name="start1",
            bin2_id_name="start2",
        )
        pixel_clust_list.append(pixel_clust)
    logging.info("Clustering is complete")

    # concatenate clustering results ...
    # indexing information persists here ...
    if not pixel_clust_list:
        logging.warning("No clusters found for any regions! Output will be empty")
        empty_output = pd.DataFrame(
            [],
            columns=list(scored_df.columns)
            + [
                assigned_regions_name + "1",
                assigned_regions_name + "2",
                "c_label",
                "c_size",
                "cstart1",
                "cstart2",
            ],
        )
        return empty_output  # Empty dataframe with the same columns as anticipated
    else:
        pixel_clust_df = pd.concat(
            pixel_clust_list, ignore_index=False
        )  # Concatenate the clustering results for different regions

    # now merge pixel_clust_df and scored_df DataFrame ...
    # TODO make a more robust merge here
    df = pd.merge(
        scored_df, pixel_clust_df, how="left", left_index=True, right_index=True
    )
    # TODO check if next str-cast is neccessary
    df[assigned_regions_name + "1"] = df[assigned_regions_name].astype(str)
    df[assigned_regions_name + "2"] = df[assigned_regions_name].astype(str)
    # report only centroids with highest Observed:
    chrom_clust_group = df.groupby(
        [assigned_regions_name + "1", assigned_regions_name + "2", "c_label"],
        observed=True,
    )
    centroids = df.loc[
        chrom_clust_group[obs_raw_name].idxmax()
    ]  # Select the brightest pixel in the cluster
    return centroids


def cluster_filtering_hiccups(
    centroids,
    obs_raw_name=observed_count_name,
    enrichment_factor_vh=1.5,
    enrichment_factor_d_and_ll=1.75,
    enrichment_factor_d_or_ll=2.0,
    FDR_orphan_threshold=0.02,
):
    """
    Centroids of enriched pixels can be filtered to further minimize
    the amount of false-positive dot-calls.

    First, centroids are filtered on enrichment relative to the
    locally-adjusted expected for the "donut", "lowleft", "vertical",
    and "horizontal" kernels. Additionally, singleton pixels
    (i.e. pixels that do not belong to a cluster) are filtered based on
    a combined q-values for all kernels. This empirical filtering approach
    was developed in Rao et al 2014 and results in a conservative dot-calls
    with the low rate of false-positive calls.

    Parameters
    ----------
    centroids : pd.DataFrame
        DataFrame that stores enriched and clustered pixels.
    obs_raw_name : str
        name of the column with raw observed pixel counts
    enrichment_factor_vh : float
        minimal enrichment factor for pixels relative to
        both "vertical" and "horizontal" kernel.
    enrichment_factor_d_and_ll : float
        minimal enrichment factor for pixels relative to
        both "donut" and "lowleft" kernels.
    enrichment_factor_d_or_ll : float
        minimal enrichment factor for pixels relative to
        either "donut" or" "lowleft" kenels.
    FDR_orphan_threshold : float
        minimal combined q-value for singleton pixels.

    Returns
    -------
    filtered_centroids : pd.DataFrame
        filtered dot-calls
    """
    # make sure input DataFrame of pixels has been clustered:
    if not "c_size" in centroids:
        raise ValueError(f"input dataframe of pixels does not seem to be clustered")

    # make sure input DataFrame of pixels has been annotated with genomic coordinates and raw counts column is present:
    if not {"chrom1", "chrom2", "start1", "start2", obs_raw_name}.issubset(centroids):
        raise ValueError(
            "input dataframe of clustered pixels provided for filtering is not annotated"
        )

    # make sureinput DataFrame of pixels was scored using 4-hiccups kernels (donut, lowleft, vertical, horizontal):
    _hiccups_kernel_cols_set = {
        "la_exp.donut.value",
        "la_exp.vertical.value",
        "la_exp.horizontal.value",
        "la_exp.lowleft.value",
        # and q-values as well
        "la_exp.donut.qval",
        "la_exp.vertical.qval",
        "la_exp.horizontal.qval",
        "la_exp.lowleft.qval",
    }
    # make sure input DataFrame of pixels has been annotated with genomic coordinates and raw counts column is present:
    if not _hiccups_kernel_cols_set.issubset(centroids):
        raise ValueError(
            "clustered pixels provided for filtering were not scored with 4 hiccups kernels"
        )

    # ad hoc filtering by enrichment, FDR for singletons etc.
    # employed in Rao et al 2014 HiCCUPS
    enrichment_fdr_comply = (
        (
            centroids[obs_raw_name]
            > enrichment_factor_d_and_ll * centroids["la_exp.lowleft.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_d_and_ll * centroids["la_exp.donut.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_vh * centroids["la_exp.vertical.value"]
        )
        & (
            centroids[obs_raw_name]
            > enrichment_factor_vh * centroids["la_exp.horizontal.value"]
        )
        & (
            (
                centroids[obs_raw_name]
                > enrichment_factor_d_or_ll * centroids["la_exp.lowleft.value"]
            )
            | (
                centroids[obs_raw_name]
                > enrichment_factor_d_or_ll * centroids["la_exp.donut.value"]
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

    logging.info(
        f"filtered {enrichment_fdr_comply.sum()} out of {len(centroids)} centroids to reduce the number of false-positives"
    )

    # use "enrichment_fdr_comply" to filter out non-satisfying pixels:
    return centroids[enrichment_fdr_comply].reset_index(drop=True)


####################################################################
# large helper functions wrapping smaller step-specific ones
####################################################################


def scoring_and_histogramming_step(
    clr,
    expected_indexed,
    expected_value_col,
    clr_weight_name,
    tiles,
    kernels,
    ledges,
    max_nans_tolerated,
    loci_separation_bins,
    nproc,
):
    """
    This implements the 1st step of the lambda-binning scoring procedure - histogramming.

    In short, this pipes a scoring operation together with histogramming into a
    single pipeline of per-chunk operations/transforms.
    """
    logging.info(f"convolving {len(tiles)} tiles to build histograms for lambda-bins")

    # to score per tile:
    to_score = partial(
        score_tile,
        clr=clr,
        expected_indexed=expected_indexed,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        kernels=kernels,
        max_nans_tolerated=max_nans_tolerated,
        band_to_cover=loci_separation_bins,
    )

    # to hist per scored chunk:
    to_hist = partial(histogram_scored_pixels, kernels=kernels, ledges=ledges)

    # compose scoring and histogramming together :
    job = lambda tile: to_hist(to_score(tile))

    # standard multiprocessing implementation
    if nproc > 1:
        logging.info(f"creating a Pool of {nproc} workers to tackle {len(tiles)} tiles")
        pool = mp.Pool(nproc)
        map_ = pool.imap
        map_kwargs = dict(chunksize=int(np.ceil(len(tiles) / nproc)))
    else:
        logging.info("fallback to serial implementation.")
        map_ = map
        map_kwargs = {}
    try:
        # consider using
        # https://github.com/mirnylab/cooler/blob/9e72ee202b0ac6f9d93fd2444d6f94c524962769/cooler/tools.py#L59
        histogram_chunks = map_(job, tiles, **map_kwargs)
    finally:
        if nproc > 1:
            pool.close()

    # now we need to combine/sum all of the histograms for different kernels:
    def _sum_hists(hx, hy):
        # perform a DataFrame summation for every value of the dictionary:
        hxy = {}
        for k in kernels:
            hxy[k] = hx[k].add(hy[k], fill_value=0).fillna(0).astype(np.int64)
        # returning the sum:
        return hxy

    # TODO consider more efficient implementation to accumulate histograms:
    final_hist = reduce(_sum_hists, histogram_chunks)
    # we have to make sure there is nothing in the last lambda-bin
    # this is a temporary implementation detail, until we implement dynamic lambda-bins
    for k in kernels:
        last_lambda_bin = final_hist[k].iloc[:, -1]
        if last_lambda_bin.sum() != 0:
            raise ValueError(
                f"There are la_exp.{k}.value in {last_lambda_bin.name}, please check the histogram"
            )
        # drop all lambda-bins that do not have pixels in them:
        final_hist[k] = final_hist[k].loc[:, final_hist[k].sum() > 0]
        # make sure index (observed pixels counts) is sorted
        if not final_hist[k].index.is_monotonic_increasing:
            raise ValueError(f"Histogram for {k}-kernel is not sorted")
    # returning filtered histogram
    return final_hist


def scoring_and_extraction_step(
    clr,
    expected_indexed,
    expected_value_col,
    clr_weight_name,
    tiles,
    kernels,
    ledges,
    thresholds,
    max_nans_tolerated,
    loci_separation_bins,
    nproc,
    bin1_id_name="bin1_id",
    bin2_id_name="bin2_id",
):
    """
    This implements the 2nd step of the lambda-binning scoring procedure,
    extracting pixels that are FDR compliant.

    In short, this combines scoring with with extraction into a
    single pipeline of per-chunk operations/transforms.

    """
    logging.info(f"convolving {len(tiles)} tiles to extract enriched pixels")

    # to score per tile:
    to_score = partial(
        score_tile,
        clr=clr,
        expected_indexed=expected_indexed,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        kernels=kernels,
        max_nans_tolerated=max_nans_tolerated,
        band_to_cover=loci_separation_bins,
    )

    # to hist per scored chunk:
    to_extract = partial(
        extract_scored_pixels,
        thresholds=thresholds,
    )

    # compose scoring and histogramming together
    job = lambda tile: to_extract(to_score(tile))

    # standard multiprocessing implementation
    if nproc > 1:
        logging.info(f"creating a Pool of {nproc} workers to tackle {len(tiles)} tiles")
        pool = mp.Pool(nproc)
        map_ = pool.imap
        map_kwargs = dict(chunksize=int(np.ceil(len(tiles) / nproc)))
    else:
        logging.info("fallback to serial implementation.")
        map_ = map
        map_kwargs = {}
    try:
        # consider using
        # https://github.com/mirnylab/cooler/blob/9e72ee202b0ac6f9d93fd2444d6f94c524962769/cooler/tools.py#L59
        filtered_pix_chunks = map_(job, tiles, **map_kwargs)
        significant_pixels = pd.concat(filtered_pix_chunks, ignore_index=True)
    finally:
        if nproc > 1:
            pool.close()
    # same pixels should never be scored >1 times with the current tiling of the interactions matrix
    if significant_pixels.duplicated().any():
        raise ValueError(
            f"Some pixels were scored more than one time, matrix tiling procedure is not correct"
        )
    # sort the result just in case and drop its index:
    return significant_pixels.sort_values(by=[bin1_id_name, bin2_id_name]).reset_index(
        drop=True
    )


# user-friendly high-level API function
def dots(
    clr,
    expected,
    expected_value_col="balanced.avg",
    clr_weight_name="weight",
    view_df=None,
    kernels=None,
    max_loci_separation=10_000_000,
    max_nans_tolerated=1,  # test if this has desired behavior
    n_lambda_bins=40,  # update this eventually
    lambda_bin_fdr=0.1,
    clustering_radius=20_000,
    cluster_filtering=None,
    tile_size=5_000_000,
    nproc=1,
):
    """
    Call dots on a cooler {clr}, using {expected} defined in regions specified
    in {view_df}.

    All convolution kernels specified in {kernels} will be all applied to the {clr},
    and statistical testing will be performed separately for each kernel. A convolutional
    kernel is a small squared matrix (e.g. 7x7) of zeros and ones
    that defines a "mask" to extract local expected around each pixel. Since the
    enrichment is calculated relative to the central pixel, kernel width should
    be an odd number >=3.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data.
    expected : DataFrame in expected format
        Diagonal summary statistics for each chromosome, and name of the column
        with the values of expected to use.
    expected_value_col : str
        Name of the column in expected that holds the values of expected
    clr_weight_name : str
        Name of the column in the clr.bins to use as balancing weights.
        Using raw unbalanced data is not supported for dot-calling.
    view_df : viewframe
        Viewframe with genomic regions, at the moment the view has to match the
        view used for generating expected. If None, generate from the cooler.
    kernels : { str:np.ndarray } | None
        A dictionary of convolution kernels to be used for calculating locally adjusted
        expected. If None the default kernels from HiCCUPS are going to be recommended
        based on the resolution of the cooler.
    max_loci_separation : int
        Miaximum loci separation for dot-calling, i.e., do not call dots for
        loci that are further than max_loci_separation basepair apart. default 10Mb.
    max_nans_tolerated : int
        Maximum number of NaNs tolerated in a footprint of every used kernel
        Adjust with caution, as large max_nans_tolerated, might lead to artifacts in
        pixels scoring.
    n_lambda_bins : int
        Number of log-spaced bins, where FDR-testing will be performed independently.
        TODO: generate lambda-bins on the fly based on the dynamic range of the data (i.e. maximum pixel count)
    lambda_bin_fdr : float
        False discovery rate (FDR) for multiple hypothesis testing BH-FDR procedure, applied per lambda bin.
    clustering_radius : None | int
        Cluster enriched pixels with a given radius. "Brightest" pixels in each group
        will be reported as the final dot-calls. If None, no clustering is performed.
    cluster_filtering : bool
        whether to apply additional filtering to centroids after clustering, using cluster_filtering_hiccups()
    tile_size : int
        Tile size for the Hi-C heatmap tiling. Typically on order of several mega-bases, and <= max_loci_separation.
        Controls tradeoff between memory consumption and speed of execution.
    nproc : int
        Number of processes to use for multiprocessing.

    Returns
    -------
    dots : pandas.DataFrame
        BEDPE-style dataFrame with genomic coordinates of called dots and additional annotations.

    Notes
    -----
    'clustering_radius' in Birch clustering algorithm corresponds to a
    double the clustering radius in the "greedy"-clustering used in HiCCUPS
    (to be tested).

    TODO describe sequence of processing steps

    """

    #### Generate viewframes ####
    if view_df is None:
        view_df = make_cooler_view(clr)
    else:
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # check balancing status
    if clr_weight_name:
        # check if cooler is balanced
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e
    else:
        raise ValueError("calling dots on raw data is not supported.")

    # add checks to make sure cis-expected is symmetric
    # make sure provided expected is compatible
    try:
        _ = is_valid_expected(
            expected,
            "cis",
            view_df,
            verify_cooler=clr,
            expected_value_cols=[
                expected_value_col,
            ],
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided expected is not compatible") from e
    expected = expected.set_index(["region1", "region2", "dist"]).sort_index()

    # Prepare some parameters.
    binsize = clr.binsize
    loci_separation_bins = bp_to_bins(max_loci_separation, binsize)
    tile_size_bins = bp_to_bins(tile_size, binsize)

    # verify provided kernels or recommend them (HiCCUPS)...
    if kernels and is_compatible_kernels(kernels, binsize, max_nans_tolerated):
        warnings.warn(
            "Compatibility checks for 'kernels' are not fully implemented yet, use at your own risk"
        )
    else:
        # recommend them (default hiccups ones for now)
        kernels = recommend_kernels(binsize)
    # deduce kernel_width - overall footprint
    kernel_width = max(len(k) for k in kernels.values())  # 2*w+1
    kernel_half_width = int((kernel_width - 1) / 2)  # former w parameter

    # try to guess required lambda bins using "max" value of pixel counts
    # statistical: lambda-binning edges ...
    if not 40 <= n_lambda_bins <= 50:
        raise ValueError(f"Incompatible n_lambda_bins={n_lambda_bins}")
    BASE = 2 ** (1 / 3)  # very arbitrary - parameterize !
    ledges = np.concatenate(
        (
            [-np.inf],
            np.logspace(
                0,
                n_lambda_bins - 1,
                num=n_lambda_bins,
                base=BASE,
                dtype=np.float64,
            ),
            [np.inf],
        )
    )

    # list of tile coordinate ranges
    tiles = list(
        generate_tiles_diag_band(
            clr, view_df, kernel_half_width, tile_size_bins, loci_separation_bins
        )
    )

    # 1. Calculate genome-wide histograms of scores.
    time_start = time.perf_counter()
    gw_hist = scoring_and_histogramming_step(
        clr,
        expected,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        tiles=tiles,
        kernels=kernels,
        ledges=ledges,
        max_nans_tolerated=max_nans_tolerated,
        loci_separation_bins=loci_separation_bins,
        nproc=nproc,
    )
    elapsed_time = time.perf_counter() - time_start
    logging.info(f"Done building histograms in {elapsed_time:.3f} sec ...")

    # 2. Determine the FDR thresholds.
    threshold_df, qvalues = determine_thresholds(gw_hist, lambda_bin_fdr)
    logging.info("Determined thresholds for every lambda-bin ...")

    # 3. Filter using FDR thresholds calculated in the histogramming step
    time_start = time.perf_counter()
    filtered_pixels = scoring_and_extraction_step(
        clr,
        expected,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        tiles=tiles,
        kernels=kernels,
        ledges=ledges,
        thresholds=threshold_df,
        max_nans_tolerated=max_nans_tolerated,
        loci_separation_bins=loci_separation_bins,
        nproc=nproc,
        bin1_id_name="bin1_id",
        bin2_id_name="bin2_id",
    )
    elapsed_time = time.perf_counter() - time_start
    logging.info(f"Done extracting enriched pixels in {elapsed_time:.3f} sec ...")

    # 4. Post-processing
    logging.info(f"Begin post-processing of {len(filtered_pixels)} filtered pixels")
    logging.info("preparing to extract needed q-values ...")

    # annotate enriched pixels
    filtered_pixels_qvals = annotate_pixels_with_qvalues(filtered_pixels, qvalues)
    filtered_pixels_annotated = cooler.annotate(
        filtered_pixels_qvals, clr.bins()[["chrom", "start", "end"]], replace=True
    )
    if not clustering_radius:
        # TODO: make sure returned DataFrame has the same columns as "postprocessed_calls"
        # columns to return before-clustering
        output_cols = []
        output_cols += bedpe_required_cols
        output_cols += [
            observed_count_name,
        ]
        output_cols += [f"la_exp.{k}.value" for k in kernels]
        output_cols += [f"la_exp.{k}.qval" for k in kernels]
        return filtered_pixels_annotated[output_cols]

    # 4a. clustering
    # Clustering is done independently for every region, therefore regions must be assigned:
    filtered_pixels_annotated = assign_regions(filtered_pixels_annotated, view_df)
    centroids = clustering_step(
        filtered_pixels_annotated,
        clustering_radius,
    ).reset_index(drop=True)

    # columns to return post-clustering
    output_cols = []
    output_cols += bedpe_required_cols
    output_cols += [
        "cstart1",
        "cstart2",
        "c_label",
        "c_size",
        observed_count_name,
    ]
    output_cols += [f"la_exp.{k}.value" for k in kernels]
    output_cols += [f"la_exp.{k}.qval" for k in kernels]

    # 4b. filter by enrichment and qval
    if (cluster_filtering is None) or cluster_filtering:
        # default - engage HiCCUPS filtering
        postprocessed_calls = cluster_filtering_hiccups(centroids)
    elif not cluster_filtering:
        postprocessed_calls = centroids

    return postprocessed_calls
