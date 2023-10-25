import numpy as np
from toolz import curry
import cooltools
import cooler
import functools
from multiprocessing import Pool
import logging
from cooltools.lib.common import pool_decorator

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s : %(message)s")
logger = logging.getLogger("data_util")
logger.propagate = False  # Disable propagation to the root logger

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

@curry
def cis_total_ratio_filter(clr, threshold=0.5):
    """
    Filter out bins with low cis-to-total coverage ratio from a Cooler object.

    Parameters
    ----------
    clr : cooler.Cooler
        A Cooler object containing Hi-C contact matrices.
    threshold : float
        The threshold cis-to-total coverage ratio below which bins are considered bad.

    Returns
    -------
    numpy.ndarray
        An array of bin mask.

    Note
    ----
    This curried function accepts partial evaluation with only providing threshold value.
    """
    if isinstance(clr, float):
        raise TypeError(
            "If only threshold value is provided, please use 'threshold' keyword to set threshold value (e.g. threshold=0.2)"
        )
    coverage = cooltools.coverage(clr)
    cis_total_cov = coverage[0] / coverage[1]
    bin_mask = cis_total_cov > threshold

    return bin_mask


def generate_bin_mask(
    clr, filters=[None], store=False, store_name="cis_total_ratio_>_0.5_thres"
):
    """
    Generates a binary mask for a given `clr` object based on a list of filters and thresholds.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler object containing Hi-C contact matrices.
    filters : list
        A list of filter functions to apply to the contact matrices.
    store : bool, optional
        If True, store the results in the input cooler file when finished. Default is False.
    store_name : str, optional
        Name of the columns of the bin table to save bin mask.

    Returns
    -------
    bin_mask : numpy.ndarray
        A binary mask indicating which genomic bins pass all filters.
    """
    if not isinstance(filters, list):
        logger.error("filter_lst parameter takes a list")

    bin_mask = np.array([True] * clr.bins().shape[0])
    for filter in filters:
        bin_mask *= filter(clr)

    if store:
        with clr.open("r+") as grp:
            if store_name in grp["bins"]:
                del grp["bins"][store_name]
            h5opts = dict(compression="gzip", compression_opts=6)
            grp["bins"].create_dataset(store_name, data=bin_mask, **h5opts, dtype=bool)

    return bin_mask


def _pixel_filter(chunk_pixels, good_bins_index):
    """
    Filters a chunk of pixels based on a list of good bin indices.

    Parameters
    ----------
    chunk_pixels : pandas.DataFrame
        A DataFrame containing the pixels to be filtered. It must have columns 'bin1_id' and 'bin2_id'.
    good_bins_index : list of int
        A list of indices representing the good bins.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only the pixels whose bin1_id and bin2_id are in good_bins_index.
    """

    pixels_mask = chunk_pixels["bin1_id"].isin(good_bins_index) * chunk_pixels[
        "bin2_id"
    ].isin(good_bins_index)
    return chunk_pixels[pixels_mask]


def pixel_iter_chunks(clr, chunksize):
    """
    Iterate over the pixels of a cooler object in chunks of a given size.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler object containing Hi-C data.
    chunksize : int
        The size of each chunk of pixels to iterate over.

    Yields
    ------
    chunk : numpy.ndarray
        A chunk of pixels of size `chunksize`.
    """
    selector = clr.pixels()
    for lo, hi in cooler.util.partition(0, len(selector), chunksize):
        chunk = selector[lo:hi]
        yield chunk

@pool_decorator
def create_filtered_cooler(
    output_uri, clr, bin_mask, chunksize=10_000_000, nproc=1, map=map
):
    """
    Create a filtered cooler file from a given cooler object and a binary mask of good bins.

    Parameters
    ----------
    output_uri : str
        The URI of the output cooler file to be created.
    clr : cooler.Cooler
        The cooler object to be filtered.
    bin_mask : numpy.ndarray
        A boolean array indicating which bins to keep (True) and which to discard (False).
        Must have the same length as the number of bins in the cooler object.
    nproc : int, optional
        The number of processes to use for parallelization. Default is 16.
    chunksize : int, optional
        The number of pixels to process per chunk. Default is 10,000,000.

    Returns
    -------
    None
    """
    if len(bin_mask) != clr.bins().shape[0]:
        raise ValueError(
            "bin_mask should have the same length as bin table in cool file"
        )
    logger.debug("Start to create cooler file...")
    bin_table = clr.bins()[:][['chrom','start','end']].copy()
    good_bins_index = np.array(range(clr.bins().shape[0]))[bin_mask]
    pixels_filter = functools.partial(_pixel_filter, good_bins_index=good_bins_index)

    cooler.create_cooler(
        output_uri,
        bins=bin_table,
        pixels=map(pixels_filter, pixel_iter_chunks(clr, chunksize)),
        ordered=True,
        columns=["count"],
    )

    logger.debug("done")
