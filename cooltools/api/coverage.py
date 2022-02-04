import numpy as np
import cooler.tools


def _zero_diags(chunk, n_diags):
    if n_diags > 0:
        mask = np.abs(chunk["pixels"]["bin1_id"] - chunk["pixels"]["bin2_id"]) < n_diags
        chunk["pixels"]["count"][mask] = 0

    return chunk


def _get_chunk_coverage(chunk, pixel_weight_key="count"):
    """
    Compute cis and total coverages of a cooler chunk.
    Every interaction is contributing to the "coverage" twice:
    at its row coordinate bin1_id, and at its column coordinate bin2_id

    Parameters
    ----------
    chunk : dict of dict/pd.DataFrame
        A cooler chunk produced by the cooler split-apply-combine pipeline.
    pixel_weight_key: str
        The key of a pixel chunk to retrieve pixel weights.

    Returns
    -------
    covs : np.array 2 x n_bins
        A numpy array with cis (the first row) and total (the 4nd) coverages.
    """

    bins = chunk["bins"]
    pixels = chunk["pixels"]
    n_bins = len(bins["chrom"])
    covs = np.zeros((2, n_bins))
    pixel_weights = pixels[pixel_weight_key]

    cis_mask = bins["chrom"][pixels["bin1_id"]] == bins["chrom"][pixels["bin2_id"]]
    covs[0] += np.bincount(
        pixels["bin1_id"], weights=pixel_weights * cis_mask, minlength=n_bins
    )
    covs[0] += np.bincount(
        pixels["bin2_id"], weights=pixel_weights * cis_mask, minlength=n_bins
    )

    covs[1] += np.bincount(pixels["bin1_id"], weights=pixel_weights, minlength=n_bins)
    covs[1] += np.bincount(pixels["bin2_id"], weights=pixel_weights, minlength=n_bins)

    return covs


def coverage(
    clr,
    ignore_diags=None,
    chunksize=int(1e7),
    map=map,
    use_lock=False,
    store=False,
    store_names=["cis_raw_cov", "tot_raw_cov"],
):

    """
    Calculate the sums of cis and genome-wide contacts (aka coverage aka marginals) for
    a sparse Hi-C contact map in Cooler HDF5 format.
    Note that the sum(tot_cov) from this function is two times the number of reads
    contributing to the cooler, as each side contributes to the coverage.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    chunksize : int, optional
        Split the contact matrix pixel records into equally sized chunks to
        save memory and/or parallelize. Default is to use all the pixels at
        once.
    map : callable, optional
        Map function to dispatch the matrix chunks to workers.
        Default is the builtin ``map``, but alternatives include parallel map
        implementations from a multiprocessing pool.
    ignore_diags : int, optional
        Drop elements occurring on the first ``ignore_diags`` diagonals of the
        matrix (including the main diagonal).
        If None, equals the number of diagonals ignored during IC balancing.
    store : bool, optional
        If True, store the results in the file when finished. Default is False.
    store_names : list, optional
        Names of the columns of the bin table to save cis and total coverages.

    Returns
    -------
    cis_cov : 1D array, whose shape is the number of bins in ``h5``. Vector of bin sums in cis.
    tot_cov : 1D array, whose shape is the number of bins in ``h5``. Vector of bin sums.

    """

    try:
        ignore_diags = (
            ignore_diags
            if ignore_diags is not None
            else clr._load_attrs(clr.root.rstrip("/") + "/bins/weight")["ignore_diags"]
        )
    except:
        raise ValueError(
            "Please, specify ignore_diags and/or IC balance this cooler! Cannot access the value used in IC balancing. "
        )

    chunks = cooler.tools.split(clr, chunksize=chunksize, map=map, use_lock=use_lock)

    if ignore_diags:
        chunks = chunks.pipe(_zero_diags, n_diags=ignore_diags)

    n_bins = clr.info["nbins"]
    covs = chunks.pipe(_get_chunk_coverage).reduce(np.add, np.zeros((2, n_bins)))

    if store:
        with clr.open("r+") as grp:
            for store_name, cov_arr in zip(store_names, covs):
                if store_name in grp["bins"]:
                    del grp["bins"][store_name]
                h5opts = dict(compression="gzip", compression_opts=6)
                grp["bins"].create_dataset(store_name, data=cov_arr, **h5opts)

    return covs
