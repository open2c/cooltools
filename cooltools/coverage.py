import numpy as np
from operator import add
from cooler.balance import _zero_diags, _zero_trans, _init
from cooler.tools import split, partition


def _mainDiag_marginalize(chunk, data):
    n = len(chunk['bins']['chrom'])
    pixels = chunk['pixels']
    marg =  np.bincount(pixels['bin1_id'], weights=data, minlength=n)
    mask =  pixels['bin1_id']  ==  pixels['bin2_id'] 
    data[mask] = 0
    marg += np.bincount(pixels['bin2_id'], weights=data, minlength=n)
    return marg


def get_cis( clr, spans, filters, chunksize, map, use_lock):
    chroms = clr.chroms()['name'][:]
    chrom_ids = np.arange(len(clr.chroms()))
    chrom_offsets = clr._load_dset('indexes/chrom_offset')
    bin1_offsets = clr._load_dset('indexes/bin1_offset')
    n_bins = clr.info['nbins']
    cis_marg = np.zeros(n_bins, dtype=float)

    for cid, lo, hi in zip(chrom_ids, chrom_offsets[:-1], chrom_offsets[1:]):
        plo, phi = bin1_offsets[lo], bin1_offsets[hi]
        spans = list(partition(plo, phi, chunksize))

        marg = (
            split(clr, spans=spans, map=map, use_lock=use_lock)
                .prepare(_init)
                .pipe(filters)
                .pipe(_zero_trans)
                .pipe(_mainDiag_marginalize)
                .reduce(add, np.zeros(n_bins))
        )
        marg = marg[lo:hi]
        cis_marg[lo:hi] = marg
    return cis_marg


def get_tot(  clr, spans, filters, chunksize, map, use_lock):
    n_bins = clr.info['nbins']
    marg = (
        split(clr, spans=spans, map=map, use_lock=use_lock)
            .prepare(_init)
            .pipe(filters)
            .pipe(_mainDiag_marginalize)
            .reduce(add, np.zeros(n_bins))
    )
    return marg


def get_cistot(clr, chunksize=None, map=map, ignore_diags=False,
                   use_lock=False, blacklist=None, 
                   store=False, store_names=['cis','tot']):

    """
    Calculation of cis and total sums (margingals) for sparse Hi-C contact map in
    Cooler HDF5 format. Adapted from Cooler balance.py.

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
    ignore_diags : int or False, optional
        Drop elements occurring on the first ``ignore_diags`` diagonals of the
        matrix (including the main diagonal).
    store : bool, optional
        Whether to store the results in the file when finished. Default is False.
    store_names : list, optional
        Names of the columns of the bin table to save to. 
        Must contain 'cis' for cis counts, and 'tot' for total counts.
    
    Returns
    -------
    cis : 1D array, whose shape is the number of bins in ``h5``. Vector of bin sums in cis.
    tot : 1D array, whose shape is the number of bins in ``h5``. Vector of bin sums.

    """

    nnz = clr.info['nnz']
    if chunksize is None:
        chunksize = nnz
        spans = [(0, nnz)]
    else:
        edges = np.arange(0, nnz+chunksize, chunksize)
        spans = list(zip(edges[:-1], edges[1:]))

    # option to ignore diagonals for calculating cis/tot
    base_filters = []
    if ignore_diags:
        base_filters.append(partial(_zero_diags, ignore_diags))

    cis_sum = get_cis(
        clr, spans, base_filters, chunksize, map, use_lock)

    tot_sum = get_tot(
        clr, spans, base_filters, chunksize, map, use_lock)

    if store:
        with clr.open('r+') as grp:
            for store_name in store_names:
                if store_name in grp['bins']:
                    del grp['bins'][store_name]
                h5opts = dict(compression='gzip', compression_opts=6)
                if 'cis' in store_name:
                    grp['bins'].create_dataset(store_name, data=cis_sum, **h5opts)
                elif 'tot' in store_name:
                    grp['bins'].create_dataset(store_name, data=tot_sum, **h5opts)

    return cis_sum, tot_sum	


