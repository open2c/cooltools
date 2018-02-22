from itertools import chain
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve
import distributed
import dask.dataframe as dd
import dask.array as da
import dask

from cooler.tools import split, partition
from cooler.contrib.dask import daskify
import cooler

where = np.flatnonzero
concat = chain.from_iterable


def bg2slice_frame(bg2, region1, region2):
    """
    Slice a dataframe with columns ['chrom1', 'start1', 'end1', 'chrom2',
    'start2', 'end2']. Assumes no proper nesting of intervals.

    """
    chrom1, start1, end1 = region1
    chrom2, start2, end2 = region2
    if end1 is None:
        end1 = np.inf
    if end2 is None:
        end2 = np.inf
    out = bg2[(bg2['chrom1'] == chrom1) & 
              (bg2['start1'] >= start1) &
              (bg2['end1'] < end1) &
              (bg2['chrom2'] == chrom2) & 
              (bg2['start2'] >= start2) &
              (bg2['end2'] < end2)]
    return out


def lattice_pdist_frequencies(n, points):
    """
    Distribution of pairwise 1D distances among a collection of distinct
    integers ranging from 0 to n-1.

    Parameters
    ----------
    n : int
        Size of the lattice on which the integer points reside.
    points : sequence of int
        Arbitrary integers between 0 and n-1, inclusive, in any order but 
        with no duplicates.

    Returns
    -------
    h : 1D array of length n
        h[d] counts the number of integer pairs that are exactly d units apart

    Notes
    -----
    This is done using a convolution via FFT. Thanks to Peter de Rivaz; see
    <http://stackoverflow.com/questions/42423823/distribution-of-pairwise-
    distances-between-many-integers>.
    
    """
    if len(np.unique(points)) != len(points):
        raise ValueError("Integers must be distinct.")
    x = np.zeros(n); x[points] = 1
    return np.round(fftconvolve(x, x[::-1], mode='full')).astype(int)[-n:]


def count_bad_pixels_per_diag(n, bad_bins):
    """
    Efficiently count the number of bad pixels on each upper diagonal of a
    matrix assuming a sequence of bad bins forms a "grid" of invalid pixels.
    
    Each bad bin bifurcates into two a row and column of bad pixels, so an 
    upper bound on number of bad pixels per diagonal is 2*k, where k is the 
    number of bad bins. For a given diagonal, we need to subtract from this
    upper estimate the contribution from rows/columns reaching "out-of-bounds"
    and the contribution of the intersection points of bad rows with bad
    columns that get double counted.
    
    o : bad bin
    * : bad pixel
    x : intersection bad pixel
    $ : out of bounds bad pixel

         $    $     $
     *--------------------------+    
      *  *    *     *           |    
       * *    *     *           |    
        **    *     *           |    
         o****x*****x***********|$    
          *   *     *           |    
           *  *     *           |    
            * *     *           |    
             o******x***********|$    
              *     *           |    
               *    *           |    
                *   *           |    
                 *  *           |    
                  * *           |    
                   **           |    
                    o***********|$    
                     *          |    
                      *         |    
    
    Parameters
    ----------
    n : int
        total number of bins
    bad_bins : 1D array of int
        sorted array of bad bin indexes

    Returns
    -------
    dcount : 1D array of length n
        dcount[d] == number of bad pixels on diagonal d
    
    """
    k = len(bad_bins)
    dcount = np.zeros(n, dtype=int)

    # Store all intersection pixels in a separate array
    # ~O(n log n) with fft
    ixn = lattice_pdist_frequencies(n, bad_bins)
    dcount[0] = ixn[0]

    # Keep track of out-of-bounds pixels by squeezing left and right bounds
    # ~O(n)
    pl = 0
    pr = k
    for diag in range(1, n):
        if pl < k:
            while (bad_bins[pl] - diag) < 0:
                pl += 1
                if pl == k:
                    break
        if pr > 0:
            while (bad_bins[pr-1] + diag) >= n:
                pr -= 1
                if pr == 0:
                    break
        dcount[diag] = 2*k - ixn[diag] - pl - (k - pr)
    return dcount


def count_all_pixels_per_diag(n):
    """
    Total number of pixels on each upper diagonal of a square matrix.

    Parameters
    ----------
    n : int
        total number of bins (dimension of square matrix)

    Returns
    -------
    dcount : 1D array of length n
        dcount[d] == total number of pixels on diagonal d

    """
    return np.arange(n, 0, -1)



def make_diag_table(bad_mask, span1, span2):
    """
    Compute the total number of elements ``n_elem`` and the number of bad 
    elements ``n_bad`` per diagonal for a single contact area encompassing 
    ``span1`` and ``span2`` on the same genomic scaffold (cis matrix).

    Follows the same principle as the algorithm for finding contact areas for
    computing scalings.

    Parameters
    ----------
    bad_mask : 1D array of bool
        Mask of bad bins for the whole genomic scaffold containing the regions
        of interest.
    span1, span2 : pair of ints
        The bin spans (not genomic coordinates) of the two regions of interest.

    Returns
    -------
    diags : pandas.DataFrame
        Table indexed by 'diag' with columns ['n_elem', 'n_bad'].

    """
    def _make_diag_table(n_bins, bad_locs):
        diags = pd.DataFrame(index=pd.Series(np.arange(n_bins), name='diag'))
        diags['n_valid'] = (count_all_pixels_per_diag(n_bins) - 
                            count_bad_pixels_per_diag(n_bins, bad_locs))
        return diags
    
    if span1 == span2:
        lo, hi = span1
        diags = _make_diag_table(hi - lo, where(bad_mask[lo:hi]))
    else:
        lo1, hi1 = span1
        lo2, hi2 = span2
        if lo2 <= lo1:
            lo1, lo2 = lo2, lo1
            hi1, hi2 = hi2, hi1
        diags = (
            _make_diag_table(hi2 - lo1, where(bad_mask[lo1:hi2]))
                .subtract(_make_diag_table(lo2 - lo1, where(bad_mask[lo1:lo2])), 
                          fill_value=0)
                .subtract(_make_diag_table(hi2 - hi1, where(bad_mask[hi1:hi2])), 
                          fill_value=0)            
        )
        if hi1 < lo2:
            diags.add(_make_diag_table(lo2 - hi1, where(bad_mask[hi1:lo2])), 
                      fill_value=0)
        diags = diags[diags['n_elem'] > 0]
    return diags.astype(int)


def _sum_diagonals(df, field):
    reduced = df.groupby('diag')[field].sum()
    reduced.name = field + '.sum'
    return reduced


def cis_expected(clr, regions, field='balanced', chunksize=1000000, 
                 use_dask=True, ignore_diags=2):
    """
    Compute the mean signal along diagonals of one or more regional blocks of
    intra-chromosomal contact matrices. Typically used as a background model 
    for contact frequencies on the same polymer chain. 

    Parameters
    ----------
    clr : cooler.Cooler
        Input Cooler
    regions : iterable of genomic regions or pairs of regions
        Iterable of genomic region strings or 3-tuples, or 5-tuples for pairs
        of regions
    field : str, optional
        Which values of the contact matrix to aggregate. This is currently a
        no-op. *FIXME*
    chunksize : int, optional
        Size of dask chunks.

    Returns
    -------
    Dataframe of diagonal statistics, indexed by region and diagonal number

    """
    if use_dask:
        pixels = daskify(clr.filename, clr.root + '/pixels', chunksize=chunksize)
    else:
        pixels = clr.pixels()[:]
    pixels = cooler.annotate(pixels, clr.bins(), replace=False)
    pixels = pixels[pixels.chrom1 == pixels.chrom2]

    named_regions = False
    if isinstance(regions, pd.DataFrame):
        named_regions = True
        chroms = regions['chrom'].values
        names = regions['name'].values
        regions = regions[['chrom', 'start', 'end']].to_records(index=False)
    else:
        chroms = [region[0] for region in regions]
        names = chroms
    cis_maps = {chrom: pixels[pixels.chrom1==chrom] for chrom in chroms}

    diag_tables = []
    data_sums = []

    for region in regions:
        if len(region) == 1:
            chrom, = region
            start1, end1 = 0, clr.chromsizes[chrom]
            start2, end2 = start1, end1
        elif len(region) == 3:
            chrom, start1, end1 = region
            start2, end2 = start1, end1
        elif len(region) == 5:
            chrom, start1, end1, start2, end2 = region
        else:
            raise ValueError("Regions must be sequences of length 1, 3 or 5")

        bins = clr.bins().fetch(chrom).reset_index(drop=True)
        bad_mask = np.array(bins['weight'].isnull())
        lo1, hi1 = clr.extent((chrom, start1, end1))
        lo2, hi2 = clr.extent((chrom, start2, end2))
        co = clr.offset(chrom)
        lo1 -= co
        lo2 -= co
        hi1 -= co
        hi2 -= co

        dt = make_diag_table(bad_mask, [lo1, hi1], [lo2, hi2])
        sel = bg2slice_frame(
            cis_maps[chrom], 
            (chrom, start1, end1), 
            (chrom, start2, end2)
        ).copy()
        sel['diag'] = sel['bin2_id'] - sel['bin1_id']
        sel['balanced'] = sel['count'] * sel['weight1'] * sel['weight2']
        agg = _sum_diagonals(sel, field)
        diag_tables.append(dt)
        data_sums.append(agg)

    # run dask scheduler
    if len(data_sums) and isinstance(data_sums[0], dd.Series):
        data_sums = dd.compute(*data_sums)

    # append to tables
    for dt, agg in zip(diag_tables, data_sums):
        dt[agg.name] = 0
        dt[agg.name] = dt[agg.name].add(agg, fill_value=0)
        dt.iloc[:ignore_diags, dt.columns.get_loc(agg.name)] = np.nan

    # merge and return
    if named_regions:
        dtable = pd.concat(
            diag_tables, 
            keys=zip(names, chroms), 
            names=['name', 'chrom'])
    else:
        dtable = pd.concat(
            diag_tables, 
            keys=list(chroms), 
            names=['chrom'])  

    # the actual expected is balanced.sum/n_valid:
    dtable['balanced.avg'] = dtable['balanced.sum'] / dtable['n_valid']
    return dtable


def trans_expected(clr, chromosomes, chunksize=1000000, use_dask=False):
    """
    Aggregate the signal in intrachromosomal blocks.
    Can be used as abackground for contact frequencies between chromosomes.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    chromosomes : list of str
        List of chromosome names
    chunksize : int, optional
        Size of dask chunks
    use_dask : bool, optional
        option to use dask
    
    Returns
    -------
    pandas.DataFrame that stores total number of
    interactions between a pair of chromosomes: 'balanced.sum',
    corresponding number of bins involved
    in the inter-chromosomal interactions: 'n_valid', 
    and a ratio 'balanced.avg = balanced.sum/n_valid', that is
    the actual value of expected for every interchromosomal pair.

    """
    def n_total_trans_elements(clr, chromosomes):
        n = len(chromosomes)
        x = [clr.extent(chrom)[1] - clr.extent(chrom)[0] 
                 for chrom in chromosomes]
        pairblock_list = []
        for i in range(n):
            for j in range(i + 1, n):
                # appending to the list of tuples
                pairblock_list.append((chromosomes[i],
                                       chromosomes[j],
                                       x[i] * x[j] ))
        return pd.DataFrame(pairblock_list, 
            columns=['chrom1', 'chrom2', 'n_total'])

    def n_bad_trans_elements(clr, chromosomes):
        n = 0
        # bad bins are ones with
        # the weight vector being NaN:
        x = [np.sum(clr.bins()['weight']
                       .fetch(chrom)
                       .isnull()
                       .astype(int)
                       .values)
                 for chrom in chromosomes]
        pairblock_list = []
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                # appending to the list of tuples
                pairblock_list.append((chromosomes[i],
                                       chromosomes[j],
                                       x[i] * x[j] ))
        return pd.DataFrame(pairblock_list,
            columns=['chrom1', 'chrom2', 'n_bad'])

    if use_dask:
        # pixels = daskify(clr.filename, clr.root + '/pixels', chunksize=chunksize)
        raise NotImplementedError("To be implemented once dask supports MultiIndex")
    else:
        pixels = clr.pixels()[:]
    # getting pixels that belong to trans-area,
    # defined by the list of chromosomes:
    pixels = cooler.annotate(pixels, clr.bins(), replace=False)
    pixels = pixels[
        (pixels.chrom1.isin(chromosomes)) &
        (pixels.chrom2.isin(chromosomes)) &
        (pixels.chrom1 != pixels.chrom2)
    ]
    pixels['balanced'] = pixels['count'] * pixels['weight1'] * pixels['weight2']
    ntot = n_total_trans_elements(clr, chromosomes).groupby(('chrom1', 'chrom2'))['n_total'].sum()
    nbad = n_bad_trans_elements(clr, chromosomes).groupby(('chrom1', 'chrom2'))['n_bad'].sum()
    trans_area = ntot - nbad
    trans_area.name = 'n_valid'
    # processing with use_dask=True is different:
    if use_dask:
        # trans_sum = pixels.groupby(('chrom1', 'chrom2'))['balanced'].sum().compute()
        pass
    else:
        trans_sum = pixels.groupby(('chrom1', 'chrom2'))['balanced'].sum()
    # for consistency with the cis_expected function:
    trans_sum.name = trans_sum.name + '.sum'

    # returning a DataFrame with MultiIndex, that stores
    # pairs of 'balanced.sum' and 'n_valid' values for each
    # pair of chromosomes.
    dtable = pd.merge(
        trans_sum.to_frame(),
        trans_area.to_frame(),
        left_index=True,
        right_index=True)

    # the actual expected is balanced.sum/n_valid:
    dtable['balanced.avg'] = dtable['balanced.sum'] / dtable['n_valid']
    return dtable
