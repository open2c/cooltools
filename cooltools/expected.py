from itertools import chain
from collections import defaultdict
from functools import partial

import warnings

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

from cooler.tools import partition
import cooler
import bioframe
from .lib import assign_supports, numutils

where = np.flatnonzero
concat = chain.from_iterable


def _contact_areas(distbins, scaffold_length):
    distbins = distbins.astype(float)
    scaffold_length = float(scaffold_length)
    outer_areas = np.maximum(scaffold_length - distbins[:-1], 0) ** 2
    inner_areas = np.maximum(scaffold_length - distbins[1:], 0) ** 2
    return 0.5 * (outer_areas - inner_areas)


def contact_areas(distbins, region1, region2):
    if region1 == region2:
        start, end = region1
        areas = _contact_areas(distbins, end - start)
    else:
        start1, end1 = region1
        start2, end2 = region2
        if start2 <= start1:
            start1, start2 = start2, start1
            end1, end2 = end2, end1
        areas = (
            _contact_areas(distbins, end2 - start1)
            - _contact_areas(distbins, start2 - start1)
            - _contact_areas(distbins, end2 - end1)
        )
        if end1 < start2:
            areas += _contact_areas(distbins, start2 - end1)

    return areas


def compute_scaling(df, region1, region2=None, dmin=int(1e1), dmax=int(1e7), n_bins=50):

    import dask.array as da

    if region2 is None:
        region2 = region1

    distbins = numutils.logbins(dmin, dmax, N=n_bins)
    areas = contact_areas(distbins, region1, region2)

    df = df[
        (df["pos1"] >= region1[0])
        & (df["pos1"] < region1[1])
        & (df["pos2"] >= region2[0])
        & (df["pos2"] < region2[1])
    ]
    dists = (df["pos2"] - df["pos1"]).values

    if isinstance(dists, da.Array):
        obs, _ = da.histogram(dists[(dists >= dmin) & (dists < dmax)], bins=distbins)
    else:
        obs, _ = np.histogram(dists[(dists >= dmin) & (dists < dmax)], bins=distbins)

    return distbins, obs, areas


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
    `<http://stackoverflow.com/questions/42423823/distribution-of-pairwise-distances-between-many-integers>`_.

    """
    if len(np.unique(points)) != len(points):
        raise ValueError("Integers must be distinct.")
    x = np.zeros(n)
    x[points] = 1
    return np.round(fftconvolve(x, x[::-1], mode="full")).astype(int)[-n:]


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

    ::

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
            while (bad_bins[pr - 1] + diag) >= n:
                pr -= 1
                if pr == 0:
                    break
        dcount[diag] = 2 * k - ixn[diag] - pl - (k - pr)
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


def count_all_pixels_per_block(x, y):
    """
    Calculate total number of pixels in a rectangular block

    Parameters
    ----------
    x : int
        block width in pixels
    y : int
        block height in pixels

    Returns
    -------
    number_of_pixels : int
        total number of pixels in a block
    """
    return x * y


def count_bad_pixels_per_block(x, y, bad_bins_x, bad_bins_y):
    """
    Calculate number of "bad" pixels per rectangular block of a contact map

    "Bad" pixels are inferred from the balancing weight column `weight_name` or
    provided directly in the form of an array `bad_bins`.

    Setting `weight_name` and `bad_bins` to `None` yields 0 bad pixels in a block.

    Parameters
    ----------
    x : int
        block width in pixels
    y : int
        block height in pixels
    bad_bins_x : int
        number of bad bins on x-side
    bad_bins_y : int
        number of bad bins on y-side

    Returns
    -------
    number_of_pixes : int
        number of "bad" pixels in a block
    """

    # Calculate the resulting bad pixels in a rectangular block:
    return (x * bad_bins_y) + (y * bad_bins_x) - (bad_bins_x * bad_bins_y)


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
        diags = pd.DataFrame(index=pd.Series(np.arange(n_bins), name="diag"))
        diags["n_elem"] = count_all_pixels_per_diag(n_bins)
        diags["n_valid"] = diags["n_elem"] - count_bad_pixels_per_diag(n_bins, bad_locs)
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
            .subtract(
                _make_diag_table(lo2 - lo1, where(bad_mask[lo1:lo2])), fill_value=0
            )
            .subtract(
                _make_diag_table(hi2 - hi1, where(bad_mask[hi1:hi2])), fill_value=0
            )
        )
        if hi1 < lo2:
            diags.add(
                _make_diag_table(lo2 - hi1, where(bad_mask[hi1:lo2])), fill_value=0
            )
        diags = diags[diags["n_elem"] > 0]

    diags = diags.drop("n_elem", axis=1)
    return diags.astype(int)


def make_diag_tables(clr, regions, regions2=None, weight_name="weight", bad_bins=None):
    """
    For every support region infer diagonals that intersect this region
    and calculate the size of these intersections in pixels, both "total" and
    "n_valid", where "n_valid" does not include "bad" bins into counting.

    "Bad" pixels are inferred from the balancing weight column `weight_name` or
    provided directly in the form of an array `bad_bins`.

    Setting `weight_name` and `bad_bins` to `None` yields 0 "bad" pixels per
    diagonal per support region.

    When `regions2` are provided, all intersecting diagonals are reported for
    each rectangular and asymmetric block defined by combinations of matching
    elements of `regions` and `regions2`.
    Otherwise only `regions`-based symmetric square blocks are considered.
    Only intra-chromosomal regions are supported.

    Parameters
    ----------
    clr : cooler.Cooler
        Input cooler
    regions : list
        a list of genomic support regions
    regions2 : list
        a list of genomic support regions for asymmetric regions
    weight_name : str
        name of the weight vector in the "bins" table,
        if weight_name is None returns 0 for each block.
        Balancing weight are used to infer bad bins.
    bad_bins : array-like
        a list of bins to ignore. Indexes of bins must
        be absolute, as in clr.bins()[:], as opposed to
        being offset by chromosome start.
        "bad_bins" will be combined with the bad bins
        masked by balancing if there are any.

    Returns
    -------
    diag_tables : dict
        dictionary with DataFrames of relevant diagonals for every support.
    """

    regions = bioframe.parse_regions(regions, clr.chromsizes).values
    if regions2 is not None:
        regions2 = bioframe.parse_regions(regions2, clr.chromsizes).values

    bins = clr.bins()[:]
    if weight_name is None:
        # ignore bad bins
        sizes = dict(bins.groupby("chrom").size())
        bad_bin_dict = {
            chrom: np.zeros(sizes[chrom], dtype=bool) for chrom in sizes.keys()
        }
    elif isinstance(weight_name, str):
        # using balacning weight to infer bad bins
        if weight_name not in clr.bins().columns:
            raise KeyError("Balancing weight {weight_name} not found!")
        groups = dict(iter(bins.groupby("chrom")[weight_name]))
        bad_bin_dict = {
            chrom: np.array(groups[chrom].isnull()) for chrom in groups.keys()
        }
    else:
        raise ValueError("`weight_name` can be `str` or `None`")

    # combine custom "bad_bins" with "bad_bin_dict":
    if bad_bins is not None:
        # check if "bad_bins" are legit:
        try:
            bad_bins_chrom = bins.iloc[bad_bins].reset_index(drop=False)
        except IndexError:
            raise ValueError("Provided `bad_bins` are incorrect or out-of-bound")
        # group them by observed chromosomes only
        bad_bins_grp = bad_bins_chrom[["index", "chrom"]].groupby(
            "chrom", observed=True
        )
        # update "bad_bin_dict" with "bad_bins" for each chrom:
        for chrom, bin_ids in bad_bins_grp["index"]:
            co = clr.offset(chrom)
            # adjust by chromosome offset
            bad_bin_dict[chrom][bin_ids.values - co] = True

    diag_tables = {}
    for i in range(len(regions)):
        chrom, start1, end1, name1 = regions[i]
        if regions2 is not None:
            chrom2, start2, end2, name2 = regions2[i]
            # cis-only for now:
            assert chrom2 == chrom
        else:
            start2, end2 = start1, end1

        # translate regions into relative bin id-s:
        lo1, hi1 = clr.extent((chrom, start1, end1))
        lo2, hi2 = clr.extent((chrom, start2, end2))
        co = clr.offset(chrom)
        lo1 -= co
        lo2 -= co
        hi1 -= co
        hi2 -= co

        bad_mask = bad_bin_dict[chrom]
        newname = name1
        if regions2 is not None:
            newname = (name1, name2)
        diag_tables[newname] = make_diag_table(bad_mask, [lo1, hi1], [lo2, hi2])

    return diag_tables


def make_block_table(clr, regions1, regions2, weight_name="weight", bad_bins=None):
    """
    Creates a table that characterizes a set of rectangular genomic blocks
    formed by combining regions from regions1 and regions2.
    For every block calculate its "area" in pixels ("n_total"), and calculate
    number of "valid" pixels in each block ("n_valid").
    "Valid" pixels exclude "bad" pixels, which in turn inferred from the balancing
    weight column `weight_name` or provided directly in the form of an array of
    `bad_bins`.

    Setting `weight_name` and `bad_bins` to `None` yields 0 "bad" pixels per
    block.

    Parameters
    ----------
    clr : cooler.Cooler
        Input cooler
    regions1 : iterable
        a collection of genomic regions
    regions2 : iterable
        a collection of genomic regions
    weight_name : str
        name of the weight vector in the "bins" table,
        if weight_name is None returns 0 for each block.
        Balancing weight are used to infer bad bins.
    bad_bins : array-like
        a list of bins to ignore. Indexes of bins must
        be absolute, as in clr.bins()[:], as opposed to
        being offset by chromosome start.
        "bad_bins" will be combined with the bad bins
        masked by balancing if there are any.

    Returns
    -------
    block_table : dict
        dictionary for blocks that are 0-indexed
    """
    if bad_bins is None:
        bad_bins = np.asarray([]).astype(int)
    else:
        bad_bins = np.asarray(bad_bins).astype(int)

    regions1 = bioframe.parse_regions(regions1, clr.chromsizes).values
    regions2 = bioframe.parse_regions(regions2, clr.chromsizes).values

    # should we check for nestedness here, or that each region1 is < region2 ?

    block_table = {}
    for r1, r2 in zip(regions1, regions2):
        chrom1, start1, end1, name1 = r1
        chrom2, start2, end2, name2 = r2
        # translate regions into relative bin id-s:
        lo1, hi1 = clr.extent((chrom1, start1, end1))
        lo2, hi2 = clr.extent((chrom2, start2, end2))
        # width and height of a block:
        x = hi1 - lo1
        y = hi2 - lo2
        # get "regional" bad_bins for each of the regions
        bx = bad_bins[(bad_bins >= lo1) & (bad_bins < hi1)] - lo1
        by = bad_bins[(bad_bins >= lo2) & (bad_bins < hi2)] - lo2

        # now we need to combine it with the balancing weights
        if weight_name is None:
            bad_bins_x = len(bx)
            bad_bins_y = len(by)
        elif isinstance(weight_name, str):
            if weight_name not in clr.bins().columns:
                raise KeyError("Balancing weight {weight_name} not found!")
            else:
                # extract "bad" bins filtered by balancing:
                cb_bins_x = clr.bins()[weight_name][lo1:hi1].isnull().values
                cb_bins_y = clr.bins()[weight_name][lo2:hi2].isnull().values
                # combine with "bad_bins" using assignment:
                cb_bins_x[bx] = True
                cb_bins_y[by] = True
                # count and yield final list of bad bins:
                bad_bins_x = np.count_nonzero(cb_bins_x)
                bad_bins_y = np.count_nonzero(cb_bins_y)
        else:
            raise ValueError("`weight_name` can be `str` or `None`")

        # calculate total and bad pixels per block:
        n_tot = count_all_pixels_per_block(x, y)
        n_bad = count_bad_pixels_per_block(x, y, bad_bins_x, bad_bins_y)

        # fill in "block_table" with number of valid pixels:
        block_table[name1, name2] = defaultdict(int)
        block_table[name1, name2]["n_valid"] = n_tot - n_bad

    return block_table


def _diagsum_symm(clr, fields, transforms, regions, span):
    """
    calculates diagonal summary for a collection of
    square symmteric regions defined by regions.
    returns a dictionary of DataFrames with diagonal
    sums as values, and 0-based indexes of square
    genomic regions as keys.
    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)

    # this could further expanded to allow for custom groupings:
    pixels["dist"] = pixels["bin2_id"] - pixels["bin1_id"]
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    diag_sums = {}
    # r define square symmetric block i:
    for i, r in enumerate(regions):
        r1 = assign_supports(pixels, [r], suffix="1")
        r2 = assign_supports(pixels, [r], suffix="2")
        # calculate diag_sums on the spot to allow for overlapping blocks:
        diag_sums[i] = pixels[(r1 == r2)].groupby("dist")[fields].sum()

    return diag_sums


def diagsum(
    clr,
    regions,
    transforms={},
    weight_name="weight",
    bad_bins=None,
    chunksize=10000000,
    ignore_diags=2,
    map=map,
):
    """

    Intra-chromosomal diagonal summary statistics.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    regions : sequence of genomic range tuples
        Support regions for intra-chromosomal diagonal summation
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    weight_name : str
        name of the balancing weight vector used to count
        "bad"(masked) pixels per diagonal.
        Use `None` to avoid masking "bad" pixels.
    bad_bins : array-like
        a list of bins to ignore per support region.
        Combines with the list of bad bins from balacning
        weight.
    chunksize : int, optional
        Size of pixel table chunks to process
    ignore_diags : int, optional
        Number of intial diagonals to exclude from statistics
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    Dataframe of diagonal statistics for all regions

    """
    spans = partition(0, len(clr.pixels()), chunksize)
    fields = ["count"] + list(transforms.keys())

    regions = bioframe.parse_regions(regions, clr.chromsizes)

    dtables = make_diag_tables(clr, regions, weight_name=weight_name, bad_bins=bad_bins)

    # combine masking with existing transforms and add a "count" transform:
    if bad_bins is not None:
        # turn bad_bins into a mask of size clr.bins:
        mask_size = len(clr.bins())
        bad_bins_mask = np.ones(mask_size, dtype=int)
        bad_bins_mask[bad_bins] = 0
        #
        masked_transforms = {}
        bin1 = "bin1_id"
        bin2 = "bin2_id"
        for field in fields:
            if field in transforms:
                # combine masking and transform, minding the scope:
                t = transforms[field]
                masked_transforms[field] = (
                    lambda p, t=t, m=bad_bins_mask: t(p) * m[p[bin1]] * m[p[bin2]]
                )
            else:
                # presumably field == "count", mind the scope as well:
                masked_transforms[field] = (
                    lambda p, f=field, m=bad_bins_mask: p[f] * m[p[bin1]] * m[p[bin2]]
                )
        # substitute transforms to the masked_transforms:
        transforms = masked_transforms

    for dt in dtables.values():
        for field in fields:
            agg_name = "{}.sum".format(field)
            dt[agg_name] = 0

    job = partial(_diagsum_symm, clr, fields, transforms, regions.values)
    results = map(job, spans)
    for result in results:
        for i, agg in result.items():
            region = regions.loc[i, "name"]
            for field in fields:
                agg_name = "{}.sum".format(field)
                dtables[region][agg_name] = dtables[region][agg_name].add(
                    agg[field], fill_value=0
                )

    if ignore_diags:
        for dt in dtables.values():
            for field in fields:
                agg_name = "{}.sum".format(field)
                j = dt.columns.get_loc(agg_name)
                dt.iloc[:ignore_diags, j] = np.nan

    # returning dataframe for API consistency
    result = []
    for i, dtable in dtables.items():
        dtable = dtable.reset_index()
        dtable.insert(0, "region", i)
        result.append(dtable)
    return pd.concat(result).reset_index(drop=True)


def _diagsum_asymm(clr, fields, transforms, regions1, regions2, span):
    """
    calculates diagonal summary for a collection of
    rectangular regions defined as combinations of
    regions1 and regions2.
    returns a dictionary of DataFrames with diagonal
    sums as values, and 0-based indexes of rectangular
    genomic regions as keys.
    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)

    # this could further expanded to allow for custom groupings:
    pixels["dist"] = pixels["bin2_id"] - pixels["bin1_id"]
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    diag_sums = {}
    # r1 and r2 define rectangular block i:
    for i, (r1, r2) in enumerate(zip(regions1, regions2)):
        r1 = assign_supports(pixels, [r1], suffix="1")
        r2 = assign_supports(pixels, [r2], suffix="2")
        # calculate diag_sums on the spot to allow for overlapping blocks:
        diag_sums[i] = pixels[(r1 == r2)].groupby("dist")[fields].sum()

    return diag_sums


def diagsum_asymm(
    clr,
    regions1,
    regions2,
    transforms={},
    weight_name="weight",
    bad_bins=None,
    chunksize=10000000,
    map=map,
):
    """

    Diagonal summary statistics.

    Matchings elements of `regions1` and  `regions2` define
    asymmetric rectangular blocks for calculating diagonal
    summary statistics.
    Only intra-chromosomal blocks are supported.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    regions1 : sequence of genomic range tuples
        "left"-side support regions for diagonal summation
    regions2 : sequence of genomic range tuples
        "right"-side support regions for diagonal summation
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    weight_name : str
        name of the balancing weight vector used to count
        "bad"(masked) pixels per diagonal.
        Use `None` to avoid masking "bad" pixels.
    bad_bins : array-like
        a list of bins to ignore per support region.
        Combines with the list of bad bins from balacning
        weight.
    chunksize : int, optional
        Size of pixel table chunks to process
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    DataFrame with summary statistic of every diagonal of every block:
    region1, region2, diag, n_valid, count.sum

    """
    spans = partition(0, len(clr.pixels()), chunksize)
    fields = ["count"] + list(transforms.keys())
    regions1 = bioframe.parse_regions(regions1, clr.chromsizes)
    regions2 = bioframe.parse_regions(regions2, clr.chromsizes)

    dtables = make_diag_tables(
        clr, regions1, regions2, weight_name=weight_name, bad_bins=bad_bins
    )

    # combine masking with existing transforms and add a "count" transform:
    if bad_bins is not None:
        # turn bad_bins into a mask of size clr.bins:
        mask_size = len(clr.bins())
        bad_bins_mask = np.ones(mask_size, dtype=int)
        bad_bins_mask[bad_bins] = 0
        #
        masked_transforms = {}
        bin1 = "bin1_id"
        bin2 = "bin2_id"
        for field in fields:
            if field in transforms:
                # combine masking and transform, minding the scope:
                t = transforms[field]
                masked_transforms[field] = (
                    lambda p, t=t, m=bad_bins_mask: t(p) * m[p[bin1]] * m[p[bin2]]
                )
            else:
                # presumably field == "count", mind the scope as well:
                masked_transforms[field] = (
                    lambda p, f=field, m=bad_bins_mask: p[f] * m[p[bin1]] * m[p[bin2]]
                )
        # substitute transforms to the masked_transforms:
        transforms = masked_transforms

    for dt in dtables.values():
        for field in fields:
            agg_name = "{}.sum".format(field)
            dt[agg_name] = 0

    job = partial(
        _diagsum_asymm, clr, fields, transforms, regions1.values, regions2.values
    )
    results = map(job, spans)
    for result in results:
        for i, agg in result.items():
            region1 = regions1.loc[i, "name"]
            region2 = regions2.loc[i, "name"]
            for field in fields:
                agg_name = "{}.sum".format(field)
                dtables[region1, region2][agg_name] = dtables[region1, region2][
                    agg_name
                ].add(agg[field], fill_value=0)

    # returning a dataframe for API consistency:
    result = []
    for (i, j), dtable in dtables.items():
        dtable = dtable.reset_index()
        dtable.insert(0, "region1", i)
        dtable.insert(1, "region2", j)
        result.append(dtable)
    return pd.concat(result).reset_index(drop=True)


def _blocksum_asymm(clr, fields, transforms, regions1, regions2, span):
    """
    calculates block summary for a collection of
    rectangular regions defined as combinations of
    regions1 and regions2.
    returns a dictionary of with block sums as values,
    and 0-based indexes of rectangular genomic regions
    as keys.
    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)

    for field, t in transforms.items():
        pixels[field] = t(pixels)

    block_sums = {}
    # r1 and r2 define rectangular block i:
    for i, (r1, r2) in enumerate(zip(regions1, regions2)):
        r1 = assign_supports(pixels, [r1], suffix="1")
        r2 = assign_supports(pixels, [r2], suffix="2")
        # calculate sum on the spot to allow for overlapping blocks:
        block_sums[i] = pixels[(r1 == r2)][fields].sum()

    return block_sums


def blocksum_asymm(
    clr,
    regions1,
    regions2,
    transforms={},
    weight_name="weight",
    bad_bins=None,
    chunksize=1000000,
    map=map,
):
    """
    Summary statistics on rectangular blocks of genomic regions.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    regions1 : sequence of genomic range tuples
        "left"-side support regions for diagonal summation
    regions2 : sequence of genomic range tuples
        "right"-side support regions for diagonal summation
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    weight_name : str
        name of the balancing weight vector used to count
        "bad"(masked) pixels per block.
        Use `None` to avoid masking "bad" pixels.
    bad_bins : array-like
        a list of bins to ignore per support region.
        Combines with the list of bad bins from balacning
        weight.
    chunksize : int, optional
        Size of pixel table chunks to process
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    DataFrame with entries for each blocks: region1, region2, n_valid, count.sum

    """

    regions1 = bioframe.parse_regions(regions1, clr.chromsizes)
    regions2 = bioframe.parse_regions(regions2, clr.chromsizes)

    spans = partition(0, len(clr.pixels()), chunksize)
    fields = ["count"] + list(transforms.keys())

    # similar with diagonal summations, pre-generate a block_table listing
    # all of the rectangular blocks and "n_valid" number of pixels per each block:
    records = make_block_table(
        clr, regions1, regions2, weight_name=weight_name, bad_bins=bad_bins
    )

    # combine masking with existing transforms and add a "count" transform:
    if bad_bins is not None:
        # turn bad_bins into a mask of size clr.bins:
        mask_size = len(clr.bins())
        bad_bins_mask = np.ones(mask_size, dtype=int)
        bad_bins_mask[bad_bins] = 0
        #
        masked_transforms = {}
        bin1 = "bin1_id"
        bin2 = "bin2_id"
        for field in fields:
            if field in transforms:
                # combine masking and transform, minding the scope:
                t = transforms[field]
                masked_transforms[field] = (
                    lambda p, t=t, m=bad_bins_mask: t(p) * m[p[bin1]] * m[p[bin2]]
                )
            else:
                # presumably field == "count", mind the scope as well:
                masked_transforms[field] = (
                    lambda p, f=field, m=bad_bins_mask: p[f] * m[p[bin1]] * m[p[bin2]]
                )
        # substitute transforms to the masked_transforms:
        transforms = masked_transforms

    job = partial(
        _blocksum_asymm, clr, fields, transforms, regions1.values, regions2.values
    )
    results = map(job, spans)
    for result in results:
        for i, agg in result.items():
            for field in fields:
                agg_name = "{}.sum".format(field)
                s = agg[field].item()
                if not np.isnan(s):
                    n1 = regions1.loc[i, "name"]
                    n2 = regions2.loc[i, "name"]
                    records[n1, n2][agg_name] += s

    # returning a dataframe for API consistency:
    return pd.DataFrame(
        [{"region1": n1, "region2": n2, **rec} for (n1, n2), rec in records.items()],
        columns=["region1", "region2", "n_valid", "count.sum"]
        + [k + ".sum" for k in transforms.keys()],
    )


def diagsum_from_array(
    A, counts=None, *, offset=0, ignore_diags=2, filter_counts=False, region_name=None
):
    """
    Calculates Open2C-formatted expected for a dense submatrix of a whole
    genome contact map.

    Parameters
    ----------
    A : 2D array
        Normalized submatrix to calculate expected (``balanced.sum``).
    counts : 2D array or None, optional
        Corresponding raw contacts to populate ``count.sum``.
    offset : int or (int, int)
        i- and j- bin offsets of A relative to the parent matrix. If a single
        offset is provided it is applied to both axes.
    ignore_diags : int, optional
        Number of initial diagonals to ignore.
    filter_counts : bool, optional
        Apply the validity mask from balanced matrix to the raw one. Ignored
        when counts is None.
    region_name : str or (str, str), optional
        A custom region name or pair of region names. If provided, region
        columns will be included in the output.

    Notes
    -----
    For regions that cross the main diagonal of the whole-genome contact map,
    the lower triangle "overhang" is ignored.

    Examples
    --------
    >>> A = clr.matrix()[:, :]  # whole genome balanced
    >>> C = clr.matrix(balance=False)[:, :]  # whole genome raw

    Using only balanced data:
    >>> exp = diagsum_from_array(A)

    Using balanced and raw counts:
    >>> exp1 = diagsum_from_array(A, C)

    Using an off-diagonal submatrix
    >>> exp2 = diagsum_from_array(A[:50, 50:], offset=(0, 50))

    """
    if isinstance(offset, (list, tuple)):
        offset1, offset2 = offset
    else:
        offset1, offset2 = offset, offset
    if isinstance(region_name, (list, tuple)):
        region1, region2 = region_name
    elif region_name is not None:
        region1, region2 = region_name, region_name
    A = np.asarray(A, dtype=float)
    if counts is not None:
        counts = np.asarray(counts)
        if counts.shape != A.shape:
            raise ValueError("`counts` must have the same shape as `A`.")

    # Compute validity mask for bins on each axis
    A[~np.isfinite(A)] = 0
    invalid_mask1 = np.sum(A, axis=1) == 0
    invalid_mask2 = np.sum(A, axis=0) == 0

    # Prepare an indicator matrix of "diagonals" (toeplitz) where the lower
    # triangle diagonals wrt the parent matrix are negative.
    # The "outer difference" operation below produces a toeplitz matrix.
    lo1, hi1 = offset1, offset1 + A.shape[0]
    lo2, hi2 = offset2, offset2 + A.shape[1]
    ar1 = np.arange(lo1, hi1, dtype=np.int32)
    ar2 = np.arange(lo2, hi2, dtype=np.int32)
    diag_indicator = ar2[np.newaxis, :] - ar1[:, np.newaxis]
    diag_lo = max(lo2 - hi1 + 1, 0)
    diag_hi = hi2 - lo1

    # Apply the validity mask to the indicator matrix.
    # Both invalid and lower triangle pixels will now have negative indicator values.
    D = diag_indicator.copy()
    D[invalid_mask1, :] = -1
    D[:, invalid_mask2] = -1
    # Drop invalid and lower triangle pixels and flatten.
    mask_per_pixel = D >= 0
    A_flat = A[mask_per_pixel]
    D_flat = D[mask_per_pixel]

    # Group by diagonal and aggregate the number of valid pixels and pixel values.
    diagonals = np.arange(diag_lo, diag_hi, dtype=int)
    n_valid = np.bincount(D_flat, minlength=diag_hi - diag_hi)[diag_lo:]
    balanced_sum = np.bincount(D_flat, weights=A_flat, minlength=diag_hi - diag_lo)[
        diag_lo:
    ]
    # Mask to ignore initial diagonals.
    mask_per_diag = diagonals >= ignore_diags

    # Populate the output dataframe.
    # Include region columns if region names are provided.
    # Include raw pixel counts for each diag if counts is provided.
    df = pd.DataFrame({"diag": diagonals, "n_valid": n_valid})

    if region_name is not None:
        df.insert(0, "region1", region1)
        df.insert(1, "region2", region2)

    if counts is not None:
        # Either count everything or apply the same filtering as A.
        if filter_counts:
            C_flat = counts[mask_per_pixel]
            count_sum = np.bincount(
                D_flat, weights=C_flat, minlength=diag_hi - diag_lo
            )[diag_lo:]
        else:
            mask_per_pixel = diag_indicator >= 0
            D_flat = diag_indicator[mask_per_pixel]
            C_flat = counts[mask_per_pixel]
            count_sum = np.bincount(
                D_flat, weights=C_flat, minlength=diag_hi - diag_lo
            )[diag_lo:]
        count_sum[~mask_per_diag] = np.nan
        df["count.sum"] = count_sum

    balanced_sum[~mask_per_diag] = np.nan
    df["balanced.sum"] = balanced_sum

    return df


def logbin_expected(
    exp,
    summary_name="balanced.sum",
    bins_per_order_magnitude=10,
    bin_layout="fixed",
    smooth=lambda x: numutils.robust_gauss_filter(x, 2),
    min_nvalid=200,
    min_count=50,
):
    """
    Logarithmically bins expected as produced by diagsum method.

    Parameters
    ----------
    exp : DataFrame
        DataFrame produced by diagsum

    summary_name : str, optional
        Name of the column of exp-DataFrame to use as a diagonal summary.
        Default is "balanced.sum".

    bins_per_order_magnitude : int, optional
        How many bins per order of magnitude. Default of 10 has a ratio of
        neighboring bins of about 1.25

    bin_layout : "fixed", "longest_region", or array
        "fixed" means that bins are exactly the same for different datasets,
        and only depend on bins_per_order_magnitude

        "longest_region" means that the last bin will end at size of the
        longest region.
            GOOD: the last bin will have as much data as possible.
            BAD: bin edges will end up different for different datasets, you
            can't divide them by each other

        array: provide your own bin edges. Can be of any size, and end at any
        value. Bins exceeding the size of the largest region will be simply
        ignored.

    smooth : callable
        A smoothing function to be applied to log(P(s)) and log(x)
        before calculating P(s) slopes for by-region data

    min_nvalid : int
        For each region, throw out bins (log-spaced) that have less than
        min_nvalid valid pixels
        This will ensure that each entree in Pc_by_region has at least n_valid
        valid pixels
        Don't set it to zero, or it will introduce bugs. Setting it to 1 is OK,
        but not recommended.

    min_count : int
        If counts are found in the data, then for each region, throw out bins
        (log-spaced)
        that have more than min_counts of counts.sum (raw Hi-C counts).
        This will ensure that each entree in Pc_by_region has at least
        min_count raw Hi-C reads

    Returns
    -------
    Pc : DataFrame
        dataframe of contact probabilities and spread across regions
    slope : ndarray
        slope of Pc(s) on a log-log plot and spread across regions
    bins : ndarray
        an array of bin edges used for calculating P(s)

    Notes
    -----
    For main Pc and slope, the algorithm is the following

    1. concatenate all the expected for all regions into a large dataframe.
    2. create logarithmically-spaced bins of diagonals (or use provided)
    3. pool together n_valid and balanced.sum for each region and for each bin
    4. calculate the average diagonal for each bucket, weighted by n_valid
    5. divide balanced.sum by n_valid after summing for each bucket (not before)
    6. calculate the slope in log space (for each region)

    X values are not midpoints of bins
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    In step 4, we calculate the average diag index weighted by n_valid. This
    seems counter-intuitive, but it actually is justified.

    Let's take the worst case scenario. Let there be a bin from 40MB to 44MB.
    Let there be a region that is exactly 41 MB long. The midpoint of the bin
    is at 42MB. But the only part of this region belonging to this bin is
    actually between 40MB and 41MB. Moreover, the "average" read in this
    little triangle of the heatmap is actually not coming even from 40.5 MB
    because the triangle is getting narrower towards 41MB. The center of mass
    of a triangle is 1/3 of the way up, or 40.33 MB. So an average read for
    this region in this bin is coming from 40.33.

    Consider the previous bin, say, from 36MB to 40MB. The heatmap there is a
    trapezoid with a long side of 5MB, the short side of 1MB, and height of
    4MB. The center of mass of this trapezoid is at 36 + 14/9 = 37.55MB,
    and not at 38MB. So the last bin center is definitely mis-assigned, and
    the second-to-last bin center is off by some 25%. This would lead to a 25%
    error of the P(s) slope estimated between the third-to-last and
    second-to-last bin.

    In presence of missing bins, this all becomes more complex, but this kind
    of averaging should take care of everything. It follows a general
    principle: when averaging the y values with some weights, one needs to
    average the x values with the same weights. The y values here are being
    added together, so per-diag means are effectively averaged with the weight
    of n_valid. Therefore, the x values (diag) should be averaged with the
    same weights.

    Other considerations
    ~~~~~~~~~~~~~~~~~~~~
    Steps #3 and #5 are important because the ratio of sums does not equal to
    the sum of ratios, and the former is more correct (the latter is more
    susceptible to noise). It is generally better to divide at the very end,
    rather than dividing things for each diagonal.

    Here we divide at the end twice: first we divide balanced.sum by n_valid
    for each region, then we effectively multiply it back up and divide it for
    each bin when combining different regions (see weighted average in the
    next function).

    Smoothing P(s) for the slope
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For calcuating the slope, we apply smoothing to the P(s) to ensure the
    slope is not too noisy. There are several caveats here: the P(s) has to
    be smoothed in logspace, and both P and s have to be smoothed. It is
    discussed in detail here

    https://gist.github.com/mimakaev/4becf1310ba6ee07f6b91e511c531e73

    Examples
    --------
    For example, see this gist: https://gist.github.com/mimakaev/e9117a7fcc318e7904702eba5b47d9e6

    """
    from cooltools.lib.numutils import logbins

    raw_summary_name = "count.sum"
    exp_summary_base, *_ = summary_name.split(".")
    Pc_name = f"{exp_summary_base}.avg"
    diag_name = "diag"
    diag_avg_name = f"{diag_name}.avg"

    exp = exp[~pd.isna(exp[summary_name])].copy()
    exp[diag_avg_name] = exp.pop(diag_name)  # "average" or weighted diagonals
    diagmax = exp[diag_avg_name].max()

    # create diag_bins based on chosen layout:
    if bin_layout == "fixed":
        diag_bins = numutils.persistent_log_bins(
            10, bins_per_order_magnitude=bins_per_order_magnitude
        )
    elif bin_layout == "longest_region":
        diag_bins = logbins(1, diagmax + 1, ratio=10 ** (1 / bins_per_order_magnitude))
    else:
        diag_bins = bin_layout

    if diag_bins[-1] < diagmax:
        raise ValueError(
            "Genomic separation bins end is less than the size of the largest region"
        )

    # assign diagonals in exp DataFrame to diag_bins, i.e. give them ids:
    exp["diag_bin_id"] = (
        np.searchsorted(diag_bins, exp[diag_avg_name], side="right") - 1
    )
    exp = exp[exp["diag_bin_id"] >= 0]

    # constructing expected grouped by region
    byReg = exp.copy()

    # this averages diag_avg with the weight equal to n_valid, and sums everything else
    byReg[diag_avg_name] *= byReg["n_valid"]
    byRegExp = byReg.groupby(["region", "diag_bin_id"]).sum()
    byRegExp[diag_avg_name] /= byRegExp["n_valid"]

    byRegExp = byRegExp.reset_index()
    byRegExp = byRegExp[byRegExp["n_valid"] > min_nvalid]  # filtering by n_valid
    byRegExp[Pc_name] = byRegExp[summary_name] / byRegExp["n_valid"]
    byRegExp = byRegExp[byRegExp[Pc_name] > 0]  # drop diag_bins with 0 counts
    if min_count:
        if raw_summary_name in byRegExp:
            byRegExp = byRegExp[byRegExp[raw_summary_name] > min_count]
        else:
            warnings.warn(
                RuntimeWarning(f"{raw_summary_name} not found in the input expected")
            )

    byRegExp["diag_bin_start"] = diag_bins[byRegExp["diag_bin_id"].values]
    byRegExp["diag_bin_end"] = diag_bins[byRegExp["diag_bin_id"].values + 1] - 1

    byRegDer = []
    for reg, subdf in byRegExp.groupby("region"):
        subdf = subdf.sort_values("diag_bin_id")
        valid = np.minimum(subdf["n_valid"].values[:-1], subdf["n_valid"].values[1:])
        mids = np.sqrt(
            subdf[diag_avg_name].values[:-1] * subdf[diag_avg_name].values[1:]
        )
        slope = np.diff(smooth(np.log(subdf[Pc_name].values))) / np.diff(
            smooth(np.log(subdf[diag_avg_name].values))
        )
        newdf = pd.DataFrame(
            {
                diag_avg_name: mids,
                "slope": slope,
                "n_valid": valid,
                "diag_bin_id": subdf["diag_bin_id"].values[:-1],
            }
        )
        newdf["region"] = reg
        byRegDer.append(newdf)
    byRegDer = pd.concat(byRegDer).reset_index(drop=True)
    return byRegExp, byRegDer, diag_bins[: byRegExp["diag_bin_id"].max() + 2]


def combine_binned_expected(
    binned_exp,
    binned_exp_slope=None,
    Pc_name="balanced.avg",
    der_smooth_function_combined=lambda x: numutils.robust_gauss_filter(x, 1.3),
    spread_funcs="logstd",
    spread_funcs_slope="std",
    minmax_drop_bins=2,
    concat_original=False,
):
    """
    Combines by-region log-binned expected and slopes into genome-wide averages,
    handling small chromosomes and "corners" in an optimal fashion, robust to
    outliers. Calculates spread of by-chromosome P(s) and slopes, also in an optimal fashion.

    Parameters
    ----------
    binned_exp: dataframe
        binned expected as outputed by logbin_expected

    binned_exp_slope : dataframe or None
        If provided, estimates spread of slopes.
        Is necessary if concat_original is True

    Pc_name : str
        Name of the column with the probability of contacts.
        Defaults to "balanced.avg".

    der_smooth_function_combined : callable
        A smoothing function for calculating slopes on combined data

    spread_funcs: "minmax", "std", "logstd" or a function (see below)
        A way to estimate the spread of the P(s) curves between regions.
        * "minmax" - use the minimum/maximum of by-region P(s)
        * "std" - use weighted standard deviation of P(s) curves (may produce negative results)
        * "logstd" (recommended) weighted standard deviation in logspace (as seen on the plot)

    spread_funcs_slope: "minmax", "std" or a funciton
        Similar to spread_func, but for slopes rather than P(s)

    concat_original: bool (default = False)
        Append original dataframe, and put combined under region "combined"

    Returns
    -------
    scal, slope_df

    Notes
    -----
    This function does not calculate errorbars. The spread is not the deviation of the mean,
    and rather is representative of variability between chromosomes.


    Calculating errorbars/spread

    1. Take all by-region P(s)
    2. For "minmax", remove the last var_drop_last_bins bins for each region
       (by default two. They are most noisy and would inflate the
       spread for the last points). Min/max are most susceptible to this.
    3. Groupby P(s) by region
    4. Apply spread_funcs to the pd.GroupBy object. Options are:
       * minimum and maximum ("minmax"),
       * weighted standard deviation ("std"),
       * weighted standard deviation in logspace ("logstd", default) or two custom functions
       We do not remove the last bins for "std" / "logstd" because we are
       doing weighted standard deviation. Therefore, noisy "ends" of regions
       would contribute very little to this.
    5. Append them to the P(s) for the same bin.

    As a result, by for minmax, we do not estimate spread for the last
    two bins. This is because there are often very few chromosomal arms there,
    and different arm measurements are noisy. For other methods, we do
    estimate the spread there, and noisy last bins are taken care of by the
    weighted standard deviation. However, the spread in the last bins may be
    noisy, and may become a 0 if only one region is contributing to the last
    pixel.
    """
    diag_avg_name = "diag.avg"
    # combine pre-logbinned expecteds
    scal = numutils.weighted_groupby_mean(
        binned_exp[
            [
                Pc_name,
                "diag_bin_id",
                "n_valid",
                diag_avg_name,
                "diag_bin_start",
                "diag_bin_end",
            ]
        ],
        group_by="diag_bin_id",
        weigh_by="n_valid",
        mode="mean",
    )

    # for every diagonal calculate the spread of expected
    if spread_funcs == "minmax":
        byRegVar = binned_exp.copy()
        byRegVar = byRegVar.loc[
            byRegVar.index.difference(
                byRegVar.groupby("region")["n_valid"].tail(minmax_drop_bins).index
            )
        ]
        low_err = byRegVar.groupby("diag_bin_id")[Pc_name].min()
        high_err = byRegVar.groupby("diag_bin_id")[Pc_name].max()
    elif spread_funcs == "std":
        var = numutils.weighted_groupby_mean(
            binned_exp[[Pc_name, "diag_bin_id", "n_valid"]],
            group_by="diag_bin_id",
            weigh_by="n_valid",
            mode="std",
        )[Pc_name]
        low_err = scal[Pc_name] - var
        high_err = scal[Pc_name] + var
    elif spread_funcs == "logstd":
        var = numutils.weighted_groupby_mean(
            binned_exp[[Pc_name, "diag_bin_id", "n_valid"]],
            group_by="diag_bin_id",
            weigh_by="n_valid",
            mode="logstd",
        )[Pc_name]
        low_err = scal[Pc_name] / var
        high_err = scal[Pc_name] * var
    else:
        low_err, high_err = spread_funcs(binned_exp, scal)

    scal["low_err"] = low_err
    scal["high_err"] = high_err

    # re-calculate slope of the combined expected (log,smooth,diff)
    f = der_smooth_function_combined
    slope = np.diff(f(np.log(scal[Pc_name].values))) / np.diff(
        f(np.log(scal[diag_avg_name].values))
    )
    valid = np.minimum(scal["n_valid"].values[:-1], scal["n_valid"].values[1:])
    mids = np.sqrt(scal[diag_avg_name].values[:-1] * scal[diag_avg_name].values[1:])
    slope_df = pd.DataFrame(
        {
            diag_avg_name: mids,
            "slope": slope,
            "n_valid": valid,
            "diag_bin_id": scal.index.values[:-1],
        }
    )
    slope_df = slope_df.set_index("diag_bin_id")

    # when pre-region slopes are provided, calculate spread of slopes
    if binned_exp_slope is not None:
        if spread_funcs_slope == "minmax":
            byRegDer = binned_exp_slope.copy()
            byRegDer = byRegDer.loc[
                byRegDer.index.difference(
                    byRegDer.groupby("region")["n_valid"].tail(minmax_drop_bins).index
                )
            ]
            low_err = byRegDer.groupby("diag_bin_id")["slope"].min()
            high_err = byRegDer.groupby("diag_bin_id")["slope"].max()
        elif spread_funcs_slope == "std":
            var = numutils.weighted_groupby_mean(
                binned_exp_slope[["slope", "diag_bin_id", "n_valid"]],
                group_by="diag_bin_id",
                weigh_by="n_valid",
                mode="std",
            )["slope"]
            low_err = slope_df["slope"] - var
            high_err = slope_df["slope"] + var

        else:
            low_err, high_err = spread_funcs_slope(binned_exp_slope, scal)
        slope_df["low_err"] = low_err
        slope_df["high_err"] = high_err

    slope_df = slope_df.reset_index()
    scal = scal.reset_index()

    # append "combined" expected/slopes to the input DataFrames (not in-place)
    if concat_original:
        scal["region"] = "combined"
        slope_df["region"] = "combined"
        scal = pd.concat([scal, binned_exp], sort=False).reset_index(drop=True)
        slope_df = pd.concat([slope_df, binned_exp_slope], sort=False).reset_index(
            drop=True
        )

    return scal, slope_df


def interpolate_expected(
    expected,
    binned_expected,
    columns=["balanced.avg"],
    kind="quadratic",
    by_region=True,
    extrapolate_small_s=False,
):
    """
    Interpolates expected to match binned_expected.
    Basically, this function smoothes the original expected according to the logbinned expected.
    It could either use by-region expected (each region will have different expected)
    or use combined binned_expected (all regions will have the same expected after that)

    Such a smoothed expected should be used to calculate observed/expected for downstream analysis.

    Parameters
    ----------
    expected: pd.DataFrame
        expected as returned by diagsum
    binned_expected: pd.DataFrame
        binned expected (combined or not)
    columns: list[str] (optional)
        Columns to interpolate. Must be present in binned_expected,
        but not necessarily in expected.
    kind: str (optional)
        Interpolation type, according to scipy.interpolate.interp1d
    by_region: bool or str (optional)
        Whether to do interpolation by-region (default=True).
        False means use one expected for all regions (use entire table).
        If a region name is provided, expected for that region is used.

    """

    exp_int = expected.copy()
    gr_exp = exp_int.groupby("region")  # groupby original expected by region

    if by_region is not False and "region" not in binned_expected:
        warnings.warn("Region column not found, assuming combined expected")
        by_region = False

    if by_region is True:
        # groupby expected
        gr_binned = binned_expected.groupby("region")
    elif by_region is not False:
        # extract a region that we want to use
        binned_expected = binned_expected[binned_expected["region"] == by_region]

    if by_region is not True:
        # check that we have no duplicates in expected
        assert len(binned_expected["diag_bin_id"].drop_duplicates()) == len(
            binned_expected
        )

    interp_dfs = []

    for reg, df_orig in gr_exp:
        if by_region is True:  # use binned expected for this region
            if reg not in gr_binned.groups:
                continue
            subdf = gr_binned.get_group(reg)
        else:
            subdf = binned_expected

        diag_orig = df_orig["diag"].values
        diag_mid = (subdf["diag_bin_start"] + subdf["diag_bin_end"]) / 2
        interp_df = pd.DataFrame(
            index=df_orig.index
        )  # df to put interpolated values in
        with np.errstate(invalid="ignore", divide="ignore"):
            for colname in columns:  # interpolate each column
                value_column = subdf[colname]
                interp = interp1d(
                    np.log(diag_mid),
                    np.log(value_column),
                    kind=kind,
                    fill_value="extrapolate",
                )
                interp_df[colname] = np.exp(interp(np.log(diag_orig)))
            if not extrapolate_small_s:
                mask = diag_orig >= subdf["diag_bin_start"].min()
                interp_df = interp_df.iloc[mask]
        interp_dfs.append(interp_df)
    interp_df = pd.concat(interp_dfs)
    for i in interp_df.columns:
        exp_int[i] = interp_df[i]
    return exp_int
