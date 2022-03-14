from itertools import chain, combinations, combinations_with_replacement
from functools import partial

import warnings
import multiprocess as mp

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d

from cooler.tools import partition
import cooler
import bioframe
from ..lib import assign_supports, numutils
from ..lib.checks import is_compatible_viewframe, is_cooler_balanced
from ..lib.common import make_cooler_view
from ..lib.schemas import diag_expected_dtypes, block_expected_dtypes

from ..sandbox import expected_smoothing

# common expected_df column names, take from schemas
_REGION1 = list(diag_expected_dtypes)[0]
_REGION2 = list(diag_expected_dtypes)[1]
_DIST = list(diag_expected_dtypes)[2]
_NUM_VALID = list(diag_expected_dtypes)[3]


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

    # alias for np.flatnonzero inside `make_diag_table`
    where = np.flatnonzero

    def _make_diag_table(n_bins, bad_locs):
        diags = pd.DataFrame(index=pd.Series(np.arange(n_bins), name=_DIST))
        diags["n_elem"] = count_all_pixels_per_diag(n_bins)
        diags[_NUM_VALID] = diags["n_elem"] - count_bad_pixels_per_diag(
            n_bins, bad_locs
        )
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


def make_diag_tables(clr, regions, regions2=None, clr_weight_name="weight"):
    """
    For every region infer diagonals that intersect this region and calculate
    the size of these intersections in pixels, both "total" and "n_valid",
    where "n_valid" does not count "bad" pixels.

    "Bad" pixels are inferred from the balancing weight column `clr_weight_name`.
    When `clr_weight_name` is None, raw data is used, and no "bad" pixels are exclued.

    When `regions2` are provided, all intersecting diagonals are reported for
    each rectangular and asymmetric block defined by combinations of matching
    elements of `regions` and `regions2`.
    Otherwise only `regions`-based symmetric square blocks are considered.
    Only intra-chromosomal regions are supported.

    Parameters
    ----------
    clr : cooler.Cooler
        Input cooler
    regions : viewframe or viewframe-like dataframe
        viewframe without repeated entries or viewframe-like dataframe with repeated entries
    regions2 : viewframe or viewframe-like dataframe
        viewframe without repeated entries or viewframe-like dataframe with repeated entries
    clr_weight_name : str
        name of the weight column in the clr bin-table,
        Balancing weight is used to infer bad bins, set to
        `None` is masking bad bins is not desired for raw data.

    Returns
    -------
    diag_tables : dict
        dictionary with DataFrames of relevant diagonals for every region.
    """

    try:  # Run regular viewframe conversion:
        regions = bioframe.make_viewframe(
            regions, check_bounds=clr.chromsizes
        ).to_numpy()
        if regions2 is not None:
            regions2 = bioframe.make_viewframe(
                regions2, check_bounds=clr.chromsizes
            ).to_numpy()
    except ValueError:  # If there are non-unique entries in regions1/2, possible only for asymmetric expected:
        regions = pd.concat(
            [
                bioframe.make_viewframe([region], check_bounds=clr.chromsizes)
                for i, region in regions.iterrows()
            ]
        ).to_numpy()
        regions2 = pd.concat(
            [
                bioframe.make_viewframe([region], check_bounds=clr.chromsizes)
                for i, region in regions2.iterrows()
            ]
        ).to_numpy()

    bins = clr.bins()[:]
    if clr_weight_name is None:
        # ignore bad bins
        sizes = dict(bins.groupby("chrom").size())
        bad_bin_dict = {
            chrom: np.zeros(sizes[chrom], dtype=bool) for chrom in sizes.keys()
        }
    elif is_cooler_balanced(clr, clr_weight_name):
        groups = dict(iter(bins.groupby("chrom")[clr_weight_name]))
        bad_bin_dict = {
            chrom: np.array(groups[chrom].isnull()) for chrom in groups.keys()
        }
    else:
        raise ValueError(
            f"provided cooler is not balanced, or weight {clr_weight_name} is missing"
        )

    diag_tables = {}
    for i, region in enumerate(regions):
        chrom, start1, end1, name1 = region
        if regions2 is not None:
            chrom2, start2, end2, name2 = regions2[i]
            # cis-only for now:
            if not (chrom2 == chrom):
                raise ValueError(
                    "regions/2 have to be on the same chrom to generate diag_tables"
                )
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


def make_block_table(clr, regions1, regions2, clr_weight_name="weight"):
    """
    Creates a table of total and valid pixels for a set of rectangular genomic blocks
    defined by regions1 and regions2.
    For every block calculate its "area" in pixels ("n_total"), and calculate
    number of "valid" pixels ("n_valid").
    Valid pixels exclude "bad" pixels, which are inferred from the balancing
    weight column `clr_weight_name`.

    When `clr_weight_name` is None, raw data is used, and no "bad" pixels are exclued.

    Parameters
    ----------
    clr : cooler.Cooler
        Input cooler
    regions1 : viewframe-like dataframe
        viewframe-like dataframe, where repeated entries are allowed
    regions2 : viewframe-like dataframe
        viewframe-like dataframe, where repeated entries are allowed
    clr_weight_name : str
        name of the weight column in the cooler bins-table, used
        for masking bad pixels.
        When clr_weight_name is None, no bad pixels are masked.

    Returns
    -------
    block_table : dict
        dictionary for blocks that are 0-indexed
    """

    try:  # Run regular viewframe conversion:
        regions1 = bioframe.make_viewframe(
            regions1, check_bounds=clr.chromsizes
        ).to_numpy()
        regions2 = bioframe.make_viewframe(
            regions2, check_bounds=clr.chromsizes
        ).to_numpy()
    except ValueError:  # Might be non-unique entries in regions:
        regions1 = pd.concat(
            [
                bioframe.make_viewframe([region], check_bounds=clr.chromsizes)
                for i, region in regions1.iterrows()
            ]
        ).to_numpy()
        regions2 = pd.concat(
            [
                bioframe.make_viewframe([region], check_bounds=clr.chromsizes)
                for i, region in regions2.iterrows()
            ]
        ).to_numpy()

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

        # now we need to combine it with the balancing weights
        if clr_weight_name is None:
            bad_bins_x = 0
            bad_bins_y = 0
        elif is_cooler_balanced(clr, clr_weight_name):
            # count "bad" bins filtered by balancing:
            bad_bins_x = clr.bins()[clr_weight_name][lo1:hi1].isnull().sum()
            bad_bins_y = clr.bins()[clr_weight_name][lo2:hi2].isnull().sum()
        else:
            raise ValueError(
                f"cooler is not balanced or weight {clr_weight_name} is missing"
            )

        # calculate total and bad pixels per block:
        n_tot = count_all_pixels_per_block(x, y)
        n_bad = count_bad_pixels_per_block(x, y, bad_bins_x, bad_bins_y)

        # fill in "block_table" with number of valid pixels:
        block_table[name1, name2] = {_NUM_VALID: n_tot - n_bad}

    return block_table


def _diagsum_symm(clr, fields, transforms, clr_weight_name, regions, span):
    """
    calculates diagonal/distance summary for a collection of
    square symmetric blocks defined by the "regions".

    Return:
    dictionary of DataFrames with diagonal/distance
    sums for the "fields", and 0-based indexes of square
    genomic regions as keys.
    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)
    # pre-filter cis-only pixels to speed up calculations
    pixels = pixels[pixels["chrom1"] == pixels["chrom2"]].copy()

    # annotate pixels with regions at once
    # book-ended regions still get reannotated
    pixels["r1"] = assign_supports(pixels, regions, suffix="1")
    pixels["r2"] = assign_supports(pixels, regions, suffix="2")
    # select symmetric pixels that have notnull weights
    if clr_weight_name is None:
        pixels = pixels.dropna(subset=["r1", "r2"])
    else:
        pixels = pixels.dropna(
            subset=["r1", "r2", clr_weight_name + "1", clr_weight_name + "2"]
        )
    pixels = pixels[pixels["r1"] == pixels["r2"]]

    # this could further expanded to allow for custom groupings:
    pixels[_DIST] = pixels["bin2_id"] - pixels["bin1_id"]
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    symm_blocks = pixels.groupby("r1")
    return {int(i): block.groupby(_DIST)[fields].sum() for i, block in symm_blocks}


def diagsum_symm(
    clr,
    view_df,
    transforms={},
    clr_weight_name="weight",
    ignore_diags=2,
    chunksize=10000000,
    map=map,
):
    """

    Intra-chromosomal diagonal summary statistics.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        view_dfof regions for intra-chromosomal diagonal summation
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    clr_weight_name : str
        name of the balancing weight vector used to count "bad"
        pixels per diagonal. Set to `None` not to mask
        "bad" pixels (raw data only).
    chunksize : int, optional
        Size of pixel table chunks to process
    ignore_diags : int, optional
        Number of intial diagonals to exclude from statistics
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    Dataframe of diagonal statistics for all regions in the view

    """
    spans = partition(0, len(clr.pixels()), chunksize)
    fields = list(chain(["count"], transforms))
    # names of summary results
    summary_fields = [f"{field}.sum" for field in fields]

    # check viewframe
    try:
        _ = is_compatible_viewframe(
            view_df,
            clr,
            check_sorting=False,  # liberal for this low-level function
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided view_df is not valid") from e

    # prepare dtables: for every region a table with diagonals, number of valid pixels, etc
    dtables = make_diag_tables(clr, view_df, clr_weight_name=clr_weight_name)

    # initialize columns to store summary results in dtables
    for dt in dtables.values():
        for agg_name in summary_fields:
            dt[agg_name] = 0

    # apply _diagsum_symm to chunks of pixels
    job = partial(
        _diagsum_symm, clr, fields, transforms, clr_weight_name, view_df.to_numpy()
    )
    results = map(job, spans)
    # accumulate every chunk of summary results to dtables
    for result in results:
        for i, agg in result.items():
            region = view_df["name"].iat[i]
            # for every field accumulate its aggregate/summary:
            for field, agg_name in zip(fields, summary_fields):
                dtables[region][agg_name] = dtables[region][agg_name].add(
                    agg[field], fill_value=0
                )

    # returning pd.DataFrame for API consistency
    result = []
    for i, dtable in dtables.items():
        dtable = dtable.reset_index()
        # conform with the new expected format, treat regions as 2D
        dtable.insert(0, _REGION1, i)
        dtable.insert(1, _REGION2, i)
        if ignore_diags:
            # fill out summary fields of ignored diagonals with NaN:
            dtable.loc[dtable[_DIST] < ignore_diags, summary_fields] = np.nan
        result.append(dtable)

    return pd.concat(result).reset_index(drop=True)


def _diagsum_pairwise(clr, fields, transforms, clr_weight_name, regions, span):
    """
    calculates diagonal/distance summary for a collection of
    rectangular blocks defined by all pairwise combinations
    of "regions" for intra-chromosomal interactions.

    Return:
    dictionary of DataFrames with diagonal/distance
    sums for the "fields", and (i,j)-like indexes of rectangular
    genomic regions as keys.
    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)
    # pre-filter cis-only pixels to speed up calculations
    pixels = pixels[pixels["chrom1"] == pixels["chrom2"]].copy()

    # annotate pixels with regions at once
    # book-ended regions still get reannotated
    pixels["r1"] = assign_supports(pixels, regions, suffix="1")
    pixels["r2"] = assign_supports(pixels, regions, suffix="2")
    # pre-filter asymetric pixels only that have notnull weights
    if clr_weight_name is None:
        pixels = pixels.dropna(subset=["r1", "r2"])
    else:
        pixels = pixels.dropna(
            subset=["r1", "r2", clr_weight_name + "1", clr_weight_name + "2"]
        )
    # pixels = pixels[ pixels["r1"] != pixels["r2"] ]

    # this could further expanded to allow for custom groupings:
    pixels[_DIST] = pixels["bin2_id"] - pixels["bin1_id"]
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    asymm_blocks = pixels.groupby(["r1", "r2"])
    return {
        (int(i), int(j)): block.groupby(_DIST)[fields].sum()
        for (i, j), block in asymm_blocks
    }


def diagsum_pairwise(
    clr,
    view_df,
    transforms={},
    clr_weight_name="weight",
    ignore_diags=2,
    chunksize=10_000_000,
    map=map,
):
    """

    Intra-chromosomal diagonal summary statistics for asymmetric blocks of
    contact matrix defined as pairwise combinations of regions in "view_df.

    Note
    ----
    This is a special case of asymmetric diagonal summary statistic that is
    efficient and covers the most important practical case of inter-chromosomal
    arms "expected" calculation.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        view_df of regions for intra-chromosomal diagonal summation, has to
        be sorted according to the order of chromosomes in cooler.
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    clr_weight_name : str
        name of the balancing weight vector used to count "bad"
        pixels per diagonal. Set to `None` not to mask
        "bad" pixels (raw data only).
    chunksize : int, optional
        Size of pixel table chunks to process
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    Dataframe of diagonal statistics for all intra-chromosomal blocks defined as
    pairwise combinations of regions in the view

    """
    spans = partition(0, len(clr.pixels()), chunksize)
    fields = list(chain(["count"], transforms))
    # names of summary results
    summary_fields = [f"{field}.sum" for field in fields]

    # check viewframe
    try:
        _ = is_compatible_viewframe(
            view_df,
            clr,
            check_sorting=True,  # required for pairwise combinations
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided view_df is not valid") from e

    # create pairwise combinations of regions from view_df
    all_combinations = combinations_with_replacement(view_df.itertuples(index=False), 2)
    # keep only intra-chromosomal combinations
    cis_combinations = ((r1, r2) for r1, r2 in all_combinations if (r1[0] == r2[0]))
    # unzip regions1 regions2 defining the blocks for summary collection
    regions1, regions2 = zip(*cis_combinations)
    regions1 = pd.DataFrame(regions1)
    regions2 = pd.DataFrame(regions2)
    # prepare dtables: for every region a table with diagonals, number of valid pixels, etc
    dtables = make_diag_tables(clr, regions1, regions2, clr_weight_name=clr_weight_name)

    # initialize columns to store summary results in dtables
    for dt in dtables.values():
        for agg_name in summary_fields:
            dt[agg_name] = 0

    # apply _diagsum_pairwise to chunks of pixels
    job = partial(
        _diagsum_pairwise, clr, fields, transforms, clr_weight_name, view_df.values
    )
    results = map(job, spans)
    # accumulate every chunk of summary results to dtables
    for result in results:
        for (i, j), agg in result.items():
            ni = view_df["name"].iat[i]
            nj = view_df["name"].iat[j]
            # for every field accumulate its aggregate/summary:
            for field, agg_name in zip(fields, summary_fields):
                dtables[ni, nj][agg_name] = dtables[ni, nj][agg_name].add(
                    agg[field], fill_value=0
                )

    # returning a pd.DataFrame for API consistency:
    result = []
    for (i, j), dtable in dtables.items():
        dtable = dtable.reset_index()
        dtable.insert(0, _REGION1, i)
        dtable.insert(1, _REGION2, j)
        if ignore_diags:
            # fill out summary fields of ignored diagonals with NaN:
            dtable.loc[dtable[_DIST] < ignore_diags, summary_fields] = np.nan
        result.append(dtable)
    return pd.concat(result).reset_index(drop=True)


def _blocksum_pairwise(clr, fields, transforms, clr_weight_name, regions, span):
    """
    calculates block summary for a collection of
    rectangular regions defined as pairwise combinations
    of all regions.

    Return:
    a dictionary of block-wide sums for all "fields":
    keys are (i,j)-like, where i and j are 0-based indexes of
    "regions", and a combination of (i,j) defines rectangular block.

    Note:
    Input pixels are assumed to be "symmetric-upper", and "regions"
    to be sorted according to the order of chromosomes in "clr", thus
    i < j.

    """
    lo, hi = span
    bins = clr.bins()[:]
    pixels = clr.pixels()[lo:hi]
    pixels = cooler.annotate(pixels, bins, replace=False)

    pixels["r1"] = assign_supports(pixels, regions, suffix="1")
    pixels["r2"] = assign_supports(pixels, regions, suffix="2")
    # pre-filter asymetric pixels only that have notnull weights
    if clr_weight_name is None:
        pixels = pixels.dropna(subset=["r1", "r2"])
    else:
        pixels = pixels.dropna(
            subset=["r1", "r2", clr_weight_name + "1", clr_weight_name + "2"]
        )
    pixels = pixels[pixels["r1"] != pixels["r2"]]

    # apply transforms, e.g. balancing etc
    for field, t in transforms.items():
        pixels[field] = t(pixels)

    # pairwise-combinations of regions define asymetric pixels-blocks
    pixel_groups = pixels.groupby(["r1", "r2"])
    return {
        (int(i), int(j)): group[fields].sum(skipna=False)
        for (i, j), group in pixel_groups
    }


def blocksum_pairwise(
    clr,
    view_df,
    transforms={},
    clr_weight_name="weight",
    chunksize=1000000,
    map=map,
):
    """
    Summary statistics on rectangular blocks of all (trans-)pairwise combinations
    of genomic regions in the view_df (aka trans-expected).

    Note
    ----
    This is a special case of asymmetric block-level summary stats, that can be
    calculated very efficiently. Regions in view_df are assigned to pixels only
    once and pixels falling into a given asymmetric block i != j are summed up.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        view_df of regions defining blocks for summary calculations,
        has to be sorted according to the order of chromosomes in clr.
    transforms : dict of str -> callable, optional
        Transformations to apply to pixels. The result will be assigned to
        a temporary column with the name given by the key. Callables take
        one argument: the current chunk of the (annotated) pixel dataframe.
    clr_weight_name : str
        name of the balancing weight column in cooler bin-table used
        to count "bad" pixels per block. Set to `None` not ot mask
        "bad" pixels (raw data only).
    chunksize : int, optional
        Size of pixel table chunks to process
    map : callable, optional
        Map functor implementation.

    Returns
    -------
    DataFrame with entries for each blocks: region1, region2, n_valid, count.sum

    """

    # check viewframe
    try:
        _ = is_compatible_viewframe(
            view_df,
            clr,
            check_sorting=False,  # required for pairwise combinations
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("provided view_df is not valid") from e

    spans = partition(0, len(clr.pixels()), chunksize)
    fields = list(chain(["count"], transforms))
    # names of summary results
    summary_fields = [f"{field}.sum" for field in fields]

    # create pairwise combinations of regions from view_df using
    # the standard zip(*bunch_of_tuples) unzipping procedure:
    regions1, regions2 = zip(*combinations(view_df.itertuples(index=False), 2))
    regions1 = pd.DataFrame(regions1)
    regions2 = pd.DataFrame(regions2)
    # similar with diagonal summations, pre-generate a block_table listing
    # all of the rectangular blocks and "n_valid" number of pixels per each block:
    btables = make_block_table(clr, regions1, regions2, clr_weight_name=clr_weight_name)

    # initialize columns to store summary results in btables
    for bt in btables.values():
        for agg_name in summary_fields:
            bt[agg_name] = 0

    # apply _diagsum_pairwise to chunks of pixels
    job = partial(
        _blocksum_pairwise, clr, fields, transforms, clr_weight_name, view_df.to_numpy()
    )
    results = map(job, spans)
    # accumulate every chunk of summary results to dtables
    for result in results:
        for (i, j), agg in result.items():
            ni = view_df["name"].iat[i]
            nj = view_df["name"].iat[j]
            # for every field accumulate its aggregate/summary:
            for field, agg_name in zip(fields, summary_fields):
                btables[ni, nj][agg_name] += np.nan_to_num(agg[field].item())

    # returning a pd.DataFrame for API consistency:
    return pd.DataFrame(
        [
            {_REGION1: n1, _REGION2: n2, **btable}
            for (n1, n2), btable in btables.items()
        ],
        columns=list(block_expected_dtypes) + summary_fields,
    )


# user-friendly wrapper for diagsum_symm and diagsum_pairwise - part of new "public" API
def expected_cis(
    clr,
    view_df=None,
    intra_only=True,
    smooth=True,
    aggregate_smoothed=True,
    smooth_sigma=0.1,
    clr_weight_name="weight",
    ignore_diags=2,  # should default to cooler info
    chunksize=10_000_000,
    nproc=1,
):
    """
    Calculate average interaction frequencies as a function of genomic
    separation between pixels i.e. interaction decay with distance.
    Genomic separation aka "dist" is measured in the number of bins,
    and defined as an index of a diagonal on which pixels reside (bin1_id - bin2_id).

    Average values are reported in the columns with names {}.avg, and they
    are calculated as a ratio between a corresponding sum {}.sum and the
    total number of "valid" pixels on the diagonal "n_valid".

    When balancing weights (clr_weight_name=None) are not applied to the data, there is no
    masking of bad bins performed.


    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        a collection of genomic intervals where expected is calculated
        otherwise expected is calculated for full chromosomes.
        view_df has to be sorted, when inter-regions expected is requested,
        i.e. intra_only is False.
    intra_only: bool
        Return expected only for symmetric intra-regions defined by view_df,
        i.e. chromosomes, chromosomal-arms, intra-domains, etc.
        When False returns expected both for symmetric intra-regions and
        assymetric inter-regions.
    smooth: bool
        Apply smoothing to cis-expected. Will be stored in an additional column
    aggregate_smoothed: bool
        When smoothing, average over all regions, ignored without smoothing.
    smooth_sigma: float
        Control smoothing with the standard deviation of the smoothing Gaussian kernel.
        Ignored without smoothing.
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    ignore_diags : int, optional
        Number of intial diagonals to exclude results
    chunksize : int, optional
        Size of pixel table chunks to process
    nproc : int, optional
        How many processes to use for calculation

    Returns
    -------
    DataFrame with summary statistic of every diagonal of every symmetric
    or asymmetric block:
    region1, region2, diag, n_valid, count.sum count.avg, etc

    """

    if view_df is None:
        if not intra_only:
            raise ValueError(
                "asymmetric regions has to be smaller then full chromosomes, use view_df"
            )
        else:
            # Generate viewframe from clr.chromsizes:
            view_df = make_cooler_view(clr)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                # must be sorted for asymmetric case
                check_sorting=(not intra_only),
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # define transforms - balanced and raw ('count') for now
    if clr_weight_name is None:
        # no transforms
        transforms = {}
    elif is_cooler_balanced(clr, clr_weight_name):
        # define balanced data transform:
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
    else:
        raise ValueError(
            "cooler is not balanced, or"
            f"balancing weight {clr_weight_name} is not available in the cooler."
        )

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly
    try:
        if intra_only:
            result = diagsum_symm(
                clr,
                view_df,
                transforms=transforms,
                clr_weight_name=clr_weight_name,
                ignore_diags=ignore_diags,
                chunksize=chunksize,
                map=map_,
            )
        else:
            result = diagsum_pairwise(
                clr,
                view_df,
                transforms=transforms,
                clr_weight_name=clr_weight_name,
                ignore_diags=ignore_diags,
                chunksize=chunksize,
                map=map_,
            )
    finally:
        if nproc > 1:
            pool.close()

    # calculate actual averages by dividing sum by n_valid:
    for key in chain(["count"], transforms):
        result[f"{key}.avg"] = result[f"{key}.sum"] / result[_NUM_VALID]

    # additional smoothing and aggregating options would add columns only, not replace them
    if smooth:
        result_smooth = expected_smoothing.agg_smooth_cvd(
            result,
            sigma_log10=smooth_sigma,
        )
        # add smoothed columns to the result (only balanced for now)
        result = result.merge(
            result_smooth[["balanced.avg.smoothed", _DIST]],
            on=[_REGION1, _REGION2, _DIST],
            how="left",
        )
        if aggregate_smoothed:
            result_smooth_agg = expected_smoothing.agg_smooth_cvd(
                result,
                groupby=None,
                sigma_log10=smooth_sigma,
            ).rename(columns={"balanced.avg.smoothed": "balanced.avg.smoothed.agg"})
            # add smoothed columns to the result
            result = result.merge(
                result_smooth_agg[["balanced.avg.smoothed.agg", _DIST]],
                on=[
                    _DIST,
                ],
                how="left",
            )

    return result


# user-friendly wrapper for diagsum_symm and diagsum_pairwise - part of new "public" API
def expected_trans(
    clr,
    view_df=None,
    clr_weight_name="weight",
    chunksize=10_000_000,
    nproc=1,
):
    """
    Calculate average interaction frequencies for inter-chromosomal
    blocks defined as pairwise combinations of regions in view_df.

    An expected level of interactions between disjoint chromosomes
    is calculated as a simple average, as there is no notion of genomic
    separation for a pair of chromosomes and contact matrix for these
    regions looks "flat".

    Average values are reported in the columns with names {}.avg, and they
    are calculated as a ratio between a corresponding sum {}.sum and the
    total number of "valid" pixels on the diagonal "n_valid".


    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        a collection of genomic intervals where expected is calculated
        otherwise expected is calculated for full chromosomes, has to be sorted.
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Use raw unbalanced data, when None.
    chunksize : int, optional
        Size of pixel table chunks to process
    nproc : int, optional
        How many processes to use for calculation

    Returns
    -------
    DataFrame with summary statistic for every trans-blocks:
    region1, region2, n_valid, count.sum count.avg, etc

    """

    if view_df is None:
        # Generate viewframe from clr.chromsizes:
        view_df = make_cooler_view(clr)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                # must be sorted for pairwise regions combinations
                # to be in the upper right of the heatmap
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # define transforms - balanced and raw ('count') for now
    if clr_weight_name is None:
        # no transforms
        transforms = {}
    elif is_cooler_balanced(clr, clr_weight_name):
        # define balanced data transform:
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
    else:
        raise ValueError(
            "cooler is not balanced, or"
            f"balancing weight {clr_weight_name} is not available in the cooler."
        )

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly
    try:
        result = blocksum_pairwise(
            clr,
            view_df,
            transforms=transforms,
            clr_weight_name=clr_weight_name,
            chunksize=chunksize,
            map=map_,
        )
    finally:
        if nproc > 1:
            pool.close()

    # keep only trans interactions for the user-friendly function:
    _name_to_region = view_df.set_index("name")
    _r1_chroms = _name_to_region.loc[result[_REGION1]]["chrom"].values
    _r2_chroms = _name_to_region.loc[result[_REGION2]]["chrom"].values
    # trans-data only:
    result = result.loc[_r1_chroms != _r2_chroms].reset_index(drop=True)

    # calculate actual averages by dividing sum by n_valid:
    for key in chain(["count"], transforms):
        result[f"{key}.avg"] = result[f"{key}.sum"] / result[_NUM_VALID]

    return result


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
    invalid_mask1 = np.sum(np.isnan(A), axis=1) == A.shape[0]
    invalid_mask2 = np.sum(np.isnan(A), axis=0) == A.shape[1]

    A[~np.isfinite(A)] = 0

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
    n_valid = np.bincount(D_flat, minlength=diag_hi - diag_lo)[diag_lo:]
    balanced_sum = np.bincount(D_flat, weights=A_flat, minlength=diag_hi - diag_lo)[
        diag_lo:
    ]
    # Mask to ignore initial diagonals.
    mask_per_diag = diagonals >= ignore_diags

    # Populate the output dataframe.
    # Include region columns if region names are provided.
    # Include raw pixel counts for each diag if counts is provided.
    df = pd.DataFrame({_DIST: diagonals, _NUM_VALID: n_valid})

    if region_name is not None:
        df.insert(0, _REGION1, region1)
        df.insert(1, _REGION2, region2)

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
    Logarithmically bins expected as produced by diagsum_symm method.

    Parameters
    ----------
    exp : DataFrame
        DataFrame produced by diagsum_symm

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

    def _get_diag_bins(bin_layout, diagmax, bins_per_order_magnitude):
        """
        create the logbins themselves based on layout, maxdiag, etc.
        """
        # create diag_bins based on chosen layout:
        if bin_layout == "fixed":
            diag_bins = numutils.persistent_log_bins(
                10, bins_per_order_magnitude=bins_per_order_magnitude
            )
        elif bin_layout == "longest_region":
            diag_bins = numutils.logbins(
                1, diagmax + 1, ratio=10 ** (1 / bins_per_order_magnitude)
            )
        elif isinstance(bin_layout, np.ndarray):
            diag_bins = bin_layout
        else:
            raise ValueError("bin_layout can be fixed, longest_region or an ndarray")

        if diag_bins[-1] < diagmax:
            raise ValueError(
                "Genomic separation bins end is less than the size of the largest region"
            )
        return diag_bins

    def _get_weighted_expected(
        exp_filtered,
        diag_bins,
        digit_id_name,
        weighted_dist_name,
        Pc_name,
        summary_name,
        raw_summary_name,
        min_nvalid=0,
        min_count=0,
    ):
        """
        given the logbins (diag_bins) and expected with digitized distances (and pre-filtered)
        calculate weighted distance per logbin and weighted expected.
        """

        # digitize dist: assign diagonals in expected df to diag_bins, - give them ids:
        exp_filtered[digit_id_name] = (
            np.searchsorted(diag_bins, exp_filtered[weighted_dist_name], side="right")
            - 1
        )
        exp_filtered = exp_filtered[
            exp_filtered[digit_id_name] >= 0
        ]  # ignore those that do not fit into diag_bins

        # constructing expected grouped by region
        byReg = exp_filtered.copy()

        # this averages diag_avg with the weight equal to n_valid, and sums everything else
        byReg[weighted_dist_name] *= byReg[_NUM_VALID]  # dist * n_valid
        byRegExp = byReg.groupby(
            [_REGION1, _REGION2, digit_id_name]
        ).sum()  # sum in each logbin
        byRegExp[weighted_dist_name] /= byRegExp[
            _NUM_VALID
        ]  # sum(dist*n_valid) / sum(n_valid)

        byRegExp = byRegExp.reset_index()
        byRegExp = byRegExp[byRegExp[_NUM_VALID] > min_nvalid]  # filtering by n_valid
        byRegExp[Pc_name] = byRegExp[summary_name] / byRegExp[_NUM_VALID]
        byRegExp = byRegExp[byRegExp[Pc_name] > 0]  # drop diag_bins with 0 counts
        # try to filter by the matching raw number of interactions
        if min_count:
            if raw_summary_name in byRegExp:
                byRegExp = byRegExp[byRegExp[raw_summary_name] > min_count]
            else:
                warnings.warn(
                    RuntimeWarning(
                        f"{raw_summary_name} not found in the input expected"
                    )
                )

        byRegExp["dist_bin_start"] = diag_bins[byRegExp[digit_id_name].to_numpy()]
        byRegExp["dist_bin_end"] = diag_bins[byRegExp[digit_id_name].to_numpy() + 1] - 1

        return byRegExp

    def _get_slopes(
        logbin_exp,
        digit_id_name,
        weighted_dist_name,
        Pc_name,
    ):
        """
        calculate derivative of P(s) (our logbinned weighted average expected)
        """

        slope_name = "slope"

        # now calculate P(s) derivatives aka slopes per region
        byRegDer = []
        for (reg1, reg2), subdf in logbin_exp.groupby([_REGION1, _REGION2]):
            subdf = subdf.sort_values(digit_id_name)
            valid = np.minimum(
                subdf[_NUM_VALID].to_numpy()[:-1], subdf[_NUM_VALID].to_numpy()[1:]
            )
            # geometric mean of each logbin - aka mids
            mids = np.sqrt(
                subdf[weighted_dist_name].to_numpy()[:-1]
                * subdf[weighted_dist_name].to_numpy()[1:]
            )
            slope = np.diff(smooth(np.log(subdf[Pc_name].to_numpy()))) / np.diff(
                smooth(np.log(subdf[weighted_dist_name].to_numpy()))
            )
            newdf = pd.DataFrame(
                {
                    weighted_dist_name: mids,
                    slope_name: slope,
                    _NUM_VALID: valid,
                    digit_id_name: subdf[digit_id_name].to_numpy()[:-1],
                }
            )
            newdf[_REGION1] = reg1
            newdf[_REGION2] = reg2
            byRegDer.append(newdf)
        byRegDer = pd.concat(byRegDer).reset_index(drop=True)

        return byRegDer

    raw_summary_name = "count.sum"
    exp_summary_base, *_ = summary_name.split(".")
    Pc_name = f"{exp_summary_base}.avg"
    diag_name = _DIST
    diag_avg_name = f"{diag_name}.avg"
    # filter expected from NaNs in summary column and copy (precaution)
    exp = exp.dropna(
        subset=[
            summary_name,
        ]
    ).copy()
    # rename "dist" column dist.avg, it'll change later
    exp[diag_avg_name] = exp.pop(diag_name)

    # generate the "logbins", i.e. uneven bins for the diagonal distances
    diag_bins = _get_diag_bins(
        bin_layout=bin_layout,
        diagmax=exp[diag_avg_name].max(),
        bins_per_order_magnitude=bins_per_order_magnitude,
    )
    # assign distances to logbins and calculate weight averages for dist and counts
    byRegExp = _get_weighted_expected(
        exp,
        diag_bins,
        digit_id_name="dist_bin_id",
        weighted_dist_name=diag_avg_name,
        Pc_name=f"{exp_summary_base}.avg",
        summary_name=summary_name,
        raw_summary_name="count.sum",
        min_nvalid=min_nvalid,
        min_count=min_count,
    )

    # now calculate P(s) derivatives aka slopes per region
    byRegDer = _get_slopes(
        byRegExp,
        digit_id_name="dist_bin_id",
        weighted_dist_name=diag_avg_name,
        Pc_name=f"{exp_summary_base}.avg",
    )

    # returning logbin expected, its derivative and lobins themselves:
    return byRegExp, byRegDer, diag_bins[: byRegExp["dist_bin_id"].max() + 2]


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
    diag_avg_name = f"{_DIST}.avg"
    # combine pre-logbinned expecteds
    scal = numutils.weighted_groupby_mean(
        binned_exp[
            [
                Pc_name,
                "dist_bin_id",
                "n_valid",
                diag_avg_name,
                "dist_bin_start",
                "dist_bin_end",
            ]
        ],
        group_by="dist_bin_id",
        weigh_by="n_valid",
        mode="mean",
    )

    # for every diagonal calculate the spread of expected
    if spread_funcs == "minmax":
        byRegVar = binned_exp.copy()
        byRegVar = byRegVar.loc[
            byRegVar.index.difference(
                byRegVar.groupby(["region1", "region2"])["n_valid"]
                .tail(minmax_drop_bins)
                .index
            )
        ]
        low_err = byRegVar.groupby("dist_bin_id")[Pc_name].min()
        high_err = byRegVar.groupby("dist_bin_id")[Pc_name].max()
    elif spread_funcs == "std":
        var = numutils.weighted_groupby_mean(
            binned_exp[[Pc_name, "dist_bin_id", "n_valid"]],
            group_by="dist_bin_id",
            weigh_by="n_valid",
            mode="std",
        )[Pc_name]
        low_err = scal[Pc_name] - var
        high_err = scal[Pc_name] + var
    elif spread_funcs == "logstd":
        var = numutils.weighted_groupby_mean(
            binned_exp[[Pc_name, "dist_bin_id", "n_valid"]],
            group_by="dist_bin_id",
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
            "dist_bin_id": scal.index.values[:-1],
        }
    )
    slope_df = slope_df.set_index("dist_bin_id")

    # when pre-region slopes are provided, calculate spread of slopes
    if binned_exp_slope is not None:
        if spread_funcs_slope == "minmax":
            byRegDer = binned_exp_slope.copy()
            byRegDer = byRegDer.loc[
                byRegDer.index.difference(
                    byRegDer.groupby(["region1", "region2"])["n_valid"]
                    .tail(minmax_drop_bins)
                    .index
                )
            ]
            low_err = byRegDer.groupby("dist_bin_id")["slope"].min()
            high_err = byRegDer.groupby("dist_bin_id")["slope"].max()
        elif spread_funcs_slope == "std":
            var = numutils.weighted_groupby_mean(
                binned_exp_slope[["slope", "dist_bin_id", "n_valid"]],
                group_by="dist_bin_id",
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
        expected as returned by diagsum_symm
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
    gr_exp = exp_int.groupby(
        ["region1", "region2"]
    )  # groupby original expected by region

    if by_region is not False and (
        ("region1" not in binned_expected) or ("region2" not in binned_expected)
    ):
        warnings.warn("Region columns not found, assuming combined expected")
        by_region = False

    if by_region is True:
        # groupby expected
        gr_binned = binned_expected.groupby(["region1", "region2"])
    elif by_region is not False:
        # extract a region that we want to use
        binned_expected = binned_expected[binned_expected["region1"] == by_region]

    if by_region is not True:
        # check that we have no duplicates in expected
        assert len(binned_expected["dist_bin_id"].drop_duplicates()) == len(
            binned_expected
        )

    interp_dfs = []

    for (reg1, reg2), df_orig in gr_exp:
        if by_region is True:  # use binned expected for this region
            if (reg1, reg2) not in gr_binned.groups:
                continue
            subdf = gr_binned.get_group((reg1, reg2))
        else:
            subdf = binned_expected

        diag_orig = df_orig[_DIST].to_numpy()
        diag_mid = (subdf["dist_bin_start"] + subdf["dist_bin_end"]) / 2
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
                mask = diag_orig >= subdf["dist_bin_start"].min()
                interp_df = interp_df.iloc[mask]
        interp_dfs.append(interp_df)
    interp_df = pd.concat(interp_dfs)
    for i in interp_df.columns:
        exp_int[i] = interp_df[i]
    return exp_int
