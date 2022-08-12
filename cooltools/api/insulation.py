import re
import logging

logging.basicConfig(level=logging.INFO)

import warnings
import multiprocess as mp
from functools import partial

import numpy as np
import pandas as pd
import cooler
from skimage.filters import threshold_li, threshold_otsu

from ..lib._query import CSRSelector
from ..lib import peaks, numutils

from ..lib.checks import is_compatible_viewframe, is_cooler_balanced
from ..lib.common import make_cooler_view


def get_n_pixels(bad_bin_mask, window=10, ignore_diags=2):
    """
    Calculate the number of "good" pixels in a diamond at each bin.

    """
    N = len(bad_bin_mask)
    n_pixels = np.zeros(N)
    loc_bad_bin_mask = np.zeros(N, dtype=bool)
    for i_shift in range(0, window):
        for j_shift in range(0, window):
            if i_shift + j_shift < ignore_diags:
                continue

            loc_bad_bin_mask[:] = False
            if i_shift == 0:
                loc_bad_bin_mask |= bad_bin_mask
            else:
                loc_bad_bin_mask[i_shift:] |= bad_bin_mask[:-i_shift]
            if j_shift == 0:
                loc_bad_bin_mask |= bad_bin_mask
            else:
                loc_bad_bin_mask[:-j_shift] |= bad_bin_mask[j_shift:]

            n_pixels[i_shift : (-j_shift if j_shift else None)] += (
                1 - loc_bad_bin_mask[i_shift : (-j_shift if j_shift else None)]
            )
    return n_pixels


def insul_diamond(
    pixel_query,
    bins,
    window=10,
    ignore_diags=2,
    norm_by_median=True,
    clr_weight_name="weight",
):
    """
    Calculates the insulation score of a Hi-C interaction matrix.

    Parameters
    ----------
    pixel_query : RangeQuery object <TODO:update description>
        A table of Hi-C interactions. Must follow the Cooler columnar format:
        bin1_id, bin2_id, count, balanced (optional)).
    bins : pandas.DataFrame
        A table of bins, is used to determine the span of the matrix
        and the locations of bad bins.
    window : int
        The width (in bins) of the diamond window to calculate the insulation
        score.
    ignore_diags : int
        If > 0, the interactions at separations < `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.
    norm_by_median : bool
        If True, normalize the insulation score by its NaN-median.
    clr_weight_name : str or None
        Name of balancing weight column from the cooler to use.
        Using raw unbalanced data is not supported for insulation.
    """
    lo_bin_id = bins.index.min()
    hi_bin_id = bins.index.max() + 1
    N = hi_bin_id - lo_bin_id
    sum_counts = np.zeros(N)
    sum_balanced = np.zeros(N)

    if clr_weight_name is None:
        # define n_pixels
        n_pixels = get_n_pixels(
            np.repeat(False, len(bins)), window=window, ignore_diags=ignore_diags
        )
    else:
        # calculate n_pixels
        n_pixels = get_n_pixels(
            bins[clr_weight_name].isnull().values,
            window=window,
            ignore_diags=ignore_diags,
        )
        # define transform - balanced and raw ('count') for now
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transform = lambda p: p["count"] * p[weight1] * p[weight2]

    for chunk_dict in pixel_query.read_chunked():
        chunk = pd.DataFrame(chunk_dict, columns=["bin1_id", "bin2_id", "count"])
        diag_pixels = chunk[chunk.bin2_id - chunk.bin1_id <= (window - 1) * 2]

        if clr_weight_name:
            diag_pixels = cooler.annotate(diag_pixels, bins[[clr_weight_name]])
            diag_pixels["balanced"] = transform(diag_pixels)
            valid_pixel_mask = ~diag_pixels["balanced"].isnull().values

        i = diag_pixels.bin1_id.values - lo_bin_id
        j = diag_pixels.bin2_id.values - lo_bin_id

        for i_shift in range(0, window):
            for j_shift in range(0, window):
                if i_shift + j_shift < ignore_diags:
                    continue

                mask = (
                    (i + i_shift == j - j_shift)
                    & (i + i_shift < N)
                    & (j - j_shift >= 0)
                )

                sum_counts += np.bincount(
                    i[mask] + i_shift, diag_pixels["count"].values[mask], minlength=N
                )

                if clr_weight_name:
                    sum_balanced += np.bincount(
                        i[mask & valid_pixel_mask] + i_shift,
                        diag_pixels["balanced"].values[mask & valid_pixel_mask],
                        minlength=N,
                    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if clr_weight_name:
            score = sum_balanced / n_pixels
        else:
            score = sum_counts / n_pixels

        if norm_by_median:
            score /= np.nanmedian(score)

    return score, n_pixels, sum_balanced, sum_counts


def calculate_insulation_score(
    clr,
    window_bp,
    view_df=None,
    ignore_diags=None,
    min_dist_bad_bin=0,
    is_bad_bin_key="is_bad_bin",
    append_raw_scores=False,
    chunksize=20000000,
    clr_weight_name="weight",
    verbose=False,
    nproc=1,
):
    """Calculate the diamond insulation scores for all bins in a cooler.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data.
    window_bp : int or list of integers
        The size of the sliding diamond window used to calculate the insulation
        score. If a list is provided, then a insulation score if calculated for each
        value of window_bp.
    view_df : bioframe.viewframe or None
        Viewframe for independent calculation of insulation scores for regions
    ignore_diags : int | None
        The number of diagonals to ignore. If None, equals the number of
        diagonals ignored during IC balancing.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin to report insulation score.
        Fills bins that have a bad bin closer than this distance by nans.
    is_bad_bin_key : str
        Name of the output column to store bad bins
    append_raw_scores : bool
        If True, append columns with raw scores (sum_counts, sum_balanced, n_pixels)
        to the output table.
    clr_weight_name : str or None
        Name of the column in the bin table with weight.
        Using unbalanced data with `None` will avoid masking "bad" pixels.
    verbose : bool
        If True, report real-time progress.
    nproc : int, optional
        How many processes to use for calculation

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins
    """

    if view_df is None:
        view_df = make_cooler_view(clr)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    # check if cooler is balanced
    if clr_weight_name:
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e

    bin_size = clr.info["bin-size"]
    # check if ignore_diags is valid
    if ignore_diags is None:
        try:
            ignore_diags = clr._load_attrs(
                clr.root.rstrip("/") + f"/bins/{clr_weight_name}"
            )["ignore_diags"]
        except:
            raise ValueError(
                f"ignore_diags not provided, and not found in cooler balancing weights {clr_weight_name}"
            )
    elif isinstance(ignore_diags, int):
        pass  # keep it as is
    else:
        raise ValueError(f"ignore_diags must be int or None, got {ignore_diags}")

    if np.isscalar(window_bp):
        window_bp = [window_bp]
    window_bp = np.array(window_bp, dtype=int)

    bad_win_sizes = window_bp % bin_size != 0
    if np.any(bad_win_sizes):
        raise ValueError(
            f"The window sizes {window_bp[bad_win_sizes]} has to be a multiple of the bin size {bin_size}"
        )

    # Calculate insulation score for each region separately.
    # Define mapper depending on requested number of threads:
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # Using try-clause to close mp.Pool properly
    try:
        # Apply get_region_insulation:
        job = partial(
            _get_region_insulation,
            clr,
            is_bad_bin_key,
            clr_weight_name,
            chunksize,
            window_bp,
            min_dist_bad_bin,
            ignore_diags,
            append_raw_scores,
            verbose,
        )
        ins_region_tables = map_(job, view_df[["chrom", "start", "end", "name"]].values)

    finally:
        if nproc > 1:
            pool.close()

    ins_table = pd.concat(ins_region_tables)
    return ins_table


def _get_region_insulation(
    clr,
    is_bad_bin_key,
    clr_weight_name,
    chunksize,
    window_bp,
    min_dist_bad_bin,
    ignore_diags,
    append_raw_scores,
    verbose,
    region,
):
    """
    Auxilary function to make calculate_insulation_score parallel.
    """

    # XXX -- Use a delayed query executor
    nbins = len(clr.bins())
    selector = CSRSelector(
        clr.open("r"), shape=(nbins, nbins), field="count", chunksize=chunksize
    )

    # Convert window sizes to bins:
    bin_size = clr.info["bin-size"]
    window_bins = window_bp // bin_size

    # Parse region and set up insulation table for the region:
    chrom, start, end, name = region
    region = [chrom, start, end]
    region_bins = clr.bins().fetch(region)
    ins_region = region_bins[["chrom", "start", "end"]].copy()
    ins_region.loc[:, "region"] = name
    ins_region[is_bad_bin_key] = (
        region_bins[clr_weight_name].isnull() if clr_weight_name else False
    )

    if verbose:
        logging.info(f"Processing region {name}")

    if min_dist_bad_bin:
        ins_region = ins_region.assign(
            dist_bad_bin=numutils.dist_to_mask(ins_region[is_bad_bin_key])
        )

    # XXX --- Create a delayed selection
    c0, c1 = clr.extent(region)
    region_query = selector[c0:c1, c0:c1]

    for j, win_bin in enumerate(window_bins):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # XXX -- updated insul_diamond
            ins_track, n_pixels, sum_balanced, sum_counts = insul_diamond(
                region_query,
                region_bins,
                window=win_bin,
                ignore_diags=ignore_diags,
                clr_weight_name=clr_weight_name,
            )
            ins_track[ins_track == 0] = np.nan
            ins_track = np.log2(ins_track)

        ins_track[~np.isfinite(ins_track)] = np.nan

        ins_region[f"log2_insulation_score_{window_bp[j]}"] = ins_track
        ins_region[f"n_valid_pixels_{window_bp[j]}"] = n_pixels

        if min_dist_bad_bin:
            mask_bad = ins_region.dist_bad_bin.values < min_dist_bad_bin
            ins_region.loc[mask_bad, f"log2_insulation_score_{window_bp[j]}"] = np.nan

        if append_raw_scores:
            ins_region[f"sum_counts_{window_bp[j]}"] = sum_counts
            ins_region[f"sum_balanced_{window_bp[j]}"] = sum_balanced

    return ins_region


def find_boundaries(
    ins_table,
    min_frac_valid_pixels=0.66,
    min_dist_bad_bin=0,
    log2_ins_key="log2_insulation_score_{WINDOW}",
    n_valid_pixels_key="n_valid_pixels_{WINDOW}",
    is_bad_bin_key="is_bad_bin",
):
    """Call insulating boundaries.

    Find all local minima of the log2(insulation score) and calculate their
    chromosome-wide topographic prominence.

    Parameters
    ----------
    ins_table : pandas.DataFrame
        A bin table with columns containing log2(insulation score),
        annotation of regions (required),
        the number of valid pixels per diamond and (optionally) the mask
        of bad bins. Normally, this should be an output of calculate_insulation_score.
    view_df : bioframe.viewframe or None
        Viewframe for independent boundary calls for regions
    min_frac_valid_pixels : float
        The minimal fraction of valid pixels in a diamond to be used in
        boundary picking and prominence calculation.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin to be used in boundary picking.
        Ignore bins that have a bad bin closer than this distance.
    log2_ins_key, n_valid_pixels_key : str
        The names of the columns containing log2_insulation_score and
        the number of valid pixels per diamond. When a template
        containing `{WINDOW}` is provided, the calculation is repeated
        for all pairs of columns matching the template.

    Returns
    -------
    ins_table : pandas.DataFrame
        A bin table with appended columns with boundary prominences.
    """

    if min_dist_bad_bin:
        ins_table = pd.concat(
            [
                df.assign(dist_bad_bin=numutils.dist_to_mask(df[is_bad_bin_key]))
                for region, df in ins_table.groupby("region")
            ]
        )

    if "{WINDOW}" in log2_ins_key:
        windows = set()
        for col in ins_table.columns:
            m = re.match(log2_ins_key.format(WINDOW=r"(\d+)"), col)
            if m:
                windows.add(int(m.groups()[0]))
    else:
        windows = set([None])

    min_valid_pixels = {
        win: ins_table[n_valid_pixels_key.format(WINDOW=win)].max()
        * min_frac_valid_pixels
        for win in windows
    }

    dfs = []
    index_name = ins_table.index.name  # Store the name of the index and soring order
    sorting_order = ins_table.index.values
    ins_table.index.name = "sorting_index"
    ins_table.reset_index(drop=False, inplace=True)
    for region, df in ins_table.groupby("region"):
        df = df.sort_values(["start"])  # Force sorting by the bin start coordinate
        for win in windows:
            mask = (
                df[n_valid_pixels_key.format(WINDOW=win)].values
                >= min_valid_pixels[win]
            )

            if min_dist_bad_bin:
                mask &= df.dist_bad_bin.values >= min_dist_bad_bin

            ins_track = df[log2_ins_key.format(WINDOW=win)].values[mask]
            poss, proms = peaks.find_peak_prominence(-ins_track)
            ins_prom_track = np.zeros_like(ins_track) * np.nan
            ins_prom_track[poss] = proms

            if win is not None:
                bs_key = f"boundary_strength_{win}"
            else:
                bs_key = "boundary_strength"

            df[bs_key] = np.nan
            df.loc[mask, bs_key] = ins_prom_track

        dfs.append(df)

    df = pd.concat(dfs)
    df = df.set_index("sorting_index")  # Restore original sorting order and name
    df.index.name = index_name
    df = df.loc[sorting_order, :]
    return df


def _insul_diamond_dense(mat, window=10, ignore_diags=2, norm_by_median=True):
    """
    Calculates the insulation score of a Hi-C interaction matrix.

    Parameters
    ----------
    mat : numpy.array
        A dense square matrix of Hi-C interaction frequencies.
        May contain nans, e.g. in rows/columns excluded from the analysis.

    window : int
        The width of the window to calculate the insulation score.

    ignore_diags : int
        If > 0, the interactions at separations < `ignore_diags` are ignored
        when calculating the insulation score. Typically, a few first diagonals
        of the Hi-C map should be ignored due to contamination with Hi-C
        artifacts.

    norm_by_median : bool
        If True, normalize the insulation score by its NaN-median.

    Returns
    -------
    score : ndarray
        an array with normalized insulation scores for provided matrix
    """
    if ignore_diags:
        mat = mat.copy()
        for i in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(mat, np.nan, i)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        N = mat.shape[0]
        score = np.nan * np.ones(N)
        for i in range(0, N):
            lo = max(0, i + 1 - window)
            hi = min(i + window, N)
            # nanmean of interactions to reduce the effect of bad bins
            score[i] = np.nanmean(mat[lo : i + 1, i:hi])
        if norm_by_median:
            score /= np.nanmedian(score)
    return score


def _find_insulating_boundaries_dense(
    clr,
    window_bp=100000,
    view_df=None,
    clr_weight_name="weight",
    min_dist_bad_bin=2,
    ignore_diags=None,
):
    """Calculate the diamond insulation scores and call insulating boundaries.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data. Balancing weights are required
        for the detection of bad_bins.
    window_bp : int
        The size of the sliding diamond window used to calculate the insulation
        score.
    view_df : bioframe.viewframe or None
        Viewframe for independent calculation of insulation scores for regions
    clr_weight_name : str
        Name of the column in bin table that stores the balancing weights.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin. Do not calculate insulation
        scores for bins having a bad bin closer than this distance.
    ignore_diags : int
        The number of diagonals to ignore. If None, equals the number of
        diagonals ignored during IC balancing.

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins and
        the insulating boundary strengths.
    """

    if view_df is None:
        view_df = make_cooler_view(clr)
    else:
        # Make sure view_df is a proper viewframe
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    bin_size = clr.info["bin-size"]

    # check if cooler is balanced
    if clr_weight_name:
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e

    # check if ignore_diags is valid
    if ignore_diags is None:
        ignore_diags = clr._load_attrs(
            clr.root.rstrip("/") + f"/bins/{clr_weight_name}"
        )["ignore_diags"]
    elif isinstance(ignore_diags, int):
        pass  # keep it as is
    else:
        raise ValueError(f"provided ignore_diags {ignore_diags} is not int or None")

    window_bins = window_bp // bin_size

    if window_bp % bin_size != 0:
        raise ValueError(
            f"The window size ({window_bp}) has to be a multiple of the bin size {bin_size}"
        )

    ins_region_tables = []
    for chrom, start, end, name in view_df[["chrom", "start", "end", "name"]].values:
        region = [chrom, start, end]
        ins_region = clr.bins().fetch(region)[["chrom", "start", "end"]]
        is_bad_bin = np.isnan(clr.bins().fetch(region)[clr_weight_name].values)
        # extract dense Hi-C heatmap for a given "region"
        m = clr.matrix(balance=clr_weight_name).fetch(region)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ins_track = _insul_diamond_dense(m, window_bins, ignore_diags)
            ins_track[ins_track == 0] = np.nan
            ins_track = np.log2(ins_track)

        bad_bin_neighbor = np.zeros_like(is_bad_bin)
        for i in range(0, min_dist_bad_bin):
            if i == 0:
                bad_bin_neighbor = bad_bin_neighbor | is_bad_bin
            else:
                bad_bin_neighbor = bad_bin_neighbor | np.r_[[True] * i, is_bad_bin[:-i]]
                bad_bin_neighbor = bad_bin_neighbor | np.r_[is_bad_bin[i:], [True] * i]

        ins_track[bad_bin_neighbor] = np.nan
        ins_region["bad_bin_masked"] = bad_bin_neighbor

        ins_track[~np.isfinite(ins_track)] = np.nan

        ins_region[f"log2_insulation_score_{window_bp}"] = ins_track

        poss, proms = peaks.find_peak_prominence(-ins_track)
        ins_prom_track = np.zeros_like(ins_track) * np.nan
        ins_prom_track[poss] = proms
        ins_region[f"boundary_strength_{window_bp}"] = ins_prom_track
        ins_region[f"boundary_strength_{window_bp}"] = ins_prom_track

        ins_region_tables.append(ins_region)

    ins_table = pd.concat(ins_region_tables)
    return ins_table


def insulation(
    clr,
    window_bp,
    view_df=None,
    ignore_diags=None,
    clr_weight_name="weight",
    min_frac_valid_pixels=0.66,
    min_dist_bad_bin=0,
    threshold="Li",
    append_raw_scores=False,
    chunksize=20000000,
    verbose=False,
    nproc=1,
):
    """Find insulating boundaries in a contact map via the diamond insulation score.

    For a given cooler, this function (a) calculates the diamond insulation score track,
    (b) detects all insulating boundaries, and (c) removes weak boundaries via an automated
    thresholding algorithm.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data.
    window_bp : int or list of integers
        The size of the sliding diamond window used to calculate the insulation
        score. If a list is provided, then a insulation score if done for each
        value of window_bp.
    view_df : bioframe.viewframe or None
        Viewframe for independent calculation of insulation scores for regions
    ignore_diags : int | None
        The number of diagonals to ignore. If None, equals the number of
        diagonals ignored during IC balancing.
    clr_weight_name : str
        Name of the column in the bin table with weight
    min_frac_valid_pixels : float
        The minimal fraction of valid pixels in a diamond to be used in
        boundary picking and prominence calculation.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin to report insulation score.
        Fills bins that have a bad bin closer than this distance by nans.
    threshold : "Li", "Otsu" or float
        Rule used to threshold the histogram of boundary strengths to exclude weak
        boundaries. "Li" or "Otsu" use corresponding methods from skimage.thresholding.
        Providing a float value will filter by a fixed threshold
    append_raw_scores : bool
        If True, append columns with raw scores (sum_counts, sum_balanced, n_pixels)
        to the output table.
    verbose : bool
        If True, report real-time progress.
    nproc : int, optional
        How many processes to use for calculation

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins
    """
    # Create view:
    if view_df is None:
        # full chromosomes:
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

    if threshold == "Li":
        thresholding_func = lambda x: x >= threshold_li(x)
    elif threshold == "Otsu":
        thresholding_func = lambda x: x >= threshold_otsu(x)
    else:
        try:
            thr = float(threshold)
            thresholding_func = lambda x: x >= thr
        except ValueError:
            raise ValueError(
                "Insulating boundary strength threshold can be Li, Otsu or a float"
            )
    # Calculate insulation score:
    ins_table = calculate_insulation_score(
        clr,
        view_df=view_df,
        window_bp=window_bp,
        ignore_diags=ignore_diags,
        min_dist_bad_bin=min_dist_bad_bin,
        append_raw_scores=append_raw_scores,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
        verbose=verbose,
        nproc=nproc,
    )

    # Find boundaries:
    ins_table = find_boundaries(
        ins_table,
        min_frac_valid_pixels=min_frac_valid_pixels,
        min_dist_bad_bin=min_dist_bad_bin,
    )
    for win in window_bp:
        strong_boundaries = thresholding_func(
            ins_table[f"boundary_strength_{win}"].values
        )
        ins_table[f"is_boundary_{win}"] = strong_boundaries

    return ins_table
