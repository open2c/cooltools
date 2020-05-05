import re
import logging
import warnings
import numpy as np
import pandas as pd
import cooler

from .lib._query import CSRSelector
from .lib import peaks, numutils

logging.basicConfig(level=logging.INFO)


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


def insul_diamond(pixel_query, bins, window=10, ignore_diags=2, norm_by_median=True):
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
    """
    lo_bin_id = bins.index.min()
    hi_bin_id = bins.index.max() + 1
    N = hi_bin_id - lo_bin_id
    sum_counts = np.zeros(N)
    sum_balanced = np.zeros(N)

    n_pixels = get_n_pixels(
        bins.weight.isnull().values, window=window, ignore_diags=ignore_diags
    )

    for chunk_dict in pixel_query.read_chunked():
        chunk = pd.DataFrame(chunk_dict, columns=["bin1_id", "bin2_id", "count"])
        diag_pixels = chunk[chunk.bin2_id - chunk.bin1_id <= (window - 1) * 2]

        diag_pixels = cooler.annotate(diag_pixels, bins[["weight"]])
        diag_pixels["balanced"] = (
            diag_pixels["count"] * diag_pixels["weight1"] * diag_pixels["weight2"]
        )
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

                sum_balanced += np.bincount(
                    i[mask & valid_pixel_mask] + i_shift,
                    diag_pixels["balanced"].values[mask & valid_pixel_mask],
                    minlength=N,
                )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        score = sum_balanced / n_pixels

        if norm_by_median:
            score /= np.nanmedian(score)

    return score, n_pixels, sum_balanced, sum_counts


def calculate_insulation_score(
    clr,
    window_bp,
    ignore_diags=None,
    chromosomes=None,
    append_raw_scores=False,
    chunksize=20000000,
    verbose=False,
):
    """Calculate the diamond insulation scores and call insulating boundaries.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data.
    window_bp : int or list
        The size of the sliding diamond window used to calculate the insulation
        score. If a list is provided, then a insulation score if done for each
        value of window_bp.
    ignore_diags : int
        The number of diagonals to ignore. If None, equals the number of
        diagonals ignored during IC balancing.
    append_raw_scores : bool
        If True, append columns with raw scores (sum_counts, sum_balanced, n_pixels)
        to the output table.
    verbose : bool
        If True, report real-time progress.

    Returns
    -------
    ins_table : pandas.DataFrame
        A table containing the insulation scores of the genomic bins and
        the insulating boundary strengths.
    """
    if chromosomes is None:
        chromosomes = clr.chromnames

    bin_size = clr.info["bin-size"]
    ignore_diags = (
        ignore_diags
        if ignore_diags is not None
        else clr._load_attrs(clr.root.rstrip("/") + "/bins/weight")["ignore_diags"]
    )

    if isinstance(window_bp, int):
        window_bp = [window_bp]
    window_bp = np.array(window_bp)
    window_bins = window_bp // bin_size

    bad_win_sizes = window_bp % bin_size != 0
    if np.any(bad_win_sizes):
        raise Exception(
            "The window sizes {} has to be a multiple of the bin size {}".format(
                window_bp[bad_win_sizes], bin_size
            )
        )

    # XXX -- Use a delayed query executor
    nbins = len(clr.bins())
    selector = CSRSelector(
        clr.open("r"), shape=(nbins, nbins), field="count", chunksize=chunksize
    )

    ins_chrom_tables = []
    for chrom in chromosomes:
        if verbose:
            logging.info("Processing {}".format(chrom))

        chrom_bins = clr.bins().fetch(chrom)
        ins_chrom = chrom_bins[["chrom", "start", "end"]].copy()
        ins_chrom["is_bad_bin"] = chrom_bins["weight"].isnull()

        # XXX --- Create a delayed selection
        c0, c1 = clr.extent(chrom)
        chrom_query = selector[c0:c1, c0:c1]

        for j, win_bin in enumerate(window_bins):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                # XXX -- updated insul_diamond
                ins_track, n_pixels, sum_balanced, sum_counts = insul_diamond(
                    chrom_query, chrom_bins, window=win_bin, ignore_diags=ignore_diags
                )
                ins_track[ins_track == 0] = np.nan
                ins_track = np.log2(ins_track)

            ins_track[~np.isfinite(ins_track)] = np.nan

            ins_chrom["log2_insulation_score_{}".format(window_bp[j])] = ins_track
            ins_chrom["n_valid_pixels_{}".format(window_bp[j])] = n_pixels

            if append_raw_scores:
                ins_chrom["sum_counts_{}".format(window_bp[j])] = sum_counts
                ins_chrom["sum_balanced_{}".format(window_bp[j])] = sum_balanced

        ins_chrom_tables.append(ins_chrom)

    ins_table = pd.concat(ins_chrom_tables)
    return ins_table


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
        the number of valid pixels per diamond and (optionally) the mask
        of bad bins.
    min_frac_valid_pixels : float
        The minimal fraction of valid pixels in a diamond to be used in
        boundary picking and prominence calculation.
    min_dist_bad_bin : int
        The minimal allowed distance to a bad bin.
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
                df.assign(dist_bad_bin=numutils.dist_to_mask(df.is_bad_bin))
                for chrom, df in ins_table.groupby("chrom")
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
    for chrom, df in ins_table.groupby("chrom"):
        df = df.reset_index(drop=True)
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
                bs_key = "boundary_strength_{win}".format(win=win)
            else:
                bs_key = "boundary_strength"

            df[bs_key] = np.nan
            df.loc[mask, bs_key] = ins_prom_track

        dfs.append(df)
    return pd.concat(dfs)


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
    balance="weight",
    min_dist_bad_bin=2,
    ignore_diags=None,
    chromosomes=None,
):
    """Calculate the diamond insulation scores and call insulating boundaries.

    Parameters
    ----------
    c : cooler.Cooler
        A cooler with balanced Hi-C data.
    window_bp : int
        The size of the sliding diamond window used to calculate the insulation
        score.
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
    if chromosomes is None:
        chromosomes = clr.chromnames

    bin_size = clr.info["bin-size"]
    ignore_diags = (
        ignore_diags
        if ignore_diags is not None
        else clr._load_attrs(clr.root.rstrip("/") + "/bins/weight")["ignore_diags"]
    )
    window_bins = window_bp // bin_size

    if window_bp % bin_size != 0:
        raise Exception(
            "The window size ({}) has to be a multiple of the bin size {}".format(
                window_bp, bin_size
            )
        )

    ins_chrom_tables = []
    for chrom in chromosomes:
        ins_chrom = clr.bins().fetch(chrom)[["chrom", "start", "end"]]
        is_bad_bin = np.isnan(clr.bins().fetch(chrom)["weight"].values)

        m = clr.matrix(balance=balance).fetch(chrom)

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
        ins_chrom["bad_bin_masked"] = bad_bin_neighbor

        ins_track[~np.isfinite(ins_track)] = np.nan

        ins_chrom["log2_insulation_score_{}".format(window_bp)] = ins_track

        poss, proms = peaks.find_peak_prominence(-ins_track)
        ins_prom_track = np.zeros_like(ins_track) * np.nan
        ins_prom_track[poss] = proms
        ins_chrom["boundary_strength_{}".format(window_bp)] = ins_prom_track
        ins_chrom["boundary_strength_{}".format(window_bp)] = ins_prom_track

        ins_chrom_tables.append(ins_chrom)

    ins_table = pd.concat(ins_chrom_tables)
    return ins_table
