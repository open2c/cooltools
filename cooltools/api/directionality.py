import warnings
import numpy as np
import pandas as pd
from ..lib import peaks, numutils


def _dirscore(
    pixels, bins, window=10, ignore_diags=2, balanced=True, signed_chi2=False
):
    lo_bin_id = bins.index.min()
    hi_bin_id = bins.index.max() + 1
    N = hi_bin_id - lo_bin_id

    bad_bin_mask = (
        bins["weight"].isnull().values if balanced else np.zeros(N, dtype=bool)
    )

    diag_pixels = pixels[pixels["bin2_id"] - pixels["bin1_id"] <= (window - 1) * 2]
    if balanced:
        diag_pixels = diag_pixels[~diag_pixels["balanced"].isnull()]

    i = diag_pixels["bin1_id"].values - lo_bin_id
    j = diag_pixels["bin2_id"].values - lo_bin_id
    val = diag_pixels["balanced"].values if balanced else diag_pixels["count"].values

    sum_pixels_left = np.zeros(N)
    n_pixels_left = np.zeros(N)
    for i_shift in range(0, window):
        if i_shift < ignore_diags:
            continue

        mask = (i + i_shift == j) & (i + i_shift < N) & (j >= 0)
        sum_pixels_left += np.bincount(i[mask] + i_shift, val[mask], minlength=N)

        loc_bad_bin_mask = np.zeros(N, dtype=bool)
        if i_shift == 0:
            loc_bad_bin_mask |= bad_bin_mask
        else:
            loc_bad_bin_mask[i_shift:] |= bad_bin_mask[:-i_shift]
            loc_bad_bin_mask |= bad_bin_mask
        n_pixels_left[i_shift:] += 1 - loc_bad_bin_mask[i_shift:]

    sum_pixels_right = np.zeros(N)
    n_pixels_right = np.zeros(N)
    for j_shift in range(0, window):
        if j_shift < ignore_diags:
            continue

        mask = (i == j - j_shift) & (i < N) & (j - j_shift >= 0)

        sum_pixels_right += np.bincount(i[mask], val[mask], minlength=N)

        loc_bad_bin_mask = np.zeros(N, dtype=bool)
        loc_bad_bin_mask |= bad_bin_mask
        if j_shift == 0:
            loc_bad_bin_mask |= bad_bin_mask
        else:
            loc_bad_bin_mask[:-j_shift] |= bad_bin_mask[j_shift:]

        n_pixels_right[: (-j_shift if j_shift else None)] += (
            1 - loc_bad_bin_mask[: (-j_shift if j_shift else None)]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        a = sum_pixels_left
        b = sum_pixels_right
        if signed_chi2:
            e = (a + b) / 2.0
            score = np.sign(b - a) * ((a - e) ** 2 + (b - e) ** 2) / e
        else:
            score = (b - a) / (a + b)

    return score


def _dirscore_dense(A, window=10, signed_chi2=False):
    N = A.shape[0]
    di = np.zeros(N)
    for i in range(0, N):
        lo = max(0, i - window)
        hi = min((i + window) + 1, N)
        b, a = np.nansum(A[i, i:hi]), np.nansum(A[i, lo : i + 1])
        if signed_chi2:
            e = (a + b) / 2.0
            if e:
                di[i] = np.sign(b - a) * ((a - e) ** 2 + (b - e) ** 2) / e
        else:
            di[i] = (b - a) / (a + b)
    mask = np.nansum(A, axis=0) == 0
    di[mask] = np.nan
    return di


def directionality(
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
    clr : cooler.Cooler
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

    dir_chrom_tables = []
    for chrom in chromosomes:
        chrom_bins = clr.bins().fetch(chrom)
        chrom_pixels = clr.matrix(as_pixels=True, balance=balance).fetch(chrom)

        # mask neighbors of bad bins
        is_bad_bin = np.isnan(chrom_bins["weight"].values)
        bad_bin_neighbor = np.zeros_like(is_bad_bin)
        for i in range(0, min_dist_bad_bin):
            if i == 0:
                bad_bin_neighbor = bad_bin_neighbor | is_bad_bin
            else:
                bad_bin_neighbor = bad_bin_neighbor | np.r_[[True] * i, is_bad_bin[:-i]]
                bad_bin_neighbor = bad_bin_neighbor | np.r_[is_bad_bin[i:], [True] * i]

        dir_chrom = chrom_bins[["chrom", "start", "end"]].copy()
        dir_chrom["bad_bin_masked"] = bad_bin_neighbor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dir_track = _dirscore(
                chrom_pixels, chrom_bins, window=window_bins, ignore_diags=ignore_diags
            )
            dir_track[bad_bin_neighbor] = np.nan
            dir_track[~np.isfinite(dir_track)] = np.nan
            dir_chrom["directionality_ratio_{}".format(window_bp)] = dir_track

            dir_track = _dirscore(
                chrom_pixels,
                chrom_bins,
                window=window_bins,
                ignore_diags=ignore_diags,
                signed_chi2=True,
            )
            dir_track[bad_bin_neighbor] = np.nan
            dir_track[~np.isfinite(dir_track)] = np.nan
            dir_chrom["directionality_index_{}".format(window_bp)] = dir_track

        dir_chrom_tables.append(dir_chrom)

    dir_table = pd.concat(dir_chrom_tables)
    return dir_table
