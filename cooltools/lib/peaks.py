# This is the Python implementation of the peakdet algorithm.
#
import warnings
import numpy as np


def find_peak_prominence(arr, max_dist=None):
    """Find the local maxima of an array and their prominence.
    The prominence of a peak is defined as the maximal difference between the
    height of the peak and the lowest point in the range until a higher peak.

    Parameters
    ----------
    arr : array_like
    max_dist : int
        If specified, the distance to the adjacent higher peaks is limited
        by `max_dist`.

    Returns
    -------
    loc_max_poss : numpy.array
        The positions of local maxima of a given array.

    proms : numpy.array
        The prominence of the detected maxima.
    """

    arr = np.asarray(arr)
    n = len(arr)
    max_dist = len(arr) if max_dist is None else int(max_dist)

    # Finding all local minima and maxima (i.e. points the are lower/higher than
    # both immediate non-nan neighbors).
    arr_nonans = arr[~np.isnan(arr)]
    idxs_nonans2idx = np.arange(arr.size)[~np.isnan(arr)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        is_min_left = np.r_[False, arr_nonans[:-1] > arr_nonans[1:]]
        is_min_right = np.r_[arr_nonans[:-1] < arr_nonans[1:], False]
        is_loc_min = is_min_left & is_min_right
        loc_min_poss = np.where(is_loc_min)[0]
        loc_min_poss = idxs_nonans2idx[loc_min_poss]

        is_max_left = np.r_[False, arr_nonans[:-1] < arr_nonans[1:]]
        is_max_right = np.r_[arr_nonans[:-1] > arr_nonans[1:], False]
        is_loc_max = is_max_left & is_max_right
        loc_max_poss = np.where(is_loc_max)[0]
        loc_max_poss = idxs_nonans2idx[loc_max_poss]

    # For each maximum, find the position of a higher peak on the left and
    # on the right. If there are no higher peaks within the `max_dist` range,
    # just use the position `max_dist` away.
    left_maxs = -1 * np.ones(len(loc_max_poss), dtype=np.int64)
    right_maxs = -1 * np.ones(len(loc_max_poss), dtype=np.int64)

    for i, pos in enumerate(loc_max_poss):
        for j in range(pos - 1, -1, -1):
            if (arr[j] > arr[pos]) or (pos - j > max_dist):
                left_maxs[i] = j
                break

        for j in range(pos + 1, n):
            if (arr[j] > arr[pos]) or (j - pos > max_dist):
                right_maxs[i] = j
                break

    # Find the prominence of each peak with respect of the lowest point
    # between the peak and the adjacent higher peaks, on the left and the right
    # separately.
    left_max_proms = np.array(
        [
            (
                arr[pos] - np.nanmin(arr[left_maxs[i] : pos])
                if (left_maxs[i] >= 0)
                else np.nan
            )
            for i, pos in enumerate(loc_max_poss)
        ]
    )

    right_max_proms = np.array(
        [
            (
                arr[pos] - np.nanmin(arr[pos : right_maxs[i]])
                if (right_maxs[i] >= 0)
                else np.nan
            )
            for i, pos in enumerate(loc_max_poss)
        ]
    )

    # In 1D, the topographic definition of the prominence of a peak reduces to
    # the minimum of the left-side and right-side prominence.

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        max_proms = np.nanmin(np.vstack([left_max_proms, right_max_proms]), axis=0)

    # The global maximum, by definition, does not have higher peaks around it and
    # thus its prominence is explicitly defined with respect to the lowest local
    # minimum. This issue arises only if max_dist was not specified, otherwise
    # the prominence of the global maximum is already calculated with respect
    # to the lowest point within the `max_dist` range.
    # If no local minima are within the `max_dist` range, just use the
    # lowest point.
    global_max_mask = (left_maxs == -1) & (right_maxs == -1)
    if (global_max_mask).sum() > 0:
        global_max_idx = np.where(global_max_mask)[0][0]
        global_max_pos = loc_max_poss[global_max_idx]
        neighbor_loc_mins = (loc_min_poss >= global_max_pos - max_dist) & (
            loc_min_poss < global_max_pos + max_dist
        )
        if np.any(neighbor_loc_mins):
            max_proms[global_max_idx] = arr[global_max_pos] - np.nanmin(
                arr[loc_min_poss[neighbor_loc_mins]]
            )
        else:
            max_proms[global_max_idx] = arr[global_max_pos] - np.nanmin(
                arr[max(global_max_pos - max_dist, 0) : global_max_pos + max_dist]
            )

    return loc_max_poss, max_proms


def peakdet(arr, min_prominence):
    """Detect local peaks in an array.
    Finds a sequence of minima and maxima such that the two consecutive extrema
    have a value difference (i.e. a prominence) >= `min_prominence`. This is
    analogous to the definition of prominence in topography:
    https://en.wikipedia.org/wiki/Topographic_prominence

    The original peakdet algorithm was designed by Eli Billauer and described in
    http://billauer.co.il/peakdet.html (v. 3.4.05, Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.
    The Python implementation was published
    by endolith on Github: https://gist.github.com/endolith/250860 .

    Here, we use the endolith's implementation with minimal to none modifications
    to the algorithm, but with significant changes in the interface and
    the documentation

    Parameters
    ----------
    arr : array_like
    min_prominence : float
        The minimal prominence of detected extrema.

    Returns
    -------
    maxidxs, minidx : numpy.array
        The indices of the maxima and minima in `arr`.

    """
    maxidxs = []
    minidxs = []

    x = np.arange(len(arr))

    arr = np.asarray(arr)

    if not np.isscalar(min_prominence):
        raise Exception("Input argument delta must be a scalar")

    if min_prominence <= 0:
        raise Exception("Input argument delta must be positive")

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in np.arange(len(arr)):
        this = arr[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - min_prominence:
                maxidxs.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + min_prominence:
                minidxs.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(minidxs), np.array(maxidxs)


def find_peak_prominence_iterative(
    arr,
    min_prom=None,
    max_prom=None,
    steps_prom=1000,
    log_space_proms=True,
    min_n_peak_pairs=5,
):
    """Finds the minima/maxima of an array using the peakdet algorithm at
    different values of the threshold prominence. For each location, returns
    the maximal threshold prominence at which it is called as a minimum/maximum.

    Note that this function is inferior in every aspect to find_peak_prominence.
    We keep it for testing purposes and will remove in the future.

    Parameters
    ----------
    arr : array_like
    min_prom, max_prom : float
        The minimal and the maximal values of prominence to probe.
        If None, these values are inferred as the minimal and the maximal
        non-zero difference between any two elements of `v`.
    steps_prom : int
        The number of threshold prominence values to probe in the range
        between `min_prom` and `max_prom`.
    log_space_proms : bool
        If True, probe logarithmically spaced values of the threshold prominence
        in the range between `min_prom` and `max_prom`.
    min_n_peak_pairs : int
        If the number of detected minima/maxima at a certain threshold
        prominence is < `min_n_peak_pairs`, the detected peaks are ignored.

    Returns
    -------
    minproms, maxproms : numpy.array
        The prominence of detected minima and maxima.
    """
    if ((min_prom is None) and (max_prom is not None)) or (
        (min_prom is not None) and (max_prom is None)
    ):
        raise Exception(
            "Please, provide either both min_prom and max_prom or "
            "none to infer these values from the data."
        )

    if (min_prom is None) and (max_prom is None):
        arr_sorted = np.sort(arr)
        arr_sorted = arr_sorted[np.isfinite(arr_sorted)]
        max_prom = arr_sorted[-1] - arr_sorted[0]

        diffs = np.diff(arr_sorted)
        min_prom = diffs[diffs != 0].min()

    if log_space_proms:
        proms = 10 ** np.linspace(np.log10(min_prom), np.log10(max_prom), steps_prom)
    else:
        proms = np.linspace(min_prom, max_prom, steps_prom)

    minproms = np.nan * np.ones_like(arr)
    maxproms = np.nan * np.ones_like(arr)
    for p in proms:
        minidxs, maxidxs = peakdet(arr, p)
        if (len(minidxs) >= min_n_peak_pairs) and (len(minidxs) >= min_n_peak_pairs):

            valid_mins = minidxs[np.isfinite(arr[minidxs])]
            minproms[valid_mins] = p

            valid_maxs = maxidxs[np.isfinite(arr[maxidxs])]
            maxproms[valid_maxs] = p

    return minproms, maxproms
