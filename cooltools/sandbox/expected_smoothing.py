from collections.abc import Iterable

import numpy as np
import numba


DEFAULT_CVD_COLS = {
    "dist": "dist",
    "n_pixels": "n_valid",
    "n_contacts": "balanced.sum",
    "contact_freq": "balanced.avg",
    "smooth_suffix": ".smoothed",
}


def _log_interp(xs, xp, fp):
    """
    Interpolate a function in the log-log space.
    Equivalent to np.exp(np.interp(np.log(xs), np.log(xp), np.log(fp))).

    Parameters
    ----------
    xs : array-like
        The x-coordinates at which to evaluate the interpolated values.
    xp : 1-D sequence of floats
        The x-coordinates of the data points, must be increasing.
    fp : 1D array
        The y-coordinates of the data points, same length as xp.

    Returns
    -------
    ys : 1D array
        The interpolated values, same shape as x.
    """
    with np.errstate(divide="ignore"):
        ys = np.exp(
            np.interp(
                np.log(xs),
                np.log(xp),
                np.log(fp),
            )
        )

    return ys


@numba.njit
def _log_thin(xs, min_log10_step=0.1):
    """
    Thin out a sorted array, by selecting a subset of elements that are uniformly spaced in log-space.

    Parameters
    ----------
    xs : array-like
        An array of elements to thin out.
    min_log10_step : float, optional
        The minimal log10 ratio between consecutive elements in the output, by default 0.1

    Returns
    -------
    xs_thinned : array-like
        A subset of elements from xs, whose logs are approx. uniformly spaced.
    """
    xs_thinned = [xs[0]]
    prev = xs[0]
    min_ratio = 10 ** min_log10_step
    for x in xs[1:]:
        if x > prev * min_ratio:
            xs_thinned.append(x)
            prev = x

    if xs_thinned[-1] != xs[-1]:
        xs_thinned.append(xs[-1])
    return np.array(xs_thinned)


@numba.njit
def _log_smooth_numba(
    xs,
    ys,
    sigma_log10=0.1,
    window_sigma=5,
    points_per_sigma=10,
):
    xs_thinned = xs
    if points_per_sigma:
        xs_thinned = _log_thin(xs, sigma_log10 / points_per_sigma)

    N = xs_thinned.size
    N_FUNCS = ys.shape[0]

    log_xs = np.log10(xs)
    log_thinned_xs = np.log10(xs_thinned)

    ys_smoothed = np.zeros((N_FUNCS, N))

    for i in range(N):
        cur_log_x = log_thinned_xs[i]
        lo = np.searchsorted(log_xs, cur_log_x - sigma_log10 * window_sigma)
        hi = np.searchsorted(log_xs, cur_log_x + sigma_log10 * window_sigma)
        smooth_weights = np.exp(
            -((cur_log_x - log_xs[lo:hi]) ** 2) / 2 / sigma_log10 / sigma_log10
        ) 
        norm = smooth_weights.sum()
        
        if norm > 0:
            smooth_weights /= norm

            for k in range(N_FUNCS):
                ys_smoothed[k, i] = np.sum(ys[k, lo:hi] * smooth_weights)

    return xs_thinned, ys_smoothed


def log_smooth(
    xs,
    ys,
    sigma_log10=0.1,
    window_sigma=5,
    points_per_sigma=10,
):
    """
    Convolve a function or multiple functions with a gaussian kernel in the log space.

    Parameters
    ----------
    xs : 1D array
        The x-coordinates (function arguments) of the data points, must be increasing.
    ys : 1D or 2D array
        The y-coordinates (function values) of the data points.
        If 2D, rows correspond to multiple functions, columns correspond to different points.
    sigma_log10 : float, optional
        The standard deviation of the smoothing Gaussian kernel, applied over log10(xs), by default 0.1
    window_sigma : int, optional
        Width of the smoothing window, expressed in sigmas, by default 5
    points_per_sigma : int, optional
        If provided, smoothing is done only for `points_per_sigma` points per sigma and the
        rest of the values are interpolated (this results in a major speed-up). By default 10

    Returns
    -------
    xs_thinned : 1D array
        The subset of arguments, uniformly spaced in log-space.
    ys_smoothed : 1D or 2D array
        The gaussian-smoothed function values.

    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if xs.ndim != 1:
        raise ValueError("xs must be a 1D vector")
    if ys.ndim not in (1, 2):
        raise ValueError('ys must be either a 1D vector or a "tall" 2D matrix')
    if xs.shape[0] != ys.shape[-1]:
        raise ValueError(
            "xs and ys must have the same number of observations"
        )

    ys = ys[np.newaxis, :] if ys.ndim == 1 else ys

    xs_thinned, ys_smoothed = _log_smooth_numba(
        xs, ys, sigma_log10, window_sigma, points_per_sigma
    )

    if points_per_sigma:
        ys_smoothed = np.asarray(
            [_log_interp(xs, xs_thinned, ys_row) for ys_row in ys_smoothed]
        )

    ys_smoothed = ys_smoothed[0] if ys.shape[0] == 1 else ys_smoothed

    return ys_smoothed


def _smooth_cvd_group(cvd, sigma_log10, window_sigma, points_per_sigma, cols=None):
    cols = dict(DEFAULT_CVD_COLS, **({} if cols is None else cols))

    cvd_smoothed = (
        cvd.groupby(cols["dist"])
        .agg(
            {
                cols["n_pixels"]: "sum",
                cols["n_contacts"]: "sum",
            }
        )
        .reset_index()
    )

    smoothed_balanced_sum, smoothed_n_valid = log_smooth(
        cvd_smoothed[cols["dist"]].values.astype(np.float64),
        [
            cvd_smoothed[cols["n_contacts"]].values.astype(np.float64),
            cvd_smoothed[cols["n_pixels"]].values.astype(np.float64),
        ],
        sigma_log10=sigma_log10,
        window_sigma=window_sigma,
        points_per_sigma=points_per_sigma,
    )

    # cvd_smoothed[cols["contact_freq"]] = cvd_smoothed[cols["n_contacts"]] / cvd_smoothed[cols["n_pixels"]]

    cvd_smoothed[cols["n_pixels"] + cols["smooth_suffix"]] = smoothed_n_valid
    cvd_smoothed[cols["n_contacts"] + cols["smooth_suffix"]] = smoothed_balanced_sum
    cvd_smoothed[cols["contact_freq"] + cols["smooth_suffix"]] = (
        cvd_smoothed[cols["n_contacts"] + cols["smooth_suffix"]]
        / cvd_smoothed[cols["n_pixels"] + cols["smooth_suffix"]]
    )

    return cvd_smoothed


def _agg_smooth_cvd(
    cvd, groupby, sigma_log10, window_sigma, points_per_sigma, cols=None
):
    if groupby:
        cvd = cvd.set_index(groupby).groupby(groupby).apply(
            _smooth_cvd_group,
            sigma_log10=sigma_log10,
            window_sigma=window_sigma,
            points_per_sigma=points_per_sigma,
            cols=cols,
        )
    else:
        cvd = _smooth_cvd_group(
            cvd,
            sigma_log10=sigma_log10,
            window_sigma=window_sigma,
            points_per_sigma=points_per_sigma,
            cols=cols,
        )

    return cvd


def agg_smooth_cvd(
    cvd,
    groupby=["region1", "region2"],
    sigma_log10=0.1,
    window_sigma=5,
    points_per_sigma=10,
    cols=None,
):
    """
    Smooth the contact-vs-distance curve in the log-space.

    Parameters
    ----------
    cvd : pandas.DataFrame
        A dataframe with the expected values in the cooltools.expected format.
    groupby : list of str
        The list of column names to split the input table before smoothing.
        This argument can be used to calculate separate smoothed CvD curves for
        each region, Hi-C read orientation, etc.
        If None or empty, a single CvD curve is calculated for the whole table.
    sigma_log10 : float, optional
        The standard deviation of the smoothing Gaussian kernel, applied over log10(diagonal), by default 0.1
    window_sigma : int, optional
        Width of the smoothing window, expressed in sigmas, by default 5
    points_per_sigma : int, optional
        If provided, smoothing is done only for `points_per_sigma` points per sigma and the
        rest of the values are interpolated (this results in a major speed-up). By default 10
    cols : dict, optional
        If provided, use the specified column names instead of the standard ones.
        See DEFAULT_CVD_COLS variable for the format of this argument.

    Returns
    -------
    cvd_smoothed : pandas.DataFrame
        A cvd table with extra column for the log-smoothed contact frequencies (by default, "balanced.avg.smoothed").
    """

    cols = dict(DEFAULT_CVD_COLS, **({} if cols is None else cols))

    if groupby is None:
        groupby = []
    elif isinstance(groupby, str):
        groupby = [groupby]
    elif isinstance(groupby, Iterable):
        groupby = list(groupby)
    else:
        raise ValueError("groupby must be a string, a list of strings, or None")

    cvd_smoothed = _agg_smooth_cvd(
        cvd,
        groupby=groupby,
        sigma_log10=sigma_log10,
        window_sigma=window_sigma,
        points_per_sigma=points_per_sigma,
        cols=cols,
    )

    cvd_smoothed.drop(
        [cols["n_pixels"], cols["n_contacts"]], axis="columns", inplace=True
    )

    # cvd = cvd.drop(cols["contact_freq"], axis='columns', errors='ignore')

    # cvd = cvd.merge(
    #     cvd_smoothed,
    #     on=groupby + [cols["dist"]],
    #     how="left",
    # )

    return cvd_smoothed
