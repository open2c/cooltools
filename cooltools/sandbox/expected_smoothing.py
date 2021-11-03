import numpy as np
import pandas as pd
import numba


DEFAULT_CVD_COLS = {
    "dist": "diag",
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
    if xs.shape[0] == ys.shape[0]:
        raise ValueError(
            "xs and ys must have the same number of elements along the 1st dimension"
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


def _smooth_cvd_group(cvd, agg, sigma_log10, window_sigma, points_per_sigma, cols=None):
    _cols = dict(DEFAULT_CVD_COLS)

    if cols:
        _cols.update(cols)

    dist_col = _cols["dist"]
    contact_freq_col = _cols["contact_freq"]
    n_pixels_col = _cols["n_pixels"]
    n_contacts_col = _cols["n_contacts"]
    smooth_col_suffix = _cols["smooth_suffix"]

    if agg:
        cvd = (
            cvd.groupby(dist_col)
            .agg(
                {
                    n_pixels_col: "sum",
                    n_contacts_col: "sum",
                }
            )
            .reset_index()
        )

    if contact_freq_col not in cvd.columns:
        cvd[contact_freq_col] = cvd[n_contacts_col] / cvd[n_pixels_col]

    balanced, areas = log_smooth(
        cvd[dist_col].values.astype(np.float64),
        [
            cvd[n_contacts_col].values.astype(np.float64),
            cvd[n_pixels_col].values.astype(np.float64),
        ],
        sigma_log10=sigma_log10,
        window_sigma=window_sigma,
        points_per_sigma=points_per_sigma,
    )

    cvd[n_pixels_col + smooth_col_suffix] = areas
    cvd[n_contacts_col + smooth_col_suffix] = balanced

    cvd[contact_freq_col + smooth_col_suffix] = (
        cvd[n_contacts_col + smooth_col_suffix] / cvd[n_pixels_col + smooth_col_suffix]
    )

    return cvd


def smooth_cvd(
    cvd,
    groupby=["region"],
    agg=False,
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
    agg : bool, optional
        If True, additionally group by dist_col and aggregate the table or each group (if groupby is provided)
        before smoothing.
        By default True.
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

    if groupby is None:
        cvd_smoothed = _smooth_cvd_group(
            cvd,
            agg=agg,
            sigma_log10=sigma_log10,
            window_sigma=window_sigma,
            points_per_sigma=points_per_sigma,
            cols=cols,
        )

    else:
        cvd_smoothed = cvd.groupby(groupby).apply(
            _smooth_cvd_group,
            agg=agg,
            sigma_log10=sigma_log10,
            window_sigma=window_sigma,
            points_per_sigma=points_per_sigma,
            cols=cols,
        )

        if agg:
            cvd_smoothed = cvd_smoothed.droplevel(1).reset_index()

    return cvd_smoothed
