import numpy as np
import pandas as pd
import numba


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
    Thin out a sorted array, selecting a subset of elements with the uniform density in log-space.

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
    steps_per_sigma=10,
):

    xs_thinned = xs
    if steps_per_sigma:
        xs_thinned = _log_thin(xs, sigma_log10 / steps_per_sigma)

    N = xs_thinned.size
    N_FUNCS = ys.shape[0]

    log_xs = np.log10(xs)
    log_thinned_xs = np.log10(xs_thinned)

    ys_smooth = np.zeros((N_FUNCS, N))

    for i in range(N):
        cur_log_x = log_thinned_xs[i]
        lo = np.searchsorted(log_xs, cur_log_x - sigma_log10 * window_sigma)
        hi = np.searchsorted(log_xs, cur_log_x + sigma_log10 * window_sigma)
        smooth_weights = np.exp(
            -((cur_log_x - log_xs[lo:hi]) ** 2) / 2 / sigma_log10 / sigma_log10
        )
        for k in range(N_FUNCS):
            ys_smooth[k, i] = np.sum(ys[k, lo:hi] * smooth_weights)

    return xs_thinned, ys_smooth


def log_smooth(
    xs,
    ys,
    sigma_log10=0.1,
    window_sigma=5,
    steps_per_sigma=10,
):
    """
    Convolve a function or multiple functions in with a gaussian filter in log space.

    Parameters
    ----------
    xs : 1D array
        The x-coordinates (function arguments) of the data points, must be increasing.
    ys : 1D or 2D array
        The y-coordinates (function values) of the data points.
        If 2D, rows correspond to multiple functions values, columns correspond to different points.
    sigma_log10 : float, optional
        The standard deviation of the smoothing Gaussian kernel, applied over log10(xs), by default 0.1
    window_sigma : int, optional
        Width of the smoothing window, expressed in sigmas, by default 5
    steps_per_sigma : int, optional
        The number of interpolation steps per sigma, by default 10

    Returns
    -------
    thinned_xs : 1D array
        The subset of function arguments, uniformly spaced in log-space.
    ys_smooth : 1D or 2D array
        The Gaussian smoothed function arguments.

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
        xs, ys, sigma_log10, window_sigma, steps_per_sigma
    )
    ys_smoothed = ys_smoothed[0] if ys.shape[0] == 1 else ys_smoothed

    return xs_thinned, ys_smoothed


def agg_smooth_cvd(cvd, sigma_log10=0.1, window_sigma=5, steps_per_sigma=10, **kwargs):

    dist_col = kwargs.get("dist_col", "diag")
    n_pairs_col = kwargs.get("n_pairs_col", "n_valid")
    n_contacts_col = kwargs.get("n_contacts_col", "balanced.sum")
    contact_freq_col = kwargs.get("contact_freq_col", "balanced.avg")
    smooth_suffix = kwargs.get("smooth_suffix", ".smoothed")

    cvd_agg = (
        cvd.groupby(dist_col)
        .agg(
            {
                n_pairs_col: "sum",
                n_contacts_col: "sum",
            }
        )
        .reset_index()
    )

    bin_mids, (balanced, areas) = log_smooth(
        cvd_agg[dist_col].values.astype(np.float64),
        [
            cvd_agg[n_contacts_col].values.astype(np.float64),
            cvd_agg[n_pairs_col].values.astype(np.float64),
        ],
        sigma_log10=sigma_log10,
        steps_per_sigma=steps_per_sigma,
    )

    if steps_per_sigma:
        cvd_agg[n_pairs_col + smooth_suffix] = _log_interp(
            cvd_agg[dist_col].values, bin_mids, areas
        )
        cvd_agg[n_contacts_col + smooth_suffix] = _log_interp(
            cvd_agg[dist_col].values, bin_mids, balanced
        )
    else:
        cvd_agg[n_pairs_col + smooth_suffix] = areas
        cvd_agg[n_contacts_col + smooth_suffix] = balanced

    cvd_agg[contact_freq_col] = cvd_agg[n_contacts_col] / cvd_agg[n_pairs_col]
    cvd_agg[contact_freq_col + smooth_suffix] = (
        cvd_agg[n_contacts_col + smooth_suffix] / cvd_agg[n_pairs_col + smooth_suffix]
    )

    return cvd_agg
