# -*- coding: utf-8 -*-
import numpy as np


def isrle(starts, lengths, values):
    if not (len(starts) == len(lengths) == len(values)):
        return False

    if np.any(np.diff(starts) < 0):
        return False

    ends = starts + lengths
    if np.any(ends[:-1] > starts[1:]):
        return False

    return True


def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle
    function from R.

    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.

    Returns
    -------
    start positions, run lengths, run values

    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=x.dtype),
        )

    isnumeric = np.issubdtype(x.dtype, np.number)

    if isnumeric:
        starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    else:
        starts = np.r_[0, where(x[1:] != x[:-1]) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]

    if isnumeric and dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]

    return starts, lengths, values


def rldecode(starts, lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.

    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.

    Returns
    -------
    1D array. Missing data will be filled with NaNs.

    """
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    # TODO: check validity of rle
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x


def iterruns(x, value=None, **kwargs):
    starts, lengths, values = rlencode(x, **kwargs)
    if value is None:
        ends = starts + lengths
        return zip(starts, ends, values)
    else:
        mask = values == value
        starts, lengths = starts[mask], lengths[mask]
        ends = starts + lengths
        return zip(starts, ends)


def fillgaps(starts, lengths, values, minlength=None, fill_value=np.nan):
    """
    Add additional runs to fill in spaces between runs. Defaults to runs of NaN.
    """
    where = np.flatnonzero
    n = starts[-1] + lengths[-1]
    if minlength is not None:
        n = max(minlength, n)

    ends = starts + lengths
    lo = np.r_[0, ends]
    hi = np.r_[starts, n]
    gap_locs = where((hi - lo) > 0)
    if len(gap_locs):
        starts = np.insert(starts, gap_locs, lo[gap_locs])
        lengths = np.insert(lengths, gap_locs, hi[gap_locs] - lo[gap_locs])
        values = np.insert(values, gap_locs, fill_value)
    return starts, lengths, values


def dropgaps(starts, lengths, values):
    """
    Discard runs of NaN.
    """
    mask = np.isnan(values)
    starts, lengths, values = starts[mask], lengths[mask], values[mask]
    return starts, lengths, values


def align(slv1, slv2, minlength=None):
    """
    Remove NaN runs and runs of length zero and stich together consecutive runs
    of the same value.

    """
    starts1, lengths1, values1 = fillgaps(*slv1)
    starts2, lengths2, values2 = fillgaps(*slv2)
    n1 = starts1[-1] + lengths1[-1]
    n2 = starts2[-1] + lengths2[-1]
    if minlength is not None:
        n = max(minlength, n1, n2)

    starts = np.concatenate([starts1, starts2])
    values = np.concatenate([values1, values2])
    idx = np.argsort(starts)
    starts = starts[idx]
    values = values[idx]
    lengths = np.diff(np.r_[starts, n])
    return starts, lengths, values


def simplify(starts, lengths, values, minlength=None):
    """
    Remove NaN runs and runs of length zero and stich together consecutive runs
    of the same value.

    """
    starts, lengths, values = fillgaps(starts, lengths, values, minlength)
    n = starts[-1] + lengths[-1]

    is_nontrivial = lengths > 0
    starts = starts[is_nontrivial]
    values = values[is_nontrivial]

    is_new_run = np.r_[True, ~np.isclose(values[:-1], values[1:], equal_nan=True)]
    starts = starts[is_new_run]
    values = values[is_new_run]

    lengths = np.r_[starts[1:] - starts[:-1], n - starts[-1]]

    mask = ~np.isnan(values)
    return starts[mask], lengths[mask], values[mask]
