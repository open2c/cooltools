import numpy as np
import pandas as pd
from scipy.stats import poisson

from cooltools.api.dotfinder import (
    histogram_scored_pixels,
    determine_thresholds,
    annotate_pixels_with_qvalues,
    extract_scored_pixels,
)


# helper functions for BH-FDR copied from www.statsmodels.org
def _fdrcorrection(pvals, alpha=0.05):
    """
    pvalue correction for false discovery rate.

    This covers Benjamini/Hochberg for independent or positively correlated tests.

    Parameters
    ----------
    pvals : np.ndarray
        Sorted set of p-values of the individual tests.
    alpha : float, optional
        Family-wise error rate. Defaults to ``0.05``.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    """

    ntests = len(pvals)
    # empirical Cumulative Distribution Function for pvals:
    ecdffactor = np.arange(1, ntests + 1) / float(ntests)

    reject = pvals <= ecdffactor * alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1
    return reject, pvals_corrected


def multipletests(pvals, alpha=0.1, is_sorted=False):
    """
    Test results and p-value correction for multiple tests
    Using FDR Benjamini-Hochberg method (non-negative)

    Parameters
    ----------
    pvals : array_like, 1-d
        uncorrected p-values. Must be 1-dimensional.
    alpha : float
        FWER, family-wise error rate, e.g. 0.1
    is_sorted : bool
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    reject : ndarray, boolean
        true for hypothesis that can be rejected for given alpha
    pvals_corrected : ndarray
        p-values corrected for multiple tests

    Notes
    -----
    the p-value correction is independent of the
    alpha specified as argument. In these cases the corrected p-values
    can also be compared with a different alpha

    All procedures that are included, control FWER or FDR in the independent
    case, and most are robust in the positively correlated case.
    """
    pvals = np.asarray(pvals)

    if not is_sorted:
        sortind = np.argsort(pvals)
        pvals = np.take(pvals, sortind)

    reject, pvals_corrected = _fdrcorrection(pvals, alpha=alpha)

    if is_sorted:
        return reject, pvals_corrected
    else:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[sortind] = reject
        return reject_, pvals_corrected_


# mock input data to perform some p-value calculation and correction on
num_pixels = 2500
max_value = 99
# fake kernels just for the sake of their names 'd' and 'v':
fake_kernels = {
    "d": np.random.randint(2, size=9).reshape(3, 3),
    "v": np.random.randint(2, size=9).reshape(3, 3),
}
# table with the "scored" pixel (as if they are returned by dotfinder-scoring function)
pixel_dict = {}
# enrich fake counts to all for more significant calls
pixel_dict["count"] = np.random.randint(max_value, size=num_pixels) + 9
for k in fake_kernels:
    pixel_dict[f"la_exp.{k}.value"] = max_value * np.random.random(num_pixels)
scored_df = pd.DataFrame(pixel_dict)

# design lambda-bins as in dot-calling:
num_lchunks = 6
ledges = np.r_[[-np.inf], np.linspace(0, max_value, num_lchunks), [np.inf]]

# set FDR parameter
FDR = 0.1


# helper functions working on a chunk of counts or pvals
# associated with a given lambda-bin:
def get_pvals_chunk(counts_series_lchunk):
    """
    Parameters:
    -----------
    counts_series_lchunk : pd.Series(int)
        Series of raw pixel counts where the name of the Series
        is pd.Interval of the lambda-bin where the pixel belong.
        I.e. counts_series_lchunk.name.right - is the upper limit of the chunk
        and is used as "expected" in Poisson distribution to estimate p-value.

    Returns:
    --------
    pvals: ndarray[float]
        array of p-values for each pixel

    Notes:
    ------
    poisson.sf = 1.0 - poisson.cdf
    """
    return poisson.sf(counts_series_lchunk.values, counts_series_lchunk.name.right)


def get_qvals_chunk(pvals_series_lchunk):
    """
    Parameters:
    -----------
    pvals_series_lchunk : pd.Series(float)
        Series of p-values calculated for each pixel, where the name
        of the Series is pd.Interval of the lambda-bin where the pixel belong.

    Returns:
    --------
    qvals: ndarray[float]
        array of q-values, i.e. p-values corrected with the multiple hypothesis
        testing procedure BH-FDR, for each pixel

    Notes:
    ------
    level of False Discore Rate (FDR) is fixed for testing
    """
    _, qvals = multipletests(pvals_series_lchunk.values, alpha=FDR, is_sorted=False)
    return qvals


def get_reject_chunk(pvals_series_lchunk):
    """
    Parameters:
    -----------
    pvals_series_lchunk : pd.Series(float)
        Series of p-values calculated for each pixel, where the name
        of the Series is pd.Interval of the lambda-bin where the pixel belong.

    Returns:
    --------
    rej: ndarray[bool]
        array of rejection statuses, i.e. for every p-values return if corresponding
        null hypothesis can be rejected or not, using multiple hypothesis testing
        procedure BH-FDR.

    Notes:
    ------
     - pixels with rejected status (not null) are considered as significantly enriched
     - level of False Discore Rate (FDR) is fixed for testing
    """
    rej, _ = multipletests(pvals_series_lchunk.values, alpha=FDR, is_sorted=False)
    return rej


# for the fake scored-pixel table calculate p-vals, q-vals, l-chunk where they belong
# and rejection status using introduced statsmodels-based helper functions:
for k in fake_kernels:
    lbin = pd.cut(scored_df[f"la_exp.{k}.value"], ledges)
    scored_df[f"{k}.pval"] = scored_df.groupby(lbin)["count"].transform(get_pvals_chunk)
    scored_df[f"{k}.qval"] = scored_df.groupby(lbin)[f"{k}.pval"].transform(
        get_qvals_chunk
    )
    scored_df[f"{k}.rej"] = scored_df.groupby(lbin)[f"{k}.pval"].transform(
        get_reject_chunk
    )


# test functions in dotfinder using this reference from statsmodels
def test_histogramming_summary():
    gw_hists = histogram_scored_pixels(
        scored_df, kernels=fake_kernels, ledges=ledges, obs_raw_name="count"
    )
    # make sure total sum of the histogram yields total number of pixels:
    for k, _hist in gw_hists.items():
        assert _hist.sum().sum() == num_pixels
        assert _hist.index.is_monotonic  # is index sorted


# test threshold and rejection tables and only then try q-values
def test_thresholding():
    # rebuild hists
    gw_hists = histogram_scored_pixels(
        scored_df, kernels=fake_kernels, ledges=ledges, obs_raw_name="count"
    )

    # # we have to make sure there is nothing in the last lambda-bin
    # # this is a temporary implementation detail, until we implement dynamic lambda-bins
    for k in fake_kernels:
        last_lambda_bin = gw_hists[k].iloc[:, -1]
        assert last_lambda_bin.sum() == 0  # should be True by construction:
        # drop that last column/bin (last_edge, +inf]:
        gw_hists[k] = gw_hists[k].drop(columns=last_lambda_bin.name)

    # calculate q-values and rejection threshold using dotfinder built-in methods
    # that are the reimplementation of HiCCUPS statistical procedures:
    threshold_df, qvalues = determine_thresholds(gw_hists, FDR)

    enriched_pixels_df = extract_scored_pixels(
        scored_df, threshold_df, obs_raw_name="count"
    )

    # all enriched pixels have their Null hypothesis rejected
    assert enriched_pixels_df["d.rej"].all()
    assert enriched_pixels_df["v.rej"].all()
    # number of enriched pixels should match that number of
    # pixels with both null-hypothesis rejected:
    assert (scored_df["d.rej"] & scored_df["v.rej"]).sum() == len(enriched_pixels_df)


def test_qvals():
    # rebuild hists
    gw_hists = histogram_scored_pixels(
        scored_df, kernels=fake_kernels, ledges=ledges, obs_raw_name="count"
    )

    # # we have to make sure there is nothing in the last lambda-bin
    # # this is a temporary implementation detail, until we implement dynamic lambda-bins
    for k in fake_kernels:
        last_lambda_bin = gw_hists[k].iloc[:, -1]
        assert last_lambda_bin.sum() == 0  # should be True by construction:
        # drop that last column/bin (last_edge, +inf]:
        gw_hists[k] = gw_hists[k].drop(columns=last_lambda_bin.name)

    # calculate q-values and rejection threshold using dotfinder built-in methods
    # that are the reimplementation of HiCCUPS statistical procedures:
    threshold_df, qvalues = determine_thresholds(gw_hists, FDR)
    # annotate scored pixels with q-values:
    scored_df_qvals = annotate_pixels_with_qvalues(
        scored_df, qvalues, obs_raw_name="count"
    )

    # our procedure in dotfiner should match these q-values exactly, including >1.0
    assert np.allclose(scored_df_qvals["v.qval"], scored_df_qvals["la_exp.v.qval"])
    assert np.allclose(scored_df_qvals["d.qval"], scored_df_qvals["la_exp.d.qval"])
