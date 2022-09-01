import numpy as np
import scipy
import scipy.stats

import pandas as pd
from ..lib import numutils
from ..lib.checks import is_compatible_viewframe, is_cooler_balanced
from ..lib.common import make_cooler_view, align_track_with_cooler

import bioframe


def _phase_eigs(eigvals, eigvecs, phasing_track, sort_metric=None):
    """
    Flip `eigvecs` to achieve a positive correlation with `phasing_track`.

    Parameters
    ----------
    sort_metric : str
        If provided, re-sort `eigenvecs` and `eigvals` in the order of
        decreasing correlation between phasing_track and eigenvector, using the
        specified measure of correlation. Possible values:
        'pearsonr' - sort by decreasing Pearson correlation.
        'var_explained' - sort by decreasing absolute amount of variation in
        `eigvecs` explained by `phasing_track` (i.e. R^2 * var(eigvec)).
        'MAD_explained' - sort by decreasing absolute amount of Median Absolute
        Deviation from the median of `eigvecs` explained by `phasing_track`
        (i.e. COMED(eigvec, phasing_track) * MAD(eigvec)).
        'spearmanr' - sort by decreasing Spearman correlation.
    """

    corrs = []
    for eigvec in eigvecs:
        mask = np.isfinite(eigvec) & np.isfinite(phasing_track)
        if sort_metric is None or sort_metric == "spearmanr":
            corr = scipy.stats.spearmanr(phasing_track[mask], eigvec[mask])[0]
        elif sort_metric == "pearsonr":
            corr = scipy.stats.pearsonr(phasing_track[mask], eigvec[mask])[0]
        elif sort_metric == "var_explained":
            corr = scipy.stats.pearsonr(phasing_track[mask], eigvec[mask])[0]
            # multiply by the sign to keep the phasing information
            corr = np.sign(corr) * corr * corr * np.var(eigvec[mask])
        elif sort_metric == "MAD_explained":
            corr = numutils.COMED(phasing_track[mask], eigvec[mask]) * numutils.MAD(
                eigvec[mask]
            )
        else:
            raise ValueError("Unknown sorting metric: {}".format(sort_metric))

        corrs.append(corr)

    # flip eigvecs
    for i in range(len(eigvecs)):
        eigvecs[i] = np.sign(corrs[i]) * eigvecs[i]

    # sort eigvecs
    if sort_metric is not None:
        idx = np.argsort(-np.abs(corrs))
        eigvals, eigvecs = eigvals[idx], eigvecs[idx]

    return eigvals, eigvecs


def cis_eig(
    A, n_eigs=3, phasing_track=None, ignore_diags=2, clip_percentile=0, sort_metric=None
):
    """
    Compute compartment eigenvector on a dense cis matrix.

    Note that the amplitude of compartment eigenvectors is weighted by their
    corresponding eigenvalue

    Parameters
    ----------
    A : 2D array
        balanced dense contact matrix
    n_eigs : int
        number of eigenvectors to compute
    phasing_track : 1D array, optional
        if provided, eigenvectors are flipped to achieve a positive correlation
        with `phasing_track`.
    ignore_diags : int
        the number of diagonals to ignore
    clip_percentile : float
        if >0 and <100, clip pixels with diagonal-normalized values
        higher than the specified percentile of matrix-wide values.
    sort_metric : str
        If provided, re-sort `eigenvecs` and `eigvals` in the order of
        decreasing correlation between phasing_track and eigenvector, using the
        specified measure of correlation. Possible values:
        'pearsonr' - sort by decreasing Pearson correlation.
        'var_explained' - sort by decreasing absolute amount of variation in
        `eigvecs` explained by `phasing_track` (i.e. R^2 * var(eigvec))
        'MAD_explained' - sort by decreasing absolute amount of Median Absolute
        Deviation from the median of `eigvecs` explained by `phasing_track`
        (i.e. COMED(eigvec, phasing_track) * MAD(eigvec)).
        'spearmanr' - sort by decreasing Spearman correlation.
        This option is designed to report the most "biologically" informative
        eigenvectors first, and prevent eigenvector swapping caused by
        translocations. In reality, however, sometimes it shows poor
        performance and may lead to reporting of non-informative eigenvectors.
        Off by default.


    Returns
    -------
    eigenvalues, eigenvectors

    .. note:: ALWAYS check your EVs by eye. The first one occasionally does
              not reflect the compartment structure, but instead describes
              chromosomal arms or translocation blowouts.

    """
    A = np.array(A)
    A[~np.isfinite(A)] = 0

    mask = A.sum(axis=0) > 0

    if A.shape[0] <= ignore_diags + 3 or mask.sum() <= ignore_diags + 3:
        return (
            np.array([np.nan for i in range(n_eigs)]),
            np.array([np.ones(A.shape[0]) * np.nan for i in range(n_eigs)]),
        )

    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(A, 1.0, d)

    OE, _, _, _ = numutils.observed_over_expected(A, mask)

    if clip_percentile and clip_percentile < 100:
        OE = np.clip(OE, 0, np.percentile(OE[mask, :][:, mask], clip_percentile))

    # subtract 1.0
    OE -= 1.0

    # empty invalid rows, so that get_eig can find them
    OE[~mask, :] = 0
    OE[:, ~mask] = 0

    eigvecs, eigvals = numutils.get_eig(OE, n_eigs, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.nansum(eigvecs ** 2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]

    # Orient and reorder
    if phasing_track is not None:
        eigvals, eigvecs = _phase_eigs(eigvals, eigvecs, phasing_track, sort_metric)

    return eigvals, eigvecs


def _filter_heatmap(A, transmask, perc_top, perc_bottom):
    # Truncate trans blowouts
    lim = np.percentile(A[transmask], perc_top)
    tdata = A[transmask]
    tdata[tdata > lim] = lim
    A[transmask] = tdata

    # Remove bins with poor coverage in trans
    marg = np.sum(A, axis=0)
    marg_nz = marg[np.sum(A, axis=0) > 0]
    min_cutoff = np.percentile(marg_nz, perc_bottom)
    dropmask = (marg > 0) & (marg < min_cutoff)
    A[dropmask, :] = 0
    A[:, dropmask] = 0
    return A


def _fake_cis(A, cismask):
    cismask = cismask.astype(np.uint8)
    s = np.abs(np.sum(A, axis=0)) <= 1e-10
    cismask[:, s] = 2
    cismask[s, :] = 2
    numutils.fake_cis(A, cismask)
    return A


def trans_eig(
    A,
    partition,
    n_eigs=3,
    perc_top=99.95,
    perc_bottom=1,
    phasing_track=None,
    sort_metric=False,
):
    """
    Compute compartmentalization eigenvectors on trans contact data

    Parameters
    ----------
    A : 2D array
        balanced whole genome contact matrix
    partition : sequence of int
        bin offset of each contiguous region to treat separately (e.g.,
        chromosomes or chromosome arms)
    n_eigs : int
        number of eigenvectors to compute; default = 3
    perc_top : float (percentile)
        filter - clip trans blowout contacts above this cutoff; default = 99.95
    perc_bottom : float (percentile)
        filter - remove bins with trans coverage below this cutoff; default=1
    phasing_track : 1D array, optional
        if provided, eigenvectors are flipped to achieve a positive correlation
        with `phasing_track`.
    sort_metric : str
        If provided, re-sort `eigenvecs` and `eigvals` in the order of
        decreasing correlation between phasing_track and eigenvector, using the
        specified measure of correlation. Possible values:
        'pearsonr' - sort by decreasing Pearson correlation.
        'var_explained' - sort by decreasing absolute amount of variation in
        `eigvecs` explained by `phasing_track` (i.e. R^2 * var(eigvec))
        'MAD_explained' - sort by decreasing absolute amount of Median Absolute
        Deviation from the median of `eigvecs` explained by `phasing_track`
        (i.e. COMED(eigvec, phasing_track) * MAD(eigvec)).
        'spearmanr' - sort by decreasing Spearman correlation.
        This option is designed to report the most "biologically" informative
        eigenvectors first, and prevent eigenvector swapping caused by
        translocations. In reality, however, sometimes it shows poor
        performance and may lead to reporting of non-informative eigenvectors.
        Off by default.


    Returns
    -------
    eigenvalues, eigenvectors

    .. note:: ALWAYS check your EVs by eye. The first one occasionally does
          not reflect the compartment structure, but instead describes
          chromosomal arms or translocation blowouts.


    """
    A = np.array(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not symmetric")

    n_bins = A.shape[0]
    if not (
        partition[0] == 0 and partition[-1] == n_bins and np.all(np.diff(partition) > 0)
    ):
        raise ValueError(
            "Not a valid partition. Must be a monotonic sequence "
            "from 0 to {}.".format(n_bins)
        )

    # Delete cis data and create trans mask
    extents = zip(partition[:-1], partition[1:])
    part_ids = []
    for n, (i0, i1) in enumerate(extents):
        A[i0:i1, i0:i1] = 0
        part_ids.extend([n] * (i1 - i0))
    part_ids = np.array(part_ids)
    is_trans = part_ids[:, None] != part_ids[None, :]

    # Filter heatmap
    is_bad_bin = np.nansum(A, axis=0) == 0
    is_good_bin = ~is_bad_bin
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    A = _filter_heatmap(A, is_trans & is_valid, perc_top, perc_bottom)
    is_bad_bin = np.nansum(A, axis=0) == 0
    is_good_bin = ~is_bad_bin
    is_valid = np.logical_and.outer(is_good_bin, is_good_bin)
    A[is_bad_bin, :] = 0
    A[:, is_bad_bin] = 0

    # Fake cis and re-balance
    A = _fake_cis(A, ~is_trans)
    A = numutils.iterative_correction_symmetric(A)[0]
    A = _fake_cis(A, ~is_trans)
    A = numutils.iterative_correction_symmetric(A)[0]

    # Compute eig
    Abar = np.mean(A[is_valid])  # center by scalar mean
    O = (A - Abar) / Abar
    O[is_bad_bin, :] = 0
    O[:, is_bad_bin] = 0
    eigvecs, eigvals = numutils.get_eig(O, n_eigs, mask_zero_rows=True)

    eigvecs /= np.sqrt(np.nansum(eigvecs ** 2, axis=1))[:, None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    if phasing_track is not None:
        eigvals, eigvecs = _phase_eigs(eigvals, eigvecs, phasing_track, sort_metric)

    return eigvals, eigvecs


def eigs_cis(
    clr,
    phasing_track=None,
    view_df=None,
    n_eigs=3,
    clr_weight_name="weight",
    ignore_diags=None,
    clip_percentile=99.9,
    sort_metric=None,
    map=map,
):
    """
    Compute compartment eigenvector for a given cooler `clr` in a number of
    symmetric intra chromosomal regions defined in view_df (cis-regions), or for each
    chromosome.

    Note that the amplitude of compartment eigenvectors is weighted by their
    corresponding eigenvalue. Eigenvectors can be oriented by passing a binned
    `phasing_track` with the same resolution as the cooler.


    Parameters
    ----------
    clr : cooler
        cooler object to fetch data from
    phasing_track : DataFrame
        binned track with the same resolution as cooler bins, the fourth column is
        used to phase the eigenvectors, flipping them to achieve a positive correlation.
    view_df : iterable or DataFrame, optional
        if provided, eigenvectors are calculated for the regions of the view only,
        otherwise chromosome-wide eigenvectors are computed, for chromosomes
        specified in phasing_track.
    n_eigs : int
        number of eigenvectors to compute
    clr_weight_name : str
        name of the column with balancing weights to be used.
    ignore_diags : int, optional
        the number of diagonals to ignore. Derived from cooler metadata
        if not specified.
    clip_percentile : float
        if >0 and <100, clip pixels with diagonal-normalized values
        higher than the specified percentile of matrix-wide values.
    sort_metric : str
        If provided, re-sort `eigenvecs` and `eigvals` in the order of
        decreasing correlation between phasing_track and eigenvector, using the
        specified measure of correlation. Possible values:
        'pearsonr' - sort by decreasing Pearson correlation.
        'var_explained' - sort by decreasing absolute amount of variation in
        `eigvecs` explained by `phasing_track` (i.e. R^2 * var(eigvec))
        'MAD_explained' - sort by decreasing absolute amount of Median Absolute
        Deviation from the median of `eigvecs` explained by `phasing_track`
        (i.e. COMED(eigvec, phasing_track) * MAD(eigvec)).
        'spearmanr' - sort by decreasing Spearman correlation.
        This option is designed to report the most "biologically" informative
        eigenvectors first, and prevent eigenvector swapping caused by
        translocations. In reality, however, sometimes it shows poor
        performance and may lead to reporting of non-informative eigenvectors.
        Off by default.
    map : callable, optional
        Map functor implementation.
    Returns
    -------
    eigvals, eigvec_table -> DataFrames with eigenvalues for each region and
    a table of eigenvectors filled in the `bins` table.
    .. note:: ALWAYS check your EVs by eye. The first one occasionally does
              not reflect the compartment structure, but instead describes
              chromosomal arms or translocation blowouts. Possible mitigations:
              employ `view_df` (e.g. arms) to avoid issues with chromosomal arms,
              consider blacklisting regions with translocations during balancing.
    """

    # get chromosomes from cooler, if view_df not specified:
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
    try:
        _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
    except Exception as e:
        raise ValueError(
            f"provided cooler is not balanced or {clr_weight_name} is missing"
        ) from e

    # ignore diags as in cooler unless specified
    ignore_diags = (
        clr._load_attrs(f"bins/{clr_weight_name}").get("ignore_diags", 2)
        if ignore_diags is None
        else ignore_diags
    )

    bins = clr.bins()[:]

    if phasing_track is not None:
        phasing_track = align_track_with_cooler(
            phasing_track,
            clr,
            view_df=view_df,
            clr_weight_name=clr_weight_name,
            mask_clr_bad_bins=True,
            drop_track_na=True # this adds check for chromosomes that have all missing values
        )

    # prepare output table for eigen vectors
    eigvec_table = bioframe.assign_view(bins, view_df).dropna(subset=["view_region"], axis=0)
    eigvec_table = eigvec_table.loc[:, bins.columns]
    eigvec_columns = [f"E{i + 1}" for i in range(n_eigs)]
    for ev_col in eigvec_columns:
        eigvec_table[ev_col] = np.nan

    # prepare output table for eigenvalues
    eigvals_table = view_df.copy()
    eigval_columns = [f"eigval{i + 1}" for i in range(n_eigs)]
    for eval_col in eigval_columns:
        eigvals_table[eval_col] = np.nan

    def _each(region):
        """
        perform eigen decomposition for a given region
        assuming safety checks are done outside of this
        function.
        Parameters
        ----------
        region: tuple-like
            tuple of the form (chroms,start,end,*)
        Returns
        -------
        _region, eigvals, eigvecs -> ndarrays
            array of eigenvalues and an array eigenvectors
        """
        _region = region[:3]  # take only (chrom, start, end)
        A = clr.matrix(balance=clr_weight_name).fetch(_region)

        # extract phasing track relevant for the _region
        if phasing_track is not None:
            phasing_track_region = bioframe.select(phasing_track, _region)
            phasing_track_region_values = phasing_track_region["value"].values
        else:
            phasing_track_region_values = None

        eigvals, eigvecs = cis_eig(
            A,
            n_eigs=n_eigs,
            ignore_diags=ignore_diags,
            phasing_track=phasing_track_region_values,
            clip_percentile=clip_percentile,
            sort_metric=sort_metric,
        )

        return _region, eigvals, eigvecs

    # eigendecompose matrix per region (can be multiprocessed)
    # output assumes that the order of results matches regions
    results = map(_each, view_df.values)

    # go through eigendecomposition results and fill in
    # output table eigvec_table and eigvals_table
    for _region, _eigvals, _eigvecs in results:
        idx = bioframe.select(eigvec_table, _region).index
        eigvec_table.loc[idx, eigvec_columns] = _eigvecs.T
        idx = bioframe.select(eigvals_table, _region).index
        eigvals_table.loc[idx, eigval_columns] = _eigvals

    return eigvals_table, eigvec_table


def eigs_trans(
    clr,
    phasing_track=None,
    n_eigs=3,
    partition=None,
    clr_weight_name="weight",
    sort_metric=None,
    **kwargs,
):

    # check if cooler is balanced
    try:
        _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
    except Exception as e:
        raise ValueError(
            f"provided cooler is not balanced or {clr_weight_name} is missing"
        ) from e

    # TODO: implement usage of view for eigs_trans
    view_df = None
    if view_df is None:
        view_df = make_cooler_view(clr)
    else:
        raise NotImplementedError("views are not currently implemented for eigs_trans")

    if partition is None:
        partition = np.r_[
            [clr.offset(chrom) for chrom in clr.chromnames], len(clr.bins())
        ]

    lo = partition[0]
    hi = partition[-1]
    A = clr.matrix(balance=clr_weight_name)[lo:hi, lo:hi]
    bins = clr.bins()[lo:hi]

    if phasing_track is not None:
        phasing_track = align_track_with_cooler(
            phasing_track,
            clr,
            view_df=view_df,
            clr_weight_name=clr_weight_name,
            mask_clr_bad_bins=True,
            drop_track_na=True # this adds check for chromosomes that have all missing values
        )
        phasing_track_values = phasing_track["value"].values[lo:hi]
    else:
        phasing_track_values = None

    eigvals, eigvecs = trans_eig(
        A,
        partition,
        n_eigs=n_eigs,
        phasing_track=phasing_track_values,
        sort_metric=sort_metric,
        **kwargs,
    )

    eigvec_table = bins.copy()
    for i, eigvec in enumerate(eigvecs):
        eigvec_table[f"E{i + 1}"] = eigvec

    eigvals = pd.DataFrame(
        data=np.atleast_2d(eigvals),
        columns=[f"eigval{i + 1}" for i in range(n_eigs)],
    )
    return eigvals, eigvec_table
