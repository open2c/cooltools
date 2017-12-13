import numpy as np
import scipy
import scipy.stats
import pandas as pd
from .num import numutils
from .num import _numutils_cy


def _orient_eigs_gc(eigvals, eigvecs, gc, sort_by_gc_corr=False):
    """
    Flip `eigvecs` to achieve a positive correlation with `gc`. 
    
    Parameters
    ----------
    gc : 1D array, optional
        GC content per bin for choosing and orienting the primary compartment 
        eigenvector.
        
    sort_by_gc_corr : bool
        if True, re-sort `eigenvecs` and `eigvals` in the order of 
        descreasing absolute correlation with `gc`. 
    
    """
    corrs = [scipy.stats.spearmanr(gc, eigvec, nan_policy='omit')[0] 
             for eigvec in eigvecs]
    # flip eigvecs 
    for i in range(len(eigvecs)):
        eigvecs[i] = np.sign(corrs[i]) * eigvecs[i]

    # sort eigvecs
    if sort_by_gc_corr:
        idx = np.argsort(-np.abs(corrs))
        eigvals, eigvecs = eigvals[idx], eigvecs[idx]

    return eigvals, eigvecs


def cis_eig(A, n_eigs=3, gc=None, ignore_diags=2, clip_percentile=0,
            sort_by_gc_corr=False):
    """
    Compute compartment eigenvector on a dense cis matrix

    Parameters
    ----------
    A : 2D array
        balanced dense contact matrix
    k : int
        number of eigenvectors to compute
    gc : 1D array, optional
        GC content per bin for choosing and orienting the primary compartment 
        eigenvector; not performed if no array is provided
    ignore_diags : int
        the number of diagonals to ignore
    clip_percentile : float
        if >0 and <100, clip pixels with diagonal-normalized values
        higher than the specified percentile of matrix-wide values.
    sort_by_gc_corr : bool
        if True, report eigenvectors in the order of descreasing absolute 
        correlation with GC and not by eigenvalue. This option is designed
        to report the most "biologically" informative eigenvectors first,
        and prevent eigenvector swapping caused by translocations.
        In reality, however, shows poor performance and may lead to 
        reporting of non-informative eigenvectors. False by default.
    

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

    if A.shape[0] <= ignore_diags + 3 or mask.sum() <= ignore_diags+3:
        return (
            np.array([np.nan for i in range(n_eigs)]),
            np.array([np.ones(A.shape[0]) * np.nan for i in range(n_eigs)]),
        )
    
    A[~mask, :] = 0
    A[:, ~mask] = 0
    
    if ignore_diags:
        for d in range(-ignore_diags + 1, ignore_diags):
            numutils.set_diag(A, 1.0, d)
    
    OE, _,_,_ = numutils.observed_over_expected(A, mask)

    if clip_percentile and clip_percentile<100:
        OE = np.clip(OE, 0, np.percentile(OE, clip_percentile))

    # subtract 1.0 from valid rows/columns 
    OE -= 1.0 * (mask[:, None] * mask[None, :])
    eigvecs, eigvals = numutils.get_eig(OE, n_eigs, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.sum(eigvecs**2, axis=1))[:,None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    
    # Orient and reorder
    if gc is not None:
        eigvals, eigvecs = _orient_eigs_gc(
                eigvals, eigvecs, gc, sort_by_gc_corr)

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
    idx = np.flatnonzero(dropmask)
    A[idx, :] = 0
    A[:, idx] = 0
    return A


def _fake_cis(A, cismask):
    cismask = cismask.astype(np.int64)
    s = np.abs(np.sum(A, axis=0)) <= 1e-10
    cismask[:, s] = 2
    cismask[s, :] = 2
    _numutils_cy.fake_cis(A, cismask)
    return A


def trans_eig(A, partition, k=3, perc_top=99.95, perc_bottom=1, gc=None,
              sort_by_gc_corr=False):
    """
    Compute compartmentalization eigenvectors on trans contact data

    Parameters
    ----------
    A : 2D array
        balanced whole genome contact matrix
    partition : sequence of int
        bin offset of each contiguous region to treat separately (e.g., 
        chromosomes or chromosome arms)
    k : int
        number of eigenvectors to compute; default = 3
    perc_top : float (percentile)
        filter - clip trans blowout contacts above this cutoff; default = 99.95
    perc_bottom : float (percentile)
        filter - remove bins with trans coverage below this cutoff; default=1
    gc : 1D array, optional
        GC content per bin for reordering and orienting the primary compartment 
        eigenvector; not performed if no array is provided
    sort_by_gc_corr : bool
        if True, report eigenvectors in the order of descreasing absolute 
        correlation with GC and not by eigenvalue. This option is designed
        to report the most "biologically" informative eigenvectors first,
        and prevent eigenvector swapping caused by translocations.
        In reality, however, shows poor performance and may lead to 
        reporting of non-informative eigenvectors. False by default.
    

    Returns
    -------
    eigenvalues, eigenvectors
    
    .. note:: ALWAYS check your EVs by eye. The first one occasionally does 
          not reflect the compartment structure, but instead describes
          chromosomal arms or translocation blowouts.


    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not symmetric")
    
    A = np.array(A)
    A[np.isnan(A)] = 0
    n_bins = A.shape[0]
    if not (partition[0] == 0 and 
            partition[-1] == n_bins and 
            np.all(np.diff(partition) > 0)):
        raise ValueError("Not a valid partition. Must be a monotonic sequence "
                         "from 0 to {}.".format(n_bins))

    # Delete cis data and create trans mask
    extents = zip(partition[:-1], partition[1:])
    part_ids = []
    for n, (i0, i1) in enumerate(extents):
        A[i0:i1, i0:i1] = 0
        part_ids.extend([n] * (i1 - i0))
    part_ids = np.array(part_ids)
    transmask = (part_ids[:, None] != part_ids[None, :])


    # Filter heatmap
    A[~transmask] = 0
    A = _filter_heatmap(A, transmask, perc_top, perc_bottom)
    A = _numutils_cy.iterative_correction_symmetric(A)[0]

    # Fake cis and re-balance
    A = _fake_cis(A, ~transmask)
    #A = numutils.iterative_correction_symmetric(A)[0]
    A = _numutils_cy.iterative_correction_symmetric(A)[0]
    A = _fake_cis(A, ~transmask)
    #A = numutils.iterative_correction_symmetric(A)[0]
    A = _numutils_cy.iterative_correction_symmetric(A)[0]
    
    # Compute eig
    Abar = A.mean()
    O = (A - Abar) / Abar
    eigvecs, eigvals = numutils.get_eig(O, k, mask_zero_rows=True)
    eigvecs /= np.sqrt(np.sum(eigvecs**2, axis=1))[:,None]
    eigvecs *= np.sqrt(np.abs(eigvals))[:, None]
    if gc is not None:
        eigvals, eigvecs = _orient_eigs_gc(
                eigvals, eigvecs, gc, sort_by_gc_corr)
    
    return eigvals, eigvecs


def cooler_cis_eig(clr, bins, n_eigs=3, gc_col='GC', **kwargs):
    bins_grouped = bins.groupby('chrom')

    ignore_diags = kwargs.get(
        'ignore_diags',
        clr._load_attrs('/bins/weight')['ignore_diags'])

    def _each(chrom):
        A = clr.matrix(balance=True).fetch(chrom)
        gc = (bins_grouped.get_group(chrom)[gc_col].values 
              if gc_col in bins else None)
        
        eigvals, eigvecs = cis_eig(
            A, n_eigs=n_eigs, ignore_diags=ignore_diags,
            gc=gc, **kwargs)
        
        eig_chrom_table = bins_grouped.get_group(chrom).copy()
        for i, eigvec in enumerate(eigvecs):
            eig_chrom_table['E{}'.format(i+1)] = eigvec
        
        return eigvals, eig_chrom_table
    
    bins_chroms = list(bins_grouped.groups.keys())
    map_chroms = [chrom for chrom in clr.chromnames if chrom in bins_chroms]

    eigvals, eigvecs = zip(*map(_each, map_chroms))
    eigvals = pd.DataFrame(
        index=map_chroms,
        data=np.vstack(eigvals),
        columns=['eigval'+str(i+1) for i in range(n_eigs)],
    )

    eigvals.index.name = 'chrom'
    eigvecs = pd.concat(eigvecs, axis=0, ignore_index=True)
    return eigvals, eigvecs


def cooler_trans_eig(clr, bins, partition=None, gc_col='GC', **kwargs):

    if partition is None:
        partition = np.r_[ 
            [clr.offset(chrom) for chrom in clr.chromnames], len(clr.bins())]
    
    lo = partition[0]
    hi = partition[-1]
    A = clr.matrix(balance=True)[lo:hi, lo:hi]

    gc = bins[gc_col].values if gc_col in bins else None
    eigvals, eigvecs = trans_eig(A, partition, gc=gc, **kwargs)

    eigvec_table = bins.copy()
    for i, eigvec in enumerate(eigvecs):
        eigvec_table['E{}'.format(i+1)] = eigvec

    return eigvals, eigvec_table

