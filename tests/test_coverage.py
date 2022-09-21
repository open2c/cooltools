import os.path as op
import cooler

import cooltools
import cooltools.api
from numpy import testing
import numpy as np
import pandas as pd


def test_coverage_symmetric_upper(request):
    # perform test:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    cis_cov, tot_cov = cooltools.api.coverage.coverage(
        clr, ignore_diags=2, chunksize=int(1e7)
    )

    # Test that minimal coverage is larger than 0.5
    assert tot_cov[tot_cov > 0].min() >= 1

    # Test that dense matrix marginal is the same:
    mtx = clr.matrix(balance=False, as_pixels=False)[:]
    np.fill_diagonal(mtx, 0)
    np.fill_diagonal(mtx[1:, :], 0)
    np.fill_diagonal(mtx[:, 1:], 0)
    cov_dense = np.sum(mtx, axis=1)
    testing.assert_allclose(
        actual=tot_cov,
        desired=cov_dense,
        equal_nan=True,
    )

    """  generate the following cooler to test coverage:
            array([[0, 1, 2],
                   [1, 0, 0],
                   [2, 0, 0]], dtype=int32)
    """

    bins = pd.DataFrame(
        [["chr1", 0, 1], ["chr1", 1, 2], ["chrX", 1, 2]],
        columns=["chrom", "start", "end"],
    )

    pixels = pd.DataFrame(
        [[0, 1, 1], [0, 2, 2]], columns=["bin1_id", "bin2_id", "count"]
    )

    clr_file = op.join(request.fspath.dirname, "data/test_coverage.cool")
    cooler.create_cooler(clr_file, bins, pixels)
    clr = cooler.Cooler(clr_file)
    cis_cov, tot_cov = cooltools.coverage(clr, ignore_diags=0, store=True)
    assert (cis_cov == np.array([1, 1, 0])).all()
    assert (tot_cov == np.array([3, 1, 2])).all()
    assert clr.info["cis"] == 1
    assert clr.info["sum"] == 3
    
def test_balanced_coverage(request):
    # perform test:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    cis_cov_weight, tot_cov_weight = cooltools.api.coverage.coverage(
        clr, ignore_diags=2, chunksize=int(1e7), clr_weight_name="weight"
    )
    
    # Test that mean total balanced coverage is 1.0
    assert np.nanmean(tot_cov_weight) == 1.0
    
    # Generate test matrix with weights
    bins=pd.DataFrame(
        [["chr1", 0, 1, 0.5], 
         ["chr1", 1, 2, 1], 
         ["chrX", 1, 2, 0.2],
         ["chrX", 2, 3, np.nan]],
    columns=["chrom", "start", "end", "weight"],
    )

    pixels = pd.DataFrame(
        [[0, 1, 1], [0, 2, 2], [1, 3, 2], [2, 3, 1]], 
        columns=["bin1_id", "bin2_id", "count"]
    )

    clr_file = op.join(request.fspath.dirname, "data/test_coverage.cool")
    cooler.create_cooler(clr_file, bins, pixels)
    clr = cooler.Cooler(clr_file)
    cis_cov_weight, tot_cov_weight = cooltools.coverage(clr, ignore_diags=0, store=True, clr_weight_name="weight")
    assert np.allclose(cis_cov_weight, np.array([0.5, 0.5, 0, np.nan]),
                       equal_nan=True)
    assert np.allclose(tot_cov_weight, np.array([0.7, 0.5, 0.2, np.nan]),
                      equal_nan=True)
