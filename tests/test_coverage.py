import os.path as op
import cooler

import cooltools
import cooltools.api
from numpy import testing
import numpy as np


def test_coverage_symmetric_upper(request):
    # perform test:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    cov = cooltools.api.coverage.coverage(clr, ignore_diags=2, chunksize=int(1e7))

    # Test that minimal coverage is larger than 0.5
    assert cov[cov > 0].min() >= 1

    # Test that dense matrix marginal is the same:
    mtx = clr.matrix(balance=False, as_pixels=False)[:]
    np.fill_diagonal(mtx, 0)
    np.fill_diagonal(mtx[1:, :], 0)
    np.fill_diagonal(mtx[:, 1:], 0)
    cov_dense = np.sum(mtx, axis=1)
    testing.assert_allclose(
        actual=cov[1],
        desired=cov_dense,
        equal_nan=True,
    )
