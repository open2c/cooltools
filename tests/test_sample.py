import os.path as op
import cooler

import cooltools
import cooltools.api
from numpy import testing


def test_sample(request):
    # perform test:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool"),
        frac=0.1,
        nproc=3
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool")
    )
    # Test that deviation from expected total is very small
    testing.assert_allclose(clr_result.info["sum"], clr.info["sum"] / 2, rtol=1e-1)

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool"),
        count=2000,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool")
    )
    # Test that deviation from expected total is very small
    testing.assert_allclose(clr_result.info["sum"], 2000, rtol=1e-1)


def test_sample_exact(request):
    # Exact sampling is very slow! So commented out
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.10000kb.cool"))

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool"),
        frac=0.1,
        exact=True,
        nproc=3
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool")
    )
    # Test that result matches expectation exactly
    testing.assert_equal(clr_result.info["sum"], round(clr.info["sum"] * 0.1))

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool"),
        count=2000,
        exact=True,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool")
    )
    # Test that result matches expectation exactly
    testing.assert_equal(clr_result.info["sum"], 2000)
