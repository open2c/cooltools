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
        frac=0.5,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool")
    )
    # Test that deviation from expected total is very small
    testing.assert_allclose(clr_result.info["sum"], clr.info["sum"] / 2, rtol=1e-3)

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool"),
        count=200000000,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.1000kb.test_sampled.cool")
    )
    # Test that deviation from expected total is very small
    testing.assert_allclose(clr_result.info["sum"], 200000000, rtol=1e-3)


def test_sample_exact(request):
    # Exact sampling is very slow! So commented out
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.10000kb.cool"))

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool"),
        frac=0.5,
        exact=True,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool")
    )
    # Test that result matches expectation exactly
    testing.assert_equal(clr_result.info["sum"], round(clr.info["sum"] * 0.5))

    cooltools.api.sample.sample(
        clr,
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool"),
        count=200000000,
        exact=True,
    )
    clr_result = cooler.Cooler(
        op.join(request.fspath.dirname, "data/CN.mm9.10000kb.test_sampled.cool")
    )
    # Test that result matches expectation exactly
    testing.assert_equal(clr_result.info["sum"], 200000000)
