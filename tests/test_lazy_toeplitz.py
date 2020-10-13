from scipy.linalg import toeplitz
import numpy as np
from cooltools.lib.numutils import LazyToeplitz


n = 100
m = 150
c = np.arange(1, n + 1)
r = np.r_[1, np.arange(-2, -m, -1)]

L = LazyToeplitz(c, r)
T = toeplitz(c, r)


def test_symmetric():
    for si in [
        slice(10, 20),
        slice(0, 150),
        slice(0, 0),
        slice(150, 150),
        slice(10, 10),
    ]:
        assert np.allclose(L[si, si], T[si, si])


def test_triu_no_overlap():
    for si, sj in [
        (slice(10, 20), slice(30, 40)),
        (slice(10, 15), slice(30, 40)),
        (slice(10, 20), slice(30, 45)),
    ]:
        assert np.allclose(L[si, sj], T[si, sj])


def test_tril_no_overlap():
    for si, sj in [
        (slice(30, 40), slice(10, 20)),
        (slice(30, 40), slice(10, 15)),
        (slice(30, 45), slice(10, 20)),
    ]:
        assert np.allclose(L[si, sj], T[si, sj])


def test_triu_with_overlap():
    for si, sj in [
        (slice(10, 20), slice(15, 25)),
        (slice(13, 22), slice(15, 25)),
        (slice(10, 20), slice(18, 22)),
    ]:
        assert np.allclose(L[si, sj], T[si, sj])


def test_tril_with_overlap():
    for si, sj in [
        (slice(15, 25), slice(10, 20)),
        (slice(15, 22), slice(10, 20)),
        (slice(15, 25), slice(10, 18)),
    ]:
        assert np.allclose(L[si, sj], T[si, sj])


def test_nested():
    for si, sj in [
        (slice(10, 40), slice(20, 30)),
        (slice(10, 35), slice(20, 30)),
        (slice(10, 40), slice(20, 25)),
        (slice(20, 30), slice(10, 40)),
    ]:
        assert np.allclose(L[si, sj], T[si, sj])
