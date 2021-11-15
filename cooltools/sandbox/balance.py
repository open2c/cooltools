from functools import partial, reduce
from multiprocess import Pool
from operator import add

import numpy as np
import pandas
import pandas
import h5py

from scipy.sparse import linalg
from cooler.tools import split, partition
import cooler


def bnewt(matvec, mask, tol=1e-6, x0=None, delta=0.1, Delta=3, fl=0):
    """
    A balancing algorithm for symmetric matrices

    X = BNEWT(A) attempts to find a vector X such that
    diag(X)*A*diag(X) is close to doubly stochastic. A must
    be symmetric and nonnegative.

    Parameters
    ----------
    matvec : callable
        Linear operator that returns the matrix-vector product with x
    mask : 1D array of bool
        Mask of good bins
    tol : float
        Error tolerance
    x0 : 1D array
        Initial guess
    delta : float
        How close balancing vectors can get to the edge of the positive cone
    Delta : float
        How far balancing vectors can get from the edge of the positive cone

    We use a relative measure on the size of elements.

    Returns
    -------
    x : 1D array
        balancing weights
    res : float
        residual error, measured by norm(diag(x)*A*x - e)

    """
    # Initialize
    n = mask.sum()

    e = np.ones(n)
    if x0 is None:
        x0 = e.copy()
    res = []

    # Inner stopping criterion parameters.
    g = 0.9
    etamax = 0.1
    eta = etamax
    stop_tol = tol * 0.5
    x = x0
    rt = tol ** 2
    v = x * matvec(x, mask)

    rk = 1 - v
    rho_km1 = np.dot(rk, rk)
    rho_km2 = None # will be defined later
    rout = rho_km1
    rold = rout

    MVP = 0  # We’ll count matrix vector products.
    i = 0  # Outer iteration count.

    if fl == 1:
        print("it in. it res", flush=True)

    # Outer iteration
    while rout > rt:
        i += 1
        k = 0
        y = e.copy()
        innertol = max((eta ** 2) * rout, rt)

        # Inner iteration by Conjugate Gradient
        while rho_km1 > innertol:
            k += 1

            if k == 1:
                Z = rk / v
                p = Z.copy()
                rho_km1 = np.dot(rk, Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            # Update search direction efficiently.
            w = x * matvec(x * p, mask) + v * p

            alpha = rho_km1 / np.dot(p, w)
            ap = alpha * p

            # Test distance to boundary of cone.
            ynew = y + ap
            if min(ynew) <= delta:
                if delta == 0:
                    break
                idx = ap < 0
                gamma = np.min((delta - y[idx]) / ap[idx])
                y = y + gamma * ap
                break

            if max(ynew) >= Delta:
                idx = ynew > Delta
                gamma = np.min((Delta - y[idx]) / ap[idx])
                y = y + gamma * ap
                break

            y = ynew.copy()
            rk = rk - alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = np.dot(rk, Z)

        x = x * y
        v = x * matvec(x, mask)

        rk = 1 - v
        rho_km1 = np.dot(rk, rk)
        rout = rho_km1
        MVP += k + 1

        # Update inner iteration stopping criterion.
        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat
        if g * (eta_o ** 2) > 0.1:
            eta = max(eta, g * (eta_o ** 2))

        eta = max(min(eta, etamax), stop_tol / res_norm)
        if fl == 1:
            print("%3d\t%6d\t%.3e" % (i, k, res_norm), flush=True)
        res.append(res_norm)

        print("Matrix-vector products = %6d" % (MVP,), flush=True)

    x_full = np.zeros(len(mask))
    x_full[mask] = x
    return x_full, np.array(res)
