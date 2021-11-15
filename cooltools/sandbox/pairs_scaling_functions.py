###################################
#
# several functions for calculating scalings using pairs
# they used to reside in cooltools.expected module
#
####################################

import numpy as np
from ..lib import numutils

def _contact_areas(distbins, scaffold_length):
    distbins = distbins.astype(float)
    scaffold_length = float(scaffold_length)
    outer_areas = np.maximum(scaffold_length - distbins[:-1], 0) ** 2
    inner_areas = np.maximum(scaffold_length - distbins[1:], 0) ** 2
    return 0.5 * (outer_areas - inner_areas)


def contact_areas(distbins, region1, region2):
    if region1 == region2:
        start, end = region1
        areas = _contact_areas(distbins, end - start)
    else:
        start1, end1 = region1
        start2, end2 = region2
        if start2 <= start1:
            start1, start2 = start2, start1
            end1, end2 = end2, end1
        areas = (
            _contact_areas(distbins, end2 - start1)
            - _contact_areas(distbins, start2 - start1)
            - _contact_areas(distbins, end2 - end1)
        )
        if end1 < start2:
            areas += _contact_areas(distbins, start2 - end1)

    return areas


def compute_scaling(df, region1, region2=None, dmin=int(1e1), dmax=int(1e7), n_bins=50):

    import dask.array as da

    if region2 is None:
        region2 = region1

    distbins = numutils.logbins(dmin, dmax, N=n_bins)
    areas = contact_areas(distbins, region1, region2)

    df = df[
        (df["pos1"] >= region1[0])
        & (df["pos1"] < region1[1])
        & (df["pos2"] >= region2[0])
        & (df["pos2"] < region2[1])
    ]
    dists = (df["pos2"] - df["pos1"]).values

    if isinstance(dists, da.Array):
        obs, _ = da.histogram(dists[(dists >= dmin) & (dists < dmax)], bins=distbins)
    else:
        obs, _ = np.histogram(dists[(dists >= dmin) & (dists < dmax)], bins=distbins)

    return distbins, obs, areas
