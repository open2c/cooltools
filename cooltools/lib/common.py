import numpy as np
import pandas as pd


def assign_supports(features, supports, labels=False, suffix=""):
    """
    Assign support regions to a table of genomic intervals.

    Parameters
    ----------
    features : DataFrame
        Dataframe with columns `chrom`, `start`, `end`
        or `chrom1`, `start1`, `end1`, `chrom2`, `start2`, `end2`
    supports : array-like
        Support areas

    """
    features = features.copy()
    supp_col = pd.Series(index=features.index, data=np.nan)

    c = "chrom" + suffix
    s = "start" + suffix
    e = "end" + suffix
    for col in (c, s, e):
        if col not in features.columns:
            raise ValueError(
                'Column "{}" not found in features data frame.'.format(col)
            )

    for i, region in enumerate(supports):
        # single-region support
        if len(region) == 3:
            sel = (features[c] == region[0]) & (features[e] > region[1])
            if region[2] is not None:
                sel &= features[s] < region[2]
        # paired-region support
        elif len(region) == 2:
            region1, region2 = region
            sel1 = (features[c] == region1[0]) & (features[e] > region1[1])
            if region1[2] is not None:
                sel1 &= features[s] < region1[2]
            sel2 = (features[c] == region2[0]) & (features[e] > region2[1])
            if region2[2] is not None:
                sel2 &= features[s] < region2[2]
            sel = sel1 | sel2
        supp_col.loc[sel] = i

    if labels:
        supp_col = supp_col.map(lambda i: supports[int(i)], na_action="ignore")

    return supp_col
