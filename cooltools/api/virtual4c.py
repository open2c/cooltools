import logging

logging.basicConfig(level=logging.INFO)

from functools import partial

import numpy as np
import pandas as pd
import bioframe


from ..lib.checks import is_cooler_balanced
from ..lib.common import pool_decorator



def _extract_profile(chrom, clr, clr_weight_name, viewpoint):
    to_return = []
    if clr_weight_name:
        colname = "balanced"
    else:
        colname = "count"
    pxls1 = clr.matrix(balance=clr_weight_name, as_pixels=True, join=True).fetch(
        chrom, viewpoint
    )
    pxls1[["chrom2"]] = viewpoint[0]
    pxls1[["start2"]] = viewpoint[1]
    pxls1[["end2"]] = viewpoint[2]

    pxls1 = (
        pxls1.groupby(["chrom1", "start1", "end1"], observed=True)[colname]
        .mean()
        .reset_index()
    )
    pxls1.columns = ["chrom", "start", "end", colname]
    if pxls1.shape[0] > 0:
        to_return.append(pxls1)

    pxls2 = clr.matrix(balance=clr_weight_name, as_pixels=True, join=True).fetch(
        viewpoint, chrom
    )
    pxls2[["chrom1"]] = viewpoint[0]
    pxls2[["start1"]] = viewpoint[1]
    pxls2[["end1"]] = viewpoint[2]
    pxls2 = (
        pxls2.groupby(["chrom2", "start2", "end2"], observed=True)[colname]
        .mean()
        .reset_index()
    )
    pxls2.columns = ["chrom", "start", "end", colname]
    if pxls2.shape[0] > 0:
        to_return.append(pxls2)
    if len(to_return) == 0:
        return pd.DataFrame(columns=["chrom", "start", "end", colname])
    else:
        return pd.concat(to_return, ignore_index=True)

@pool_decorator
def virtual4c(
    clr,
    viewpoint,
    clr_weight_name="weight",
    nproc=1,
    map_functor=map,
):
    """Generate genome-wide contact profile for a given viewpoint.

    Extract all contacts of a given viewpoint from a cooler file.

    Parameters
    ----------
    clr : cooler.Cooler
        A cooler with balanced Hi-C data.
    viewpoint : tuple or str
        Coordinates of the viewpoint.
    clr_weight_name : str
        Name of the column in the bin table with weight
    nproc : int, optional
        How many processes to use for calculation. Ignored if map_functor is passed.
    map_functor : callable, optional
        Map function to dispatch the matrix chunks to workers.
        If left unspecified, pool_decorator applies the following defaults: if nproc>1 this defaults to multiprocess.Pool;
        If nproc=1 this defaults the builtin map. 

    Returns
    -------
    v4C_table : pandas.DataFrame
        A table containing the interaction frequency of the viewpoint with the rest of
        the genome

    Note
    ----
    Note: this is a new (experimental) function, the interface or output might change in
    a future version.
    """
    if clr_weight_name not in [None, False]:
        # check if cooler is balanced
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)

        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e
        colname = "balanced"
    else:
        colname = "count"
    viewpoint = bioframe.core.stringops.parse_region(viewpoint)

    f = partial(
        _extract_profile, clr=clr, clr_weight_name=clr_weight_name, viewpoint=viewpoint
    )

    counts = list(map_functor(f, clr.chromnames))

    # Concatenate all chrompsome dfs into one
    v4c = pd.concat(counts, ignore_index=True)
    if v4c.shape[0] == 0:
        logging.warn(f"No contacts found for viewpoint {viewpoint}")
        v4c = clr.bins()[:][["chrom", "start", "end"]]
        v4c[colname] = np.nan
    else:
        v4c["chrom"] = v4c["chrom"].astype("category")
        v4c["start"] = v4c["start"].astype(int)
        v4c["end"] = v4c["end"].astype(int)
        bioframe.sort_bedframe(
            v4c,
            view_df=bioframe.make_viewframe(clr.chromsizes),
        )  # sort it according clr.chromsizes order
        v4c.loc[
            (v4c["chrom"] == viewpoint[0])
            & (v4c["start"] >= viewpoint[1])
            & (v4c["end"] <= viewpoint[2]),
            colname,
        ] = np.nan  # Set within-viewpoint bins to nan
        v4c = (
            pd.merge(
                clr.bins()[:][["chrom", "start", "end"]],
                v4c,
                on=["chrom", "start", "end"],
                how="left",
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )  # Ensure we return all bins even if empty
    return v4c
