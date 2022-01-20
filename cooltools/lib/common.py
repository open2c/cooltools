import warnings
from itertools import tee, starmap
from operator import gt
from copy import copy

import numpy as np
import pandas as pd

import bioframe


def assign_view_2D(
    features,
    view_df,
    cols=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
    cols_view=["chrom", "start", "end"],
    features_view_cols=["region1", "region2"],
    view_name_col="name",
    drop_unassigned=False,
):
    """Assign region names from the view to each feature

    Will add two columns

    Parameters
    ----------
    features : pd.DataFrame
        bedpe-style dataframe
    view_df : pandas.DataFrame
        ViewFrame specifying region start and ends for assignment. Attempts to
        convert dictionary and pd.Series formats to viewFrames.
    cols : list of str
        he names of columns containing the chromosome, start and end of the
        genomic intervals. The default values are 'chrom', 'start', 'end'.
    cols_view : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals in the view. The default values are 'chrom', 'start', 'end'.
    features_view_cols : list of str
        Names of the columns where to save the assigned region names
    view_name_col : str
        Column of ``view_df`` with region names. Default 'name'.
    drop_unassignes : bool
        If True, drop intervals in df that do not overlap a region in the view.
        Default False.
    """
    features = features.copy()
    features.reset_index(inplace=True, drop=True)

    bioframe.core.checks.is_bedframe(features, raise_errors=True, cols=cols[:3])
    bioframe.core.checks.is_bedframe(features, raise_errors=True, cols=cols[3:])
    view_df = bioframe.core.construction.make_viewframe(
        view_df, view_name_col=view_name_col, cols=cols_view
    )
    features = bioframe.assign_view(
        features,
        view_df,
        drop_unassigned=drop_unassigned,
        df_view_col=features_view_cols[0],
        view_name_col=view_name_col,
        cols=cols[:3],
        cols_view=cols_view,
    )
    features[cols[-2:]] = features[cols[-2:]].astype(int)  # gets cast to float above...
    features = bioframe.assign_view(
        features,
        view_df,
        drop_unassigned=drop_unassigned,
        df_view_col=features_view_cols[1],
        view_name_col=view_name_col,
        cols=cols[3:],
        cols_view=cols_view,
    )
    return features


def assign_regions(features, supports):
    """
    DEPRECATED. Will be removed in the future versions and replaced with bioframe.overlap()
    For each feature in features dataframe assign the genomic region (support)
    that overlaps with it. In case if feature overlaps multiple supports, the
    region with largest overlap will be reported.
    """

    index_name = features.index.name  # Store the name of index
    features = (
        features.copy()
        .reset_index()
        .rename({"index" if index_name is None else index_name: "native_order"}, axis=1)
    )  # Store the original features' order as a column with original index

    if "chrom" in features.columns:
        overlap = bioframe.overlap(
            features,
            supports,
            how="left",
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            keep_order=True,
            return_overlap=True,
            suffixes=("_1", "_2"),
        )
        overlap_columns = overlap.columns  # To filter out duplicates later
        overlap["overlap_length"] = overlap["overlap_end"] - overlap["overlap_start"]
        # Filter out overlaps with multiple regions:
        overlap = (
            overlap.sort_values("overlap_length", ascending=False)
            .drop_duplicates(overlap_columns, keep="first")
            .sort_index()
        )
        # Copy single column with overlapping region name:
        features["region"] = overlap["name_2"]

    if "chrom1" in features.columns:
        for idx in ("1", "2"):
            overlap = bioframe.overlap(
                features,
                supports,
                how="left",
                cols1=[f"chrom{idx}", f"start{idx}", f"end{idx}"],
                cols2=[f"chrom", f"start", f"end"],
                keep_order=True,
                return_overlap=True,
                suffixes=("_1", "_2"),
            )
            overlap_columns = overlap.columns  # To filter out duplicates later
            overlap[f"overlap_length{idx}"] = (
                overlap[f"overlap_end{idx}"] - overlap[f"overlap_start{idx}"]
            )
            # Filter out overlaps with multiple regions:
            overlap = (
                overlap.sort_values(f"overlap_length{idx}", ascending=False)
                .drop_duplicates(overlap_columns, keep="first")
                .sort_index()
            )
            # Copy single column with overlapping region name:
            features[f"region{idx}"] = overlap["name_2"]

        # Form a single column with region names where region1 == region2, and np.nan in other cases:
        features["region"] = np.where(
            features["region1"] == features["region2"], features["region1"], np.nan
        )
        features = features.drop(
            ["region1", "region2"], axis=1
        )  # Remove unnecessary columns

    features = features.set_index("native_order")  # Restore the original index
    features.index.name = index_name  # Restore original index title
    return features


def assign_supports(features, supports, labels=False, suffix=""):
    """
    Assign support regions to a table of genomic intervals.
    Obsolete, replaced by assign_regions now.

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
        if len(region) in [3, 4]:
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


def assign_regions_to_bins(bin_ids, regions_span):
    regions_binsorted = (
        regions_span[(regions_span["bin_start"] >= 0) & (regions_span["bin_end"] >= 0)]
        .sort_values(["bin_start", "bin_end"])
        .reset_index()
    )

    bin_reg_idx_lo = regions_span["bin_start"].searchsorted(bin_ids, "right") - 1
    bin_reg_idx_hi = regions_span["bin_end"].searchsorted(bin_ids, "right")
    mask_assigned = (bin_reg_idx_lo == bin_reg_idx_hi) & (bin_reg_idx_lo >= 0)

    region_ids = pd.array([pd.NA] * len(bin_ids))
    region_ids[mask_assigned] = regions_span["name"][bin_reg_idx_lo[mask_assigned]]

    return region_ids


def make_cooler_view(clr, ucsc_names=False):
    """
    Generate a full chromosome viewframe
    using cooler's chromsizes

    Parameters
    ----------
    clr :  cooler
        cooler-object to extract chromsizes
    ucsc_names : bool
        Use full UCSC formatted names instead
        of short chromosome names.

    Returns
    -------
    cooler_view : viewframe
        full chromosome viewframe
    """
    cooler_view = bioframe.make_viewframe(clr.chromsizes)
    if ucsc_names:
        # UCSC formatted names
        return cooler_view
    else:
        # rename back to short chromnames
        cooler_view["name"] = cooler_view["chrom"]
        return cooler_view


def view_from_track(track_df):
    bioframe.core.checks._verify_columns(track_df, ["chrom", "start", "end"])
    return bioframe.make_viewframe(
        [
            (chrom, df.start.min(), df.end.max())
            for chrom, df in track_df.groupby("chrom")
        ]
    )


def mask_cooler_bad_bins(track, bintable):
    """
    Mask (set to NaN) values in track where bin is masked in bintable.

    Currently used in `cli.get_saddle()`.
    TODO: determine if this should be used elsewhere.

    Parameters
    ----------
    track : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the value column.
    bintable : tuple of (DataFrame, str)
        bedGraph-like dataframe along with the name of the weight column.

    Returns
    -------
    track : DataFrame
        New bedGraph-like dataframe with bad bins masked in the value column
    """
    # TODO: update to new track format

    track, name = track

    bintable, clr_weight_name = bintable

    track = pd.merge(
        track[["chrom", "start", "end", name]], bintable, on=["chrom", "start", "end"]
    )
    track.loc[~np.isfinite(track[clr_weight_name]), name] = np.nan
    track = track[["chrom", "start", "end", name]]

    return track
