import warnings
from itertools import tee, starmap
from operator import gt
from copy import copy

import numpy as np
import pandas as pd

import bioframe


def assign_view_paired(
    features,
    view_df,
    cols_paired=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
    cols_view=["chrom", "start", "end"],
    features_view_cols=["region1", "region2"],
    view_name_col="name",
    drop_unassigned=False,
):
    """Assign region names from the view to each feature

    Assigns a regular 1D view independently to each side of a bedpe-style dataframe.
    Will add two columns with region names (`features_view_cols`)

    Parameters
    ----------
    features : pd.DataFrame
        bedpe-style dataframe
    view_df : pandas.DataFrame
        ViewFrame specifying region start and ends for assignment. Attempts to
        convert dictionary and pd.Series formats to viewFrames.
    cols_paired : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals. The default values are `"chrom1", "start1", "end1", "chrom2",
        "start2", "end2"`.
    cols_view : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals in the view. The default values are `"chrom", "start", "end"`.
    features_view_cols : list of str
        Names of the columns where to save the assigned region names
    view_name_col : str
        Column of ``view_df`` with region names. Default "name".
    drop_unassigned : bool
        If True, drop intervals in df that do not overlap a region in the view.
        Default False.
    """
    features = features.copy()
    features.reset_index(inplace=True, drop=True)

    cols_left = cols_paired[:3]
    cols_right = cols_paired[3:]

    bioframe.core.checks.is_bedframe(features, raise_errors=True, cols=cols_left)
    bioframe.core.checks.is_bedframe(features, raise_errors=True, cols=cols_right)
    view_df = bioframe.core.construction.make_viewframe(
        view_df, view_name_col=view_name_col, cols=cols_view
    )
    features = bioframe.assign_view(
        features,
        view_df,
        drop_unassigned=drop_unassigned,
        df_view_col=features_view_cols[0],
        view_name_col=view_name_col,
        cols=cols_left,
        cols_view=cols_view,
    )
    features[cols_right[1:]] = features[cols_right[1:]].astype(
        int
    )  # gets cast to float above...
    features = bioframe.assign_view(
        features,
        view_df,
        drop_unassigned=drop_unassigned,
        df_view_col=features_view_cols[1],
        view_name_col=view_name_col,
        cols=cols_right,
        cols_view=cols_view,
    )
    return features


def assign_view_auto(
    features,
    view_df,
    cols_unpaired=["chrom", "start", "end"],
    cols_paired=["chrom1", "start1", "end1", "chrom2", "start2", "end2"],
    cols_view=["chrom", "start", "end"],
    features_view_col_unpaired="region",
    features_view_cols_paired=["region1", "region2"],
    view_name_col="name",
    drop_unassigned=False,
    combined_assignments_column="region",
):
    """Assign region names from the view to each feature

    Determines whether the `features` are unpaired (1D, bed-like) or paired (2D,
    bedpe-like) based on presence of column names (`cols_unpaired` vs `cols_paired`)
    Assigns a regular 1D view, independently to each side in case of paired features.
    Will add one or two columns with region names (`features_view_col_unpaired` or
    `features_view_cols_paired`) respectively, in case of unpaired and paired features.

    Parameters
    ----------
    features : pd.DataFrame
        bedpe-style dataframe
    view_df : pandas.DataFrame
        ViewFrame specifying region start and ends for assignment. Attempts to
        convert dictionary and pd.Series formats to viewFrames.
    cols_unpaired : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals for unpaired features.
        The default values are `"chrom", "start", "end"`.
    cols_paired : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals for paired features.
        The default values are `"chrom1", "start1", "end1", "chrom2", "start2", "end2"`.
    cols_view : list of str
        The names of columns containing the chromosome, start and end of the
        genomic intervals in the view. The default values are `"chrom", "start", "end"`.
    features_view_cols_unpaired : str
        Name of the column where to save the assigned region name for unpaired features
    features_view_cols_paired : list of str
        Names of the columns where to save the assigned region names for paired features
    view_name_col : str
        Column of ``view_df`` with region names. Default "name".
    drop_unassigned : bool
        If True, drop intervals in `features` that do not overlap a region in the view.
        Default False.
    combined_assignments_column : str or None
        If set to a string value, will combine assignments from two sides of paired
        features when they match into column with this name: region name when regions
        assigned to both sides match, np.nan if not.
        Default "region"
    """
    if set(cols_unpaired).issubset(features.columns.astype(str)):
        features = bioframe.assign_view(
            features,
            view_df,
            df_view_col=features_view_col_unpaired,
            view_name_col=view_name_col,
            cols=cols_unpaired,
            cols_view=cols_view,
            drop_unassigned=drop_unassigned,
        )
    elif set(cols_paired).issubset(features.columns.astype(str)):
        features = assign_view_paired(
            features=features,
            view_df=view_df,
            cols_paired=cols_paired,
            cols_view=cols_view,
            features_view_cols=features_view_cols_paired,
            view_name_col=view_name_col,
            drop_unassigned=drop_unassigned,
        )
        if combined_assignments_column:
            features[combined_assignments_column] = np.where(
                features[features_view_cols_paired[0]]
                == features[features_view_cols_paired[1]],
                features[features_view_cols_paired[0]],
                np.nan,
            )
    else:
        raise ValueError(
            "Could not determine whether features are paired using provided column names"
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


def align_track_with_cooler(
    track,
    clr,
    view_df=None,
    clr_weight_name="weight",
    mask_clr_bad_bins=True,
    drop_track_na=True,
):
    """
    Sync a track dataframe with a cooler bintable.

    Checks that bin sizes match between a track and a cooler,
    merges the cooler bintable with the track, and
    propagates masked regions from a cooler bintable to a track.

    Parameters
    ----------
    track : pd.DataFrame
        bedGraph-like track DataFrame to check
    clr : cooler
        cooler object to check against
    view_df : bioframe.viewframe or None
        Optional viewframe of regions to check for their number of bins with assigned track values.
        If None, constructs a view_df from cooler chromsizes.
    clr_weight_name : str
        Name of the column in the bin table with weight
    mask_clr_bad_bins : bool
        Whether to propagate null bins from cooler bintable column clr_weight_name
        to the 'value' column of the output clr_track. Default True.
    drop_track_na : bool
        Whether to ignore missing values in the track (as if they are absent).
        Important for raising errors for unassigned regions and warnings for partial assignment.
        Default True, so NaN values are treated as not assigned.
        False means that NaN values are treated as assigned.

    Returns
    -------
    clr_track
        track dataframe that has been aligned with the cooler bintable
        and has columns ['chrom','start','end','value']


    """
    from .checks import is_track, is_cooler_balanced

    try:
        is_track(track, raise_errors=True)
    except Exception as e:
        raise ValueError("invalid input track") from e

    # since tracks are currently allowed to have flexible column names
    c, s, e, v = track.columns[:4]

    # using median to allow for shorter / longer last bin on any chromosome
    track_bin_width = int((track[e] - track[s]).median())
    if not (track_bin_width == clr.binsize):
        raise ValueError(
            "mismatch between track and cooler bin size, check track resolution"
        )

    clr_track = (
        (clr.bins()[:])
        .copy()
        .merge(
            track.rename(columns={c: "chrom", s: "start", e: "end", v: "value"}),
            how="left",
            on=["chrom", "start"],
            suffixes=("", "_"),
            indicator=True,
        )
    )

    if clr_weight_name:
        try:
            is_cooler_balanced(clr, clr_weight_name=clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"no column {clr_weight_name} detected in input cooler bintable"
            ) from e
    else:
        clr_track[clr_weight_name] = 1.0

    valid_bins = clr_track[clr_weight_name].notna()
    num_valid_bins = valid_bins.sum()
    if drop_track_na:
        num_assigned_bins = (clr_track["value"][valid_bins].notna()).sum()
    else:
        num_assigned_bins = len(clr_track.query("_merge=='both'")["value"][valid_bins])
    if num_assigned_bins == 0:
        raise ValueError("no track values assigned to cooler bintable")
    elif num_assigned_bins < 0.5 * num_valid_bins:
        warnings.warn("less than 50% of valid bins have been assigned a value")

    view_df = make_cooler_view(clr) if view_df is None else view_df
    for region in view_df.itertuples(index=False):
        track_region = bioframe.select(clr_track, region)
        if drop_track_na:
            num_assigned_region_bins = track_region["value"].notna().sum()
        else:
            num_assigned_region_bins = len(track_region["value"])
        if num_assigned_region_bins == 0:
            raise ValueError(
                f"no track values assigned to region {bioframe.to_ucsc_string(region)}"
            )
    if mask_clr_bad_bins:
        clr_track.loc[~valid_bins, "value"] = np.nan

    return clr_track[["chrom", "start", "end", "value"]]
