from functools import partial
import warnings

import numpy as np
import pandas as pd
import bioframe

from .lib.numutils import LazyToeplitz
import warnings


def make_bin_aligned_windows(
    binsize, chroms, centers_bp, flank_bp=0, region_start_bp=0, ignore_index=False
):
    """
    Convert genomic loci into bin spans on a fixed bin-size segmentation of a
    genomic region. Window limits are adjusted to align with bin edges.

    Parameters
    -----------
    binsize : int
        Bin size (resolution) in base pairs.
    chroms : 1D array-like
        Column of chromosome names.
    centers_bp : 1D or nx2 array-like
        If 1D, center points of each window. If 2D, the starts and ends.
    flank_bp : int
        Distance in base pairs to extend windows on either side.
    region_start_bp : int, optional
        If region is a subset of a chromosome, shift coordinates by this amount.
        Default is 0.

    Returns
    -------
    DataFrame with columns:
        'chrom'        - chromosome
        'start', 'end' - window limits in base pairs
        'lo', 'hi'     - window limits in bins

    """
    if not (flank_bp % binsize == 0):
        raise ValueError("Flanking distance must be divisible by the bin size.")

    if isinstance(chroms, pd.Series) and not ignore_index:
        index = chroms.index
    else:
        index = None

    chroms = np.asarray(chroms)
    centers_bp = np.asarray(centers_bp)
    if len(centers_bp.shape) == 2:
        left_bp = centers_bp[:, 0]
        right_bp = centers_bp[:, 1]
    else:
        left_bp = right_bp = centers_bp

    if np.any(left_bp > right_bp):
        raise ValueError("Found interval with end > start.")

    left = left_bp - region_start_bp
    right = right_bp - region_start_bp
    left_bin = (left / binsize).astype(int)
    right_bin = (right / binsize).astype(int)
    flank_bin = flank_bp // binsize

    lo = left_bin - flank_bin
    hi = right_bin + flank_bin + 1
    windows = pd.DataFrame(index=index)
    windows["chrom"] = chroms
    windows["start"] = lo * binsize
    windows["end"] = hi * binsize
    windows["lo"] = lo
    windows["hi"] = hi
    return windows


def assign_regions(features, supports):
    """
    For each feature in features dataframe assign the genomic region (support)
    that overlaps with it. In case if feature overlaps multiple supports, the 
    region with largest overlap will be reported.
    """

    index_name = features.index.name  # Store the name of index
    features = (
        features.copy().reset_index()
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

    features = features.set_index(
        index_name if not index_name is None else "index"
    )  # Restore the original index
    features.index.name = index_name  # Restore original index title
    return features


def _pileup(data_select, data_snip, arg):
    support, feature_group = arg

    # return empty snippets if region is unannotated:
    if len(support) == 0:
        if "start" in feature_group:  # on-diagonal off-region case:
            lo = feature_group["lo"].values
            hi = feature_group["hi"].values
            s = hi - lo  # Shape of individual snips
            assert (
                s.max() == s.min()
            ), "Pileup accepts only the windows of the same size"
            stack = np.full((s[0], s[0], len(feature_group)), np.nan)
        else:  # off-diagonal off-region case:
            lo1 = feature_group["lo1"].values
            hi1 = feature_group["hi1"].values
            lo2 = feature_group["lo2"].values
            hi2 = feature_group["hi2"].values
            s1 = hi1 - lo1  # Shape of individual snips
            s2 = hi1 - lo1
            assert (
                s1.max() == s1.min()
            ), "Pileup accepts only the windows of the same size"
            assert (
                s2.max() == s2.min()
            ), "Pileup accepts only the windows of the same size"
            stack = np.full((s1[0], s2[0], len(feature_group)), np.nan)

        return stack, feature_group["_rank"].values

    # check if support region is on- or off-diagonal
    if len(support) == 2:
        region1, region2 = support
    else:
        region1 = region2 = support

    # check if features are on- or off-diagonal
    if "start" in feature_group:
        s1 = feature_group["start"].values
        e1 = feature_group["end"].values
        s2, e2 = s1, e1
    else:
        s1 = feature_group["start1"].values
        e1 = feature_group["end1"].values
        s2 = feature_group["start2"].values
        e2 = feature_group["end2"].values

    data = data_select(region1, region2)
    stack = list(map(partial(data_snip, data, region1, region2), zip(s1, e1, s2, e2)))

    return np.dstack(stack), feature_group["_rank"].values


def pileup(features, data_select, data_snip, map=map):
    """
    Handles on-diagonal and off-diagonal cases.

    Parameters
    ----------
    features : DataFrame
        Table of features. Requires columns ['chrom', 'start', 'end'].
        Or ['chrom1', 'start1', 'end1', 'chrom1', 'start2', 'end2'].
        start, end are bp coordinates.
        lo, hi are bin coordinates.

    data_select : callable
        Callable that takes a region as argument and returns
        the data, mask and bin offset of a support region

    data_snip : callable
        Callable that takes data, mask and a 2D bin span (lo1, hi1, lo2, hi2)
        and returns a snippet from the selected support region


    """
    if features.region.isnull().any():
        warnings.warn(
            "Some features do not have regions assigned! Some snips will be empty."
        )

    features = features.copy()
    features["region"] = features.region.fillna(
        ""
    )  # fill in unanotated regions with empty string
    features["_rank"] = range(len(features))

    # cumul_stack = []
    # orig_rank = []
    cumul_stack, orig_rank = zip(
        *map(
            partial(_pileup, data_select, data_snip),
            # Note that unannotated regions will form a separate group
            features.groupby("region", sort=False),
        )
    )

    # Restore the original rank of the input features
    cumul_stack = np.dstack(cumul_stack)
    orig_rank = np.concatenate(orig_rank)

    idx = np.argsort(orig_rank)
    cumul_stack = cumul_stack[:, :, idx]
    return cumul_stack


def pair_sites(sites, separation, slop):
    """
    Create "hand" intervals to the right and to the left of each site.
    Then join right hands with left hands to pair sites together.

    """
    from bioframe.tools import tsv, bedtools

    mids = (sites["start"] + sites["end"]) // 2
    left_hand = sites[["chrom"]].copy()
    left_hand["start"] = mids - separation - slop
    left_hand["end"] = mids - separation + slop
    left_hand["site_id"] = left_hand.index
    left_hand["direction"] = "L"
    left_hand["snip_mid"] = mids
    left_hand["snip_strand"] = sites["strand"]

    right_hand = sites[["chrom"]].copy()
    right_hand["start"] = mids + separation - slop
    right_hand["end"] = mids + separation + slop
    right_hand["site_id"] = right_hand.index
    right_hand["direction"] = "R"
    right_hand["snip_mid"] = mids
    right_hand["snip_strand"] = sites["strand"]

    # ignore out-of-bounds hands
    mask = (left_hand["start"] > 0) & (right_hand["start"] > 0)
    left_hand = left_hand[mask].copy()
    right_hand = right_hand[mask].copy()

    # intersect right hands (left anchor site)
    # with left hands (right anchor site)
    with tsv(right_hand) as R, tsv(left_hand) as L:
        out = bedtools.intersect(a=R.name, b=L.name, wa=True, wb=True)
        out.columns = [c + "_r" for c in right_hand.columns] + [
            c + "_l" for c in left_hand.columns
        ]
    return out


class CoolerSnipper:
    def __init__(self, clr, cooler_opts=None, regions=None):

        if regions is None:
            regions = pd.DataFrame(
                [(chrom, 0, l, chrom) for chrom, l in clr.chromsizes.items()],
                columns=["chrom", "start", "end", "name"],
            )
        self.regions = regions.set_index("name")

        self.clr = clr
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)

    def select(self, region1, region2):
        region1_coords = self.regions.loc[region1]
        region2_coords = self.regions.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        matrix = self.clr.matrix(**self.cooler_opts).fetch(
            region1_coords, region2_coords
        )
        self._isnan1 = np.isnan(self.clr.bins()["weight"].fetch(region1_coords).values)
        self._isnan2 = np.isnan(self.clr.bins()["weight"].fetch(region2_coords).values)
        if self.cooler_opts["sparse"]:
            matrix = matrix.tocsr()
        return matrix

    def snip(self, matrix, region1, region2, tup):
        s1, e1, s2, e2 = tup
        offset1 = self.offsets[region1]
        offset2 = self.offsets[region2]
        binsize = self.binsize
        lo1, hi1 = (s1 // binsize) - offset1, (e1 // binsize) - offset1
        lo2, hi2 = (s2 // binsize) - offset2, (e2 // binsize) - offset2
        assert hi1 >= 0
        assert hi2 >= 0

        m, n = matrix.shape
        dm, dn = hi1 - lo1, hi2 - lo2
        out_of_bounds = False
        pad_left = pad_right = pad_bottom = pad_top = None
        if lo1 < 0:
            pad_bottom = -lo1
            out_of_bounds = True
        if lo2 < 0:
            pad_left = -lo2
            out_of_bounds = True
        if hi1 > m:
            pad_top = dm - (hi1 - m)
            out_of_bounds = True
        if hi2 > n:
            pad_right = dn - (hi2 - n)
            out_of_bounds = True

        if out_of_bounds:
            i0 = max(lo1, 0)
            i1 = min(hi1, m)
            j0 = max(lo2, 0)
            j1 = min(hi2, n)
            snippet = np.full((dm, dn), np.nan)
        #             snippet[pad_bottom:pad_top,
        #                     pad_left:pad_right] = matrix[i0:i1, j0:j1].toarray()
        else:
            snippet = matrix[lo1:hi1, lo2:hi2].toarray().astype("float")
            snippet[self._isnan1[lo1:hi1], :] = np.nan
            snippet[:, self._isnan2[lo2:hi2]] = np.nan
        return snippet


class ObsExpSnipper:
    def __init__(self, clr, expected, cooler_opts=None, regions=None):
        self.clr = clr
        self.expected = expected

        # Detecting the columns for the detection of regions
        columns = expected.columns
        assert len(columns) > 0
        if "region" not in columns:
            if "chrom" in columns:
                self.expected = self.expected.rename(columns={"chrom": "region"})
                warnings.warn(
                    "The expected dataframe appears to be in the old format."
                    "It should have a `region` column, not `chrom`."
                )
            else:
                raise ValueError(
                    "Please check the expected dataframe, it has no `region` column"
                )
        if regions is None:
            if set(self.expected["region"]).issubset(clr.chromnames):
                regions = pd.DataFrame(
                    [(chrom, 0, l, chrom) for chrom, l in clr.chromsizes.items()],
                    columns=["chrom", "start", "end", "name"],
                )
            else:
                raise ValueError(
                    "Please provide the regions table, if region names"
                    "are not simply chromosome names."
                )
        self.regions = regions.set_index("name")

        try:
            for region_name, group in self.expected.groupby("region"):
                n_diags = group.shape[0]
                region = self.regions.loc[region_name]
                lo, hi = self.clr.extent(region)
                assert n_diags == (hi - lo)
        except AssertionError:
            raise ValueError(
                "Region shape mismatch between expected and cooler. "
                "Are they using the same resolution?"
            )

        self.binsize = self.clr.binsize
        self.offsets = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)

    def select(self, region1, region2):
        assert region1 == region2, "ObsExpSnipper is implemented for cis contacts only."
        region1_coords = self.regions.loc[region1]
        region2_coords = self.regions.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        matrix = self.clr.matrix(**self.cooler_opts).fetch(
            region1_coords, region2_coords
        )
        if self.cooler_opts["sparse"]:
            matrix = matrix.tocsr()
        self._isnan1 = np.isnan(self.clr.bins()["weight"].fetch(region1_coords).values)
        self._isnan2 = np.isnan(self.clr.bins()["weight"].fetch(region2_coords).values)
        self._expected = LazyToeplitz(
            self.expected.groupby("region").get_group(region1)["balanced.avg"].values
        )
        return matrix

    def snip(self, matrix, region1, region2, tup):
        s1, e1, s2, e2 = tup
        offset1 = self.offsets[region1]
        offset2 = self.offsets[region2]
        binsize = self.binsize
        lo1, hi1 = (s1 // binsize) - offset1, (e1 // binsize) - offset1
        lo2, hi2 = (s2 // binsize) - offset2, (e2 // binsize) - offset2
        assert hi1 >= 0
        assert hi2 >= 0

        m, n = matrix.shape
        dm, dn = hi1 - lo1, hi2 - lo2
        out_of_bounds = False
        pad_left = pad_right = pad_bottom = pad_top = None
        if lo1 < 0:
            pad_bottom = -lo1
            out_of_bounds = True
        if lo2 < 0:
            pad_left = -lo2
            out_of_bounds = True
        if hi1 > m:
            pad_top = dm - (hi1 - m)
            out_of_bounds = True
        if hi2 > n:
            pad_right = dn - (hi2 - n)
            out_of_bounds = True

        if out_of_bounds:
            i0 = max(lo1, 0)
            i1 = min(hi1, m)
            j0 = max(lo2, 0)
            j1 = min(hi2, n)
            return np.full((dm, dn), np.nan)
        #             snippet[pad_bottom:pad_top,
        #                     pad_left:pad_right] = matrix[i0:i1, j0:j1].toarray()
        else:
            snippet = matrix[lo1:hi1, lo2:hi2].toarray()
            snippet[self._isnan1[lo1:hi1], :] = np.nan
            snippet[:, self._isnan2[lo2:hi2]] = np.nan

        e = self._expected[lo1:hi1, lo2:hi2]
        return snippet / e


class ExpectedSnipper:
    def __init__(self, clr, expected, regions=None):
        self.clr = clr
        self.expected = expected
        # Detecting the columns for the detection of regions
        columns = expected.columns
        assert len(columns) > 0
        if "region" not in columns:
            if "chrom" in columns:
                self.expected = self.expected.rename(columns={"chrom": "region"})
                warnings.warn(
                    "The expected dataframe appears to be in the old format."
                    "It should have a `region` column, not `chrom`."
                )
            else:
                raise ValueError(
                    "Please check the expected dataframe, it has no `region` column"
                )
        if regions is None:
            if set(self.expected["region"]).issubset(clr.chromnames):
                regions = pd.DataFrame(
                    [(chrom, 0, l, chrom) for chrom, l in clr.chromsizes.items()],
                    columns=["chrom", "start", "end", "name"],
                )
            else:
                raise ValueError(
                    "Please provide the regions table, if region names"
                    "are not simply chromosome names."
                )
        self.regions = regions.set_index("name")

        try:
            for region_name, group in self.expected.groupby("region"):
                n_diags = group.shape[0]
                region = self.regions.loc[region_name]
                lo, hi = self.clr.extent(region)
                assert n_diags == (hi - lo)
        except AssertionError:
            raise ValueError(
                "Region shape mismatch between expected and cooler. "
                "Are they using the same resolution?"
            )

        self.binsize = self.clr.binsize
        self.offsets = {}

    def select(self, region1, region2):
        assert (
            region1 == region2
        ), "ExpectedSnipper is implemented for cis contacts only."
        region1_coords = self.regions.loc[region1]
        region2_coords = self.regions.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        self.m = np.diff(self.clr.extent(region1_coords))
        self.n = np.diff(self.clr.extent(region2_coords))
        self._expected = LazyToeplitz(
            self.expected.groupby("region").get_group(region1)["balanced.avg"].values
        )
        return self._expected

    def snip(self, exp, region1, region2, tup):
        s1, e1, s2, e2 = tup
        offset1 = self.offsets[region1]
        offset2 = self.offsets[region2]
        binsize = self.binsize
        lo1, hi1 = (s1 // binsize) - offset1, (e1 // binsize) - offset1
        lo2, hi2 = (s2 // binsize) - offset2, (e2 // binsize) - offset2
        assert hi1 >= 0
        assert hi2 >= 0
        dm, dn = hi1 - lo1, hi2 - lo2

        if lo1 < 0 or lo2 < 0 or hi1 > self.m or hi2 > self.n:
            return np.full((dm, dn), np.nan)

        snippet = exp[lo1:hi1, lo2:hi2]
        return snippet
