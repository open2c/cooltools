from collections import OrderedDict
from functools import partial

import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import scipy.sparse as sps
import numpy as np
import pandas as pd
import bioframe
import cooler

from .lib.numutils import LazyToeplitz


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

    """
    features = features.copy()

    # on-diagonal features
    if "chrom" in features.columns:
        for i, region in enumerate(supports):
            if len(region) == 3:
                sel = features.chrom == region[0]
                sel &= features.end >= region[1]
                if region[2] is not None:
                    sel &= features.start < region[2]

                features.loc[sel, "region"] = i

            elif len(region) == 2:
                region1, region2 = region
                sel1 = features.chrom == region1[0]
                sel1 &= features.end >= region1[1]
                if region1[2] is not None:
                    sel1 &= features.start < region1[2]

                sel2 = features.chrom == region2[0]
                sel2 &= features.end >= region2[1]
                if region2[2] is not None:
                    sel2 &= features.start < region2[2]

                features.loc[(sel1 | sel2), "region"] = i

    # off-diagonal features
    elif "chrom1" in features.columns:
        for i, region in enumerate(supports):
            if len(region) == 3:
                region1, region2 = region, region
            elif len(region) == 2:
                region1, region2 = region[0], region[1]

            sel1 = features.chrom1 == region1[0]
            sel1 &= features.end1 >= region1[1]
            if region1[2] is not None:
                sel1 &= features.start1 < region1[2]

            sel2 = features.chrom2 == region2[0]
            sel2 &= features.end2 >= region2[1]
            if region2[2] is not None:
                sel2 &= features.start2 < region2[2]

            features.loc[(sel1 | sel2), "region"] = i
    else:
        raise ValueError("Could not parse `features` data frame.")

    features["region"] = features["region"].map(
        lambda i: "{}:{}-{}".format(*supports[int(i)]), na_action="ignore"
    )
    return features


def _pileup(data_select, data_snip, arg):
    support, feature_group = arg
    # check if support region is on- or off-diagonal
    if len(support) == 2:
        region1, region2 = map(bioframe.parse_region_string, support)
    else:
        region1 = region2 = bioframe.parse_region_string(support)

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
        raise ValueError(
            "Drop features with no region assignment before calling pileup!"
        )

    features = features.copy()
    features["_rank"] = range(len(features))

    # cumul_stack = []
    # orig_rank = []
    cumul_stack, orig_rank = zip(
        *map(
            partial(_pileup, data_select, data_snip),
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
    def __init__(self, clr, cooler_opts=None):
        self.clr = clr
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)

    def select(self, region1, region2):
        self.offsets[region1] = self.clr.offset(region1) - self.clr.offset(region1[0])
        self.offsets[region2] = self.clr.offset(region2) - self.clr.offset(region2[0])
        self._isnan1 = np.isnan(self.clr.bins()["weight"].fetch(region1).values)
        self._isnan2 = np.isnan(self.clr.bins()["weight"].fetch(region2).values)
        matrix = self.clr.matrix(**self.cooler_opts).fetch(region1, region2)
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
            snippet = matrix[lo1:hi1, lo2:hi2].toarray()
            snippet[self._isnan1[lo1:hi1], :] = np.nan
            snippet[:, self._isnan2[lo2:hi2]] = np.nan
        return snippet


class ObsExpSnipper:
    def __init__(self, clr, expected, cooler_opts=None):
        self.clr = clr
        self.expected = expected

        # Detecting the columns for the detection of regions
        columns = expected.columns
        assert len(columns) > 0
        if "chrom" in columns and "start" in columns and "end" in columns:
            self.regions_columns = [
                "chrom",
                "start",
                "end",
            ]  # Chromosome arms encoded by multiple columns
        elif "chrom" in columns:
            self.regions_columns = [
                "chrom"
            ]  # Chromosomes or regions encoded in string mode: "chr3:XXXXXXX-YYYYYYYY"
        elif "region" in columns:
            self.regions_columns = [
                "region"
            ]  # Regions encoded in string mode: "chr3:XXXXXXX-YYYYYYYY"
        elif len(columns) > 0:
            self.regions_columns = columns[
                0
            ]  # The first columns is treated as chromosome/region annotation
        else:
            raise ValueError("Expected dataframe has no columns.")

        self.binsize = self.clr.binsize
        self.offsets = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)

    def select(self, region1, region2):
        assert region1 == region2, "ObsExpSnipper is implemented for cis contacts only."
        self.offsets[region1] = self.clr.offset(region1) - self.clr.offset(region1[0])
        self.offsets[region2] = self.clr.offset(region2) - self.clr.offset(region2[0])
        matrix = self.clr.matrix(**self.cooler_opts).fetch(region1, region2)
        if self.cooler_opts["sparse"]:
            matrix = matrix.tocsr()
        self._isnan1 = np.isnan(self.clr.bins()["weight"].fetch(region1).values)
        self._isnan2 = np.isnan(self.clr.bins()["weight"].fetch(region2).values)
        self._expected = LazyToeplitz(
            self.expected.groupby(self.regions_columns)
            .get_group(region1[0] if len(self.regions_columns) > 0 else region1)[
                "balanced.avg"
            ]
            .values
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
    def __init__(self, clr, expected):
        self.clr = clr
        self.expected = expected

        # Detecting the columns for the detection of regions
        columns = expected.columns
        assert len(columns) > 0
        if "chrom" in columns and "start" in columns and "end" in columns:
            self.regions_columns = [
                "chrom",
                "start",
                "end",
            ]  # Chromosome arms encoded by multiple columns
        elif "chrom" in columns:
            self.regions_columns = [
                "chrom"
            ]  # Chromosomes or regions encoded in string mode: "chr3:XXXXXXX-YYYYYYYY"
        elif "region" in columns:
            self.regions_columns = [
                "region"
            ]  # Regions encoded in string mode: "chr3:XXXXXXX-YYYYYYYY"
        elif len(columns) > 0:
            self.regions_columns = columns[
                0
            ]  # The first columns is treated as chromosome/region annotation
        else:
            raise ValueError("Expected dataframe has no columns.")

        try:
            for region, group in self.expected.groupby(self.regions_columns):
                assert group.shape[0]==np.diff(self.clr.extent(region))[0]
        except AssertionError:
            raise ValueError("Region shape mismatch between expected and cooler. "
                             "Are they using the same resolution?")

        self.binsize = self.clr.binsize
        self.offsets = {}

    def select(self, region1, region2):
        assert (
            region1 == region2
        ), "ExpectedSnipper is implemented for cis contacts only."
        self.offsets[region1] = self.clr.offset(region1) - self.clr.offset(region1[0])
        self.offsets[region2] = self.clr.offset(region2) - self.clr.offset(region2[0])
        self.m = np.diff(self.clr.extent(region1))
        self.n = np.diff(self.clr.extent(region2))
        self._expected = LazyToeplitz(
            self.expected.groupby(self.regions_columns)
            .get_group(region1[0] if len(self.regions_columns) > 0 else region1)[
                "balanced.avg"
            ]
            .values
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
