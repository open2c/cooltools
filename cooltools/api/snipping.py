"""
Collection of classes and functions used for snipping and creation of pileups
(averaging of multiple small 2D regions)
The main user-facing function of this module is `pileup`, it performs pileups using
snippers and other functions defined in the module.  The concept is the following:

- First, the provided features are annotated with the regions from a view (or simply  
  whole chromosomes, if no view is provided). They are assigned to the region that
  contains it, or the one with the largest overlap.
- Then the features are expanded using the `flank` argument, and aligned to the bins  
  of the cooler
- Depending on the requested operation (whether the normalization to expected is  
  required), the appropriate snipper object is created
- A snipper can `select` a particular region of a genome-wide matrix, meaning it  
  stores its sparse representation in memory. This could be whole chromosomes or
  chromosome arms, for example
- A snipper can `snip` a small area of a selected region, meaning it will extract  
  and return a dense representation of this area
- For each region present, it is first `select`ed, and then all features within it are  
  `snip`ped, creating a stack: a 3D array containing all snippets for this region
- For features that are not assigned to any region, an empty snippet is returned  
- All per-region stacks are then combined into one, which then can be averaged to create  
  a single pileup
- The order of snippets in the stack matches the order of features, this way the stack  
  can also be used for analysis of any subsets of original features

This procedure achieves a good tradeoff between speed and RAM. Extracting each
individual snippet directly from disk would be extremely slow due to slow IO.
Extracting the whole chromosomes into dense matrices is not an option due to huge
memory requirements. As a warning, deeply sequenced data can still require a
substantial amount of RAM at high resolution even as a sparse matrix, but typically
it's not a problem.
"""
from functools import partial
import warnings

import numpy as np
import pandas as pd
import bioframe

from ..lib.checks import (
    is_compatible_viewframe,
    is_cooler_balanced,
    is_valid_expected,
)
from ..lib.common import assign_view_auto, make_cooler_view

from ..lib.numutils import LazyToeplitz
import warnings

import multiprocessing


def expand_align_features(features_df, flank, resolution, format="bed"):
    """Short summary.

    Parameters
    ----------
    features_df : pd.DataFrame
        Dataframe with feature coordinates.
    flank : int
        Flank size to add to the central bin of each feature.
    resolution : int
        Size of the bins to use.
    format : str
        "bed" or "bedpe" format: has to have 'chrom', 'start', 'end'
        or 'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end1' columns, repectively.

    Returns
    -------
    pd.DataFrame
        DataFrame with features with new columns
           "center",  "orig_start"   "orig_end"
        or "center1", "orig_start1", "orig_end1",
           "center2", "orig_start2", "orig_rank_end2", depending on format.

    """
    features_df = features_df.copy()
    if format == "bed":
        features_df[["orig_start", "orig_end"]] = features_df[["start", "end"]]
        features_df["center"] = (features_df["start"] + features_df["end"]) / 2
        features_df["lo"] = (
            np.floor(features_df["center"] / resolution) - flank // resolution
        ).astype(int)
        features_df["hi"] = (
            np.floor(features_df["center"] / resolution) + flank // resolution + 1
        ).astype(int)
        features_df["start"] = features_df["lo"] * resolution
        features_df["end"] = features_df["hi"] * resolution
    elif format == "bedpe":
        features_df[
            ["orig_start1", "orig_end1", "orig_start2", "orig_end2"]
        ] = features_df[["start1", "end1", "start2", "end2"]]
        features_df["center1"] = (features_df["start1"] + features_df["end1"]) / 2
        features_df["center2"] = (features_df["start2"] + features_df["end2"]) / 2

        features_df["lo1"] = (
            np.floor(features_df["center1"] / resolution) - flank // resolution
        ).astype(int)
        features_df["hi1"] = (
            np.floor(features_df["center1"] / resolution) + flank // resolution + 1
        ).astype(int)
        features_df["start1"] = features_df["lo1"] * resolution
        features_df["end1"] = features_df["hi1"] * resolution

        features_df["lo2"] = (
            np.floor(features_df["center2"] / resolution) - flank // resolution
        ).astype(int)
        features_df["hi2"] = (
            np.floor(features_df["center2"] / resolution) + flank // resolution + 1
        ).astype(int)
        features_df["start2"] = features_df["lo2"] * resolution
        features_df["end2"] = features_df["hi2"] * resolution
    return features_df


def make_bin_aligned_windows(
    binsize,
    chroms,
    centers_bp,
    flank_bp=0,
    region_start_bp=0,
    ignore_index=False,
):
    """
    Convert genomic loci into bin spans on a fixed bin-segmentation of a
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
        raise ValueError("Flanking distance must be divisible by binsize.")

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
    windows["lo"] = lo.astype(int)
    windows["hi"] = hi.astype(int)
    return windows


def _extract_stack(data_select, data_snip, arg):
    support, feature_group = arg
    # return empty snippets if region is unannotated:
    if len(support) == 0:
        if "start" in feature_group:  # on-diagonal off-region case:
            lo = feature_group["lo"].values
            hi = feature_group["hi"].values
            s = (hi - lo).astype(int)  # Shape of individual snips
            assert s.max() == s.min(), "Pileup accepts only windows of the same size"
            stack = np.full((s[0], s[0], len(feature_group)), np.nan)
        else:  # off-diagonal off-region case:
            lo1 = feature_group["lo1"].values
            hi1 = feature_group["hi1"].values
            lo2 = feature_group["lo2"].values
            hi2 = feature_group["hi2"].values
            s1 = (hi1 - lo1).astype(int)  # Shape of individual snips
            s2 = (hi2 - lo2).astype(int)
            assert s1.max() == s1.min(), "Pileup accepts only windows of the same size"
            assert s2.max() == s2.min(), "Pileup accepts only windows of the same size"
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


def _pileup(features, data_select, data_snip, map=map):
    """
    Creates a stackup of snippets (a 3D array) by selecting each region present in the
    `features` (using the `data_select` function) and then extracting all snippets from
    the region (using `data_snip`).
    Handles on-diagonal and off-diagonal cases.

    Internal, so assumes correctly formatted input created by `pileup`.

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

    map : callable
        Callable that works like builtin `map`.

    """
    if features["region"].isnull().any():
        warnings.warn(
            "Some features do not have view regions assigned! Some snips will be empty."
        )

    features = features.copy()
    features["region"] = features["region"].fillna(
        ""
    )  # fill in unanotated view regions with empty string
    features["_rank"] = range(len(features))

    # cumul_stack = []
    # orig_rank = []
    cumul_stack, orig_rank = zip(
        *map(
            partial(_extract_stack, data_select, data_snip),
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


class CoolerSnipper:
    def __init__(self, clr, cooler_opts=None, view_df=None, min_diag=2):
        """Class for generating snips with "observed" data from a cooler

        Parameters
        ----------
        clr : cooler.Cooler
            Cooler object with data to use
        cooler_opts : dict, optional
            Options to pass to the clr.matrix() method, by default None
            Can be used to choose the cooler weight name, e.g.
            cooler_opts={balance='non-standard-weight'}, or use unbalanced data with
            cooler_opts={balance=False}
        view_df : pd.DataFrame, optional
            Genomic view to constrain the analysis, by default None and uses all
            chromosomes present in the cooler
        min_diag : int, optional
            This number of short-distance diagonals is ignored, by default 2
        """

        # get chromosomes from cooler, if view_df not specified:
        if view_df is None:
            view_df = make_cooler_view(clr)
        else:
            # Make sure view_df is a proper viewframe
            try:
                _ = is_compatible_viewframe(
                    view_df,
                    clr,
                    check_sorting=True,
                    raise_errors=True,
                )
            except Exception as e:
                raise ValueError(
                    "view_df is not a valid viewframe or incompatible"
                ) from e

        self.view_df = view_df.set_index("name")
        self.clr = clr
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.diag_indicators = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)

        if "balance" in self.cooler_opts:
            if self.cooler_opts["balance"] is True:
                self.clr_weight_name = "weight"
            elif (
                self.cooler_opts["balance"] is False
                or self.cooler_opts["balance"] is None
            ):
                self.clr_weight_name = None
            else:
                self.clr_weight_name = self.cooler_opts["balance"]
        else:
            self.clr_weight_name = "weight"
        self.min_diag = min_diag

    def select(self, region1, region2):
        """Select a portion of the cooler for snipping based on two regions in the view

        In addition to returning the selected portion of the data, stores necessary
        information about it in the snipper object for future snipping

        Parameters
        ----------
        region1 : str
            Name of a region from the view
        region2 : str
            Name of another region from the view.

        Returns
        -------
        CSR matrix
            Sparse matrix of the selected portion of the data from the cooler
        """
        region1_coords = self.view_df.loc[region1]
        region2_coords = self.view_df.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        matrix = self.clr.matrix(**self.cooler_opts).fetch(
            region1_coords, region2_coords
        )
        if self.clr_weight_name:
            self._isnan1 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region1_coords).values
            )
            self._isnan2 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region2_coords).values
            )
        else:
            self._isnan1 = np.zeros_like(
                self.clr.bins()["start"].fetch(region1_coords).values
            ).astype(bool)
            self._isnan2 = np.zeros_like(
                self.clr.bins()["start"].fetch(region2_coords).values
            ).astype(bool)
        if self.cooler_opts["sparse"]:
            matrix = matrix.tocsr()
        if self.min_diag is not None:
            diags = np.arange(np.diff(self.clr.extent(region1_coords)), dtype=np.int32)
            self.diag_indicators[region1] = LazyToeplitz(-diags, diags)
        return matrix

    def snip(self, matrix, region1, region2, tup):
        """Extract a snippet from the matrix

        Returns a NaN-filled array for out-of-bounds regions. Fills in NaNs based on the
        cooler weight, if using balanced data. Fills NaNs in all diagonals below min_diag

        Parameters
        ----------
        matrix : SCR matrix
            Output of the .select() method
        region1 : str
            Name of a region from the view corresponding to the matrix
        region2 : str
            Name of the other regions from the view corresponding to the matrix
        tup : tuple
            (start1, end1, start2, end2) coordinates of the requested snippet in bp

        Returns
        -------
        np.array
            Requested snippet.
        """
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
        if self.min_diag is not None:
            D = self.diag_indicators[region1][lo1:hi1, lo2:hi2] < self.min_diag
            snippet[D] = np.nan
        return snippet


class ObsExpSnipper:
    def __init__(
        self,
        clr,
        expected,
        cooler_opts=None,
        view_df=None,
        min_diag=2,
        expected_value_col="balanced.avg",
    ):
        """Class for generating expected-normalised snips from a cooler

        Parameters
        ----------
        clr : cooler.Cooler
            Cooler object with data to use
        expected : pd.DataFrame
            Dataframe containing expected interactions in the cooler
        cooler_opts : dict, optional
            Options to pass to the clr.matrix() method, by default None
            Can be used to choose the cooler weight name, e.g.
            cooler_opts={balance='non-standard-weight'}, or use unbalanced data with
            cooler_opts={balance=False}
        view_df : pd.DataFrame, optional
            Genomic view to constrain the analysis, by default None and uses all
            chromosomes present in the cooler
        min_diag : int, optional
            This number of short-distance diagonals is ignored, by default 2
        expected_value_col : str, optional
            Name of the column in the expected dataframe that contains the expected
            interaction values, by default "balanced.avg"
        """
        self.clr = clr
        self.expected = expected
        self.expected_value_col = expected_value_col
        # get chromosomes from cooler, if view_df not specified:
        if view_df is None:
            view_df = make_cooler_view(clr)
        else:
            # Make sure view_df is a proper viewframe
            try:
                _ = is_compatible_viewframe(
                    view_df,
                    clr,
                    check_sorting=True,
                    raise_errors=True,
                )
            except Exception as e:
                raise ValueError(
                    "view_df is not a valid viewframe or incompatible"
                ) from e
        # make sure expected is compatible
        try:
            _ = is_valid_expected(
                expected,
                "cis",
                view_df,
                verify_cooler=clr,
                expected_value_cols=[
                    self.expected_value_col,
                ],
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("provided expected is not valid") from e

        self.view_df = view_df.set_index("name")
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.diag_indicators = {}
        self.pad = True
        self.cooler_opts = {} if cooler_opts is None else cooler_opts
        self.cooler_opts.setdefault("sparse", True)
        if "balance" in self.cooler_opts:
            if self.cooler_opts["balance"] is True:
                self.clr_weight_name = "weight"
            elif (
                self.cooler_opts["balance"] is False
                or self.cooler_opts["balance"] is None
            ):
                self.clr_weight_name = None
            else:
                self.clr_weight_name = self.cooler_opts["balance"]
        else:
            self.clr_weight_name = "weight"
        self.min_diag = min_diag

    def select(self, region1, region2):
        """Select a portion of the cooler for snipping based on two regions in the view

        In addition to returning the selected portion of the data, stores necessary
        information about it in the snipper object for future snipping

        Parameters
        ----------
        region1 : str
            Name of a region from the view
        region2 : str
            Name of another region from the view.

        Returns
        -------
        CSR matrix
            Sparse matrix of the selected portion of the data from the cooler
        """
        if not region1 == region2:
            raise ValueError("ObsExpSnipper is implemented for cis contacts only.")
        region1_coords = self.view_df.loc[region1]
        region2_coords = self.view_df.loc[region2]
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
        if self.clr_weight_name:
            self._isnan1 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region1_coords).values
            )
            self._isnan2 = np.isnan(
                self.clr.bins()[self.clr_weight_name].fetch(region2_coords).values
            )
        else:
            self._isnan1 = np.zeros_like(
                self.clr.bins()["start"].fetch(region1_coords).values
            ).astype(bool)
            self._isnan2 = np.zeros_like(
                self.clr.bins()["start"].fetch(region2_coords).values
            ).astype(bool)
        self._expected = LazyToeplitz(
            self.expected.groupby(["region1", "region2"])
            .get_group((region1, region2))[self.expected_value_col]
            .values
        )
        if self.min_diag is not None:
            diags = np.arange(np.diff(self.clr.extent(region1_coords)), dtype=np.int32)
            self.diag_indicators[region1] = LazyToeplitz(-diags, diags)
        return matrix

    def snip(self, matrix, region1, region2, tup):
        """Extract an expected-normalised snippet from the matrix

        Returns a NaN-filled array for out-of-bounds regions. Fills in NaNs based on the
        cooler weight, if using balanced data. Fills NaNs in all diagonals below min_diag

        Parameters
        ----------
        matrix : SCR matrix
            Output of the .select() method
        region1 : str
            Name of a region from the view corresponding to the matrix
        region2 : str
            Name of the other regions from the view corresponding to the matrix
        tup : tuple
            (start1, end1, start2, end2) coordinates of the requested snippet in bp

        Returns
        -------
        np.array
            Requested snippet.
        """
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
            snippet = matrix[lo1:hi1, lo2:hi2].toarray().astype("float")
            snippet[self._isnan1[lo1:hi1], :] = np.nan
            snippet[:, self._isnan2[lo2:hi2]] = np.nan

        e = self._expected[lo1:hi1, lo2:hi2]
        if self.min_diag is not None:
            D = self.diag_indicators[region1][lo1:hi1, lo2:hi2] < self.min_diag
            snippet[D] = np.nan
        return snippet / e


class ExpectedSnipper:
    def __init__(
        self, clr, expected, view_df=None, min_diag=2, expected_value_col="balanced.avg"
    ):
        """Class for generating expected snips

        Parameters
        ----------
        clr : cooler.Cooler
            Cooler object to which the data corresponds
        expected : pd.DataFrame
            Dataframe containing expected interactions in the cooler
        view_df : pd.DataFrame, optional
            Genomic view to constrain the analysis, by default None and uses all
            chromosomes present in the cooler
        min_diag : int, optional
            This number of short-distance diagonals is ignored, by default 2
        expected_value_col : str, optional
            Name of the column in the expected dataframe that contains the expected
            interaction values, by default "balanced.avg"
        """
        self.clr = clr
        self.expected = expected
        self.expected_value_col = expected_value_col
        # get chromosomes from cooler, if view_df not specified:
        if view_df is None:
            view_df = make_cooler_view(clr)
        else:
            # Make sure view_df is a proper viewframe
            try:
                _ = is_compatible_viewframe(
                    view_df,
                    clr,
                    check_sorting=True,
                    raise_errors=True,
                )
            except Exception as e:
                raise ValueError(
                    "view_df is not a valid viewframe or incompatible"
                ) from e
        # make sure expected is compatible
        try:
            _ = is_valid_expected(
                expected,
                "cis",
                view_df,
                verify_cooler=clr,
                expected_value_cols=[
                    self.expected_value_col,
                ],
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("provided expected is not valid") from e

        self.view_df = view_df.set_index("name")
        self.binsize = self.clr.binsize
        self.offsets = {}
        self.diag_indicators = {}
        self.min_diag = min_diag

    def select(self, region1, region2):
        """Select a portion of the expected matrix for snipping based on two regions
        in the view

        In addition to returning the selected portion of the data, stores necessary
        information about it in the snipper object for future snipping

        Parameters
        ----------
        region1 : str
            Name of a region from the view
        region2 : str
            Name of another region from the view.

        Returns
        -------
        CSR matrix
            Sparse matrix of the selected portion of the data from the cooler
        """
        if not region1 == region2:
            raise ValueError("ExpectedSnipper is implemented for cis contacts only.")
        region1_coords = self.view_df.loc[region1]
        region2_coords = self.view_df.loc[region2]
        self.offsets[region1] = self.clr.offset(region1_coords) - self.clr.offset(
            region1_coords[0]
        )
        self.offsets[region2] = self.clr.offset(region2_coords) - self.clr.offset(
            region2_coords[0]
        )
        self.m = np.diff(self.clr.extent(region1_coords))
        self.n = np.diff(self.clr.extent(region2_coords))
        self._expected = LazyToeplitz(
            self.expected.groupby(["region1", "region2"])
            .get_group((region1, region2))[self.expected_value_col]
            .values
        )
        if self.min_diag is not None:
            diags = np.arange(np.diff(self.clr.extent(region1_coords)), dtype=np.int32)
            self.diag_indicators[region1] = LazyToeplitz(-diags, diags)
        return self._expected

    def snip(self, exp, region1, region2, tup):
        """Extract an expected snippet

        Returns a NaN-filled array for out-of-bounds regions.
        Fills NaNs in all diagonals below min_diag

        Parameters
        ----------
        exp : SCR matrix
            Output of the .select() method
        region1 : str
            Name of a region from the view corresponding to the matrix
        region2 : str
            Name of the other regions from the view corresponding to the matrix
        tup : tuple
            (start1, end1, start2, end2) coordinates of the requested snippet in bp

        Returns
        -------
        np.array
            Requested snippet.
        """
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
        if self.min_diag is not None:
            D = self.diag_indicators[region1][lo1:hi1, lo2:hi2] < self.min_diag
            snippet[D] = np.nan
        return snippet


def pileup(
    clr,
    features_df,
    view_df=None,
    expected_df=None,
    expected_value_col="balanced.avg",
    flank=100_000,
    min_diag="auto",
    clr_weight_name="weight",
    nproc=1,
):
    """
    Pileup features over the cooler.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler with Hi-C data
    features_df : pd.DataFrame
        Dataframe in bed or bedpe format: has to have 'chrom', 'start', 'end'
        or 'chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2' columns.
    view_df : pd.DataFrame
        Dataframe with the genomic view for this operation (has to match the
        expected_df, if provided)
    expected_df : pd.DataFrame
        Dataframe with the expected level of interactions at different
        genomic separations
    expected_value_col : str
        Name of the column in expected used for normalizing.
    flank : int
        How much to flank the center of the features by, in bp
    min_diag: str or int
        All diagonals of the matrix below this value are ignored. 'auto'
        tries to extract the value used during the matrix balancing,
        if it fails defaults to 2
    clr_weight_name : str
        Value of the column that contains the balancing weights
    force : bool
        Allows start>end in the features (not implemented)
    nproc : str
        How many cores to use

    Returns
    -------
        np.ndarray: a stackup of all snippets corresponding to the features

    """

    if {"chrom", "start", "end"}.issubset(features_df.columns):
        feature_type = "bed"
    elif {"chrom1", "start1", "end1", "chrom2", "start2", "end1"}.issubset(
        features_df.columns
    ):
        feature_type = "bedpe"
    else:
        raise ValueError("Unknown feature_df format")

    features_df = assign_view_auto(features_df, view_df)
    # TODO: switch to bioframe.assign_view upon update

    if flank is not None:
        features_df = expand_align_features(
            features_df, flank, clr.binsize, format=feature_type
        )
    else:
        features_df = features_df.copy()
        if feature_type == "bed":
            features_df["lo"] = (features_df["start"] / clr.binsize).astype(int)
            features_df["hi"] = (features_df["end"] / clr.binsize).astype(int)
        else:
            features_df["lo1"] = (features_df["start1"] / clr.binsize).astype(int)
            features_df["hi1"] = (features_df["end1"] / clr.binsize).astype(int)
            features_df["lo2"] = (features_df["start2"] / clr.binsize).astype(int)
            features_df["hi2"] = (features_df["end2"] / clr.binsize).astype(int)

    if view_df is None:
        view_df = make_cooler_view(clr)
    else:
        try:
            _ = is_compatible_viewframe(
                view_df,
                clr,
                check_sorting=True,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("view_df is not a valid viewframe or incompatible") from e

    if clr_weight_name not in [None, False]:
        # check if cooler is balanced
        try:
            _ = is_cooler_balanced(clr, clr_weight_name, raise_errors=True)
        except Exception as e:
            raise ValueError(
                f"provided cooler is not balanced or {clr_weight_name} is missing"
            ) from e

    if min_diag == "auto" and clr_weight_name not in [None, False]:
        min_diag = dict(clr.open()[f"bins/{clr_weight_name}"].attrs).get(
            "ignore_diags", 2
        )
    elif clr_weight_name in [None, False]:
        min_diag = 2

    # Find region offsets and then subtract them from the feature extents

    region_offsets = view_df[["chrom", "start", "end"]].apply(clr.offset, axis=1)
    region_offsets_dict = dict(zip(view_df["name"].values, region_offsets))

    features_df["region_offset"] = features_df["region"].replace(region_offsets_dict)

    if feature_type == "bed":
        features_df[["lo", "hi"]] = (
            features_df[["lo", "hi"]]
            .subtract(
                features_df["region_offset"].fillna(0),
                axis=0,
            )
            .astype(int)
        )
    else:
        features_df[["lo1", "hi1"]] = (
            features_df[["lo1", "hi1"]]
            .subtract(
                features_df["region_offset"].fillna(0),
                axis=0,
            )
            .astype(int)
        )
        features_df[["lo2", "hi2"]] = (
            features_df[["lo2", "hi2"]]
            .subtract(
                features_df["region_offset"].fillna(0),
                axis=0,
            )
            .astype(int)
        )

    # TODO move view, expected and other checks in the user-facing functions, add tests

    if expected_df is None:
        snipper = CoolerSnipper(
            clr,
            view_df=view_df,
            cooler_opts={"balance": clr_weight_name},
            min_diag=min_diag,
        )
    else:
        snipper = ObsExpSnipper(
            clr,
            expected_df,
            view_df=view_df,
            cooler_opts={"balance": clr_weight_name},
            min_diag=min_diag,
            expected_value_col=expected_value_col,
        )

    if nproc > 1:
        pool = multiprocessing.Pool(nproc)
        mymap = pool.map
    else:
        mymap = map
    stack = _pileup(features_df, snipper.select, snipper.snip, map=mymap)
    if feature_type == "bed":
        stack = np.nansum([stack, np.transpose(stack, axes=(1, 0, 2))], axis=0)

    if nproc > 1:
        pool.close()
    return stack
