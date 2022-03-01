import warnings
from itertools import tee, starmap
from operator import gt
from copy import copy

import numpy as np
import pandas as pd
import bioframe

from . import schemas
from .common import make_cooler_view


def _is_sorted_ascending(iterable):
    # code copied from "more_itertools" package
    """Returns ``True`` if the items of iterable are in sorted order, and
    ``False`` otherwise.

    The function returns ``False`` after encountering the first out-of-order
    item. If there are no out-of-order items, the iterable is exhausted.
    """

    it0, it1 = tee(iterable)  # duplicate the iterator
    next(it1, None)  # skip 1st element in "it1" copy
    # check if all values in iterable are in ascending order
    # similar to all(array[:-1] < array[1:])
    _pairs_out_of_order = starmap(gt, zip(it0, it1))
    # no pairs out of order returns True, i.e. iterator is sorted
    return not any(_pairs_out_of_order)


def _is_expected(
    expected_df,
    contact_type="cis",
    expected_value_cols=["count.avg", "balanced.avg"],
    raise_errors=False,
):
    """
    Check if a expected_df looks like an expected
    DataFrame, i.e.:
     - has neccessary columns
     - there are no Nulls in the regions1, regions2, diag columns
     - every trans pair region1 region2 has a single value
     - every cis pair region1, region2 has at least one value

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    contact_type : str
        'cis' or 'trans': run contact type specific checks
    expected_value_cols : list of str
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_expected : bool
        True when expected_df passes the checks, False otherwise
    """

    if contact_type == "cis":
        expected_dtypes = copy(schemas.diag_expected_dtypes)  # mutable copy
    elif contact_type == "trans":
        expected_dtypes = copy(schemas.block_expected_dtypes)  # mutable copy
    else:
        raise ValueError(
            f"Incorrect contact_type: {contact_type}, only cis and trans are supported."
        )

    # that's what we expect as column names:
    expected_columns = [col for col in expected_dtypes]
    # separate "structural" columns: region1, region2, diag if "cis":
    grouping_columns = expected_columns[:-1]

    # add columns with values and their dtype (float64):
    for name in expected_value_cols:
        expected_columns.append(name)
        expected_dtypes[name] = "float"

    # try a bunch of assertions about expected
    try:
        # make sure expected is a DataFrame
        if not isinstance(expected_df, pd.DataFrame):
            raise ValueError(
                f"expected_df must be DataFrame, it is {type(expected_df)} instead"
            )
        # make sure required columns are present and can be cast to the dtypes
        if set(expected_columns).issubset(expected_df.columns):
            try:
                expected_df = expected_df.astype(expected_dtypes)
            except Exception as e:
                raise ValueError(
                    "expected_df does not match the expected schema:\n"
                    f"columns {expected_columns} cannot be cast to required data types."
                ) from e
        # raise special message for the old formatted expected_df :
        elif set(["region", "chrom"]).intersection(expected_df.columns):
            warnings.warn(
                "The expected dataframe appears to be in the old format."
                "It should have `region1` and `region2` columns instead of `region` or `chrom`."
                "Please recalculated your expected using current vestion of cooltools."
            )
            raise ValueError(
                "The expected dataframe appears to be in the old format."
                "It should have `region1` and `region2` columns instead of `region` or `chrom`."
                "Please recalculated your expected using current vestion of cooltools."
            )
        # does not look like expected at all :
        else:
            missing_columns = set(expected_columns) - set(expected_df.columns)
            raise ValueError(
                "expected_df does not match the expected schema:\n"
                f"columns {missing_columns} are missing"
            )

        # make sure there is no missing data in grouping columns
        if expected_df[grouping_columns].isna().any().any():
            raise ValueError(f"There are missing values in columns {grouping_columns}")

        # make sure "grouping" columns are unique:
        if expected_df.duplicated(subset=grouping_columns).any():
            raise ValueError(f"Values in {grouping_columns} columns must be unique")

        # for trans expected, ensure pairs region1, region2 have 1 value
        # for cis expected, ensure pairs region1, region2 have >=1 value
        region1_col, region2_col = grouping_columns[:2]
        for (r1, r2), df in expected_df.groupby([region1_col, region2_col]):
            if contact_type == "trans":
                if len(df) != 1:
                    ValueError(
                        f"region {r1},{r2} has more than a single value.\n"
                        "It has to be single for trans-expected"
                    )
                if r1 == r2:
                    ValueError(
                        f"region {r1},{r2} is symmetric\n"
                        "trans expected is caluclated for asymmetric regions only"
                    )
            if contact_type == "cis":
                # generally there shoud be >1 values per region in cis-expected, but
                # tiny regions smaller than a binsize could have 1 value
                if len(df) < 1:
                    ValueError(
                        f"region {r1},{r2} has to have at least one values for cis-expected"
                    )

    except Exception as e:
        if raise_errors:
            raise e
        else:
            # does not look like proper expected
            return False
    else:
        # if no exceptions were raised, it looks like expected_df
        return True


def _is_expected_cataloged(expected_df, verify_view):
    _all_expected_regions = expected_df[["region1", "region2"]].values.flatten()
    if not np.all(verify_view["name"].isin(_all_expected_regions)):
        raise ValueError(
            "View regions are not in the expected table. Provide expected table for the same regions"
        )


def _is_compatible_cis_expected(
    expected_df,
    verify_view=None,
    verify_cooler=None,
    expected_value_cols=["count.avg", "balanced.avg"],
    raise_errors=False,
):
    """
    Verify expected_df to make sure it is compatible.

    Expected tables can be verified against both
    a view and a cooler. This includes ensuring:
        - the entries in columns region1, region2 match names from view
        - number of diagonals per pair of region1, region2 matches cooler

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    try:
        # make sure it looks like cis-expected in the first place
        try:
            _ = _is_expected(
                expected_df,
                "cis",
                expected_value_cols,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError(
                "expected_df does not agree with schema for cis-expected"
            ) from e

        # Check that view regions are named as in expected table.
        if verify_view is not None:
            _is_expected_cataloged(expected_df, verify_view)

        # check if the number of diagonals is correct:
        if verify_cooler is not None:
            verify_view = (
                make_cooler_view(verify_cooler) if verify_view is None else verify_view
            )
            # check number of bins per region in cooler and expected table
            # compute # of bins by comparing matching indexes
            for (name1, name2), group in expected_df.groupby(["region1", "region2"]):
                n_diags_expected = len(group)
                if name1 == name2:
                    region = verify_view.set_index("name").loc[name1]
                    lo, hi = verify_cooler.extent(region)
                    n_diags_cooler = hi - lo
                else:
                    region1 = verify_view.set_index("name").loc[name1]
                    region2 = verify_view.set_index("name").loc[name2]
                    lo1, hi1 = verify_cooler.extent(region1)
                    lo2, hi2 = verify_cooler.extent(region2)
                    if not _is_sorted_ascending([lo1, hi1, lo2, hi2]):
                        raise ValueError(
                            f"Only upper right cis regions are supported, {name1}:{name2} is not"
                        )
                    # rectangle that is fully contained within upper-right part of the heatmap
                    n_diags_cooler = (hi1 - lo1) + (hi2 - lo2) - 1
                if n_diags_expected != n_diags_cooler:
                    raise ValueError(
                        "Region shape mismatch between expected and cooler. "
                        "Are they using the same resolution?"
                    )
    except Exception as e:
        if raise_errors:
            raise e
        else:
            # expected_df is not compatible
            return False
    else:
        return True


def _is_compatible_trans_expected(
    expected_df,
    verify_view=None,
    verify_cooler=None,
    expected_value_cols=["count.avg", "balanced.avg"],
    raise_errors=False,
):
    """
    Verify expected_df to make sure it is compatible
    with its view (viewframe) and cooler, i.e.:
        - entries in region1 and region2 match names from view

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    try:
        # make sure it looks like trans-expected in the first place
        try:
            _ = _is_expected(
                expected_df,
                "trans",
                expected_value_cols,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("expected_df does not look like trans-expected") from e

        # Check that view regions are named as in expected table.
        # Check region names:
        if verify_view is not None:
            _is_expected_cataloged(expected_df, verify_view)

        if verify_cooler is not None:
            verify_view = (
                make_cooler_view(verify_cooler) if verify_view is None else verify_view
            )
            # check number of bins per region in cooler and expected table
            # compute # of bins by comparing matching indexes
            for (name1, name2), group in expected_df.groupby(["region1", "region2"]):
                n_valid_expected = group["n_valid"].iat[
                    0
                ]  # extract single `n_valid` from group
                region1 = verify_view.set_index("name").loc[name1]
                region2 = verify_view.set_index("name").loc[name2]
                lo1, hi1 = verify_cooler.extent(region1)
                lo2, hi2 = verify_cooler.extent(region2)
                if not _is_sorted_ascending([lo1, hi1, lo2, hi2]):
                    raise ValueError(
                        f"Only upper right trans regions are supported, {name1}:{name2} is not"
                    )
                # compare n_valid per trans block and make sure it make sense:
                n_valid_cooler = (hi1 - lo1) * (hi2 - lo2)
                if n_valid_cooler < n_valid_expected:
                    warnings.warn(
                        "trans expected was calculated for a cooler with higher resolution."
                        "make sure this is intentional."
                    )
                # consider adding a proper check here - which requires using balancing weights
    except Exception as e:
        if raise_errors:
            raise e
        else:
            # expected_df is not compatible
            return False
    else:
        return True


def is_valid_expected(
    expected_df,
    contact_type,
    verify_view=None,
    verify_cooler=None,
    expected_value_cols=["count.avg", "balanced.avg"],
    raise_errors=False,
):
    """
    Verify expected_df to make sure it is compatible

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    contact_type : str
        'cis' or 'trans': run contact type specific checks
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    if contact_type == "cis":
        return _is_compatible_cis_expected(
            expected_df,
            verify_view=verify_view,
            verify_cooler=verify_cooler,
            expected_value_cols=expected_value_cols,
            raise_errors=raise_errors,
        )
    elif contact_type == "trans":
        return _is_compatible_trans_expected(
            expected_df,
            verify_view=verify_view,
            verify_cooler=verify_cooler,
            expected_value_cols=expected_value_cols,
            raise_errors=raise_errors,
        )
    else:
        raise ValueError("contact_type can be only cis or trans")


def is_compatible_viewframe(
    view_df, verify_cooler, check_sorting=False, raise_errors=False
):
    """
    Check if view_df is a viewframe and if
    it is compatible with the provided cooler.

    Parameters
    ----------
    view_df :  DataFrame
        view_df DataFrame to be validated
    verify_cooler : cooler
        cooler object to use for verification
    check_sorting : bool
        Check is regions in view_df are sorted as in
        chromosomes in cooler.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_compatible_viewframe : bool
        True when view_df is compatible, False otherwise
    """
    try:
        try:
            _ = bioframe.is_viewframe(view_df, raise_errors=True)
        except Exception as error_not_viewframe:
            try:
                _ = bioframe.make_viewframe(view_df)
            except Exception as error_cannot_make_viewframe:
                # view_df is not viewframe and cannot be easily converted
                raise ValueError(
                    "view_df is not a valid viewframe and cannot be recovered"
                ) from error_cannot_make_viewframe
            else:
                # view_df is not viewframe, but can be converted - formatting issue ? name-column ?
                raise ValueError(
                    "view_df is not a valid viewframe, apply bioframe.make_viewframe to convert"
                ) from error_not_viewframe

        # is view_df contained inside cooler-chromosomes ?
        cooler_view = make_cooler_view(verify_cooler)
        if not bioframe.is_contained(view_df, cooler_view, raise_errors=False):
            raise ValueError(
                "View table is out of the bounds of chromosomes in cooler."
            )

        # is view_df sorted by coord and chrom order as in cooler ?
        if check_sorting:
            if not bioframe.is_sorted(view_df, cooler_view, df_view_col="chrom"):
                raise ValueError(
                    "regions in the view_df must be sorted by coordinate"
                    " and chromosomes order as as in the verify_cooler."
                )

    except Exception as e:
        if raise_errors:
            raise ValueError("view_df is not compatible, or not a viewframe") from e
        else:
            # something went wrong: it's not a viewframe
            return False
    else:
        # no exceptions were raised: it's a compatible viewframe
        return True


def is_cooler_balanced(clr, clr_weight_name="weight", raise_errors=False):
    """
    Check if cooler is balanced, by checking
    if the requested weight column exist in the bin table.

    Parameters
    ----------
    clr : cooler
        cooler object to check
    clr_weight_name : str
        name of the weight column to check
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_balanced : bool
        True if weight column is present, False otherwise
    """

    if not isinstance(clr_weight_name, str):
        raise TypeError(
            "clr_weight_name has to be str that specifies name of balancing weight in clr"
        )

    if clr_weight_name in schemas.DIVISIVE_WEIGHTS_4DN:
        raise KeyError(
            f"clr_weight_name: {clr_weight_name} is reserved as divisive by 4DN"
            "cooltools supports multiplicative weights at this time."
        )

    # check if clr_weight_name is in cooler
    if clr_weight_name not in clr.bins().columns:
        if raise_errors:
            raise ValueError(
                f"specified balancing weight {clr_weight_name} is not available in cooler"
            )
        else:
            return False
    else:
        return True


def is_track(track, raise_errors=False):
    """
    Check if an input is a valid track dataframe.

    Specifically:
    - the first three columns satisfy requirements for a bedframe
    - the fourth column has a numeric dtype
    - intervals are non-overlapping
    - intervals are sorted within chromosome

    Parameters
    ----------
    track : pd.DataFrame
        track dataframe to check
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_track : bool
        True if satisfies requirements, False otherwise
    """
    if not bioframe.is_bedframe(track, cols=track.columns[:3]):
        if raise_errors:
            raise ValueError("track must have bedFrame-like interval columns")
        else:
            return False
    if not pd.core.dtypes.common.is_numeric_dtype(track[track.columns[3]]):
        if raise_errors:
            raise ValueError("track signal column must be numeric")
        else:
            return False
    if bioframe.is_overlapping(track, cols=track.columns[:3]):
        if raise_errors:
            raise ValueError("track intervals must not be overlapping")
        else:
            return False

    for name, group in track.groupby([track.columns[0]]):
        if not _is_sorted_ascending(group[track.columns[1]].values):
            if raise_errors:
                raise ValueError(
                    "track intervals must be sorted by ascending order within chromosomes"
                )
            else:
                return False
    return True
