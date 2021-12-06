import os.path as op
import numpy as np
import pandas as pd
from numpy import testing

import bioframe
import cooler
import cooltools

import pytest


def test_is_valid_expected(request, tmpdir):

    expected_file = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    expected_df = pd.read_csv(expected_file, sep="\t")

    # false because need to specify that this expected only has balanced.avg not count.avg
    assert cooltools.lib.checks.is_valid_expected(expected_df, "cis") == False

    # true, because passing expected_value_cols that match what is in the expected table
    assert cooltools.lib.checks.is_valid_expected(
        expected_df, "cis", expected_value_cols=["balanced.avg"]
    )

    expected_df_incompat = expected_df.copy()
    expected_df_incompat.drop("dist", axis=1, inplace=True)
    # false, because this is cis expected and there is no dist column
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df_incompat,
            "cis",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # raises a value error because non-unique region pairs
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df_incompat,
            "trans",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # raises a value error because of the contact type
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df,
            "other",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # raises a value error because not a dataframe
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df.values,
            "other",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # raise error w/ old column names
    expected_df_incompat = expected_df.copy()
    expected_df_incompat.rename(columns={"region1": "region"}, inplace=True)
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df_incompat,
            "cis",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # alternate method of loading:
    expected_df = cooltools.lib.read_expected_from_file(
        expected_file, expected_value_cols=["balanced.avg"]
    )

    ### testing expected compatibility with a view as well
    view_file = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    view_df = cooltools.lib.read_viewframe_from_file(view_file)

    # true, because this view has regions named foo and bar, just like the expected table
    assert cooltools.lib.is_valid_expected(
        expected_df, "cis", verify_view=view_df, expected_value_cols=["balanced.avg"]
    )

    view_df_incompatible = view_df.copy()
    view_df_incompatible["name"] = ["totally", "wrong"]
    # false, because of mismatching view region names
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df,
            "cis",
            verify_view=view_df_incompatible,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    ### TODO: test with cooler as input


# def test_is_compatible_viewframe(request, tmpdir):

# def test_is_cooler_balanced(request, tmpdir):
# good if its good!
# value error for missing weight
# key error for 4DN divisive


# def test_is_track():
