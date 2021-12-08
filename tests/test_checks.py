import os.path as op
import pandas as pd
import cooler
import cooltools
import pytest


def test_is_valid_expected(request, tmpdir):

    expected_file = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    expected_df = pd.read_csv(expected_file, sep="\t")

    # false because need to specify that this expected only has balanced.avg not count.avg
    assert cooltools.lib.checks.is_valid_expected(expected_df, "cis") is False

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


def test_is_compatible_viewframe(request, tmpdir):
    clr_file = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_file)

    view_file = op.join(request.fspath.dirname, "data/mm9.named_nonoverlap_regions.bed")
    view_df = cooltools.lib.read_viewframe_from_file(view_file)

    # true for base case
    assert cooltools.lib.is_compatible_viewframe(view_df, clr)

    # false if out of order & sorting is on
    with pytest.raises(ValueError):
        cooltools.lib.is_compatible_viewframe(
            view_df[::-1], clr, check_sorting=True, raise_errors=True
        )

    # false if out of order & sorting is on
    with pytest.raises(ValueError):
        cooltools.lib.is_compatible_viewframe(
            view_df[::-1], clr, check_sorting=True, raise_errors=True
        )

    # false if out of bounds of chromosomes
    view_df_incompatible = view_df.copy()
    view_df_incompatible.iloc[1, 2] += 1000000
    with pytest.raises(ValueError):
        cooltools.lib.is_compatible_viewframe(
            view_df_incompatible, clr, raise_errors=True
        )

    # false if not a view (e.g. name column is missing)
    with pytest.raises(ValueError):
        cooltools.lib.is_compatible_viewframe(
            view_df.iloc[:, :3], clr, raise_errors=True
        )


def test_is_cooler_balanced(request, tmpdir):

    clr_file = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_file)

    assert cooltools.lib.is_cooler_balanced(clr)

    with pytest.raises(ValueError):
        cooltools.lib.is_cooler_balanced(
            clr, clr_weight_name="weight_name_not_in_bins", raise_errors=True
        )

    with pytest.raises(KeyError):
        cooltools.lib.is_cooler_balanced(clr, clr_weight_name="KR")


def test_is_track():
    track = pd.DataFrame(
        [
            ["chr3", 0, 10, 0.3],
            ["chr1", 0, 10, 0.1],
            ["chr1", 10, 20, 0.1],
            ["chr2", 0, 10, 0.2],
        ],
        columns=["chrom", "start", "end", "value"],
    )
    track.index = [5, 2, 1, 3]

    assert cooltools.lib.is_track(track)

    track_incompat = track.copy()
    track_incompat.iloc[:, 0] = 10

    # not bedframe in first three columns
    assert cooltools.lib.is_track(track_incompat) is False

    track_incompat = track.copy()
    track_incompat.iloc[:, -1] = ["a", "b", "c", "d"]
    # not numeric type in column4
    assert cooltools.lib.is_track(track_incompat) is False

    track_incompat = track.copy()
    track_incompat.iloc[0, 0] = "chr1"
    # overlapping
    assert cooltools.lib.is_track(track_incompat) is False
