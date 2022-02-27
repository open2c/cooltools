import os.path as op
import pandas as pd
import cooler
import cooltools
import pytest
import bioframe


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

    # raises a value error because the contact type is not cis or trans
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            expected_df,
            "other",
            verify_view=None,
            expected_value_cols=["balanced.avg"],
            raise_errors=True,
        )

    # raises a value error because input is not a dataframe
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

    # tests with sin_eigs_mat cooler
    cooler_file = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    clr = cooler.Cooler(cooler_file)
    exp_cis = cooltools.expected_cis(clr)

    # cis with no verify_view should work!
    assert cooltools.lib.is_valid_expected(
        exp_cis, "cis", verify_view=None, verify_cooler=clr, raise_errors=True
    )
    # tests with sin_eigs_mat cooler and custom armwise view as input
    view_df = pd.DataFrame(
        [
            ["chr1", 0, 500, "chr1L"],
            ["chr1", 500, 1000, "chr1R"],
            ["chr2", 0, 1000, "chr2L"],
            ["chr2", 1000, 2000, "chr2R"],
            ["chr3", 0, 1500, "chr3L"],
            ["chr3", 1500, 3000, "chr3R"],
        ],
        columns=["chrom", "start", "end", "name"],
    )

    exp_cis = cooltools.expected_cis(clr, view_df=view_df[:1])

    # cis with intra_only=True does not raise ValueError with swapped region columns
    # because region names are identical
    assert cooltools.lib.is_valid_expected(
        exp_cis.rename(columns={"region1": "region2", "region2": "region1"}),
        "cis",
        verify_view=view_df[:1],
        verify_cooler=clr,
    )

    # cis that is shortened does not have enough diagonals
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            exp_cis[::2],
            "cis",
            verify_view=view_df[:1],
            verify_cooler=clr,
            raise_errors=True,
        )

    exp_cis = cooltools.expected_cis(clr, view_df=view_df[:2], intra_only=False)
    # cis with intra_only=False raises ValueError with swapped region columns,
    # because this tries to query lower triangular part of cooler
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            exp_cis.rename(columns={"region1": "region2", "region2": "region1"}),
            "cis",
            verify_view=view_df[:2],
            verify_cooler=clr,
            raise_errors=True,
        )

    # trans raises ValueError with swapped region columns
    exp_trans = cooltools.expected_trans(clr, view_df=view_df)
    with pytest.raises(ValueError):
        cooltools.lib.is_valid_expected(
            exp_trans.rename(columns={"region1": "region2", "region2": "region1"}),
            "trans",
            verify_view=view_df[::-1],
            verify_cooler=clr,
            raise_errors=True,
        )


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

    # index shouldn't matter
    assert cooltools.lib.is_track(track)

    track_incompat = bioframe.sort_bedframe(track.copy())
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

    # not sorted
    track_incompat = pd.DataFrame(
        [
            ["chr3", 0, 10, 0.3],
            ["chr1", 10, 20, 0.1],
            ["chr1", 0, 10, 0.1],
            ["chr2", 0, 10, 0.2],
        ],
        columns=["chrom", "start", "end", "value"],
    )
    assert cooltools.lib.is_track(track_incompat) is False

    # not sorted
    track = pd.DataFrame(
        [
            ["chr3", 0, 10, 0.3],
            ["chr1", 0, 10, 0.1],
            ["chr1", 10, 20, 0.1],
            ["chr2", 0, 10, 0.2],
        ],
        columns=["chr", "chromStart", "chr_end", "quant"],
    )
    assert cooltools.lib.is_track(track)
