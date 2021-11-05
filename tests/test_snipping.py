import cooler
import bioframe
import os.path as op

import cooltools.snipping
import numpy as np
import pandas as pd


from click.testing import CliRunner
from cooltools.cli import cli


def test_pileup_cli_npz(request):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_features = op.join(request.fspath.dirname, "data/CN.mm9.toy_features.bed")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    in_expected = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    runner = CliRunner()
    # Output npz file:
    result = runner.invoke(
        cli,
        [
            "compute-pileup",
            in_cool,
            in_features,
            "--view",
            in_regions,
            "--expected",
            in_expected,
            "--flank",
            300000,
            "--ignore-diags",
            2,
            "--weight-name",
            "weight",
            "--features-format",
            "bed",
            "--nproc",
            1,
            "--store-snips",
            "--out",
            "tmp.npz",
        ],
    )
    assert result.exit_code == 0


def test_pileup_cli_hdf5(request):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_features = op.join(request.fspath.dirname, "data/CN.mm9.toy_features.bed")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    in_expected = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    runner = CliRunner()

    # Output hdf5 file:
    result = runner.invoke(
        cli,
        [
            "compute-pileup",
            in_cool,
            in_features,
            "--view",
            in_regions,
            "--expected",
            in_expected,
            "--flank",
            300000,
            "--ignore-diags",
            2,
            "--weight-name",
            "weight",
            "--features-format",
            "bed",
            "--nproc",
            1,
            "--store-snips",
            "--out",
            "tmp.hdf5",
            "--out-format",
            "HDF5",
        ],
    )
    assert result.exit_code == 0


def test_ondiag_pileups_with_expected(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    for snipper_class in (
        cooltools.snipping.ObsExpSnipper,
        cooltools.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(clr, exp, view_df=view_df)

        # I.
        # Example region with windows, two regions from annotated genomic regions:
        windows = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
        )
        windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(
            drop=True
        )
        stack = cooltools.snipping.pileup_legacy(
            windows, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (5, 5, 2)

        # II.
        # Example region with windows, second window comes from unannotated genomic region:
        windows = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
        )
        windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.snipping.pileup_legacy(
            windows, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (5, 5, 2)
        assert np.all(np.isnan(stack[:, :, 1]))


def test_ondiag_pileups_without_expected(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )

    # I.
    # Example region with windows, two regions from annotated genomic regions:
    windows = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
    )

    windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(drop=True)

    snipper = cooltools.snipping.CoolerSnipper(clr, view_df=view_df, min_diag=None)
    stack = cooltools.snipping.pileup_legacy(
        windows, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # II.
    # Example region with windows, second window comes from unannotated genomic region:
    windows = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
    )
    windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(drop=True)

    stack = cooltools.snipping.pileup_legacy(
        windows, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (5, 5, 2)
    assert np.all(np.isfinite(stack[:, :, 0]))
    assert np.all(np.isnan(stack[:, :, 1]))


def test_offdiag_pileups_with_expected(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    for snipper_class in (
        cooltools.snipping.ObsExpSnipper,
        cooltools.snipping.ExpectedSnipper,
    ):

        snipper = snipper_class(clr, exp, view_df=view_df)

        # I.
        # Example region with windows, two off-diagonal features from annotated genomic regions:
        windows1 = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
        )
        windows2 = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
        )
        windows = pd.merge(
            windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
        )
        windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.snipping.pileup_legacy(
            windows, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (5, 5, 2)

        # II.
        # Example region with windows, second window is between two different regions:
        windows1 = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
        )
        windows2 = cooltools.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
        )
        windows = pd.merge(
            windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
        )
        windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.snipping.pileup_legacy(
            windows, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (5, 5, 2)
        assert np.all(np.isnan(stack[:, :, 1]))


def test_offdiag_pileups_without_expected(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )

    # I.
    # Example region with windows, two regions from annotated genomic regions:
    windows1 = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(drop=True)

    snipper = cooltools.snipping.CoolerSnipper(clr, view_df=view_df, min_diag=None)
    stack = cooltools.snipping.pileup_legacy(
        windows, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # II.
    # Example region with windows, second window comes from unannotated genomic region:
    windows1 = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.snipping.assign_regions(windows, view_df).reset_index(drop=True)

    stack = cooltools.snipping.pileup_legacy(
        windows, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (5, 5, 2)
    assert np.all(np.isfinite(stack[:, :, 0]))
    assert np.all(np.isnan(stack[:, :, 1]))


def test_snipper_with_view_and_expected(request):
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    for snipper_class in (
        cooltools.snipping.ObsExpSnipper,
        cooltools.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(clr, exp, view_df=view_df)
        matrix = snipper.select("foo", "foo")
        snippet = snipper.snip(
            matrix, "foo", "foo", (110_000_000, 120_000_000, 110_000_000, 120_000_000)
        )
        assert snippet.shape is not None


def test_cooler_snipper_with_view(request):
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    snipper = cooltools.snipping.CoolerSnipper(clr, view_df=view_df)
    matrix = snipper.select("foo", "foo")
    snippet = snipper.snip(
        matrix, "foo", "foo", (110_000_000, 120_000_000, 110_000_000, 120_000_000)
    )
    assert snippet.shape is not None
