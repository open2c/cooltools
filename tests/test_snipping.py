import cooler
import bioframe
import os.path as op

import cooltools.api
import numpy as np
import pandas as pd
import pytest

from click.testing import CliRunner
from cooltools.cli import cli


def test_pileup_cli_npz(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_features = op.join(request.fspath.dirname, "data/CN.mm9.toy_features.bed")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    in_expected = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    out_file = op.join(tmpdir, "tmp.npz")
    runner = CliRunner()
    # Output npz file:
    result = runner.invoke(
        cli,
        [
            "pileup",
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
            "--clr-weight-name",
            "weight",
            "--features-format",
            "bed",
            "--nproc",
            1,
            "--store-snips",
            "--out",
            out_file,
        ],
    )
    assert result.exit_code == 0


def test_pileup_cli_hdf5(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_features = op.join(request.fspath.dirname, "data/CN.mm9.toy_features.bed")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    in_expected = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    out_file = op.join(tmpdir, "tmp.hdf5")
    runner = CliRunner()

    # Output hdf5 file:
    result = runner.invoke(
        cli,
        [
            "pileup",
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
            "--clr-weight-name",
            "weight",
            "--features-format",
            "bed",
            "--nproc",
            1,
            "--store-snips",
            "--out",
            out_file,
            "--out-format",
            "HDF5",
        ],
    )
    assert result.exit_code == 0


def test_pileup(request):

    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )

    # I.
    # Example on-diagonal features, two regions from annotated genomic regions:
    windows = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [102_000_000, 108_000_000],
            "end": [107_000_000, 113_000_000],
        }
    )

    stack = cooltools.api.snipping.pileup(
        clr, windows, view_df=view_df, expected_df=exp, flank=None
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # II.
    # Example off-diagonal features, two features from annotated genomic regions:
    windows = pd.DataFrame(
        {
            "chrom1": ["chr1", "chr1"],
            "start1": [102_000_000, 107_000_000],
            "end1": [107_000_000, 112_000_000],
            "chrom2": ["chr1", "chr1"],
            "start2": [107_000_000, 113_000_000],
            "end2": [112_000_000, 118_000_000],
        }
    )
    stack = cooltools.api.snipping.pileup(
        clr, windows, view_df=view_df, expected_df=exp, flank=None
    )
    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # III.
    # Example off-diagonal features, one region outside the view:
    windows = pd.DataFrame(
        {
            "chrom1": ["chr1", "chr1"],
            "start1": [90_000_000, 105_000_000],
            "end1": [95_000_000, 110_000_000],
            "chrom2": ["chr1", "chr1"],
            "start2": [105_000_000, 110_000_000],
            "end2": [110_000_000, 115_000_000],
        }
    )
    stack = cooltools.api.snipping.pileup(
        clr, windows, view_df=view_df, expected_df=exp, flank=None
    )
    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    assert np.all(np.isnan(stack[:, :, 0]))

    # IV.
    # Example on-diagonal features, not valid bedframes (start>end):
    windows = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [107_000_000, 108_000_000],
            "end": [102_000_000, 113_000_000],
        }
    )
    with pytest.raises(ValueError):
        stack = cooltools.api.snipping.pileup(clr, windows, view_df, exp, flank=None)

    # DRAFT # Should work with force=True:
    # stack = cooltools.api.snipping.pileup(clr, windows, view_df, exp, flank=None, force=True)
    # # Check that the size of snips is OK and there are two of them:
    # assert stack.shape == (5, 5, 2)

    # IV.
    # Example of-diagonal features not valid bedframes (start>end):
    windows = pd.DataFrame(
        {
            "chrom1": ["chr1", "chr1"],
            "start1": [107_000_000, 107_000_000],
            "end1": [102_000_000, 112_000_000],
            "chrom2": ["chr1", "chr1"],
            "start2": [107_000_000, 113_000_000],
            "end2": [112_000_000, 118_000_000],
        }
    )
    with pytest.raises(ValueError):
        stack = cooltools.api.snipping.pileup(
            clr, windows, view_df=view_df, expected_df=exp, flank=None
        )

    # DRAFT # Should work with force=True:
    # stack = cooltools.api.snipping.pileup(clr, windows, view_df, exp, flank=0, force=True)
    # # Check that the size of snips is OK and there are two of them:
    # assert stack.shape == (5, 5, 2)


def test_ondiag__pileup_with_expected(request):
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
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(clr, exp, view_df=view_df)

        # I.
        # Example region with windows, two regions from annotated genomic regions:
        windows = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
        )
        windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
            drop=True
        )
        stack = cooltools.api.snipping._pileup(
            windows, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (5, 5, 2)

        # II.
        # Example region with windows, second window comes from unannotated genomic region:
        windows = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
        )
        windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (5, 5, 2)
        assert np.all(np.isnan(stack[:, :, 1]))


def test_ondiag__pileup_without_expected(request):
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
    windows = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
    )

    windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
        drop=True
    )
    snipper = cooltools.api.snipping.CoolerSnipper(clr, view_df=view_df, min_diag=None)
    stack = cooltools.api.snipping._pileup(
        windows, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # II.
    # Example region with windows, second window comes from unannotated genomic region:
    windows = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
    )
    windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
        drop=True
    )

    stack = cooltools.api.snipping._pileup(
        windows, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (5, 5, 2)
    assert np.all(np.isfinite(stack[:, :, 0]))
    assert np.all(np.isnan(stack[:, :, 1]))


def test_offdiag__pileup_with_expected(request):
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
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
    ):

        snipper = snipper_class(clr, exp, view_df=view_df)

        # I.
        # Example region with windows, two off-diagonal features from annotated genomic regions:
        windows1 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
        )
        windows2 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
        )
        windows = pd.merge(
            windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
        )
        windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (5, 5, 2)

        # II.
        # Example region with windows, second window is between two different regions:
        windows1 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
        )
        windows2 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
        )
        windows = pd.merge(
            windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
        )
        windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (5, 5, 2)
        assert np.all(np.isnan(stack[:, :, 1]))


def test_offdiag__pileup_without_expected(request):
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
    windows1 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
        drop=True
    )

    snipper = cooltools.api.snipping.CoolerSnipper(clr, view_df=view_df, min_diag=None)
    stack = cooltools.api.snipping._pileup(
        windows, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # II.
    # Example region with windows, second window comes from unannotated genomic region:
    windows1 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [105_000_000, 109_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.lib.common.assign_view_auto(windows, view_df).reset_index(
        drop=True
    )

    stack = cooltools.api.snipping._pileup(
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
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
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
    snipper = cooltools.api.snipping.CoolerSnipper(clr, view_df=view_df)
    matrix = snipper.select("foo", "foo")
    snippet = snipper.snip(
        matrix, "foo", "foo", (110_000_000, 120_000_000, 110_000_000, 120_000_000)
    )
    assert snippet.shape is not None
