import cooler
import bioframe
import os.path as op

import cooltools.api
import numpy as np
import pandas as pd
import pytest

from click.testing import CliRunner
from cooltools.lib import numutils
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

# define fixture with clr, exp and view_df to be reused across many tests
# fixture return a dict and values can be accessed e.g. as data["clr"]
@pytest.fixture
def data(request):
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    exp = pd.read_table(op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv"))
    view_df = bioframe.read_table(
        op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed"), schema="bed4"
    )
    return {
        "clr": clr,
        "exp": exp,
        "view": view_df,
    }


def test_pileup(request, data):

    # I.
    # Example on-diagonal features, two regions from annotated genomic regions:
    windows = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [83_000_000, 108_000_000],
            "end": [88_000_000, 113_000_000],
        }
    )

    stack = cooltools.api.snipping.pileup(data["clr"], windows, view_df=None, flank=None)
    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (2, 5, 5)
    # Check that NaNs were propagated
    assert np.all(np.isnan(stack[0, 2, :]))
    assert not np.all(np.isnan(stack))

    stack = cooltools.api.snipping.pileup(
        data["clr"], windows, view_df=data["view"], expected_df=data["exp"], flank=None
    )
    # Check that the size of snips is OK and there are two of them.
    # Now with view and expected:
    assert stack.shape == (2, 5, 5)

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
        data["clr"], windows, view_df=data["view"], expected_df=data["exp"], flank=None
    )
    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (2, 5, 5)

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
        data["clr"], windows, view_df=data["view"], expected_df=data["exp"], flank=None
    )
    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (2, 5, 5)

    assert np.all(np.isnan(stack[0]))

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
        stack = cooltools.api.snipping.pileup(data["clr"], windows, data["view"], data["exp"], flank=None)

    # DRAFT # Should work with force=True:
    # stack = cooltools.api.snipping.pileup(data["clr"], windows, view_df, data["exp"], flank=None, force=True)
    # # Check that the size of snips is OK and there are two of them:
    # assert stack.shape == (2, 5, 5,)

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
            data["clr"], windows, view_df=data["view"], expected_df=data["exp"], flank=None
        )

    # DRAFT # Should work with force=True:
    # stack = cooltools.api.snipping.pileup(data["clr"], windows, data["view"], data["exp"], flank=0, force=True)
    # # Check that the size of snips is OK and there are two of them:
    # assert stack.shape == (2, 5, 5,)


def test_ondiag__pileup_with_expected(request, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bed"
    for snipper_class in (
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(data["clr"], data["exp"], data["view"])

        # I.
        # Example region with windows, two regions from annotated genomic regions:
        windows = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 105_000_000], flank_bp=2_000_000
        )
        windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
            drop=True
        )
        stack = cooltools.api.snipping._pileup(
            windows, feature_type, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (2, 5, 5)

        # II.
        # Example region with windows, second window comes from unannotated genomic region:
        windows = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
        )
        windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, feature_type, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (2, 5, 5)
        assert np.all(np.isnan(stack[1]))


def test_ondiag__pileup_without_expected(request, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bed"

    # I.
    # Example region with windows, two regions from annotated genomic regions:
    windows = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 140_000_000], flank_bp=2_000_000
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )
    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=None)
    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (2, 5, 5)

    # II.
    # Example region with windows, second window comes from unannotated genomic region:
    windows = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [120_000_000, 160_000_000], flank_bp=2_000_000
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (2, 5, 5)
    assert np.all(np.isfinite(stack[0]))
    assert np.all(np.isnan(stack[1]))


def test_ondiag__pileup_without_expected_with_result_validation(request, allclose, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bed"
    min_diag = 1

    # Example region with windows, two regions from annotated genomic regions:
    windows = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [110_000_000, 140_000_000], flank_bp=2_000_000
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    # generate reference stack using naive cooler.matrix.fetch extraction:
    ref_stack = []
    for _win in windows[["chrom","start","end"]].itertuples(index=False):
        _mat = data["clr"].matrix().fetch(_win)
        # fill min_diag diagonals with NaNs
        for d in range(-min_diag + 1, min_diag):
            _mat = numutils.set_diag(_mat, np.nan, d)
        ref_stack.append(_mat)
    ref_stack = np.asarray(ref_stack)

    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=min_diag)
    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (len(windows), 5, 5)
    assert allclose(stack, ref_stack, equal_nan=True)


def test_offdiag__pileup_with_expected(request, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bedpe"
    for snipper_class in (
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(data["clr"], data["exp"], data["view"])

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
        windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, feature_type, snipper.select, snipper.snip, map=map
        )

        # Check that the size of snips is OK and there are two of them:
        assert stack.shape == (2, 5, 5)

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
        windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, feature_type, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (2, 5, 5)
        assert np.all(np.isnan(stack[1]))

        # III.
        # Example region with windows on diagonal, treated as off-diagonal. Check bottom
        # triangle is all NaN
        windows1 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
        )
        windows2 = cooltools.api.snipping.make_bin_aligned_windows(
            1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
        )
        windows = pd.merge(
            windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
        )
        windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
            drop=True
        )

        stack = cooltools.api.snipping._pileup(
            windows, feature_type, snipper.select, snipper.snip, map=map
        )

        assert stack.shape == (2, 5, 5)
        assert np.all(
            [np.all(np.isnan(snip[np.tril_indices_from(snip, 1)])) for snip in stack]
        )

def test_offdiag__pileup_without_expected_with_result_validation(request, allclose, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bedpe"
    min_diag = 2

    # I.
    # Example region with windows, two regions from annotated genomic regions:
    windows1 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 115_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [115_000_000, 140_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    # generate reference stack using naive cooler.matrix.fetch extraction:
    ref_stack = []
    for _win in windows[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].itertuples(index=False):
        _reg1 = _win[:3]
        _reg2 = _win[3:]
        _mat = data["clr"].matrix().fetch(_reg1, _reg2)
        # assuming regions are away from the diagonal - no set_diag needed
        ref_stack.append(_mat)
    ref_stack = np.asarray(ref_stack)

    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=min_diag)
    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK, there are two of them and their value are OK:
    assert stack.shape == (len(windows), 5, 5)
    assert allclose(stack, ref_stack, equal_nan = True)

    # II.
    # Example trans features
    min_diag = None
    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=min_diag)
    windows1 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 120_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr2", "chr2"], [102_000_000, 130_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    # generate reference stack using naive cooler.matrix.fetch extraction:
    ref_stack = []
    for _win in windows[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].itertuples(index=False):
        _reg1 = _win[:3]
        _reg2 = _win[3:]
        _mat = data["clr"].matrix().fetch(_reg1, _reg2)
        ref_stack.append(_mat)
    ref_stack = np.asarray(ref_stack)

    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    # check shape and values
    assert stack.shape == (len(windows), 5, 5)
    assert allclose(stack, ref_stack, equal_nan = True)



def test_offdiag__pileup_without_expected(request, data):
    """
    Test the snipping on matrix:
    """
    feature_type = "bedpe"

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
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=None)
    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (2, 5, 5)

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
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (2, 5, 5)
    assert np.all(np.isfinite(stack[0]))
    assert np.all(np.isnan(stack[1]))

    # III.
    # Example region with windows on diagonal, treated as off-diagonal. Check bottom
    # triangle is all NaN
    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"], min_diag=2)
    windows1 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
    )
    windows2 = cooltools.api.snipping.make_bin_aligned_windows(
        1_000_000, ["chr1", "chr1"], [102_000_000, 10_000_000], flank_bp=2_000_000
    )
    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows = cooltools.lib.common.assign_view_auto(windows, data["view"]).reset_index(
        drop=True
    )

    stack = cooltools.api.snipping._pileup(
        windows, feature_type, snipper.select, snipper.snip, map=map
    )

    assert stack.shape == (2, 5, 5)
    assert np.all(
        [np.all(np.isnan(snip[np.tril_indices_from(snip, 1)])) for snip in stack]
    )


def test_snipper_with_view_and_expected(request, data):
    for snipper_class in (
        cooltools.api.snipping.ObsExpSnipper,
        cooltools.api.snipping.ExpectedSnipper,
    ):
        snipper = snipper_class(data["clr"], data["exp"], data["view"])
        matrix = snipper.select("foo", "foo")
        snippet = snipper.snip(
            matrix, "foo", "foo", (110_000_000, 120_000_000, 110_000_000, 120_000_000)
        )
        assert snippet.shape is not None


def test_cooler_snipper_with_view(request, data):
    snipper = cooltools.api.snipping.CoolerSnipper(data["clr"], data["view"])
    matrix = snipper.select("foo", "foo")
    snippet = snipper.snip(
        matrix, "foo", "foo", (110_000_000, 120_000_000, 110_000_000, 120_000_000)
    )
    assert snippet.shape is not None
