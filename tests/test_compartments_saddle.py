import os.path as op

import numpy as np
import pandas as pd
from click.testing import CliRunner
from cooltools.cli import cli

import cooltools.saddle as saddle
import pytest


### TODO tests for non-covered click arguments:
# clr-weight-name
# expected_path with '::' syntax to specify expected_value_col column name.
# max-dist
# min-dist
#


def test_compartment_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    out_eig_prefix = op.join(tmpdir, "test.eigs")
    runner = CliRunner()
    result = runner.invoke(cli, ["call-compartments", "-o", out_eig_prefix, in_cool])
    assert result.exit_code == 0
    test_eigs = pd.read_table(out_eig_prefix + ".cis.vecs.tsv", sep="\t")
    gb = test_eigs.groupby("chrom")
    for chrom in gb.groups:
        chrom_eigs = gb.get_group(chrom)
        r = np.abs(
            np.corrcoef(
                chrom_eigs.E1.values, np.sin(chrom_eigs.start * 2 * np.pi / 500)
            )[0, 1]
        )
        assert r > 0.95


def test_saddle_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    out_eig_prefix = op.join(tmpdir, "test.eigs")
    out_expected = op.join(tmpdir, "test.expected")
    out_saddle_prefix = op.join(tmpdir, "test.saddle")

    runner = CliRunner()
    result = runner.invoke(cli, ["call-compartments", "-o", out_eig_prefix, in_cool])
    assert result.exit_code == 0

    result = runner.invoke(cli, ["compute-expected", "-o", out_expected, in_cool])
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-saddle",
            "-o",
            out_saddle_prefix,
            "--vrange",
            "-0.5",
            "0.5",
            "--n-bins",
            "30",
            "--scale",
            "log",
            in_cool,
            f"{out_eig_prefix}.cis.vecs.tsv",
            out_expected,
        ],
    )
    assert result.exit_code == 0

    log2_sad = np.log2(np.load(out_saddle_prefix + ".saddledump.npz")["saddledata"])
    bins = np.load(out_saddle_prefix + ".saddledump.npz")["binedges"]
    binmids = (bins[:-1] + bins[1:]) / 2
    log2_theor_sad = np.log2(1 + binmids[None, :] * binmids[:, None])

    log2_sad_flat = log2_sad[1:-1, 1:-1].flatten()
    log2_theor_sad_flat = log2_theor_sad.flatten()

    mask = np.isfinite(log2_sad_flat) & np.isfinite(log2_theor_sad_flat)

    cc = np.abs(np.corrcoef(log2_sad_flat[mask], log2_theor_sad_flat[mask])[0][1])

    assert cc > 0.9


def test_trans_compartment_cli(request, tmpdir):
    # somehow - it is E3 that captures sin-like plaid
    # pattern, instead of E1 - we'll keep it like that for now:
    in_cool = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    out_eig_prefix = op.join(tmpdir, "test.eigs")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "call-compartments",
            "--contact-type",
            "trans",
            "-o",
            out_eig_prefix,
            in_cool,
        ],
    )
    assert result.exit_code == 0
    test_trans_eigs = pd.read_table(out_eig_prefix + ".trans.vecs.tsv", sep="\t")
    r = np.abs(
        np.corrcoef(
            test_trans_eigs.E1.values, np.sin(test_trans_eigs.start * 2 * np.pi / 500)
        )[0, 1]
    )
    assert r > 0.95


def test_trans_saddle_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    out_eig_prefix = op.join(tmpdir, "test.eigs")
    out_expected = op.join(tmpdir, "test.trans.expected")
    out_saddle_prefix = op.join(tmpdir, "test.trans.saddle")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["call-compartments", "--contact-type", "trans", "-o", out_eig_prefix, in_cool],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli,
        ["compute-expected", "--contact-type", "trans", "-o", out_expected, in_cool],
    )
    assert result.exit_code == 0

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-saddle",
            "-o",
            out_saddle_prefix,
            "--contact-type",
            "trans",
            "--vrange",
            "-0.5",
            "0.5",
            "--n-bins",
            "30",
            "--scale",
            "log",
            in_cool,
            f"{out_eig_prefix}.trans.vecs.tsv",
            out_expected,
        ],
    )
    assert result.exit_code == 0

    log2_sad = np.log2(np.load(out_saddle_prefix + ".saddledump.npz")["saddledata"])
    bins = np.load(out_saddle_prefix + ".saddledump.npz")["binedges"]
    binmids = (bins[:-1] + bins[1:]) / 2
    log2_theor_sad = np.log2(1 + binmids[None, :] * binmids[:, None])

    log2_sad_flat = log2_sad[1:-1, 1:-1].flatten()
    log2_theor_sad_flat = log2_theor_sad.flatten()

    mask = np.isfinite(log2_sad_flat) & np.isfinite(log2_theor_sad_flat)

    cc = np.abs(np.corrcoef(log2_sad_flat[mask], log2_theor_sad_flat[mask])[0][1])

    assert cc > 0.9


def test_get_digitized():
    # np.nan and pd.NA get digitized to -1, suffix should be added
    df = pd.DataFrame(
        [["chr1", 0, 10, np.nan]],
        columns=["chrom", "start", "end", "value"],
    )
    digitized = saddle.get_digitized(df, 10, vrange=(-1, 1), digitized_suffix=".test")[
        0
    ]
    assert -1 == digitized["value.test"].values

    df = pd.DataFrame(
        [["chr1", 0, 10, pd.NA]],
        columns=["chrom", "start", "end", "value"],
    ).astype({"value": pd.Float64Dtype()})
    digitized = saddle.get_digitized(df, 10, vrange=(-1, 1), digitized_suffix=".test")[
        0
    ]
    assert -1 == digitized["value.test"].values

    n_bins = 10
    digitized = saddle.get_digitized(df, n_bins, vrange=(-1, 1))[0]
    # the dtype of the returned column should be a categorical
    assert type(digitized["value.d"].dtype) is pd.core.dtypes.dtypes.CategoricalDtype

    # the number of categories should be equal to the number of bins +3
    assert (n_bins + 3) == digitized["value.d"].dtype.categories.shape[0]

    df = pd.DataFrame(
        [
            ["chr1", 0, 10, -0.5],
            ["chr1", 10, 20, 0.5],
        ],
        columns=["chrom", "start", "end", "value"],
    )

    # values out of the range should be in the 0 and n+1 bins
    digitized = saddle.get_digitized(df, n_bins, vrange=(-0.1, 0.1))[0]
    assert 0 == digitized["value.d"].values[0]
    assert (n_bins + 1) == digitized["value.d"].values[1]

    # for an input dataframe of ten elements between -1 and 1,
    # and 5 bins, each bin should have 2 digitized values
    # this test will need an update after input checking
    df_linspace = pd.DataFrame(
        (np.linspace(-1, 1, 10) * np.ones((4,))[:, None]).T,
        columns=["chrom", "start", "end", "value"],
    )
    df_linspace["start"] += 1
    df_linspace["start"] *= 10
    df_linspace["end"] += 2
    df_linspace["end"] *= 10
    df_linspace["chrom"] = "chrX"
    df_linspace = df_linspace.astype({"chrom": "str", "start": int, "end": int})

    x = saddle.get_digitized(df_linspace, 5, vrange=(-1, 1.001),)[
        0
    ]["value.d"]
    assert (2 == np.histogram(x, np.arange(1, 7))[0]).all()

    # if the bottom and top quantiles are 25 and 75 with 3 bins, then
    # the low outlier and high outlier bins should each have 3 values
    x = saddle.get_digitized(df_linspace, 1, qrange=(0.25, 0.75),)[
        0
    ]["value.d"]
    assert 3 == np.sum(x == 0)
    assert 3 == np.sum(x == 2)

    # bins[-1] max value should remain in bin N,
    # not get pushed to outlier bin.

    # raises error if not provided with a track
    # (i.e. bedframe with a numeric fourth column)
    df_not_track = pd.DataFrame(
        [["chr1", 20, 40, "non-numeric"]],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        saddle.get_digitized(df_not_track, n_bins, vrange=(0, 2))

    df_not_track = pd.DataFrame(
        [[0, 20, 40, 0]],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        saddle.get_digitized(df_not_track, n_bins, vrange=(0, 2))

    # raises error if both or none of vrange, qrange provided
    with pytest.raises(ValueError):
        saddle.get_digitized(df, n_bins, vrange=(0, 2), qrange=(0.1, 0.9))
    with pytest.raises(ValueError):
        saddle.get_digitized(df, n_bins, vrange=None, qrange=None)

    # raises error if vrange lo>hi, qrange lo >hi, or qrange out of (0,1)
    with pytest.raises(ValueError):
        saddle.get_digitized(df, n_bins, vrange=(2, 1))
    with pytest.raises(ValueError):
        saddle.get_digitized(df, n_bins, qrange=(0, 2.1))
    with pytest.raises(ValueError):
        saddle.get_digitized(df, n_bins, qrange=(0.5, 0.25))


def test_get_saddle(request, tmpdir):

    in_cool = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    out_eig_prefix = op.join(tmpdir, "test.eigs")
    out_expected = op.join(tmpdir, "test.cis.expected")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["call-compartments", "-o", out_eig_prefix, in_cool],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli,
        ["compute-expected", "-o", out_expected, in_cool],
    )
    assert result.exit_code == 0

    track = pd.read_csv(f"{out_eig_prefix}.cis.vecs.tsv", sep="\t")[["chrom", "start", "end", "E1"]]
    expected = pd.read_csv(out_expected, sep="\t")

    import cooler
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/sin_eigs_mat.cool"))

    # non-digitized track should raise an error
    with pytest.raises(ValueError):
        saddle.get_saddle(clr, expected, track, "cis")

    # contact_type that is not cis or trans should raise an error
    with pytest.raises(ValueError):
        saddle.get_saddle(clr, expected, track, "unknown")

    # TODO: tests after adding input agreement, e.g.
    # asserting saddle.get_saddle(clr, cis-type-expected, track, "trans")
    # throws an error
