"""
Tests for cooltools.sandbox.obs_over_exp_cooler.create_obs_over_exp_cooler
and cooltools.sandbox.cli_obs_over_exp.

Uses the 10 Mb-resolution test cooler (CN.mm9.10000kb.cool, ~38K pixels)
for fast execution.
"""
import os.path as op
import numpy as np
import pandas as pd
from numpy import testing

import bioframe
import cooler
import pytest

from click.testing import CliRunner

import cooltools.api.expected
from cooltools.lib.common import make_cooler_view, assign_supports

# --- Common test parameters -------------------------------------------------
ignore_diags = 2
clr_weight_name = "weight"
chunksize = 10_000
# Use the coarse 10 Mb cooler for speed (278 bins, 38K pixels)
_COOL_FILE = "data/CN.mm9.10000kb.cool"
assumed_binsize = 10_000_000

chromsizes_file = op.join(
    op.dirname(op.realpath(__file__)),
    "data/mm9.chrom.sizes.reduced",
)
chromsizes = bioframe.read_chromsizes(chromsizes_file)
chromosomes = list(chromsizes.index)

# Build a view that splits each of the first 4 chroms in half (aligned to 10 Mb)
common_regions = []
for i in range(4):
    chrom = chromosomes[i]
    halfway_chrom = int(chromsizes[chrom] / 2)
    halfway_chrom = round(halfway_chrom / assumed_binsize) * assumed_binsize
    reg1 = (chrom, 0, halfway_chrom)
    reg2 = (chrom, halfway_chrom, chromsizes[chrom])
    common_regions.append(reg1)
    common_regions.append(reg2)

view_df = bioframe.make_viewframe(common_regions, name_style="ucsc")


# ---------------------------------------------------------------------------
# Fixtures – compute expected once per module and share across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clr(request):
    """Cooler object for the 10 Mb test dataset."""
    return cooler.Cooler(op.join(request.fspath.dirname, _COOL_FILE))


@pytest.fixture(scope="module")
def cool_path(request):
    """Absolute path to the test cooler file."""
    return op.join(request.fspath.dirname, _COOL_FILE)


@pytest.fixture(scope="module")
def cis_exp(clr):
    """Pre-computed cis expected (non-smoothed)."""
    return cooltools.api.expected.expected_cis(
        clr,
        view_df=view_df,
        intra_only=False,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
        ignore_diags=ignore_diags,
    )


@pytest.fixture(scope="module")
def cis_exp_smoothed(clr):
    """Pre-computed cis expected with smoothing + aggregation."""
    return cooltools.api.expected.expected_cis(
        clr,
        view_df=view_df,
        intra_only=False,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
        ignore_diags=ignore_diags,
        smooth=True,
        aggregate_smoothed=True,
    )


@pytest.fixture(scope="module")
def trans_exp(clr):
    """Pre-computed trans expected."""
    return cooltools.api.expected.expected_trans(
        clr,
        view_df=view_df,
        clr_weight_name=clr_weight_name,
        chunksize=chunksize,
    )


def _verify_oe_cooler(
    oe_clr,
    original_clr,
    expected_cis_df,
    expected_trans_df,
    view_df,
    expected_value_col="balanced.avg",
    atol=1e-6,
    check_trans=True,
):
    """
    Verify that the O/E cooler has correct values by manually computing O/E
    for a few region pairs and comparing with the values stored in the cooler.

    Parameters
    ----------
    oe_clr : cooler.Cooler
        The output O/E cooler to verify.
    original_clr : cooler.Cooler
        The original input cooler.
    expected_cis_df : pd.DataFrame
        Cis expected used to create the O/E cooler.
    expected_trans_df : pd.DataFrame or None
        Trans expected, or None if trans was not included.
    view_df : pd.DataFrame
        The viewframe.
    expected_value_col : str
        Column name used for expected values.
    atol : float
        Absolute tolerance for float comparisons.
    check_trans : bool
        Whether to also verify trans (inter-chromosomal) pixels.
    """
    # 1. Basic structural checks
    assert oe_clr.binsize == original_clr.binsize
    assert len(oe_clr.bins()) == len(original_clr.bins())
    assert list(oe_clr.chromnames) == list(original_clr.chromnames)

    # 2. Check that pixels have float count values (O/E ratios)
    oe_pixels = oe_clr.pixels()[:]
    assert np.issubdtype(oe_pixels["count"].dtype, np.floating)

    # 3. Verify a cis region: pick the first intra-chromosomal block in the view
    region_name = view_df.iloc[0]["name"]
    region_ucsc = (
        view_df.iloc[0].iloc[0],
        view_df.iloc[0].iloc[1],
        view_df.iloc[0].iloc[2],
    )

    # Get balanced matrix from original
    orig_matrix = original_clr.matrix(balance=clr_weight_name).fetch(region_ucsc)
    oe_matrix = oe_clr.matrix(balance=False).fetch(region_ucsc)

    # Get expected for this region
    mask = (
        (expected_cis_df["region1"] == region_name)
        & (expected_cis_df["region2"] == region_name)
    )
    region_expected = expected_cis_df.loc[mask].set_index("dist")[expected_value_col]

    # For each element in the matrix, verify O/E = balanced / expected[dist]
    n = orig_matrix.shape[0]
    for d in range(ignore_diags, min(n, 10)):  # check a few diagonals
        exp_val = region_expected.get(d, np.nan)
        if np.isnan(exp_val) or exp_val == 0:
            continue
        # Get the diagonal from both matrices
        orig_diag = np.diag(orig_matrix, d)
        oe_diag = np.diag(oe_matrix, d)
        # Where original has valid values, O/E should be balanced / expected
        valid = ~np.isnan(orig_diag) & ~np.isnan(oe_diag)
        if valid.any():
            expected_oe = orig_diag[valid] / exp_val
            testing.assert_allclose(
                oe_diag[valid],
                expected_oe,
                atol=atol,
                rtol=1e-4,
                err_msg=f"O/E mismatch at diagonal {d} for region {region_name}",
            )

    # 4. Verify O/E values: finite values should be non-negative
    # NaN values are expected (bad bins, ignored diagonals, etc.)
    oe_all = oe_clr.pixels()[:]["count"]
    finite_mask = np.isfinite(oe_all)
    assert finite_mask.sum() > 0, "No finite O/E values found"
    assert np.all(oe_all[finite_mask] >= 0), "Found negative O/E values"

    # 5. If checking trans, verify that trans pixels exist (or don't)
    if check_trans and expected_trans_df is not None:
        # There should be some inter-chromosomal pixels in the O/E cooler
        oe_pix = oe_clr.pixels()[:]
        bins = oe_clr.bins()[:]
        annotated = cooler.annotate(oe_pix, bins)
        trans_mask = annotated["chrom1"] != annotated["chrom2"]
        assert trans_mask.sum() > 0, "Expected trans pixels in O/E cooler but found none"
        # Some trans O/E values should be finite
        trans_vals = oe_pix.loc[trans_mask, "count"]
        assert np.isfinite(trans_vals).sum() > 0, (
            "Trans pixels exist but none have finite O/E values"
        )
    elif not check_trans:
        # With no-trans, trans pixels may still be present (streamed from
        # the cooler) but should all have NaN O/E values since there was
        # no trans expected to divide by.
        oe_pix = oe_clr.pixels()[:]
        bins = oe_clr.bins()[:]
        annotated = cooler.annotate(oe_pix, bins)
        trans_mask = annotated["chrom1"] != annotated["chrom2"]
        if trans_mask.sum() > 0:
            trans_vals = oe_pix.loc[trans_mask, "count"]
            assert np.all(np.isnan(trans_vals)), (
                "Cis-only O/E cooler has finite trans pixel values"
            )


# ---------------------------------------------------------------------------
# Tests for create_obs_over_exp_cooler (Python API)
# ---------------------------------------------------------------------------

class TestCreateObsOverExpCooler:
    """Tests for the create_obs_over_exp_cooler function."""

    def test_basic_with_precomputed_expected(self, clr, cis_exp, trans_exp, tmp_path):
        """Test basic O/E cooler creation with pre-computed cis and trans expected."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_basic.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp,
            expected_trans_df=trans_exp,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        # Verify the output
        oe_clr = cooler.Cooler(out_path)
        _verify_oe_cooler(
            oe_clr, clr, cis_exp, trans_exp, view_df,
            expected_value_col="balanced.avg",
        )

    def test_cis_only_no_trans(self, clr, cis_exp, tmp_path):
        """Test O/E cooler with expected_trans_df=False (cis only)."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_cis_only.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp,
            expected_trans_df=False,  # skip trans
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        _verify_oe_cooler(
            oe_clr, clr, cis_exp, None, view_df,
            expected_value_col="balanced.avg",
            check_trans=False,
        )

    def test_smoothed_expected_value_col(self, clr, cis_exp_smoothed, tmp_path):
        """Test using a smoothed expected value column."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_smoothed.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp_smoothed,
            expected_trans_df=False,  # skip trans (smoothed cols don't exist there)
            view_df=view_df,
            expected_value_col="balanced.avg.smoothed.agg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        # Structural check
        assert len(oe_clr.bins()) == len(clr.bins())
        oe_pixels = oe_clr.pixels()[:]
        assert np.issubdtype(oe_pixels["count"].dtype, np.floating)
        assert len(oe_pixels) > 0

    def test_compute_expected_on_the_fly(self, clr, tmp_path):
        """Test that expected is computed on the fly when not provided."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_auto.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=None,  # compute on the fly
            expected_trans_df=None,  # compute on the fly
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
            nproc=1,
        )

        oe_clr = cooler.Cooler(out_path)
        assert len(oe_clr.bins()) == len(clr.bins())
        oe_pixels = oe_clr.pixels()[:]
        assert np.issubdtype(oe_pixels["count"].dtype, np.floating)
        assert len(oe_pixels) > 0

    def test_no_view_uses_full_chroms(self, clr, tmp_path):
        """Test that view_df=None defaults to full chromosomes."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_noview.cool")

        # Compute expected with full-chromosome view for comparison
        full_view = make_cooler_view(clr)
        full_cis_exp = cooltools.api.expected.expected_cis(
            clr,
            view_df=full_view,
            intra_only=False,
            clr_weight_name=clr_weight_name,
            chunksize=chunksize,
            ignore_diags=ignore_diags,
        )

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=full_cis_exp,
            expected_trans_df=False,
            view_df=None,  # should default to full chroms
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        assert len(oe_clr.bins()) == len(clr.bins())
        oe_pixels = oe_clr.pixels()[:]
        assert len(oe_pixels) > 0

    def test_trans_fallback_column(self, clr, cis_exp_smoothed, trans_exp, tmp_path):
        """Test that trans expected falls back when the requested column is missing."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_trans_fallback.cool")

        # Request "balanced.avg.smoothed.agg" which exists in cis but not in trans.
        # The function should fall back to "balanced.avg" for trans.
        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp_smoothed,
            expected_trans_df=trans_exp,
            view_df=view_df,
            expected_value_col="balanced.avg.smoothed.agg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        oe_pixels = oe_clr.pixels()[:]
        assert len(oe_pixels) > 0

    def test_invalid_expected_value_col_raises(self, clr, cis_exp, tmp_path):
        """Test that an invalid expected_value_col raises ValueError."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_bad_col.cool")

        with pytest.raises(ValueError, match="not found in cis-expected"):
            create_obs_over_exp_cooler(
                clr,
                out_path,
                expected_cis_df=cis_exp,
                expected_trans_df=False,
                view_df=view_df,
                expected_value_col="nonexistent_column",
                clr_weight_name=clr_weight_name,
                ignore_diags=ignore_diags,
            )

    def test_output_bins_have_no_weights(self, clr, cis_exp, tmp_path):
        """Test that output cooler bins only have chrom/start/end (no weight cols)."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_bins.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp,
            expected_trans_df=False,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        bin_cols = list(oe_clr.bins().columns)
        assert bin_cols == ["chrom", "start", "end"]

    def test_oe_values_are_around_one(self, clr, cis_exp, tmp_path):
        """Test that O/E values are centered roughly around 1."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / "oe_around_one.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp,
            expected_trans_df=False,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        oe_clr = cooler.Cooler(out_path)
        oe_vals = oe_clr.pixels()[:]["count"].values
        # Median of finite O/E should be somewhat near 1 for a well-balanced dataset.
        # NaN pixels (bad bins, trans in cis-only, ignored diags) are excluded.
        finite_vals = oe_vals[np.isfinite(oe_vals)]
        assert len(finite_vals) > 0, "No finite O/E values"
        median_oe = np.median(finite_vals)
        assert 0.01 < median_oe < 100, f"Median O/E = {median_oe}, expected near 1"

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_output_dtype(self, clr, cis_exp, tmp_path, dtype):
        """Test that output_dtype controls the storage precision of O/E values."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_path = str(tmp_path / f"oe_{dtype}.cool")

        create_obs_over_exp_cooler(
            clr,
            out_path,
            expected_cis_df=cis_exp,
            expected_trans_df=False,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
            output_dtype=dtype,
        )

        oe_clr = cooler.Cooler(out_path)
        oe_pixels = oe_clr.pixels()[:]
        assert oe_pixels["count"].dtype == np.dtype(dtype), (
            f"Expected dtype {dtype}, got {oe_pixels['count'].dtype}"
        )

    def test_parallel_matches_sequential(self, clr, cis_exp, trans_exp, tmp_path):
        """Test that nproc>1 gives identical results to nproc=1."""
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

        out_seq = str(tmp_path / "oe_seq.cool")
        out_par = str(tmp_path / "oe_par.cool")

        common_kwargs = dict(
            expected_cis_df=cis_exp,
            expected_trans_df=trans_exp,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
            output_dtype="float64",  # use float64 to avoid precision loss in comparison
        )

        create_obs_over_exp_cooler(clr, out_seq, nproc=1, **common_kwargs)
        create_obs_over_exp_cooler(clr, out_par, nproc=2, **common_kwargs)

        seq_pixels = cooler.Cooler(out_seq).pixels()[:].sort_values(
            ["bin1_id", "bin2_id"]
        ).reset_index(drop=True)
        par_pixels = cooler.Cooler(out_par).pixels()[:].sort_values(
            ["bin1_id", "bin2_id"]
        ).reset_index(drop=True)

        assert len(seq_pixels) == len(par_pixels), (
            f"Pixel count mismatch: sequential={len(seq_pixels)}, parallel={len(par_pixels)}"
        )
        testing.assert_array_equal(
            seq_pixels["bin1_id"].values, par_pixels["bin1_id"].values
        )
        testing.assert_array_equal(
            seq_pixels["bin2_id"].values, par_pixels["bin2_id"].values
        )
        testing.assert_allclose(
            seq_pixels["count"].values,
            par_pixels["count"].values,
            rtol=1e-12,
            err_msg="Parallel and sequential O/E values differ",
        )


# ---------------------------------------------------------------------------
# Tests for CLI (cooltools.sandbox.cli_obs_over_exp)
# ---------------------------------------------------------------------------

class TestObsOverExpCLI:
    """Tests for the standalone CLI tool."""

    def test_cli_with_precomputed_expected(self, clr, cis_exp, trans_exp, cool_path, tmp_path):
        """Test CLI with pre-computed cis and trans expected TSV files."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        cis_exp_path = str(tmp_path / "cis_exp.tsv")
        trans_exp_path = str(tmp_path / "trans_exp.tsv")
        view_path = str(tmp_path / "view.bed")
        out_path = str(tmp_path / "oe_cli.cool")

        cis_exp.to_csv(cis_exp_path, sep="\t", index=False)
        trans_exp.to_csv(trans_exp_path, sep="\t", index=False)
        view_df.to_csv(view_path, sep="\t", index=False, header=False)

        runner = CliRunner()
        result = runner.invoke(
            obs_over_exp,
            [
                cool_path,
                "--cis-expected", cis_exp_path,
                "--trans-expected", trans_exp_path,
                "--view", view_path,
                "--expected-value-col", "balanced.avg",
                "--clr-weight-name", clr_weight_name,
                "--ignore-diags", str(ignore_diags),
                "-o", out_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        oe_clr = cooler.Cooler(out_path)
        assert len(oe_clr.pixels()) > 0

    def test_cli_cis_only(self, clr, cis_exp, cool_path, tmp_path):
        """Test CLI with cis-expected only and --no-trans flag."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        cis_exp_path = str(tmp_path / "cis_exp.tsv")
        view_path = str(tmp_path / "view.bed")
        out_path = str(tmp_path / "oe_cis_cli.cool")

        cis_exp.to_csv(cis_exp_path, sep="\t", index=False)
        view_df.to_csv(view_path, sep="\t", index=False, header=False)

        runner = CliRunner()
        result = runner.invoke(
            obs_over_exp,
            [
                cool_path,
                "--cis-expected", cis_exp_path,
                "--no-trans",
                "--view", view_path,
                "--expected-value-col", "balanced.avg",
                "-o", out_path,
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        oe_clr = cooler.Cooler(out_path)
        # Verify trans pixels have NaN O/E values (cis-only mode)
        oe_pix = oe_clr.pixels()[:]
        bins = oe_clr.bins()[:]
        annotated = cooler.annotate(oe_pix, bins)
        trans_mask = annotated["chrom1"] != annotated["chrom2"]
        if trans_mask.sum() > 0:
            trans_vals = oe_pix.loc[trans_mask, "count"]
            assert np.all(np.isnan(trans_vals)), (
                "Cis-only O/E cooler has finite trans pixel values"
            )

    def test_cli_compute_on_fly(self, cool_path, tmp_path):
        """Test CLI computing expected on the fly (no --cis-expected)."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        view_path = str(tmp_path / "view.bed")
        out_path = str(tmp_path / "oe_auto_cli.cool")

        view_df.to_csv(view_path, sep="\t", index=False, header=False)

        runner = CliRunner()
        result = runner.invoke(
            obs_over_exp,
            [
                cool_path,
                "--view", view_path,
                "--expected-value-col", "balanced.avg",
                "-o", out_path,
                "-p", "1",
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.output}"
        oe_clr = cooler.Cooler(out_path)
        assert len(oe_clr.pixels()) > 0

    def test_cli_smooth_auto_adjusts_col(self, cool_path, tmp_path):
        """Test that --smooth auto-adjusts expected_value_col when computing on the fly."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        view_path = str(tmp_path / "view.bed")
        out_path = str(tmp_path / "oe_smooth_cli.cool")

        view_df.to_csv(view_path, sep="\t", index=False, header=False)

        runner = CliRunner()
        result = runner.invoke(
            obs_over_exp,
            [
                cool_path,
                "--view", view_path,
                "--smooth",
                "--aggregate-smoothed",
                "--no-trans",
                "-o", out_path,
                "-p", "1",
            ],
        )

        assert result.exit_code == 0, f"CLI failed with: {result.output}"

    def test_cli_help(self):
        """Test that --help works."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        runner = CliRunner()
        result = runner.invoke(obs_over_exp, ["--help"])
        assert result.exit_code == 0
        assert "obs-over-exp" in result.output.lower() or "COOL_PATH" in result.output

    def test_cli_missing_output(self, cool_path):
        """Test that CLI fails if -o is not provided."""
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp

        runner = CliRunner()
        result = runner.invoke(
            obs_over_exp,
            [cool_path],
        )

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Tests for consistency between API and CLI
# ---------------------------------------------------------------------------

class TestAPIvsCLIConsistency:
    """Verify that the Python API and the CLI produce identical results."""

    def test_api_and_cli_produce_same_cooler(self, clr, cis_exp, cool_path, tmp_path):
        """
        Run create_obs_over_exp_cooler via Python API and via CLI, then
        compare the resulting coolers pixel-by-pixel.
        """
        from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler
        from cooltools.sandbox.cli_obs_over_exp import obs_over_exp as cli_cmd

        # --- Python API ---
        api_out = str(tmp_path / "oe_api.cool")
        create_obs_over_exp_cooler(
            clr,
            api_out,
            expected_cis_df=cis_exp,
            expected_trans_df=False,
            view_df=view_df,
            expected_value_col="balanced.avg",
            clr_weight_name=clr_weight_name,
            ignore_diags=ignore_diags,
            chunksize=chunksize,
        )

        # --- CLI ---
        cis_exp_path = str(tmp_path / "cis_exp.tsv")
        view_path = str(tmp_path / "view.bed")
        cli_out = str(tmp_path / "oe_cli.cool")

        cis_exp.to_csv(cis_exp_path, sep="\t", index=False)
        view_df.to_csv(view_path, sep="\t", index=False, header=False)

        runner = CliRunner()
        result = runner.invoke(
            cli_cmd,
            [
                cool_path,
                "--cis-expected", cis_exp_path,
                "--no-trans",
                "--view", view_path,
                "--expected-value-col", "balanced.avg",
                "--clr-weight-name", clr_weight_name,
                "--ignore-diags", str(ignore_diags),
                "-o", cli_out,
            ],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # --- Compare ---
        api_clr = cooler.Cooler(api_out)
        cli_clr = cooler.Cooler(cli_out)

        api_pixels = api_clr.pixels()[:].sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
        cli_pixels = cli_clr.pixels()[:].sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)

        assert len(api_pixels) == len(cli_pixels), (
            f"Pixel count mismatch: API={len(api_pixels)}, CLI={len(cli_pixels)}"
        )
        testing.assert_array_equal(api_pixels["bin1_id"].values, cli_pixels["bin1_id"].values)
        testing.assert_array_equal(api_pixels["bin2_id"].values, cli_pixels["bin2_id"].values)
        testing.assert_allclose(
            api_pixels["count"].values,
            cli_pixels["count"].values,
            rtol=1e-10,
            err_msg="API and CLI produced different O/E values",
        )
