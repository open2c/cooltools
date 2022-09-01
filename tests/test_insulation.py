import os.path as op

import numpy as np
import pandas as pd
from click.testing import CliRunner
from cooltools.cli import cli

from cooltools.api.insulation import (
    calculate_insulation_score,
    find_boundaries,
    insul_diamond,
    _find_insulating_boundaries_dense,
)
import cooler

def test_insulation_cli(request, tmpdir):

    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    window = 10_000_000
    out_prefix = op.join(tmpdir, "CN.insulation.tsv")
    runner = CliRunner()
    result = runner.invoke(cli, ["insulation", "-o", out_prefix, in_cool, window])
    assert result.exit_code == 1


def test_insulation_cli_nobalance(request, tmpdir):

    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    window = 10_000_000
    out_prefix = op.join(tmpdir, "CN.insulation.tsv")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "insulation",
            "-o",
            out_prefix,
            "--clr-weight-name",
            "",
            "--ignore-diags",
            1,
            in_cool,
            window,
        ],
    )
    assert result.exit_code == 1


def test_calculate_insulation_score(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)
    windows = [10_000_000, 20_000_000]

    # I. Regular insulation, check presence of columns for each window:
    insulation = calculate_insulation_score(clr, windows)
    assert {f"log2_insulation_score_{window}" for window in windows}.issubset(
        insulation.columns
    )
    assert {f"n_valid_pixels_{window}" for window in windows}.issubset(
        insulation.columns
    )

    # II. Insulation with masking bad bins
    insulation = calculate_insulation_score(clr, 10_000_000, min_dist_bad_bin=1)
    # All bins closer than 1 to bad bins are filled with np.nans:
    assert np.all(
        np.isnan(insulation.query("dist_bad_bin==0")["log2_insulation_score_10000000"])
    )
    # Some of the bins at the distance 1 (above threshold) are not np.nans:
    assert np.any(
        ~np.isnan(insulation.query("dist_bad_bin==1")["log2_insulation_score_10000000"])
    )

    # III. Insulation for separate view:
    region = pd.DataFrame(
        {"chrom": ["chr1"], "start": [0], "end": [10_000_000], "name": ["fragment01"]}
    )
    insulation = calculate_insulation_score(
        clr, 10_000_000, min_dist_bad_bin=0, view_df=region
    )
    assert len(insulation) == 10

    # IV. Insulation with string or float inputs for window sizes should work.
    calculate_insulation_score(clr, '10_000_000')
    


def test_find_boundaries(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)
    windows = [10_000_000, 20_000_000]

    # I. Regular boundaries, check presence of columns for each window:
    insulation = calculate_insulation_score(clr, windows)
    boundaries = find_boundaries(insulation)
    assert {f"boundary_strength_{window}" for window in windows}.issubset(
        boundaries.columns
    )


def test_insul_diamond(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)

    # Pixel query
    from cooltools.lib._query import CSRSelector

    nbins = len(clr.bins())
    chunksize = 10_000
    selector = CSRSelector(
        clr.open("r"), shape=(nbins, nbins), field="count", chunksize=chunksize
    )
    c0 = 0
    c1 = 10
    pixel_query = selector[c0:c1, c0:c1]

    # Define bins with different weights:
    bins = pd.DataFrame(
        [
            ["chr1", 0, 1000000, 1, 0.1, 0.01],
            ["chr1", 1000000, 2000000, 1, 0.1, 0.01],
            ["chr1", 2000000, 3000000, 1, 0.1, 0.01],
            ["chr1", 3000000, 4000000, 1, 0.1, 0.01],
            ["chr1", 4000000, 5000000, 1, 0.1, 0.01],
            ["chr1", 5000000, 6000000, 1, 0.1, 0.01],
            ["chr1", 6000000, 7000000, 1, 0.1, 0.01],
            ["chr1", 7000000, 8000000, 1, 0.1, 0.01],
            ["chr1", 8000000, 9000000, 1, 0.1, 0.01],
            ["chr1", 9000000, 10000000, 1, 0.1, 0.01],
        ],
        columns=["chrom", "start", "end", "weight", "weight_cis", "weight_trans"],
    )

    # Run insul_diamond:
    score, n_pixels, sum_balanced, sum_counts = insul_diamond(
        pixel_query,
        bins,
        window=3,
        ignore_diags=2,
        norm_by_median=False,
        clr_weight_name="weight",
    )

    assert np.allclose(sum_balanced, sum_counts)

    score, n_pixels, sum_balanced, sum_counts = insul_diamond(
        pixel_query,
        bins,
        window=3,
        ignore_diags=2,
        norm_by_median=False,
        clr_weight_name="weight_cis",
    )

    assert np.allclose(sum_balanced, 0.01 * sum_counts)


def test_insulation_sparse_vs_dense(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)
    insul_dense = _find_insulating_boundaries_dense(
        clr,
        10_000_000,
        clr_weight_name="weight",
        min_dist_bad_bin=0,
        ignore_diags=2,
    )

    insulation_sparse = calculate_insulation_score(
        clr, 10_000_000, clr_weight_name="weight", min_dist_bad_bin=0, ignore_diags=2
    )
    boundaries_sparse = find_boundaries(insulation_sparse)

    assert np.allclose(
        insul_dense["log2_insulation_score_10000000"],
        boundaries_sparse["log2_insulation_score_10000000"],
        equal_nan=True,
    )
