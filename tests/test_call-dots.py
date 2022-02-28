import os.path as op

from click.testing import CliRunner
from cooltools.cli import cli
import cooler
import numpy as np
from cooltools import api
from cooltools.lib.io import read_viewframe_from_file, read_expected_from_file


# test user-facing API for calling dots
def test_dots(request):
    # Note that call-dots requires ucsc named expected and view
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")

    # read data for the test:
    clr = cooler.Cooler(in_cool)
    view_df = read_viewframe_from_file(in_regions, clr, check_sorting=True)
    expected_df = read_expected_from_file(
        in_exp,
        expected_value_cols=["balanced.avg"],
        verify_view=view_df,
        verify_cooler=clr,
    )

    # generate dot-calls
    dot_calls_df = api.dotfinder.dots(
        clr,
        expected_df,
        view_df=view_df,
        kernels={
            "d": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]]),
            "v": np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]),
            "h": np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
        },
        max_loci_separation=100_000_000,
        max_nans_tolerated=1,
        n_lambda_bins=50,
        lambda_bin_fdr=0.1,
        clustering_radius=False,
        cluster_filtering=None,
        tile_size=50_000_000,
        nproc=1,
    )

    # no comparison with reference results yet
    # just checking if it runs without errors
    assert not dot_calls_df.empty


def test_call_dots_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.chromnamed.tsv")
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dots",
            "-p",
            1,
            "--tile-size",
            60_000_000,
            "--max-loci-separation",
            100_000_000,
            "--output",
            out_dots,
            in_cool,
            in_exp,
        ],
    )
    # This command should fail because viewframe interpreted from cooler does not correspond to toy_expected:
    assert result.exit_code == 1


# comment this test for now, until we swap out input data and/or allow for custom kernels

# def test_call_dots_view_cli(request, tmpdir):
#     # Note that call-dots requires ucsc named expected and view
#     in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
#     in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
#     in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
#     out_dots = op.join(tmpdir, "test.dots")

#     runner = CliRunner()
#     cmd = [
#         "dots",
#         "--view",
#         in_regions,
#         "-p",
#         1,
#         "--tile-size",
#         60_000_000,
#         "--max-loci-separation",
#         100_000_000,
#         "--output",
#         out_dots,
#         in_cool,
#         in_exp,
#     ]
#     result = runner.invoke(cli, cmd)
#     assert result.exit_code == 0
#     # make sure output is generated:
#     assert op.isfile(out_dots)
