import os.path as op

from click.testing import CliRunner
from cooltools.cli import cli


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
            "--kernel-width",
            2,
            "--kernel-peak",
            1,
            "--tile-size",
            60_000_000,
            "--max-loci-separation",
            100_000_000,
            "--out-prefix",
            out_dots,
            in_cool,
            in_exp,
        ],
    )
    # This command should fail because viewframe interpreted from cooler does not correspond to toy_expected:
    assert result.exit_code == 1


def test_call_dots_view_cli(request, tmpdir):
    # Note that call-dots requires ucsc named expected and view
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    cmd = [
        "dots",
        "--view",
        in_regions,
        "-p",
        1,
        "--kernel-width",
        2,
        "--kernel-peak",
        1,
        "--tile-size",
        60_000_000,
        "--max-loci-separation",
        100_000_000,
        "--out-prefix",
        out_dots,
        in_cool,
        in_exp,
    ]
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0
    # make sure output is generated:
    assert op.isfile(f"{out_dots}.enriched.tsv")
    assert op.isfile(f"{out_dots}.postproc.bedpe")


# TODO: Remove this test once "regions" are deprecated altogether:
def test_call_dots_regions_deprecated_cli(request, tmpdir):
    # Note that call-dots requires ucsc named expected and view
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dots",
            "--regions",
            in_regions,
            "-p",
            1,
            "--kernel-width",
            2,
            "--kernel-peak",
            1,
            "--tile-size",
            60_000_000,
            "--max-loci-separation",
            100_000_000,
            "--out-prefix",
            out_dots,
            in_cool,
            in_exp,
        ],
    )
    assert result.exit_code == 0
    # make sure output is generated:
    assert op.isfile(f"{out_dots}.enriched.tsv")
    assert op.isfile(f"{out_dots}.postproc.bedpe")
