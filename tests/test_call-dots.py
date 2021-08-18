import os.path as op

from click.testing import CliRunner
from cooltools.cli import cli


def test_call_dots_cli(request, tmpdir):
    # Note that call-dots requires ucsc named expected
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.ucsc_named.tsv")
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "call-dots",
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


def test_call_dots_view_cli(request, tmpdir):
    # Note that call-dots requires ucsc named expected and view
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.ucsc_named.tsv")
    in_regions = op.join(
        request.fspath.dirname, "data/CN.mm9.toy_regions.ucsc_named.bed"
    )
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "call-dots",
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
        ],
    )
    assert result.exit_code == 0
    # make sure output is generated:
    assert op.isfile(f"{out_dots}.enriched.tsv")
    assert op.isfile(f"{out_dots}.postproc.bedpe")


# TODO: Remove this test once "regions" are deprecated altogether:
def test_call_dots_regions_deprecated_cli(request, tmpdir):
    # Note that call-dots requires ucsc named expected and view
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.ucsc_named.tsv")
    in_regions = op.join(
        request.fspath.dirname, "data/CN.mm9.toy_regions.ucsc_named.bed"
    )
    out_dots = op.join(tmpdir, "test.dots")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "call-dots",
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
