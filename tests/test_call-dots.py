import os.path as op

from click.testing import CliRunner
from cooltools.cli import cli


def test_call_dots_cli(request, tmpdir):
    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    in_exp = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.tsv")
    in_regions = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
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
            "--output-calls",
            out_dots,
            in_cool,
            in_exp,
        ],
    )
    assert result.exit_code == 0
