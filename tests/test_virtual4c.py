import os.path as op

from click.testing import CliRunner
from cooltools.cli import cli

from cooltools.api import virtual4c
import cooler


def test_virtual4c(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)
    viewpoint = "chr1:30000000-40000000"

    v4c = virtual4c.virtual4c(clr, viewpoint)

    assert v4c.shape[0] == clr.bins()[:].shape[0]


def test_virtual4c_cli(request, tmpdir):

    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    out_prefix = op.join(tmpdir, "CN.virtual4c")
    viewpoint = "chr1:30000000-40000000"

    runner = CliRunner()
    result = runner.invoke(cli, ["virtual4c", "-o", out_prefix, in_cool, viewpoint])
    assert result.exit_code == 0


def test_virtual4c_cli_nobalance(request, tmpdir):

    in_cool = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    out_prefix = op.join(tmpdir, "CN.virtual4c")
    viewpoint = "chr1:30000000-40000000"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["virtual4c", "--clr-weight-name", "", "-o", out_prefix, in_cool, viewpoint],
    )
    assert result.exit_code == 0


def test_pooled_virtual4c(request):
    clr_path = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_path)
    viewpoint = "chr1:30000000-40000000"

    v4c = virtual4c.virtual4c(clr, viewpoint)
    pooled2_v4c = virtual4c.virtual4c(clr, viewpoint, nproc=2)
    pooled3_v4c = virtual4c.virtual4c(clr, viewpoint, nproc=3)

    assert v4c.equals(pooled2_v4c)
    assert v4c.equals(pooled3_v4c)
