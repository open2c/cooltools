import multiprocess as mp
import click

import cooler

from . import cli
from .. import api


@cli.command()
@click.argument("in_path", metavar="IN_PATH", type=str, nargs=1)
@click.argument("out_path", metavar="OUT_PATH", type=str, nargs=1)
@click.option(
    "-c",
    "--count",
    help="The target number of contacts in the sample. "
    "The resulting sample size will not match it precisely. "
    "Mutually exclusive with --frac and --cis-count",
    type=int,
    default=None,
    show_default=False,
)
@click.option(
    "--cis-count",
    help="The target number of cis contacts in the sample. "
    "The resulting sample size will not match it precisely. "
    "Mutually exclusive with --count and --frac",
    type=int,
    default=None,
    show_default=False,
)
@click.option(
    "-f",
    "--frac",
    help="The target sample size as a fraction of contacts in the original dataset. "
    "Mutually exclusive with --count and --cis-count",
    type=float,
    default=None,
    show_default=False,
)
@click.option(
    "--exact",
    help="If specified, use exact sampling that guarantees the size of the output sample. "
    "Otherwise, binomial sampling will be used and the sample size will be distributed around the target value. ",
    is_flag=True,
)
@click.option(
    "--nproc",
    "-p",
    help="Number of processes to split the work between."
    "[default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
@click.option(
    "--chunksize",
    help="The number of pixels loaded and processed per step of computation.",
    type=int,
    default=int(1e7),
    show_default=True,
)
def random_sample(in_path, out_path, count, cis_count, frac, exact, nproc, chunksize):
    """
    Pick a random sample of contacts from a Hi-C map.

    IN_PATH : Input cooler path or URI.

    OUT_PATH : Output cooler path or URI.

    Specify the target sample size with either --count or --frac.

    """

    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    try:
        api.sample.sample(
            in_path,
            out_path,
            count=count,
            cis_count=cis_count,
            frac=frac,
            exact=exact,
            chunksize=chunksize,
            map_func=map_,
        )
    finally:
        if nproc > 1:
            pool.close()
