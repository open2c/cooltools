import click

import cooler

from . import cli
from .. import sample

@cli.command()
@click.argument(
    "in_path",
    metavar="IN_PATH",
    type=str,
    nargs=1)

@click.argument(
    "out_path",
    metavar="OUT_PATH",
    type=str,
    nargs=1)


@click.option(
    '-c',
    '--count',
    help='The target number of contacts in the sample. '
         'The resulting sample size will not match it precisely. '
         'Mutually exclusive with --frac',
    type=int,
    default=None,
    show_default=False)


@click.option(
    '-f',
    '--frac',
    help='The target sample size as a fraction of contacts in the original dataset. '
          'Mutually exclusive with --count',
    type=float,
    default=None,
    show_default=False)


@click.option(
    '--chunksize',
    help='The number of pixels loaded and processed per step of computation.',
    type=int,
    default=int(1e7),
    show_default=True)


def random_sample(in_path, out_path, count, frac, chunksize):
    """
    Pick a random sample of contacts from a Hi-C map, w/o replacement.

    IN_PATH : Input cooler path or URI.
    
    OUT_PATH : Output cooler path or URI.
    
    Specify the target sample size with either --target-count or --target-frac.

    """
    
    sample.sample_cooler(in_path, out_path, count=count, 
                     frac=frac, chunksize=chunksize, map_func=map)
    
