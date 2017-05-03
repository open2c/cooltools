import click

import cooler

from . import cli
from .. import insulation

@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=click.Path(exists=True),
    nargs=1)
@click.argument(
    'window', 
    metavar="WINDOW",
    type=int,
    )
@click.option(
    '--min-dist-bad-bin', 
    help='The minimal allowed distance to a bad bin. Do not calculate insulation scores '
         'for bins having a bad bin closer than this distance.',
    type=int,
    default=2,
    show_default=True,
    )
@click.option(
    '--ignore-diags', 
    help='The number of diagonals to ignore. By default, equals'
        ' the number of diagonals ignored during IC balancing.',
    type=int,
    default=None,
    show_default=True,
    )


def diamond_insulation(cool_path, window, min_dist_bad_bin, ignore_diags):
    """
    Calculate the diamond insulation scores and call insulating boundaries.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    WINDOW : The window size for the insulation score calculations.
    """

    c = cooler.Cooler(cool_path)
    ins_table = insulation.find_insulating_boundaries(
        c, window_bp = window, min_dist_bad_bin = min_dist_bad_bin,
        ignore_diags=ignore_diags)

    print(ins_table.to_csv(sep='\t', index=False, na_rep='nan'))
