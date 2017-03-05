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
    '--max-bad-bins', 
    help="The maximal allowed number of bad bins on each side of the sliding "
        "window.",
    type=int,
    default=2,
    show_default=True,
    )


def diamond_insulation(cool_path, window, max_bad_bins):
    """
    Calculate the diamond insulation scores and call insulating boundaries.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    WINDOW : The window size for the insulation score calculations.
    """

    c = cooler.Cooler(cool_path)
    ins_table = insulation.find_insulating_boundaries(
        c, window_bp = window, max_bad_bins = max_bad_bins)

    print(ins_table.to_csv(sep='\t', index=False, na_rep='nan'))
