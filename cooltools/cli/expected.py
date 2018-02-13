import click

import cooler

from . import cli
from .. import expected

@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str,
    nargs=1)
@click.option(
    '--cis-trans-type',
    help="compute expected for cis or trans region"
    "of a Hi-C map.",
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True,
    )



def expected(cool_path, window, min_dist_bad_bin, ignore_diags):
    """
    Calculate either cis- or trans- expected.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    cis-trans-type

    number of cores to split the work between.

    """

    c = cooler.Cooler(cool_path)

    if cis_trans_type == 'cis':
        # expected_result = cis_expected(clr, regions, field='balanced', chunksize=1000000, 
        #                  use_dask=True, ignore_diags=2):
    elif cis_trans_type == 'trans':
        # expected_result = trans_expected(clr, chromosomes, chunksize=1000000, use_dask=False)
    else:
        # cis_trans_type could be either cis or trans ...
        raise
    # ins_table = insulation.find_insulating_boundaries(
    #     c, window_bp = window, min_dist_bad_bin = min_dist_bad_bin,
    #     ignore_diags=ignore_diags)

    print(expected_result.to_csv(sep='\t', index=True, na_rep='nan'))



