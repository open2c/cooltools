import multiprocess as mp
import pandas as pd
from itertools import combinations
import cooler
import bioframe
from .. import api
from ..lib.common import make_cooler_view
from ..lib.io import read_viewframe_from_file

import click
from . import cli


@cli.command()
@click.argument("cool_path", metavar="COOL_PATH", type=str, nargs=1)
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
    "-c",
    help="Control the number of pixels handled by each worker process at a time.",
    type=int,
    default=int(10e6),
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    help="Specify output file name to store the expected in a tsv format.",
    type=str,
    required=False,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a 3 or 4-column BED file with genomic regions"
    " to calculated cis-expected on. When region names are not provided"
    " (no 4th column), UCSC-style region names are generated."
    " Cis-expected is calculated for all chromosomes, when this is not specified."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead.",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--smooth",
    help="If set, cis-expected is smoothed and result stored in an additional column"
    " e.g. balanced.avg.smoothed",
    is_flag=True,
)
@click.option(
    "--aggregate-smoothed",
    help="If set, cis-expected is averaged over all regions and then smoothed."
    " Result is stored in an additional column, e.g. balanced.avg.smoothed.agg."
    " Ignored without smoothing",
    is_flag=True,
)
@click.option(
    "--smooth-sigma",
    help="Control smoothing with the standard deviation of the smoothing Gaussian kernel,"
    " ignored without smoothing.",
    type=float,
    default=0.1,
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name stored in cooler."
    "Provide empty argument to calculate cis-expected on raw data",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "--ignore-diags",
    help="Number of diagonals to neglect for cis contact type",
    type=int,
    default=2,
    show_default=True,
)
def expected_cis(
    cool_path,
    nproc,
    chunksize,
    output,
    view,
    smooth,
    aggregate_smoothed,
    smooth_sigma,
    clr_weight_name,
    ignore_diags,
):
    """
    Calculate expected Hi-C signal for cis regions of chromosomal interaction map:
    average of interactions separated by the same genomic distance, i.e.
    are on the same diagonal on the cis-heatmap.

    When balancing weights are not applied to the data, there is no
    masking of bad bins performed.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    """

    clr = cooler.Cooler(cool_path)

    if view is None:
        # full chromosome case
        view_df = make_cooler_view(clr)
    else:
        # Read view_df dataframe, and verify against cooler
        view_df = read_viewframe_from_file(view, clr, check_sorting=True)

    result = api.expected.expected_cis(
        clr,
        view_df=view_df,
        intra_only=True,
        smooth=smooth,
        aggregate_smoothed=aggregate_smoothed,
        smooth_sigma=smooth_sigma,
        clr_weight_name=clr_weight_name if clr_weight_name else None,
        ignore_diags=ignore_diags,
        chunksize=chunksize,
        nproc=nproc,
    )

    # output to file if specified:
    if output:
        result.to_csv(output, sep="\t", index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(result.to_csv(sep="\t", index=False, na_rep="nan"))
