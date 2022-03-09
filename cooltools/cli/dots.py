import os.path as op
from functools import partial
import pandas as pd
import numpy as np
import cooler
import bioframe
import logging

import click
from . import cli
from .. import api

from ..lib.common import make_cooler_view, assign_regions
from ..lib.io import read_viewframe_from_file, read_expected_from_file

from .util import validate_csv

logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str,
    nargs=1,
)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=str,
    callback=partial(validate_csv, default_column="balanced.avg"),
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file with the definition of viewframe (regions)"
    " used in the calculation of EXPECTED_PATH. Dot-calling will be"
    " performed for these regions independently e.g. chromosome arms."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=False, dir_okay=False),
    default=None,
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use cooler balancing weight with this name.",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "-p",
    "--nproc",
    help="Number of processes to split the work between."
    " [default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
@click.option(
    "--max-loci-separation",
    help="Limit loci separation for dot-calling, i.e., do not call dots for"
    " loci that are further than max_loci_separation basepair apart."
    " 2-20MB is reasonable and would capture most of CTCF-dots.",
    type=int,
    default=2000000,
    show_default=True,
)
@click.option(
    "--max-nans-tolerated",
    help="Maximum number of NaNs tolerated in a footprint of every used filter."
    " Must be controlled with caution, as large max-nans-tolerated, might lead to"
    ' pixels scored in the padding area of the tiles to "penetrate" to the list'
    " of scored pixels for the statistical testing. [max-nans-tolerated <= 2*w ]",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--tile-size",
    help="Tile size for the Hi-C heatmap tiling."
    " Typically on order of several mega-bases, and <= max_loci_separation.",
    type=int,
    default=6000000,
    show_default=True,
)
@click.option(
    "--num-lambda-bins",
    help="Number of log-spaced bins to divide your adjusted expected"
    " between. Same as HiCCUPS_W1_MAX_INDX (40) in the original HiCCUPS.",
    type=int,
    default=45,
    show_default=True,
)
@click.option(
    "--fdr",
    help="False discovery rate (FDR) to control in the multiple"
    " hypothesis testing BH-FDR procedure.",
    type=float,
    default=0.02,
    show_default=True,
)
@click.option(
    "--clustering-radius",
    help="Radius for clustering dots that have been called too close to each other."
    "Typically on order of 40 kilo-bases, and >= binsize.",
    type=int,
    default=39000,
    show_default=True,
)
@click.option(
    "-v", "--verbose", help="Enable verbose output", is_flag=True, default=False
)
@click.option(
    "-o",
    "--output",
    help="Specify output file name to store called dots in a BEDPE-like format",
    type=str,
    required=True,
)
def dots(
    cool_path,
    expected_path,
    view,
    clr_weight_name,
    nproc,
    max_loci_separation,
    max_nans_tolerated,
    tile_size,
    num_lambda_bins,
    fdr,
    clustering_radius,
    verbose,
    output,
):
    """
    Call dots on a Hi-C heatmap that are not larger than max_loci_separation.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    EXPECTED_PATH : The paths to a tsv-like file with expected signal,
    including a header. Use the '::' syntax to specify a column name.

    Analysis will be performed for chromosomes referred to in EXPECTED_PATH, and
    therefore these chromosomes must be a subset of chromosomes referred to in
    COOL_PATH. Also chromosomes refered to in EXPECTED_PATH must be non-trivial,
    i.e., contain not-NaN signal. Thus, make sure to prune your EXPECTED_PATH
    before applying this script.

    COOL_PATH and EXPECTED_PATH must be binned at the same resolution.

    EXPECTED_PATH must contain at least the following columns for cis contacts:
    'region1/2', 'dist', 'n_valid', value_name. value_name is controlled using
    options. Header must be present in a file.

    """
    clr = cooler.Cooler(cool_path)
    expected_path, expected_value_col = expected_path

    # Either use view from file or all chromosomes in the provided cooler
    if view is None:
        view_df = make_cooler_view(clr)
    else:
        view_df = read_viewframe_from_file(view, clr, check_sorting=True)

    expected = read_expected_from_file(
        expected_path,
        contact_type="cis",
        expected_value_cols=[expected_value_col],
        verify_view=view_df,
        verify_cooler=clr,
    )

    dot_calls_df = api.dotfinder.dots(
        clr,
        expected,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        view_df=view_df,
        kernels=None,  # engaging default HiCCUPS kernels
        max_loci_separation=max_loci_separation,
        max_nans_tolerated=max_nans_tolerated,  # test if this has desired behavior
        n_lambda_bins=num_lambda_bins,  # update this eventually
        lambda_bin_fdr=fdr,
        clustering_radius=clustering_radius,
        cluster_filtering=None,
        tile_size=tile_size,
        nproc=nproc,
    )

    # output results in a file, when specified
    if output:
        dot_calls_df.to_csv(output, sep="\t", header=True, index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(
            dot_calls_df.to_csv(
                output, sep="\t", header=True, index=False, na_rep="nan"
            )
        )
