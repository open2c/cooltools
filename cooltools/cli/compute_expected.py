import multiprocess as mp
import pandas as pd
from itertools import combinations
import cooler
import bioframe
from .. import expected

import click
from . import cli
from . import util

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
    "--contact-type",
    "-t",
    help="compute expected for cis or trans region of a Hi-C map."
    "trans-expected is calculated for pairwise combinations of specified regions.",
    type=click.Choice(["cis", "trans"]),
    default="cis",
    show_default=True,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a 3 or 4-column BED file containing genomic regions"
    " for which expected will be calculated. Region names are stored"
    " optionally in a 4th column, otherwise UCSC notaion is generated."
    " When not specified, expected is calculated for all chromosomes."
    " Trans-expected is calculated for all pairwise combinations of regions,"
    " provided regions have to be sorted."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--balance/--no-balance",
    help="Apply balancing weights to data before calculating expected."
    "Bins masked in the balancing weights are ignored from calcualtions.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name stored in cooler.",
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
def compute_expected(
    cool_path,
    nproc,
    chunksize,
    output,
    contact_type,
    view,
    balance,
    clr_weight_name,
    ignore_diags,
):
    """
    Calculate expected Hi-C signal either for cis or for trans regions
    of chromosomal interaction map.

    When balancing weights are not applied to the data, there is no
    masking of bad bins performed.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    """

    clr = cooler.Cooler(cool_path)
    if view is None:
        # Generate viewframe from clr.chromsizes:
        view_df = bioframe.make_viewframe(
            [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]
        )
    else:
        # Make viewframe out of table:
        # Read view_df dataframe:
        try:
            view_df = bioframe.read_table(view, schema="bed4", index_col=False)
        except Exception:
            view_df = bioframe.read_table(view, schema="bed3", index_col=False)
        # Convert view dataframe to viewframe:
        try:
            view_df = bioframe.make_viewframe(view_df, check_bounds=clr.chromsizes)
        except ValueError as e:
            raise ValueError(
                "View table is incorrect, please, comply with the format. "
            ) from e

    # define transofrms - balanced and raw ('count') for now
    if balance:
        weight1 = clr_weight_name + "1"
        weight2 = clr_weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
    else:
        # no masking bad bins of any kind, when balancing is not applied
        clr_weight_name = None
        transforms = {}

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly
    try:
        if contact_type == "cis":
            result = expected.diagsum(
                clr,
                view_df,
                transforms=transforms,
                weight_name=clr_weight_name,
                bad_bins=None,
                chunksize=chunksize,
                ignore_diags=ignore_diags,
                map=map_,
            )
        elif contact_type == "trans":
            # trans-expected is calculated on an assymetric pairwise combinations
            # of regions in view_df (special-case for faster calculations):
            result = expected.blocksum_pairwise(
                clr,
                view_df=view_df,
                transforms=transforms,
                weight_name=clr_weight_name,
                bad_bins=None,
                chunksize=chunksize,
                map=map_,
            )
    finally:
        if nproc > 1:
            pool.close()

    # calculate actual averages by dividing sum by n_valid:
    result["count.avg"] = result["count.sum"] / result["n_valid"]
    for key in transforms.keys():
        result[key + ".avg"] = result[key + ".sum"] / result["n_valid"]

    # output to file if specified:
    if output:
        result.to_csv(output, sep="\t", index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(result.to_csv(sep="\t", index=False, na_rep="nan"))
