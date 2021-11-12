import click
import cooler

from . import cli
from .. import api 
from ..lib import common
import bioframe


@cli.command()
@click.argument("in_path", metavar="IN_PATH", type=str, nargs=1)
@click.argument("window", nargs=-1, metavar="WINDOW", type=int)
@click.option(
    "--output",
    "-o",
    help="Specify output file name to store the insulation in a tsv format.",
    type=str,
    required=False,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file containing genomic regions "
    "for which insulation scores will be calculated. Region names can "
    "be provided in a 4th column and should match regions and "
    "their names in expected."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--ignore-diags",
    help="The number of diagonals to ignore. By default, equals"
    " the number of diagonals ignored during IC balancing.",
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--min-frac-valid-pixels",
    help="The minimal fraction of valid pixels in a sliding diamond. "
    "Used to mask bins during boundary detection.",
    type=float,
    default=0.66,
    show_default=True,
)
@click.option(
    "--min-dist-bad-bin",
    help="The minimal allowed distance to a bad bin. "
    "Use to mask bins after insulation calculation and during boundary detection.",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--window-pixels",
    help="If set then the window sizes are provided in units of pixels.",
    is_flag=True,
)
@click.option(
    "--append-raw-scores",
    help="Append columns with raw scores (sum_counts, sum_balanced, n_pixels) "
    "to the output table.",
    is_flag=True,
)
@click.option("--chunksize", help="", type=int, default=20000000, show_default=True)
@click.option("--verbose", help="Report real-time progress.", is_flag=True)
@click.option(
    "--bigwig",
    help="Also save insulation tracks as a bigWig files for different window sizes"
    " with the names output.<window-size>.bw",
    is_flag=True,
    default=False,
)
def insulation(
    in_path,
    window,
    output,
    view,
    ignore_diags,
    min_frac_valid_pixels,
    min_dist_bad_bin,
    window_pixels,
    append_raw_scores,
    chunksize,
    verbose,
    bigwig
):
    """
    Calculate the diamond insulation scores and call insulating boundaries.

    IN_PATH : The paths to a .cool file with a balanced Hi-C map.

    WINDOW : The window size for the insulation score calculations.
             Multiple space-separated values can be provided.
             By default, the window size must be provided in units of bp.
             When the flag --window-pixels is set, the window sizes must
             be provided in units of pixels instead.
    """

    clr = cooler.Cooler(in_path)

    # Create view:
    cooler_view_df = common.make_cooler_view(clr)
    if view is None:
        # full chromosomes:
        view_df = cooler_view_df
    else:
        # read view_df dataframe, and verify against cooler
        view_df = common.read_viewframe(view, clr, check_sorting=True)

    # Read list with windows:
    if window_pixels:
        window = [win * clr.info["bin-size"] for win in window]

    # Calculate insulation score:
    ins_table = api.insulation.calculate_insulation_score(
        clr,
        view_df=view_df,
        window_bp=window,
        ignore_diags=ignore_diags,
        min_dist_bad_bin=min_dist_bad_bin,
        append_raw_scores=append_raw_scores,
        chunksize=chunksize,
        verbose=verbose,
    )

    # Find boundaries:
    ins_table = api.insulation.find_boundaries(
        ins_table,
        min_frac_valid_pixels=min_frac_valid_pixels,
        min_dist_bad_bin=min_dist_bad_bin,
    )

    # output to file if specified:
    if output:
        ins_table.to_csv(output, sep="\t", index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(ins_table.to_csv(sep="\t", index=False, na_rep="nan"))

    # Write the insulation track as a bigwig:
    if bigwig:
        for w in window:
            bioframe.to_bigwig(
                ins_table,
                clr.chromsizes,
                output + "." + str(w) + ".bw",
                value_field=f"log2_insulation_score_{w}",
            )