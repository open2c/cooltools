import click
import cooler

from . import cli
from .. import insulation


@cli.command()
@click.argument(
    "in_path",
    metavar="IN_PATH",
    type=str,
    nargs=1
)
@click.argument(
    "window",
    nargs=-1,
    metavar="WINDOW",
    type=int
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
    "Used to mask bins during boundary detection.",
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
@click.option(
    "--chunksize",
    help="",
    type=int,
    default=20000000,
    show_default=True
)
@click.option(
    "--verbose",
    help="Report real-time progress.",
    is_flag=True
)
def diamond_insulation(
    in_path,
    window,
    ignore_diags,
    min_frac_valid_pixels,
    min_dist_bad_bin,
    window_pixels,
    append_raw_scores,
    chunksize,
    verbose,
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
    if window_pixels:
        window = [win * clr.info["bin-size"] for win in window]

    ins_table = insulation.calculate_insulation_score(
        clr,
        window_bp=window,
        ignore_diags=ignore_diags,
        append_raw_scores=append_raw_scores,
        chunksize=chunksize,
        verbose=verbose,
    )

    ins_table = insulation.find_boundaries(
        ins_table,
        min_frac_valid_pixels=min_frac_valid_pixels,
        min_dist_bad_bin=min_dist_bad_bin,
    )

    print(ins_table.to_csv(sep="\t", index=False, na_rep="nan"))
