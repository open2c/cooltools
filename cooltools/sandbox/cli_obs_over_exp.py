"""
CLI tool for creating observed/expected cooler files.

This file lives in the sandbox and can be moved to cooltools/cli/
once the obs_over_exp_cooler module is promoted from sandbox to api.

Usage (standalone):
    python -m cooltools.sandbox.cli_obs_over_exp --help

Usage (as part of cooltools CLI, when registered):
    cooltools obs-over-exp COOL_PATH --cis-expected CIS_EXP.tsv -o output_oe.cool
"""
import click
import logging

logging.basicConfig(level=logging.INFO)


@click.command("obs-over-exp")
@click.argument("cool_path", metavar="COOL_PATH", type=str, nargs=1)
@click.option(
    "--cis-expected",
    help="Path to a TSV file with cis-expected in cooltools format "
    "(as produced by `cooltools expected-cis`). "
    "If not provided, cis-expected is computed from the cooler.",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.option(
    "--trans-expected",
    help="Path to a TSV file with trans-expected in cooltools format "
    "(as produced by `cooltools expected-trans`). "
    "If not provided, trans-expected is computed from the cooler.",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.option(
    "--output",
    "-o",
    help="Path to the output .cool file with O/E values.",
    type=str,
    required=True,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a 3 or 4-column BED file with genomic regions. "
    "Must match the regions used to compute expected. "
    "When not specified, full chromosomes from the cooler are used.",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.option(
    "--expected-value-col",
    help="Name of the column in the expected file(s) that contains "
    "the expected interaction values to divide by.",
    type=str,
    default="balanced.avg",
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Name of the balancing weight column in the cooler. "
    "Provide empty string to use raw (unbalanced) data.",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "--ignore-diags",
    help="Number of short-distance diagonals to ignore.",
    type=int,
    default=2,
    show_default=True,
)
@click.option(
    "--smooth",
    help="If set and expected is computed on the fly, apply smoothing "
    "to cis-expected before division.",
    is_flag=True,
)
@click.option(
    "--aggregate-smoothed",
    help="If set and expected is computed on the fly, aggregate smoothed "
    "expected over all regions before division.",
    is_flag=True,
)
@click.option(
    "--smooth-sigma",
    help="Sigma for smoothing (only used when computing expected on the fly).",
    type=float,
    default=0.1,
    show_default=True,
)
@click.option(
    "--chunksize",
    "-c",
    help="Number of pixels per chunk for streaming computation.",
    type=int,
    default=int(1e6),
    show_default=True,
)
@click.option(
    "--nproc",
    "-p",
    help="Number of processes for expected computation "
    "(only used when expected is computed on the fly).",
    type=int,
    default=4,
    show_default=True,
)
@click.option(
    "--no-trans",
    help="If set, skip trans (inter-chromosomal) pixels entirely. "
    "Only cis O/E values will be written to the output cooler. "
    "This is useful when only cis-expected is available.",
    is_flag=True,
)
def obs_over_exp(
    cool_path,
    cis_expected,
    trans_expected,
    output,
    view,
    expected_value_col,
    clr_weight_name,
    ignore_diags,
    smooth,
    aggregate_smoothed,
    smooth_sigma,
    chunksize,
    nproc,
    no_trans,
):
    """
    Divide a cooler's contact matrix by expected and save as a new cooler.

    Creates a new .cool file where pixel values are observed/expected (O/E)
    ratios. Cis (intra-chromosomal) and trans (inter-chromosomal) expected
    can be provided as pre-computed TSV files, or computed on the fly from
    the cooler.

    COOL_PATH : Path to a .cool file with a Hi-C contact map.

    Examples:

    \b
    Using pre-computed expected files:
        cooltools obs-over-exp data.cool \\
            --cis-expected cis_exp.tsv \\
            --trans-expected trans_exp.tsv \\
            --expected-value-col balanced.avg.smoothed.agg \\
            -o data_oe.cool

    \b
    Computing expected on the fly:
        cooltools obs-over-exp data.cool \\
            --smooth --aggregate-smoothed \\
            --expected-value-col balanced.avg.smoothed.agg \\
            -o data_oe.cool
    """
    import cooler
    import pandas as pd
    from cooltools.lib.common import make_cooler_view
    from cooltools.lib.io import read_viewframe_from_file, read_expected_from_file
    from cooltools.sandbox.obs_over_exp_cooler import create_obs_over_exp_cooler

    clr = cooler.Cooler(cool_path)

    # Handle empty string for clr_weight_name (means raw data)
    if clr_weight_name == "":
        clr_weight_name = None

    # ---- Load view ----
    if view is not None:
        view_df = read_viewframe_from_file(view, verify_cooler=clr, check_sorting=True)
    else:
        view_df = make_cooler_view(clr)

    # ---- Load expected files ----
    # Only validate the column(s) the user actually needs.
    # For cis: we need expected_value_col (e.g. "balanced.avg.smoothed.agg").
    # For trans: smoothed columns don't exist, so we'll try the requested col
    # first, and fall back to "balanced.avg" / "count.avg" at stitch-time.
    expected_cis_df = None
    expected_trans_df = None

    if cis_expected is not None:
        logging.info(f"Reading cis-expected from {cis_expected} ...")
        expected_cis_df = read_expected_from_file(
            cis_expected,
            contact_type="cis",
            expected_value_cols=[expected_value_col],
            verify_view=view_df,
            verify_cooler=clr,
        )

    if trans_expected is not None:
        logging.info(f"Reading trans-expected from {trans_expected} ...")
        # Trans expected typically has "balanced.avg" / "count.avg" only,
        # so validate with the base column; create_obs_over_exp_cooler will
        # handle the fallback logic.
        trans_base_col = "balanced.avg" if clr_weight_name else "count.avg"
        expected_trans_df = read_expected_from_file(
            trans_expected,
            contact_type="trans",
            expected_value_cols=[trans_base_col],
            verify_view=view_df,
            verify_cooler=clr,
        )
    elif no_trans:
        # User explicitly skips trans O/E
        expected_trans_df = False
    # else: expected_trans_df stays None => computed on the fly

    # ---- Determine the right expected_value_col ----
    # When computing on the fly and smooth is requested, auto-adjust
    # the expected_value_col to the smoothed column name
    if cis_expected is None and smooth:
        base_col = "balanced.avg" if clr_weight_name else "count.avg"
        auto_col = f"{base_col}.smoothed"
        if aggregate_smoothed:
            auto_col = f"{auto_col}.agg"
        if expected_value_col in ("balanced.avg", "count.avg"):
            logging.info(
                f"Smoothing requested; auto-adjusting expected_value_col "
                f"from '{expected_value_col}' to '{auto_col}'"
            )
            expected_value_col = auto_col

    # ---- Run ----
    create_obs_over_exp_cooler(
        clr=clr,
        out_path=output,
        expected_cis_df=expected_cis_df,
        expected_trans_df=expected_trans_df,
        view_df=view_df,
        expected_value_col=expected_value_col,
        clr_weight_name=clr_weight_name,
        ignore_diags=ignore_diags,
        smooth_cis=smooth,
        aggregate_smoothed=aggregate_smoothed,
        smooth_sigma=smooth_sigma,
        chunksize=chunksize,
        nproc=nproc,
    )

    logging.info(f"Done! O/E cooler written to: {output}")


# ---- Standalone entry point ----
# This allows running:  python -m cooltools.sandbox.cli_obs_over_exp --help
if __name__ == "__main__":
    obs_over_exp()
