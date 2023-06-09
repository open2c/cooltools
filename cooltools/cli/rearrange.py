import click

import cooler

from . import cli
from .. import api

import pandas as pd


@cli.command()
@click.argument("in_path", metavar="IN_PATH", type=str, nargs=1)
@click.argument("out_path", metavar="OUT_PATH", type=str, nargs=1)
@click.option(
    "--view",
    help="Path to a BED-like file which defines which regions of the chromosomes to use"
    " and in what order. Using --new-chrom-col and --orientation-col you can specify the"
    " new chromosome names and whether to invert each region (optional)",
    default=None,
    required=True,
    type=str,
)
@click.option(
    "--new-chrom-col",
    help="Column name in the view with new chromosome names."
    " If not provided, uses original chromosome names",
    default=None,
    type=str,
)
@click.option(
    "--orientation-col",
    help="Columns name in the view with orientations of each region (+ or -)."
    " If not provided, assumes all are forward oriented",
    default=None,
    type=str,
)
@click.option(
    "--assembly",
    help="The name of the assembly for the new cooler. If None, uses the same as in the"
    " original cooler.",
    default=None,
    type=str,
)
@click.option(
    "--chunksize",
    help="The number of pixels loaded and processed per step of computation.",
    type=int,
    default=int(1e7),
    show_default=True,
)
def rearrange(
    in_path, out_path, view, new_chrom_col, orientation_col, assembly, chunksize
):
    """Rearrange data from a cooler according to a new genomic view

    Parameters
    ----------
    IN_PATH : str
        .cool file (or URI) with data to rearrange.
    OUT_PATH : str
        .cool file (or URI) to save the rearrange data.
    view : str
        Path to a BED-like file which defines which regions of the chromosomes to use
        and in what order. Has to be a valid viewframe (columns corresponding to region
        coordinates followed by the region name), with potential additional columns.
        Must have a header with column names.
        Using --new-chrom-col and --orientation-col you can specify the new chromosome
        names and whether to invert each region (optional).
    new_chrom_col : str
        Column name in the view with new chromosome names.
        If not provided, uses original chromosome names.
    orientation_col : str
        Columns name in the view with orientations of each region (+ or -). - means the
        region will be inverted.
        If not provided, assumes all are forward oriented.
    assembly : str
        The name of the assembly for the new cooler. If None, uses the same as in the
        original cooler.
    chunksize : int
        The number of pixels loaded and processed per step of computation.
    """
    clr = cooler.Cooler(in_path)
    view_df = pd.read_table(view, header=0)
    view_df.columns = ["chrom", "start", "end", "name"] + list(view_df.columns[4:])
    if new_chrom_col is not None and new_chrom_col not in view_df.columns:
        raise ValueError(f"New chrom col {new_chrom_col} not found in view columns")
    if orientation_col is not None and orientation_col not in view_df.columns:
        raise ValueError(f"Orientation col {orientation_col} not found in view columns")
    api.rearrange.rearrange_cooler(
        clr,
        view_df,
        out_path,
        new_chrom_col=new_chrom_col,
        orientation_col=orientation_col,
        assembly=assembly,
        chunksize=chunksize,
    )
