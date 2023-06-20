import click
import cooler
import pandas as pd

from .. import api
from . import cli
from .util import sniff_for_header


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
    " If not provided and there is no column named 'new_chrom' in the view file, uses"
    " original chromosome names",
    default=None,
    type=str,
)
@click.option(
    "--orientation-col",
    help="Columns name in the view with orientations of each region (+ or -)."
    " If not providedand there is no column named 'strand' in the view file, assumes"
    " all are forward oriented",
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
@click.option(
    "--mode",
    help="(w)rite or (a)ppend to the output file (default: w)",
    default="w",
    type=click.Choice(["w", "a"], case_sensitive=False),
)
def rearrange(
    in_path, out_path, view, new_chrom_col, orientation_col, assembly, chunksize, mode
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
        Using --new-chrom-col and --orientation-col you can specify the new chromosome
        names and whether to invert each region (optional).
        If has no header with column names, assumes the `new-chrom-col` is the fifth
        column and `--orientation-col` is the sixth, if they exist.
    new_chrom_col : str
        Column name in the view with new chromosome names.
        If not provided and there is no column named 'new_chrom' in the view file, uses
        original chromosome names.
    orientation_col : str
        Columns name in the view with orientations of each region (+ or -). - means the
        region will be inverted.
        If not providedand there is no column named 'strand' in the view file, assumes
        all are forward oriented.
    assembly : str
        The name of the assembly for the new cooler. If None, uses the same as in the
        original cooler.
    chunksize : int
        The number of pixels loaded and processed per step of computation.
    mode : str
        (w)rite or (a)ppend to the output file (default: w)
    """
    clr = cooler.Cooler(in_path)
    default_names = ["chrom", "start", "end", "name", "new_chrom", "strand"]
    buf, names = sniff_for_header(view)
    if names is not None:
        # Simply take column names from the file
        view_df = pd.read_table(buf, header=0, sep="\t")
    else:
        # Use default names
        # If some are missing, pandas creates them with all NaNs
        view_df = pd.read_csv(buf, names=default_names, sep="\t")
        names = view_df.columns
    # If additinal column names not provided, set them to defaults
    # If additional columns are not in the view, raise
    if new_chrom_col is None:
        new_chrom_col = "new_chrom"
    elif new_chrom_col not in view_df.columns:
        raise ValueError(f"New chrom col {new_chrom_col} not found in view columns")
    if orientation_col is None:
        orientation_col = "strand"
    elif orientation_col not in view_df.columns:
        raise ValueError(f"Orientation col {orientation_col} not found in view columns")

    # Fill NaNs in additional columns: if they were created here, will be filled with
    # default values. Allows not specifying default values in the file, i.e. only
    # regions that need to be inverted need to have "-" in orientation_col
    view_df[new_chrom_col] = view_df[new_chrom_col].fillna(view_df["chrom"])
    view_df[orientation_col] = view_df[orientation_col].fillna("+")
    api.rearrange.rearrange_cooler(
        clr,
        view_df,
        out_path,
        new_chrom_col=new_chrom_col,
        orientation_col=orientation_col,
        assembly=assembly,
        chunksize=chunksize,
        mode=mode,
    )
