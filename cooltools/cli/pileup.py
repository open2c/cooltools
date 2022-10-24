import pandas as pd
import numpy as np
import cooler
import bioframe

from .. import api

from ..lib.common import make_cooler_view
from ..lib.io import read_viewframe_from_file, read_expected_from_file

import click
from functools import partial
from . import cli
from .util import sniff_for_header, validate_csv
import h5py


@cli.command()
@click.argument("cool_path", metavar="COOL_PATH", type=str)
@click.argument("features", metavar="FEATURES_PATH", type=str)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file which defines which regions of the chromosomes to use. "
    " Required if EXPECTED_PATH is provided"
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    default=None,
    type=str,
)
@click.option(
    "--expected",
    help="Path to the expected table. If provided, outputs OOE pileup. "
    " if not provided, outputs regular pileup. ",
    type=str,
    default=None,
    callback=partial(validate_csv, default_column="balanced.avg"),
)
@click.option(
    "--flank",
    help="Size of flanks.",
    type=int,
    default=100000,
    show_default=True,
)
@click.option(
    "--features-format",
    help="Input features format.",
    default="auto",
    type=click.Choice(["auto", "BED", "BEDPE"], case_sensitive=False),
)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name.",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "-o",
    "--out",
    help="Save output pileup as NPZ/HDF5 file.",
    required=True,
)
@click.option(
    "--out-format",
    help="Type of output.",
    default="NPZ",
    type=click.Choice(["NPZ", "HDF5"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--store-snips",
    help="Flag indicating whether snips should be stored. ",
    is_flag=True,
    default=False,
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
    "--ignore-diags",
    help="The number of diagonals to ignore. By default, equals"
    " the number of diagonals ignored during IC balancing.",
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--aggregate",
    help="Function for calculating aggregate signal. ",
    default="none",
    type=click.Choice(
        ["none", "mean", "median", "std", "min", "max"], case_sensitive=False
    ),
    show_default=True,
)
@click.option(
    "-v", "--verbose", help="Enable verbose output", is_flag=True, default=False
)
def pileup(
    cool_path,
    features,
    view,
    expected,
    flank,
    features_format,
    clr_weight_name,
    out,
    out_format,
    store_snips,
    nproc,
    ignore_diags,
    aggregate,
    verbose,
):
    """
    Perform retrieval of the snippets from .cool file.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map. Use the
    '::' syntax to specify a group path in a multicooler file.

    FEATURES_PATH : the path to a BED or BEDPE-like file that contains features for snipping windows.
    If BED, then the features are on-diagonal. If BEDPE, then the features
    can be off-diagonal (but not in trans or between different regions in the view).

    """

    clr = cooler.Cooler(cool_path)
    cooler_view_df = make_cooler_view(clr)

    #### Read the features:
    buf, names = sniff_for_header(features)
    if features_format.lower() == "bedpe":
        default_cols = [0, 1, 2, 3, 4, 5]
        bedpe_cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        dtypes = {
            "chrom1": str,
            "start1": np.int64,
            "end1": np.int64,
            "chrom2": str,
            "start2": np.int64,
            "end2": np.int64,
        }
        if names is None:
            kwargs = dict(
                header=None,
                usecols=default_cols,
                dtype=dtypes,
                names=bedpe_cols,
            )
        else:
            if len(names) < 6:
                raise ValueError("Too few columns for BEDPE")
            kwargs = dict(header="infer", usecols=bedpe_cols, dtype=dtypes)

    elif features_format.lower() == "bed":
        default_cols = [0, 1, 2]
        bed_cols = ["chrom", "start", "end"]
        dtypes = {"chrom": str, "start": np.int64, "end": np.int64}
        if names is None:
            kwargs = dict(
                header=None,
                usecols=default_cols, 
                names=bed_cols,
                dtype=dtypes
            )
        else:
            if len(names) < 3:
                raise ValueError("Too few columns for BED")
            kwargs = dict(header="infer", usecols=bed_cols, dtype=dtypes)

    else:
        raise NotImplementedError(
            "Automatic detection of features format is not implemented yet. "
            "Please provide BED or BEDPE as --features-format"
        )

    features_df = pd.read_table(
        buf, comment="#", verbose=verbose, **kwargs
    )

    ###### Define view
    if view is None:
        # full chromosome case
        view_df = cooler_view_df
    else:
        # Read view_df dataframe, and verify against cooler
        view_df = read_viewframe_from_file(view, clr, check_sorting=True)

    # make sure feature are compatible with the view_df
    # TODO: uncomment when bioframe issue is resolved https://github.com/open2c/bioframe/issues/126
    # if features_format.lower() == "bed":
    #     if not bioframe.is_contained(features_df, view_df, cols=bed_cols):
    #         raise ValueError("Features are not contained in view bounds")
    # else:
    #     if not bioframe.is_contained(
    #         features_df, view_df, cols=bedpe_cols[:3]
    #     ) or not bioframe.is_contained(features_df, view_df, cols=bedpe_cols[3:]):
    #         raise ValueError("Features are not contained in view bounds")

    ##### Read expected, should be cis-expected:
    if expected is None:
        expected_value_col = None
    else:
        expected_path, expected_value_col = expected
        expected_value_cols = [
            expected_value_col,
        ]
        expected = read_expected_from_file(
            expected_path,
            contact_type="cis",
            expected_value_cols=expected_value_cols,
            verify_view=view_df,
            verify_cooler=clr,
        )

    ##### Create the pileup:
    stack = api.snipping.pileup(
        clr,
        features_df,
        view_df=view_df,
        expected_df=expected,
        expected_value_col=expected_value_col,
        flank=flank,
        min_diag=ignore_diags,
        clr_weight_name=clr_weight_name,
        nproc=nproc,
    )

    ##### Aggregate the signal:
    aggregate = aggregate.lower()
    if aggregate is None or aggregate == "mean" or aggregate == "none":
        agg_func = np.nanmean
    elif aggregate == "median":
        agg_func = np.nanmedian
    elif aggregate == "min":
        agg_func = np.nanmin
    elif aggregate == "max":
        agg_func = np.nanmax
    elif aggregate == "std":
        agg_func = np.nanstd
    else:
        raise ValueError(
            f"Aggregation mode {aggregate} not supported. Please use mean/median/min/max/std."
        )

    pileup = agg_func(stack, axis=2)

    ##### Store the data as NPZ file:
    if out_format.lower() == "npz":
        if store_snips:
            np.savez(out, pileup=pileup)
        else:
            np.savez(out, pileup=pileup, stack=stack)
    elif out_format.lower() == "hdf5":
        h5 = h5py.File(out, "w")
        h5.create_dataset("pileup", data=pileup)
        if store_snips:
            h5.create_dataset("stack", data=stack)
