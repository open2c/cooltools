# inspired by:
# saddles.py by @nvictus
# https://github.com/nandankita/labUtilityTools
from functools import partial
import os.path as op
import sys
import pandas as pd
import numpy as np
import cooler
import bioframe
from .. import api

import click
from .util import validate_csv

from ..lib.common import make_cooler_view, mask_cooler_bad_bins, align_track_with_cooler
from ..lib.io import read_viewframe_from_file, read_expected_from_file
from ..lib.checks import is_track

from . import util
from . import cli


@cli.command()
@click.argument(
    "cool_path", metavar="COOL_PATH", type=str
)  # click.Path(exists=True, dir_okay=False),)
@click.argument(
    "track_path",
    metavar="TRACK_PATH",
    type=str,
    callback=partial(validate_csv, default_column="E1"),
)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=str,
    callback=partial(validate_csv, default_column="balanced.avg"),
)
@click.option(
    "-t",
    "--contact-type",
    help="Type of the contacts to aggregate",
    type=click.Choice(["cis", "trans"]),
    default="cis",
    show_default=True,
)
@click.option(
    "--min-dist",
    help="Minimal distance between bins to consider, bp. If negative, removes"
    "the first two diagonals of the data. Ignored with --contact-type trans.",
    type=int,
    default=-1,
    show_default=True,
)
@click.option(
    "--max-dist",
    help="Maximal distance between bins to consider, bp. Ignored, if negative."
    " Ignored with --contact-type trans.",
    type=int,
    default=-1,
    show_default=True,
)
@click.option(
    "-n",
    "--n-bins",
    help="Number of bins for digitizing track values.",
    type=int,
    default=50,
    show_default=True,
)
@click.option(
    "--vrange",
    "vrange",
    help="Low and high values used for binning genome-wide track values, e.g. "
    "if `range`=(-0.05, 0.05), `n-bins` equidistant bins would be generated. "
    "Use to prevent extreme track values from exploding the bin range and "
    "to ensure consistent bins across several runs of `compute_saddle` command "
    "using different track files.",
    type=(float, float),
    default=(None, None),
    nargs=2,
)
@click.option(
    "--qrange",
    help="Low and high values used for quantile bins of genome-wide track values,"
    "e.g. if `qrange`=(0.02, 0.98) the lower bin would "
    "start at the 2nd percentile and the upper bin would end at the 98th "
    "percentile of the genome-wide signal. "
    "Use to prevent the extreme track values from exploding the bin range.",
    type=(float, float),
    default=(None, None),
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name.",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "--strength/--no-strength",
    help="Compute and save compartment 'saddle strength' profile",
    is_flag=True,
    default=False,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file containing genomic regions "
    "for which saddleplot will be calculated. Region names can "
    "be provided in a 4th column and should match regions and "
    "their names in expected."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "-o",
    "--out-prefix",
    help="Dump 'saddledata', 'binedges' and 'hist' arrays in a numpy-specific "
    ".npz container. Use numpy.load to load these arrays into a "
    "dict-like object. The digitized signal values are saved to a "
    "bedGraph-style TSV.",
    required=True,
)
@click.option(
    "--fig",
    type=click.Choice(["png", "jpg", "svg", "pdf", "ps", "eps"]),
    multiple=True,
    help="Generate a figure and save to a file of the specified format. "
    "If not specified - no image is generated. Repeat for multiple "
    "output formats.",
)
@click.option(
    "--scale",
    help="Value scale for the heatmap",
    type=click.Choice(["linear", "log"]),
    default="log",
    show_default=True,
)
@click.option(
    "--cmap", help="Name of matplotlib colormap", default="coolwarm", show_default=True
)
@click.option(
    "--vmin",
    help="Low value of the saddleplot colorbar. "
    "Note: value in original units irrespective of used scale, "
    "and therefore should be positive for both vmin and vmax.",
    type=float,
    default=0.5,
)
@click.option(
    "--vmax", help="High value of the saddleplot colorbar", type=float, default=2
)
@click.option("--hist-color", help="Face color of histogram bar chart")
@click.option(
    "-v", "--verbose", help="Enable verbose output", is_flag=True, default=False
)
def saddle(
    cool_path,
    track_path,
    expected_path,
    contact_type,
    min_dist,
    max_dist,
    n_bins,
    vrange,
    qrange,
    clr_weight_name,
    strength,
    view,
    out_prefix,
    fig,
    scale,
    cmap,
    vmin,
    vmax,
    hist_color,
    verbose,
):
    """
    Calculate saddle statistics and generate saddle plots for an arbitrary
    signal track on the genomic bins of a contact matrix.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map. Use the
    '::' syntax to specify a group path in a multicooler file.

    TRACK_PATH : The path to bedGraph-like file with a binned compartment track
    (eigenvector), including a header. Use the '::' syntax to specify a column
    name.

    EXPECTED_PATH : The paths to a tsv-like file with expected signal,
    including a header. Use the '::' syntax to specify a column name.

    Analysis will be performed for chromosomes referred to in TRACK_PATH, and
    therefore these chromosomes must be a subset of chromosomes referred to in
    COOL_PATH and EXPECTED_PATH.

    COOL_PATH, TRACK_PATH and EXPECTED_PATH must be binned at the same
    resolution (expect for  EXPECTED_PATH in case of trans contact type).

    EXPECTED_PATH must contain at least the following columns for cis contacts:
    'chrom', 'diag', 'n_valid', value_name and the following columns for trans
    contacts: 'chrom1', 'chrom2', 'n_valid', value_name value_name is controlled
    using options. Header must be present in a file.

    """
    #### Read inputs: ####
    clr = cooler.Cooler(cool_path)

    expected_path, expected_value_col = expected_path
    track_path, track_name = track_path

    #### Read track: ####
    # read bedGraph-file :
    track_columns = ["chrom", "start", "end", track_name]
    # specify dtype as a rudimentary form of validation:
    track_dtype = {
        "chrom": np.str_,
        "start": np.int64,
        "end": np.int64,
        track_name: np.float64,
    }
    track = pd.read_table(
        track_path,
        usecols=track_columns,
        dtype=track_dtype,
        comment=None,
        verbose=verbose,
    )

    #### Generate viewframes ####
    # 1:cooler_view_df. Generate viewframe from clr.chromsizes:
    cooler_view_df = make_cooler_view(clr)

    # 2:view_df. Define global view for calculating calling dots
    # use input "view" BED file or all chromosomes :
    if view is None:
        view_df = cooler_view_df
    else:
        view_df = read_viewframe_from_file(view, clr, check_sorting=True)

    # 3:track_view_df. Generate viewframe from track table:
    track_view_df = bioframe.make_viewframe(
        [
            (group.chrom.iloc[0], np.nanmin(group.start), np.nanmax(group.end))
            for i, group in track.reset_index().groupby("chrom")
        ]
    )

    #### Read expected: ####

    expected_summary_cols = [
        expected_value_col,
    ]

    expected = read_expected_from_file(
        expected_path,
        contact_type=contact_type,
        expected_value_cols=expected_summary_cols,
        verify_view=view_df,
        verify_cooler=clr,
    )

    if min_dist < 0:
        min_diag = 3
    else:
        min_diag = int(np.ceil(min_dist / clr.binsize))

    if max_dist >= 0:
        max_diag = int(np.floor(max_dist / clr.binsize))
    else:
        max_diag = -1

    if clr_weight_name:
        track = mask_cooler_bad_bins(
            (track, track_name), (clr.bins()[:], clr_weight_name)
        )

    if vrange[0] is None:
        vrange = None
    if qrange[0] is None:
        qrange = None
    if (qrange is not None) and (vrange is not None):
        raise ValueError("only one of vrange or qrange can be supplied")

    # digitize outside of saddle so that we have binedges to save below
    track = align_track_with_cooler(
        track,
        clr,
        view_df=view_df,
        clr_weight_name=clr_weight_name,
        mask_clr_bad_bins=True,
        drop_track_na=False,
    )
    digitized_track, binedges = api.saddle.digitize(
        track.iloc[:, :4],
        n_bins,
        vrange=vrange,
        qrange=qrange,
        digitized_suffix=".d",
    )

    S, C = api.saddle.saddle(
        clr,
        expected,
        digitized_track,
        contact_type,
        None,
        vrange=None,
        qrange=None,
        view_df=view_df,
        clr_weight_name=clr_weight_name,
        expected_value_col=expected_value_col,
        view_name_col="name",
        min_diag=min_diag,
        max_diag=max_diag,
        verbose=verbose,
    )
    saddledata = S / C

    to_save = dict(
        saddledata=saddledata,
        binedges=binedges,
        digitized=digitized_track,
        saddlecounts=C,
    )

    if strength:
        ratios = api.saddle.saddle_strength(S, C)
        ratios = ratios[1:-1]  # drop outlier bins
        to_save["saddle_strength"] = ratios

    # Save data
    np.savez(out_prefix + ".saddledump", **to_save)  # .npz auto-added
    digitized_track.to_csv(out_prefix + ".digitized.tsv", sep="\t", index=False)

    # Generate figure
    if len(fig):
        try:
            import matplotlib as mpl

            mpl.use("Agg")  # savefig only for now:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Install matplotlib to use ", file=sys.stderr)
            sys.exit(1)

        if hist_color is None:
            color = (
                0.41568627450980394,
                0.8,
                0.39215686274509803,
            )  # sns.color_palette('muted')[2]
        else:
            color = mpl.colors.colorConverter.to_rgb(hist_color)
        title = op.basename(cool_path) + " ({})".format(contact_type)

        if qrange is not None:
            track_label = track_name + " quantiles"
        else:
            track_label = track_name

        clabel = "(contact frequency / expected)"

        api.saddle.saddleplot(
            track,
            saddledata,
            n_bins,
            vrange=vrange,
            qrange=qrange,
            scale=scale,
            vmin=vmin,
            vmax=vmax,
            color=color,
            title=title,
            xlabel=track_label,
            ylabel=track_label,
            clabel=clabel,
            cmap=cmap,
        )

        for ext in fig:
            plt.savefig(out_prefix + "." + ext, bbox_inches="tight")
