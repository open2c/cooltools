# inspired by:
# saddles.py by @nvictus
# https://github.com/nandankita/labUtilityTools
from functools import partial
import os.path as op
import sys
import pandas as pd
import numpy as np
import cooler
from .. import saddle

import click
from .util import validate_csv
from . import cli


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str
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
    "-t", "--contact-type",
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
    "-n", "--n-bins",
    help="Number of bins for digitizing track values.",
    type=int,
    default=50,
    show_default=True,
)
@click.option(
    "--quantiles",
    help="Bin the signal track into quantiles rather than by value.",
    is_flag=True,
    default=False,
)
@click.option(
    "--range",
    "range_",
    help="Low and high values used for binning genome-wide track values, e.g. "
    "if `range`=(-0.05, 0.05), `n-bins` equidistant bins would be generated. "
    "Use to prevent the extreme track values from exploding the bin range and "
    "to ensure consistent bins across several runs of `compute_saddle` command "
    "using different track files.",
    nargs=2,
    type=float,
)
@click.option(
    "--qrange",
    help="The fraction of the genome-wide range of the track values used to "
    "generate bins. E.g., if `qrange`=(0.02, 0.98) the lower bin would "
    "start at the 2nd percentile and the upper bin would end at the 98th "
    "percentile of the genome-wide signal. "
    "Use to prevent the extreme track values from exploding the bin range.",
    type=(float, float),
    default=(0.0, 1.0),
    show_default=True,
)
@click.option(
    "--weight-name",
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
    "-o", "--out-prefix",
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
    "--cmap",
    help="Name of matplotlib colormap",
    default="coolwarm",
    show_default=True
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
    "--vmax",
    help="High value of the saddleplot colorbar",
    type=float,
    default=2
)
@click.option(
    "--hist-color",
    help="Face color of histogram bar chart"
)
@click.option(
    "-v", "--verbose",
    help="Enable verbose output",
    is_flag=True,
    default=False
)
def compute_saddle(
    cool_path,
    track_path,
    expected_path,
    contact_type,
    min_dist,
    max_dist,
    n_bins,
    quantiles,
    range_,
    qrange,
    weight_name,
    strength,
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
    c = cooler.Cooler(cool_path)
    expected_path, expected_name = expected_path
    track_path, track_name = track_path

    if vmin <= 0 or vmax <= 0:
        raise ValueError(
            "vmin and vmax values are in original units irrespective "
            "of used scale, and therefore should be positive"
        )

    # read expected and make preparations for validation,
    # it's contact_type dependent:
    if contact_type == "cis":
        # that's what we expect as column names:
        expected_columns = ["chrom", "diag", "n_valid", expected_name]
        # what would become a MultiIndex:
        expected_index = ["chrom", "diag"]
        # expected dtype as a rudimentary form of validation:
        expected_dtype = {
            "chrom": np.str,
            "diag": np.int64,
            "n_valid": np.int64,
            expected_name: np.float64,
        }
        # unique list of chroms mentioned in expected_path:
        get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
        # compute # of bins by comparing matching indexes:
        get_exp_bins = lambda df, ref_chroms, _: (
            df.index.get_level_values("chrom").isin(ref_chroms).sum()
        )
    elif contact_type == "trans":
        # that's what we expect as column names:
        expected_columns = ["chrom1", "chrom2", "n_valid", expected_name]
        # what would become a MultiIndex:
        expected_index = ["chrom1", "chrom2"]
        # expected dtype as a rudimentary form of validation:
        expected_dtype = {
            "chrom1": np.str,
            "chrom2": np.str,
            "n_valid": np.int64,
            expected_name: np.float64,
        }
        # unique list of chroms mentioned in expected_path:
        get_exp_chroms = lambda df: np.union1d(
            df.index.get_level_values("chrom1").unique(),
            df.index.get_level_values("chrom2").unique(),
        )
        # no way to get bins from trans-expected, so just get the number:
        get_exp_bins = lambda _1, _2, correct_bins: correct_bins
    else:
        raise ValueError(
            "Incorrect contact_type: {}, ".format(contact_type),
            "Should have been caught by click.",
        )

    if min_dist < 0:
        min_diag = 3
    else:
        min_diag = int(np.ceil(min_dist / c.binsize))

    if max_dist >= 0:
        max_diag = int(np.floor(max_dist / c.binsize))
    else:
        max_diag = -1

    # use 'usecols' as a rudimentary form of validation,
    # and dtype. Keep 'comment' and 'verbose' - explicit,
    # as we may use them later:
    expected = pd.read_table(
        expected_path,
        usecols=expected_columns,
        index_col=expected_index,
        dtype=expected_dtype,
        comment=None,
        verbose=False,
    )

    # read bedGraph-file :
    track_columns = ["chrom", "start", "end", track_name]
    # specify dtype as a rudimentary form of validation:
    track_dtype = {
        "chrom": np.str,
        "start": np.int64,
        "end": np.int64,
        track_name: np.float64,
    }
    track = pd.read_table(
        track_path,
        usecols=track_columns,
        dtype=track_dtype,
        comment=None,
        verbose=False,
    )

    #############################################
    # CROSS-VALIDATE COOLER, EXPECTED AND TRACK:
    #############################################
    # TRACK vs COOLER:
    track_chroms = track["chrom"].unique()
    # We might want to try this eventually:
    # https://github.com/TMiguelT/PandasSchema
    # do simple column-name validation for now:
    if not set(track_chroms).issubset(c.chromnames):
        raise ValueError(
            "Chromosomes in {} must be subset of ".format(track_path)
            + "chromosomes in cooler {}".format(cool_path)
        )
    # check number of bins:
    track_bins = len(track)
    cool_bins = c.bins()[:]["chrom"].isin(track_chroms).sum()
    if not (track_bins == cool_bins):
        raise ValueError(
            "Number of bins is not matching: ",
            "{} in {}, and {} in {} for chromosomes {}".format(
                track_bins, track_path, cool_bins, cool_path, track_chroms
            ),
        )
    # EXPECTED vs TRACK:
    # validate expected a bit as well:
    expected_chroms = get_exp_chroms(expected)
    # do simple column-name validation for now:
    if not set(track_chroms).issubset(expected_chroms):
        raise ValueError(
            "Chromosomes in {} must be subset of ".format(track_path)
            + "chromosomes in expected {}".format(expected_path)
        )
    # and again bins are supposed to match up:
    # only for cis though ...
    expected_bins = get_exp_bins(expected, track_chroms, track_bins)
    if not (track_bins == expected_bins):
        raise ValueError(
            "Number of bins is not matching: ",
            "{} in {}, and {} in {} for chromosomes {}".format(
                track_bins, track_path, expected_bins, expected_path, track_chroms
            ),
        )
    #############################################
    # CROSS-VALIDATION IS COMPLETE.
    #############################################

    track = saddle.mask_bad_bins((track, track_name), (c.bins()[:], weight_name))

    if contact_type == "cis":
        getmatrix = saddle.make_cis_obsexp_fetcher(
            c, (expected, expected_name), weight_name=weight_name
        )
    elif contact_type == "trans":
        getmatrix = saddle.make_trans_obsexp_fetcher(
            c, (expected, expected_name), weight_name=weight_name
        )

    if quantiles:
        if len(range_):
            qlo, qhi = saddle.ecdf(track[track_name], range_)
        elif len(qrange):
            qlo, qhi = qrange
        else:
            qlo, qhi = 0.0, 1.0
        q_edges = np.linspace(qlo, qhi, n_bins)
        binedges = saddle.quantile(track[track_name], q_edges)
    else:
        if len(range_):
            lo, hi = range_
        elif len(qrange):
            lo, hi = saddle.quantile(track[track_name], qrange)
        else:
            lo, hi = track[track_name].min(), track[track_name].max()
        binedges = np.linspace(lo, hi, n_bins)

    digitized, hist = saddle.digitize_track(
        binedges, track=(track, track_name), regions=track_chroms
    )

    S, C = saddle.make_saddle(
        getmatrix,
        binedges,
        (digitized, track_name + ".d"),
        contact_type=contact_type,
        min_diag=min_diag,
        max_diag=max_diag,
    )

    saddledata = S / C

    to_save = dict(saddledata=saddledata, binedges=binedges, hist=hist)

    if strength:
        ratios = saddle.saddle_strength(S, C)
        ratios = ratios[1:-1]  # drop outlier bins
        to_save["saddle_strength"] = ratios

    # Save data
    np.savez(out_prefix + ".saddledump", **to_save)  # .npz auto-added
    digitized.to_csv(out_prefix + ".digitized.tsv", sep="\t", index=False)

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
        if quantiles:
            edges = q_edges
            track_label = track_name + " quantiles"
        else:
            edges = binedges
            track_label = track_name
        clabel = "(contact frequency / expected)"

        saddle.saddleplot(
            edges,
            hist,
            saddledata,
            scale=scale,
            vmin=vmin,
            vmax=vmax,
            color=color,
            title=title,
            xlabel=track_label,
            ylabel=track_label,
            clabel=clabel,
        )

        for ext in fig:
            plt.savefig(out_prefix + "." + ext, bbox_inches="tight")
