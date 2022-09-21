import pandas as pd
import numpy as np
import cooler
import bioframe
from ..api import eigdecomp
from ..lib.common import make_cooler_view
from ..lib.io import read_viewframe_from_file

import click
from .util import TabularFilePath, sniff_for_header
from . import cli


@cli.command()
@click.argument("cool_path", metavar="COOL_PATH", type=str)
@click.option(
    "--phasing-track",
    help="Phasing track for orienting and ranking eigenvectors,"
    "provided as /path/to/track::track_value_column_name.",
    type=TabularFilePath(exists=True, default_column_index=3),
    metavar="TRACK_PATH",
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file which defines which regions of the chromosomes to use"
    " (only implemented for cis contacts)."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    default=None,
    type=str,
)
@click.option(
    "--n-eigs",
    help="Number of eigenvectors to compute.",
    type=int,
    default=3,
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name. "
    "Using raw unbalanced data is not currently supported for eigenvectors.",
    type=str,
    default="weight",
    show_default=True,
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
    "-v", "--verbose", help="Enable verbose output", is_flag=True, default=False
)
@click.option(
    "-o",
    "--out-prefix",
    help="Save compartment track as a BED-like file."
    " Eigenvectors and corresponding eigenvalues are stored in"
    " out_prefix.contact_type.vecs.tsv and out_prefix.contact_type.lam.txt",
    required=True,
)
@click.option(
    "--bigwig",
    help="Also save compartment track (E1) as a bigWig file"
    " with the name out_prefix.contact_type.bw",
    is_flag=True,
    default=False,
)
def eigs_cis(
    cool_path,
    phasing_track,
    view,
    n_eigs,
    clr_weight_name,
    ignore_diags,
    verbose,
    out_prefix,
    bigwig,
):
    """
    Perform eigen value decomposition on a cooler matrix to calculate
    compartment signal by finding the eigenvector that correlates best with the
    phasing track.


    COOL_PATH : the paths to a .cool file with a balanced Hi-C map. Use the
    '::' syntax to specify a group path in a multicooler file.

    TRACK_PATH : the path to a BedGraph-like file that stores phasing track as
    track-name named column.

    BedGraph-like format assumes tab-separated columns chrom, start, stop and
    track-name.

    """
    clr = cooler.Cooler(cool_path)

    if phasing_track is not None:

        # TODO: This all needs to be refactored into a more generic tabular file parser
        # Needs to handle stdin case too.
        track_path, col = phasing_track
        buf, names = sniff_for_header(track_path)

        if names is None:
            if not isinstance(col, int):
                raise click.BadParameter(
                    "No header found. "
                    'Cannot find "{}" column without a header.'.format(col)
                )

            track_name = "ref"
            kwargs = dict(
                header=None,
                usecols=[0, 1, 2, col],
                names=["chrom", "start", "end", track_name],
            )
        else:
            if isinstance(col, int):
                try:
                    col = names[col]
                except IndexError:
                    raise click.BadParameter(
                        'Column #{} not compatible with header "{}".'.format(
                            col, ",".join(names)
                        )
                    )
            else:
                if col not in names:
                    raise click.BadParameter(
                        'Column "{}" not found in header "{}"'.format(
                            col, ",".join(names)
                        )
                    )

            track_name = col
            kwargs = dict(header="infer", usecols=["chrom", "start", "end", track_name])

        track_df = pd.read_table(
            buf,
            dtype={
                "chrom": str,
                "start": np.int64,
                "end": np.int64,
                track_name: np.float64,
            },
            comment="#",
            verbose=verbose,
            **kwargs
        )
        phasing_track = track_df

    # define view for cis compartment-calling
    # use input "view" BED file or all chromosomes mentioned in "track":
    if view is None:
        cooler_view_df = make_cooler_view(clr)
        view_df = cooler_view_df
    else:
        view_df = read_viewframe_from_file(view, clr, check_sorting=True)

    # TODO: Add check that view_df has the same bins as track
    eigvals, eigvec_table = eigdecomp.eigs_cis(
        clr=clr,
        phasing_track=phasing_track,
        view_df=view_df,
        n_eigs=n_eigs,
        clr_weight_name=clr_weight_name,
        ignore_diags=ignore_diags,
        clip_percentile=99.9,
        sort_metric=None,
    )

    # Output
    eigvals.to_csv(out_prefix + ".cis" + ".lam.txt", sep="\t", index=False)
    eigvec_table.to_csv(out_prefix + ".cis" + ".vecs.tsv", sep="\t", index=False)
    if bigwig:
        bioframe.to_bigwig(
            eigvec_table,
            clr.chromsizes,
            out_prefix + ".cis" + ".bw",
            value_field="E1",
        )
