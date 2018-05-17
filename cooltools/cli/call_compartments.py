from functools import partial
import pandas as pd
import numpy as np
import cooler
from .. import eigdecomp

import click
from .util import validate_csv
from . import cli


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str)
@click.option(
    "--orient-track",
    type=str,
    callback=partial(validate_csv, default_column=3))
@click.option(
    '--contact-type',
    help="Type of the contacts perform eigen-value decomposition on.",
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True)
@click.option(
    '--n-eigs',
    help="Number of eigenvectors to compute.",
    type=int,
    default=3,
    show_default=True)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--out-prefix", "-o",
    help="Save compartment track as a BED-like file.",
    required=True)
def call_compartments(cool_path, orient_track, contact_type, n_eigs,
                      verbose, out_prefix):
    """
    Perform eigen value decomposition on a cooler matrix to calculate
    compartment signal by finding the eigenvector that correlates best with the
    phasing track.


    COOL_PATH : the paths to a .cool file with a balanced Hi-C map.

    TRACK_PATH : the path to a BedGraph-like file that stores phasing track as
    track-name named column.

    BedGraph-like format assumes tab-separated columns chrom, start, stop and
    track-name.

    """
    clr = cooler.Cooler(cool_path)

    if orient_track is not None:
        kwargs = {}

        track_path, arg = orient_track
        if isinstance(track_name, int):
            track_name = 'orient'
            kwargs['usecols'] = [0, 1, 2, arg]
        else:
            track_name = arg
        dtypes = {
            'chrom': np.str,
            'start': np.int64,
            'end': np.int64,
            track_name: np.float64
        }
        track_df = pd.read_table(
            track_path,
            names=["chrom", "start", "end", track_name],
            dtype=dtyes,
            comment=None,
            verbose=verbose,
            **kwargs)

        # we need to merge phasing track DataFrame with the cooler bins to get
        # a DataFrame with phasing info aligned and validated against bins inside of
        # the cooler file.
        track = pd.merge(
            left=clr.bins()[:],
            right=track_df,
            how="left",
            on=["chrom", "start", "end"],
            validate=None)
        # consider using validate ?!

        # sanity check would be to check if len(bins) becomes > than nbins ...
        # that would imply there was something in the track_df that didn't match
        # ["chrom", "start", "end"] - keys from the c.bins()[:] .
        if len(track) > len(clr.bins()):
            ValueError(
                "There is something in the {} that couldn't ".format(track_path) +
                "be merged with cooler-bins {}".format(cool_path))
    else:
        track = clr.bins()[['chrom', 'start', 'end']][:]
        track_name = None


    # define OBS/EXP getter functions,
    # it's contact_type dependent:
    if contact_type == "cis":
        eigvals, eigvec_table = eigdecomp.cooler_cis_eig(
            clr=clr,
            bins=track,
            regions=None,
            n_eigs=n_eigs,
            phasing_track_col=track_name,
            ignore_diags=None,
            clip_percentile=99.9,
            sort_metric=None)
    elif contact_type == "trans":
        eigvals, eigvec_table = eigdecomp.cooler_trans_eig(
            clr=clr,
            bins=track,
            n_eigs=n_eigs,
            partition=None,
            phasing_track_col=track_name,
            sort_metric=None)

    eigvals.to_csv(out_prefix + '.' + contact_type + '.lam.txt', sep='\t', index=False)
    eigvec_table.to_csv(out_prefix + '.' + contact_type + '.vecs.tsv', sep='\t', index=False)
