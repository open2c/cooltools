from functools import partial
import pandas as pd
import numpy as np
import cooler
import bioframe
from .. import eigdecomp

import click
from .util import TabularFilePath, sniff_for_header
from . import cli


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str)
@click.option(
    "--reference-track",
    help="Reference track for orienting and ranking eigenvectors",
    type=TabularFilePath(exists=True, default_column_index=3))
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
@click.option(
    "--bigwig",
    help="Also save compartment track as a bigWig file.",
    is_flag=True,
    default=False)
def call_compartments(cool_path, reference_track, contact_type, n_eigs,
                      verbose, out_prefix, bigwig):
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

    if reference_track is not None:

        # TODO: This all needs to be refactored into a more generic tabular file parser
        # Needs to handle stdin case too.
        track_path, col = reference_track
        buf, names = sniff_for_header(track_path)

        if names is None:
            if not isinstance(col, int):
                raise click.BadParameter(
                    'No header found. '
                    'Cannot find "{}" column without a header.'.format(col))
            
            track_name = 'ref'
            kwargs = dict(
                header=None,
                usecols=[0, 1, 2, col],
                names=['chrom', 'start', 'end', track_name])
        else:
            if isinstance(col, int):
                try:
                    col = names[col]
                except IndexError:
                    raise click.BadParameter(
                        'Column #{} not compatible with header "{}".'.format(
                            col, ','.join(names)))
            else:
                if col not in names:
                    raise click.BadParameter(
                        'Column "{}" not found in header "{}"'.format(
                            col, ','.join(names)))
            
            track_name = col
            kwargs = dict(
                header='infer',
                usecols=['chrom', 'start', 'end', track_name])
        
        track_df = pd.read_table(
            buf, 
            dtype={
                'chrom': str,
                'start': np.int64,
                'end': np.int64,
                track_name: np.float64,
            },
            comment='#',
            verbose=verbose,
            **kwargs
        )

        # we need to merge phasing track DataFrame with the cooler bins to get
        # a DataFrame with phasing info aligned and validated against bins inside of
        # the cooler file.
        track = pd.merge(
            left=clr.bins()[:],
            right=track_df,
            how="left",
            on=["chrom", "start", "end"])

        # sanity check would be to check if len(bins) becomes > than nbins ...
        # that would imply there was something in the track_df that didn't match
        # ["chrom", "start", "end"] - keys from the c.bins()[:] .
        if len(track) > len(clr.bins()):
            ValueError(
                "There is something in the {} that ".format(track_path) +
                "couldn't be merged with cooler-bins {}".format(cool_path))
    else:
        track = clr.bins()[['chrom', 'start', 'end']][:]
        track_name = None

    # it's contact_type dependent:
    if contact_type == "cis":
        eigvals, eigvec_table = eigdecomp.cooler_cis_eig(
            clr=clr,
            bins=track,
            regions=None,
            n_eigs=n_eigs,
            phasing_track_col=track_name,
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

    # Output
    eigvals.to_csv(out_prefix + '.' + contact_type + '.lam.txt', sep='\t', index=False)
    eigvec_table.to_csv(out_prefix + '.' + contact_type + '.vecs.tsv', sep='\t', index=False)
    if bigwig:
        bioframe.to_bigwig(
            eigvec_table, 
            clr.chromsizes, 
            out_prefix + '.' + contact_type + '.bw', 
            value_field='E1')
