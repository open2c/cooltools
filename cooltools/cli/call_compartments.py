import click

import cooler

from . import cli
from .. import eigdecomp

import pandas as pd
import numpy as np



@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    # type=click.Path(exists=True, dir_okay=False),
    type=str,
    nargs=1)
#     track : pandas.DataFrame
#         bedGraph-like data frame ('chrom', 'start', 'end', <name>).
@click.argument(
    "track_path",
    metavar="TRACK_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
# # options ...
# phasing_name
@click.option(
    '--track-name',
    help="Name of phasing track column"
         " in the TRACK_PATH",
    default='gc',
    show_default=True,
    type=str)
@click.option(
    '--contact-type',
    help="Type of the contacts"
         " perform eigen-value"
         " decomposition on.",
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--output",
    help="Save compartment track"
         " as a BED-like file.",
    type=str)


def call_compartments(
            cool_path,
            track_path,
            track_name,
            contact_type,
            verbose,
            output):
    """

    Perform eigen value decomposition on
    a cooler matrix to calculate compartment
    signal by finding the eigenvector
    that correlates best with the phasing track.

    
    COOL_PATH : the paths to a .cool file with
                a balanced Hi-C map.

    TRACK_PATH : the path to a BedGraph-like file
                 that stores phasing track as
                 track-name named column.

    BedGraph-like format assumes tab-separated
    columns chrom, start, stop and track-name.


    """


    # load cooler file to process:
    c = cooler.Cooler(cool_path)



    # load phasing track, using some rudimenary
    # forms of validation, like usecols and dtype:
    # 
    # track columns, BedGraph style:
    track_cols = ["chrom", "start", "end", track_name]
    # track dtype per column:
    track_dtype = {'chrom' : np.str,
                   'start' : np.int64,
                   'end' : np.int64,
                   track_name : np.float64}
    # load the track into DataFrame:
    track_df = pd.read_table(
                    track_path,
                    usecols  = track_cols,
                    dtype    = track_dtype,
                    comment  = None,
                    verbose  = verbose)


    # we need to merge phasing track DataFrame
    # with the cooler.bins()[:] to get a
    # DataFrame with phasing info
    # aligned and validated against bins
    # inside of the cooler file.
    phasing_df = track_df.merge(
                    right    = c.bins()[:],
                    how      = "outer",
                    on       = ["chrom", "start", "end"],
                    validate = None)
    # consider using validate ?!

    # sanity check would be to check if
    # len(phasing_df) becomes > than nbins ...
    # that would imply there was something
    # in the track_df that didn't match 
    # ["chrom", "start", "end"] - keys from
    # the c.bins()[:] .
    if len(phasing_df) > c.nbins:
        ValueError(
            "There is something in the {} that couldn't be merged with cooler-bins {}" \
                .format(track_path, cool_path) )

    # once that's done and we are sure in our 'phasing_df':
    # go ahead and use 'cooler_cis_eig' or 'cooler_trans_eig' ...

    # define OBS/EXP getter functions,
    # it's contact_type dependent:
    if contact_type == "cis":
        eigvals, eigvec_table = eigdecomp.cooler_cis_eig(
                                    clr = c,
                                    bins = phasing_df,
                                    regions=None, 
                                    n_eigs=3, 
                                    phasing_track_col=track_name, 
                                    ignore_diags=None,
                                    clip_percentile = 99.9,
                                    sort_metric = None)
    elif contact_type == "trans":
        eigvals, eigvec_table =  eigdecomp.cooler_trans_eig(
                                    clr = c,
                                    bins = phasing_df, 
                                    n_eigs=3, 
                                    partition=None, 
                                    phasing_track_col=track_name, 
                                    sort_metric=None)

    # "cooler_cis_eig"
    # is expecting  "phasing track"
    # column in c.bins()[:]
    # should we expect a cooler like that or what ?


    ##############################
    # OUTPUT
    ##############################
    # no stdout output, since there are several
    # distinct data-structures that needs to be saved,
    # let's pack them in a single container:
    if output is not None:
        # pack saddledata, binedges, and digitized 
        # in a single .npz container,
        # use it as a dump to redraw a saddleplot.
        pass
        # output eigvals, eigvec_table ...
    else:
        pass
