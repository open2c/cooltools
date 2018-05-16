# import numpy as np
# import pandas as pd
# import cooler

from bioframe.io import formats
from bioframe import tools

import click
from . import cli

@cli.command()
@click.argument(
    "fasta_path",
    metavar="FASTA_PATH",
    type=str,
    nargs=1)
@click.argument(
    "chromsizes_path",
    metavar="CHROMSIZES_PATH",
    type=str,
    nargs=1)
@click.argument(
    "binsize",
    metavar="BINSIZE",
    type=int,
    nargs=1)
@click.option(
    '--phasing-type',
    help="Type of phasing information to compute",
    type=click.Choice(['gc', 'gene']),
    default='gc',
    show_default=True)

def compute_phasing(fasta_path, chromsizes_path, binsize, phasing_type):
    """
    Compute binned phasing track and output in a BedGraph-like format.
    
    FASTA_PATH : Paths to a fasta file with a reference genome.

    CHROMSIZES_PATH : Path to a tab-separated file with
                      chromosome names and sizes.

    BINSIZE : Size of the bins in the resultant phasing track.


    A file with the chromsizes must refer to a subset of
    chromosomes available in FASTA_PATH.

    """

    # chromsizes is pandas.Series with index "name" and values "length"
    chromsizes = formats.read_chromsizes(chromsizes_path,
                                        natsort=True,
                                        all_names=True)

    # ref_genome is a lazy OrderedDict of the fasta records:
    ref_genome = formats.load_fasta(fasta_path,
                                    engine='pyfaidx',
                                    as_raw=True)
    # should we use this instead ?:
    # ref_genome = cooler.util.load_fasta(chromsizes.index, fasta_path)

    # validate chromosomes "chromsizes" and "ref_genome":
    if not chromsizes.index.isin(ref_genome).all():
        raise ValueError("Some chromosomes mentioned in {}"
                         " are not found in {}".format(chromsizes_path,fasta_path))

    # generate a DataFrame with bins:
    bins = tools.binnify(chromsizes, binsize)
    # should we use this instead ?:
    # bins = cooler.util.binnify(chromsizes, binsize)

    # should binned GC and/or gene density be their
    # own separate CLI tools or kept as one under
    # the phasing track umbrella ?
    if phasing_type == "gc":
        # calculate Series of binned GC content:
        binned_gc = tools.frac_gc(bins, ref_genome)
        # and store it right in the "bins" DataFrame:
        bins[phasing_type] = binned_gc
    elif phasing_type == "gene":
        raise NotImplementedError("Gene density phasing calculations"
                                            "are not implemented yet.")
    else:
        raise ValueError("This tools supports only 'gc' and 'gene'"
                                        "phasing tracks (Click should have caught it).")

    # output to stdout
    # also preserving the header here:
    print(bins.to_csv(sep='\t', index=False, na_rep='nan'))

