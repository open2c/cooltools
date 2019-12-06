import multiprocess as mp
import numpy as np
import pandas as pd
import cooler
from .. import expected

import click
from . import cli

# might be relevant to us ...
# https://stackoverflow.com/questions/46577535/how-can-i-run-a-dask-distributed-local-cluster-from-the-command-line
# http://distributed.readthedocs.io/en/latest/setup.html#using-the-command-line

@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str,
    nargs=1)
@click.option(
    '--nproc', '-p',
    help="Number of processes to split the work between."
         "[default: 1, i.e. no process pool]",
    default=1,
    type=int)
@click.option(
    "--chunksize", "-c",
    help="Control the number of pixels handled by each worker process at a time.",
    type=int,
    default=int(10e6),
    show_default=True)
@click.option(
    "--output", "-o",
    help="Specify output file name to store"
         " the expected in a tsv format.",
    type=str,
    required=False)
@click.option(
    "--hdf",
    help="Use hdf5 format instead of tsv."
         " Output file name must be specified [Not Implemented].",
    is_flag=True,
    default=False)
@click.option(
    '--contact-type', "-t",
    help="compute expected for cis or trans region"
    "of a Hi-C map.",
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True)
@click.option(
    "--balance/--no-balance",
    help="Apply balancing weights to data before"
    "calculating expected",
    is_flag=True,
    default=True,
    show_default=True)
@click.option(
    '--weight-name',
    help="Use balancing weight with this name.",
    type=str,
    default='weight',
    show_default=True)
@click.option(
    "--ignore-diags",
    help="Number of diagonals to neglect for cis contact type",
    type=int,
    default=2,
    show_default=True)
def compute_expected(cool_path,
                    nproc,
                    chunksize,
                    output,
                    hdf,
                    contact_type,
                    balanced,
                    weight_name,
                    ignore_diags):
    """
    Calculate expected Hi-C signal either for cis or for trans regions
    of chromosomal interaction map.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    """
    clr = cooler.Cooler(cool_path)
    supports = [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]

    # define transofrms - balanced and raw ('count') for now
    if balanced:
        weight1 = weight_name+"1"
        weight2 = weight_name+"2"
        transforms = {
            'balanced': lambda p: p['count'] * p[weight1] * p[weight2] }
    else:
        transforms = {}

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly
    try:
        if contact_type == 'cis':
            tables = expected.diagsum(
                clr,
                supports,
                transforms=transforms,
                chunksize=chunksize,
                ignore_diags=ignore_diags,
                map=map_)
            result = pd.concat(
                [tables[support] for support in supports],
                keys=[support[0] for support in supports],
                names=['chrom'])
            result = result.reset_index()

        elif contact_type == 'trans':
            records = expected.blocksum_pairwise(
                clr,
                supports,
                transforms=transforms,
                chunksize=chunksize,
                map=map_)
            result = pd.DataFrame(
                [{'chrom1': s1[0], 'chrom2': s2[0], **rec}
                    for (s1, s2), rec in records.items()],
                columns=['chrom1', 'chrom2', 'n_valid',
                         'count.sum', 'balanced.sum'])
    finally:
        if nproc > 1:
            pool.close()

    # calculate actual averages by dividing sum by n_valid:
    result['count.avg'] = result['count.sum'] / result['n_valid']
    for key in transforms.keys():
        result[key+'.avg'] = result[key+'.sum'] / result['n_valid']

    # output to file if specified:
    if output:
        result.to_csv(output,sep='\t', index=False, na_rep='nan')
    # or print into stdout otherwise:
    else:
        print(result.to_csv(sep='\t', index=False, na_rep='nan'))

    # would be nice to have some binary output to preserve precision.
    # to_hdf/read_hdf should work in this case as the file is small .
    if hdf:
        raise NotImplementedError("hdf output is to be implemented")
