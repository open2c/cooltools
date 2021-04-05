import multiprocess as mp
import pandas as pd
from itertools import combinations
import cooler
from bioframe import parse_regions
from .. import expected

import click
from . import cli
from . import util

# might be relevant to us ...
# https://stackoverflow.com/questions/46577535/how-can-i-run-a-dask-distributed-local-cluster-from-the-command-line
# http://distributed.readthedocs.io/en/latest/setup.html#using-the-command-line


@cli.command()
@click.argument("cool_path", metavar="COOL_PATH", type=str, nargs=1)
@click.option(
    "--nproc", "-p",
    help="Number of processes to split the work between."
    "[default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
@click.option(
    "--chunksize", "-c",
    help="Control the number of pixels handled by each worker process at a time.",
    type=int,
    default=int(10e6),
    show_default=True,
)
@click.option(
    "--output", "-o",
    help="Specify output file name to store the expected in a tsv format.",
    type=str,
    required=False,
)
@click.option(
    "--hdf",
    help="Use hdf5 format instead of tsv."
    " Output file name must be specified [Not Implemented].",
    is_flag=True,
    default=False,
)
@click.option(
    "--contact-type", "-t",
    help="compute expected for cis or trans region of a Hi-C map."
    "trans-expected is calculated for pairwise combinations of specified regions.",
    type=click.Choice(["cis", "trans"]),
    default="cis",
    show_default=True,
)
@click.option(
    "--regions",
    help="Path to a BED file containing genomic regions "
    "for which expected will be calculated. Region names can"
    "be provided in a 4th column, otherwise UCSC notaion is used."
    "When not specified, expected is calculated for all chromosomes",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--balance/--no-balance",
    help="Apply balancing weights to data before calculating expected."
    "Bins masked in the balancing weights are ignored from calcualtions.",
    is_flag=True,
    default=True,
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
    "--blacklist",
    help="Path to a 3-column BED file containing genomic regions to mask "
    "out during calculation of expected. Overwrites inference of "
    "'bad' regions from balancing weights. [Not Implemented]",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--ignore-diags",
    help="Number of diagonals to neglect for cis contact type",
    type=int,
    default=2,
    show_default=True,
)
def compute_expected(
    cool_path,
    nproc,
    chunksize,
    output,
    hdf,
    contact_type,
    regions,
    balance,
    weight_name,
    blacklist,
    ignore_diags,
):
    """
    Calculate expected Hi-C signal either for cis or for trans regions
    of chromosomal interaction map.

    When balancing weights are not applied to the data, there is no
    masking of bad bins performed.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    """
    if blacklist is not None:
        raise NotImplementedError(
            "Custom genomic regions for masking from calculation of expected"
            "are not implemented."
        )
        # use blacklist-ing from cooler balance module
        # https://github.com/mirnylab/cooler/blob/843dadca5ef58e3b794dbaf23430082c9a634532/cooler/cli/balance.py#L175

    clr = cooler.Cooler(cool_path)
    if regions is None:
        regions = [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]
        regions = parse_regions(regions)
        regions["name"] = clr.chromnames
    else:
        regions_buf, names = util.sniff_for_header(regions)
        regions = pd.read_csv(regions_buf, sep="\t", header=None)
        if regions.shape[1] not in (3, 4):
            raise ValueError(
                "The region file does not have three or four tab-delimited columns."
                "We expect a bed file with columns chrom, start, end, and optional name"
            )
        if regions.shape[1] == 4:
            regions = regions.rename(columns={0:"chrom",1:"start",2:"end",3:"name"})
            regions = parse_regions(regions)
        else:
            regions = regions.rename(columns={0:"chrom",1:"start",2:"end"})
            regions["name"] = list(regions.apply(lambda x: "{}:{}-{}".format(*x), axis=1))
            regions = parse_regions(regions)

    # define transofrms - balanced and raw ('count') for now
    if balance:
        weight1 = weight_name + "1"
        weight2 = weight_name + "2"
        transforms = {"balanced": lambda p: p["count"] * p[weight1] * p[weight2]}
    else:
        # no masking bad bins of any kind, when balancing is not applied
        weight_name = None
        transforms = {}

    # execution details
    if nproc > 1:
        pool = mp.Pool(nproc)
        map_ = pool.map
    else:
        map_ = map

    # using try-clause to close mp.Pool properly
    try:
        if contact_type == "cis":
            result = expected.diagsum(
                clr,
                regions,
                transforms=transforms,
                weight_name=weight_name,
                bad_bins=None,
                chunksize=chunksize,
                ignore_diags=ignore_diags,
                map=map_,
            )
        elif contact_type == "trans":
            # prepare pairwise combinations of regions for trans-expected (blocksum):
            regions_pairwise = combinations(regions.itertuples(index=False), 2)
            regions1, regions2 = zip(*regions_pairwise)
            result = expected.blocksum_asymm(
                clr,
                regions1 = pd.DataFrame(regions1),
                regions2 = pd.DataFrame(regions2),
                transforms=transforms,
                weight_name=weight_name,
                bad_bins=None,
                chunksize=chunksize,
                map=map_,
            )
    finally:
        if nproc > 1:
            pool.close()

    # calculate actual averages by dividing sum by n_valid:
    result["count.avg"] = result["count.sum"] / result["n_valid"]
    for key in transforms.keys():
        result[key + ".avg"] = result[key + ".sum"] / result["n_valid"]

    # output to file if specified:
    if output:
        result.to_csv(output, sep="\t", index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(result.to_csv(sep="\t", index=False, na_rep="nan"))

    # would be nice to have some binary output to preserve precision.
    # to_hdf/read_hdf should work in this case as the file is small .
    # still debated as to how should we store it - store in cooler seems
    # to be consensus:
    if hdf:
        raise NotImplementedError("hdf output is to be implemented")
