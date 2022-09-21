import click
import cooler

from .. import coverage

from . import cli
from .. import api
import bioframe
import multiprocessing as mp


@cli.command()
@click.argument(
    "cool_path", metavar="COOL_PATH", type=str, nargs=1,
)
@click.option(
    "--output",
    "-o",
    help="Specify output file name to store the coverage in a tsv format.",
    type=str,
    required=False,
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
    "--store",
    help="Append columns with coverage (cov_cis_raw, cov_tot_raw), or"
    " (cov_cis_clr_weight_name, cov_tot_clr_weight_name) if calculating"
    " balanced coverage, to the cooler bin table. If clr_weight_name=None,"
    " also stores total cis counts in the cooler info",
    is_flag=True,
)
@click.option(
    "--chunksize",
    help="Split the contact matrix pixel records into equally sized chunks to"
    " save memory and/or parallelize. Default is 10^7",
    type=int,
    default=1e7,
    show_default=True,
)
@click.option(
    "--bigwig",
    help="Also save output as bigWig files for cis and total coverage"
    " with the names <output>.<cis/tot>.bw",
    is_flag=True,
    default=False,
)
@click.option(
    "--clr_weight_name",
    help="Name of the weight column. Specify to calculate coverage of"
    " balanced cooler.",
    type=str,
    default=None,
    show_default=False,
)
@click.option(
    "-p",
    "--nproc",
    help="Number of processes to split the work between."
    " [default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
def coverage(
    cool_path, output, ignore_diags, store, chunksize, bigwig, clr_weight_name, nproc,
):
    """
    Calculate the sums of cis and genome-wide contacts (aka coverage aka marginals) for
    a sparse Hi-C contact map in Cooler HDF5 format.
    Note that the sum(tot_cov) from this function is two times the number of reads
    contributing to the cooler, as each side contributes to the coverage.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    """

    clr = cooler.Cooler(cool_path)

    if nproc > 1:
        pool = mp.Pool(nproc)
        _map = pool.imap
    else:
        _map = map

    try:
        cis_cov, tot_cov = api.coverage.coverage(
            clr, ignore_diags=ignore_diags, chunksize=chunksize, map=_map, store=store, clr_weight_name=clr_weight_name
        )
    finally:
        if nproc > 1:
            pool.close()

    coverage_table = clr.bins()[:][["chrom", "start", "end"]]
    if clr_weight_name is None:
        store_names = ["cov_cis_raw", "cov_tot_raw"]
        coverage_table[store_names[0]] = cis_cov.astype(int)
        coverage_table[store_names[1]] = tot_cov.astype(int)
    else:
        store_names = [f"cov_cis_{clr_weight_name}", f"cov_tot_{clr_weight_name}"]
        coverage_table[store_names[0]] = cis_cov.astype(float)
        coverage_table[store_names[1]] = tot_cov.astype(float)

    # output to file if specified:
    if output:
        coverage_table.to_csv(output, sep="\t", index=False, na_rep="nan")
    # or print into stdout otherwise:
    else:
        print(coverage_table.to_csv(sep="\t", index=False, na_rep="nan"))

    # Write the coverage tracks as a bigwigs:
    if bigwig:
        bioframe.to_bigwig(
            coverage_table,
            clr.chromsizes,
            f"{output}.cis.bw",
            value_field=store_names[0],
        )
        bioframe.to_bigwig(
            coverage_table,
            clr.chromsizes,
            f"{output}.tot.bw",
            value_field=store_names[1],
        )
