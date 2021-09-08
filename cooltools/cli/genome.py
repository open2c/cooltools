import sys
import click
from . import cli


@cli.group()
def genome():
    """
    Utilities for binned genome assemblies.

    """


@genome.command()
@click.argument("db")
def fetch_chromsizes(db):
    import bioframe

    chromsizes = bioframe.fetch_chromsizes(db)
    print(chromsizes.to_csv(sep="\t"))


@genome.command()
@click.argument("chromsizes_path")
@click.argument("binsize", type=int)
@click.option(
    "--all-names",
    help='Parse all chromosome names from file, not only default r"^chr[0-9]+$", r"^chr[XY]$", r"^chrM$". ',
    is_flag=True,
)
def binnify(chromsizes_path, binsize, all_names):
    import bioframe

    chromsizes = bioframe.read_chromsizes(
        chromsizes_path, filter_chroms=not (all_names)
    )
    bins = bioframe.binnify(chromsizes, binsize)
    print(bins.to_csv(sep="\t", index=False))


@genome.command()
@click.argument("chromsizes_path")
@click.argument("fasta_path")
@click.argument("enzyme_name")
def digest(chromsizes_path, fasta_path, enzyme_name):
    import bioframe

    chromsizes = bioframe.read_chromsizes(chromsizes_path, all_names=True)
    fasta_records = bioframe.load_fasta(fasta_path, engine="pyfaidx", as_raw=True)
    if not chromsizes.index.isin(fasta_records).all():
        raise ValueError(
            "Some chromosomes mentioned in {}"
            " are not found in {}".format(chromsizes_path, fasta_path)
        )
    frags = bioframe.digest(fasta_records, enzyme_name)
    print(frags.to_csv(sep="\t", index=False))


@genome.command()
@click.argument("bins_path")
@click.argument("fasta_path")
@click.option("--mapped-only", is_flag=True, default=True)
def gc(bins_path, fasta_path, mapped_only):
    import bioframe
    import pandas as pd

    if bins_path == "-":
        bins_path = sys.stdin
    bins = pd.read_table(bins_path)
    chromosomes = bins["chrom"].unique()
    fasta_records = bioframe.load_fasta(fasta_path, engine="pyfaidx", as_raw=True)
    if any(chrom not in fasta_records.keys() for chrom in chromosomes):
        raise ValueError(
            "Some chromosomes mentioned in {}"
            " are not found in {}".format(bins_path, fasta_path)
        )
    bins = bioframe.frac_gc(bins, fasta_records, mapped_only)
    print(bins.to_csv(sep="\t", index=False))


@genome.command()
@click.argument("bins_path")
@click.argument("db")
def genecov(bins_path, db):
    """
    BINS_PATH is the path to bintable.

    DB is the name of the genome assembly.
    The gene locations will be automatically downloaded from teh UCSC goldenPath.
    """
    import bioframe
    import pandas as pd

    bins = pd.read_table(bins_path)
    bins = bioframe.frac_gene_coverage(bins, db)
    print(bins.to_csv(sep="\t", index=False))
