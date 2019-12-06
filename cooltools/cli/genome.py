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
def binnify(chromsizes_path, binsize):
    import bioframe

    chromsizes = bioframe.read_chromsizes(chromsizes_path)
    bins = bioframe.tools.binnify(chromsizes, binsize)
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
    frags = bioframe.tools.digest(fasta_records, enzyme_name)
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
    bins["GC"] = bioframe.tools.frac_gc(bins, fasta_records, mapped_only)
    print(bins.to_csv(sep="\t", index=False))


@genome.command()
@click.argument("bins_path")
@click.argument("db")
def genecov(bins_path, db):
    import bioframe
    import pandas as pd

    bins = pd.read_table(bins_path)
    bins = bioframe.tools.frac_gene_coverage(bins, db)
    print(bins.to_csv(sep="\t", index=False))
