import cooler
import bioframe
from .. import api


import click
from . import cli


@cli.command()
@click.argument("cool_path", metavar="COOL_PATH", type=str, nargs=1)
@click.argument("viewpoint", metavar="VIEWPOINT", type=str, nargs=1)
@click.option(
    "--clr-weight-name",
    help="Use balancing weight with this name. "
    "Provide empty argument to calculate insulation on raw data (no masking bad pixels).",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "-o",
    "--out-prefix",
    help="Save virtual 4C track as a BED-like file."
    " Contact frequency is stored in out_prefix.v4C.tsv",
    required=True,
)
@click.option(
    "--bigwig",
    help="Also save virtual 4C track as a bigWig file with the name out_prefix.v4C.bw",
    is_flag=True,
    default=False,
)
@click.option(
    "-p",
    "--nproc",
    help="Number of processes to split the work between."
    " [default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
def virtual4c(
    cool_path,
    viewpoint,
    clr_weight_name,
    out_prefix,
    bigwig,
    nproc,
):
    """
    Generate virtual 4C profile from a contact map by extracting all interactions of a
    given viewpoint with the rest of the genome.


    COOL_PATH : the paths to a .cool file with a Hi-C map. Use the '::' syntax to
    specify a group path in a multicooler file.

    VIEWPOINT : the viewpoint to use for the virtual 4C profile. Provide as a UCSC-string
    (e.g. chr1:1-1000)
    

    Note: this is a new (experimental) tool, the interface or output might change in a
    future version.
    """
    clr = cooler.Cooler(cool_path)

    viewpoint = bioframe.core.stringops.parse_region_string(viewpoint)
    v4c = api.virtual4c.virtual4c(
        clr,
        viewpoint,
        clr_weight_name=clr_weight_name if clr_weight_name else None,
        nproc=nproc,
    )
    # Output
    if out_prefix:
        v4c.to_csv(out_prefix + ".tsv", sep="\t", index=False, na_rep="nan")
        if bigwig:
            bioframe.to_bigwig(
                v4c.dropna(),
                clr.chromsizes,
                out_prefix + ".bw",
                value_field=v4c.columns[3],
            )
    else:
        print(v4c.to_csv(sep="\t", index=False, na_rep="nan"))
    return
