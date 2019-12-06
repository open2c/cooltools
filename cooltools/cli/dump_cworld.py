import click
from . import cli
from .. import io


@cli.command()
@click.argument(
    "cool_paths",
    metavar="COOL_PATHS",
    type=str,
    nargs=-1
)
@click.argument(
    "out_path",
    metavar="OUT_PATH",
    type=click.Path(exists=False, writable=True),
    nargs=1,
)
@click.option(
    "--cworld-type",
    help="The format of the CWorld output. "
    "'matrix' converts a single .cool file into the .matrix.txt.gz tab-separated format. "
    "'tar' dumps all specified cooler files into a "
    "single .tar archive containing multiple .matrix.txt.gz files (use to make "
    "multi-resolution archives).",
    type=click.Choice(["matrix", "tar"]),
    default="matrix",
    show_default=True,
)
@click.option(
    "--region",
    help="The coordinates of a genomic region to dump, in the UCSC format. "
    "If empty (by default), dump a genome-wide matrix. This option can be used "
    "only when dumping a single cooler file.",
    type=str,
    default="",
    show_default=True,
)
@click.option(
    "--balancing-type",
    help="The type of the matrix balancing. 'IC_unity' - iteratively corrected "
    "for the total number of contacts per locus=1.0; 'IC' - same, but preserving "
    "the average total number of contacts; 'raw' - no balancing",
    type=click.Choice(["IC_unity", "IC", "raw"]),
    default="IC_unity",
    show_default=True,
)
def dump_cworld(cool_paths, out_path, cworld_type, region, balancing_type):
    """
    Convert a cooler or a group of coolers into the Dekker' lab CWorld text format.

    COOL_PATHS : Paths to one or multiple .cool files
    OUT_PATH : Output CWorld file path
    """
    if (cworld_type == "matrix") and (len(cool_paths) > 1):
        raise click.ClickException(
            "Only one .cool file can be converted into the matrix " "format at a time."
        )

    if cworld_type == "matrix":
        io.dump_cworld(
            cool_paths[0],
            out_path,
            region=region,
            iced=(balancing_type != "raw"),
            iced_unity=(balancing_type == "IC_unity"),
            buffer_size=int(1e8),
        )
    elif cworld_type == "tar":
        if region:
            raise Exception(
                "Only genome-wide matrices and not specific regions can be dumpled"
                " into a .tar CWorld archive."
            )
        io.dump_cworld_tar(cool_paths, out_path)
