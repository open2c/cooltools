import os
import io
import gzip
import tarfile
import tempfile

from . import fastsavetxt

import cooler


def dump_cworld(
    in_cooler,
    out=None,
    region=None,
    iced=False,
    iced_unity=False,
    buffer_size=int(1e8),
):
    """
    Dump a genome-wide contact matrix from cooler into a CWorld-format
    text matrix.

    Parameters
    ----------
    in_cooler : str or cooler
        A cooler object or the path to the file.

    out : str or file object
        Either:
        -- a path to the output file. If ends with .gz the output is gzipped
        -- a file object
        -- a stdin of a Popen object
        -- None, in which case the data is dumped into a string and returned
        TIP: when using files/stdin do not forget to flush()/communicate().

    region : str
        The region to dump. By default is None, dump the genome-wide matrix.

    iced : bool, optional
        If True, dump the balanced matrix.

    iced_unity : bool, optional
        If True and `iced` is True, dump the matrix balanced to a unity.

    buffer_size : int
        The chunk size for iterating over the rows of the Hi-C matrix.
    """

    # Prepare the out pipe and the clean-up function.
    if not (out):
        out = io.BytesIO(b"")
    if issubclass(type(out), str) or issubclass(type(out), bytearray):
        if out.endswith(".gz"):
            writer = fastsavetxt.gzipWriter(out)
            out_pipe = writer.stdin
            close_out_func = writer.communicate
        else:
            writer = open(out, "wb")
            out_pipe = writer
            close_out_func = writer.flush
    elif hasattr(out, "write"):
        out_pipe = out
        close_out_func = fastsavetxt.empty_func

    # Make headers
    if not issubclass(type(in_cooler), cooler.Cooler):
        c = cooler.Cooler(in_cooler)
    else:
        c = in_cooler

    res = c.info["bin-size"]
    gname = c.info["genome-assembly"]

    bins = c.bins()[:] if not (region) else c.bins().fetch(region)
    nbins = len(bins)

    col_headers = "\t".join(
        ["{}x{}".format(nbins, nbins)]
        + [
            "{}|{}|{}:{}-{}".format(binidx, gname, b.chrom, b.start + 1, b.end)
            for binidx, b in bins.iterrows()
        ]
    ).encode()

    row_headers = [
        "{}|{}|{}:{}-{}".format(binidx1, gname, b1.chrom, b1.start + 1, b1.end).encode()
        for binidx1, b1 in bins.iterrows()
    ]

    # Iterate over a matrix one block at a time.
    nrows_per_step = max(1, buffer_size // nbins)
    for i in range(nbins // nrows_per_step + 1):
        lo = min(nbins, i * nrows_per_step)
        hi = min(nbins, (i + 1) * nrows_per_step)
        if hi <= lo:
            break
        mat = (
            c.matrix(balance=iced)
            if not (region)
            else c.matrix(balance=iced).fetch(region)
        )[lo:hi]
        if iced and (not iced_unity):
            mat *= c._load_attrs("/bins/weight")["scale"]

        fastsavetxt.array2txt(
            mat,
            out_pipe,
            format_string=b"%.8f" if iced_unity else b"%.4lf",
            header=col_headers if i == 0 else None,
            row_headers=row_headers[lo:hi],
        )

    if issubclass(type(out), io.BytesIO):
        return out.getvalue()
    else:
        close_out_func()


def dump_cworld_tar(
    cooler_paths,
    out_path,
):
    """
    Makes a CWorld .tar archive with binned contact maps at multiple resolutions
    in .matrix.txt.gz format.

    Parameters
    ----------
    cooler_paths : a list of str
        The paths to all coolers to dump into a single CWorld tar archive.
        Must correspond to the same dataset and have different resolutions.

    out_path : str
        The path to the output file.

    """

    dataset_name = os.path.splitext(os.path.split(out_path)[1])[0]

    with tempfile.TemporaryDirectory() as cworld_tmp_path:
        for cooler_path in cooler_paths:
            res = cooler.Cooler(cooler_path).info["bin-size"]
            os.mkdir(os.path.join(cworld_tmp_path, "C-" + str(res)))
            for iced, iced_label in [(True, "iced"), (False, "raw")]:
                folder_path = os.path.join(cworld_tmp_path, "C-" + str(res), iced_label)
                os.mkdir(folder_path)

                mat_path = os.path.join(
                    folder_path,
                    "{}__C-{}-{}.matrix.gz".format(dataset_name, res, iced_label),
                )

                dump_cworld(
                    in_cooler=cooler_path, out=mat_path, iced=iced, iced_unity=False
                )

        with tarfile.open(out_path, mode="w") as archive:
            archive.add(cworld_tmp_path, arcname=dataset_name, recursive=True)
