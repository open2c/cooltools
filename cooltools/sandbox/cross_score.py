from operator import index
import pathlib
import itertools
import multiprocessing as mp

import numpy as np
import bioframe
import cooler

import argparse
import logging

import pandas as pd

import pybigtools # we will need it for writing bigwig files

parser = argparse.ArgumentParser(
    description="""Calculate distance-dependent contact marginals of a Hi-C map.

    These contact marginals are analogous to per-bin contact-frequency-vs-distance 
    curves, but over a small number of distance bins. Such marginals are useful to 
    reveal genomic sites participating in frequent long-distance contacts, e.g. 
    anchors of loops and stripes.
    """
)

parser.add_argument(
    "COOL_URI", 
    metavar="COOL_URI", 
    type=str, 
    help="input cooler URI")

parser.add_argument(
    "--dist-bins",
    default="1e3,3e4,1e6,3e7",
    help="a comma-separated list of distance bins",
)

parser.add_argument(
    "--clr-weight-name",
    type=str,
    default="weight",
    help="Name of the column to use for data balancing",
)

parser.add_argument(
    "--ignore-diags", 
    type=int, 
    default=2, 
    help="How many diagonals to ignore"
)

parser.add_argument(
    "--outfolder", 
    type=str, 
    default="./", 
    help="The output folder")

parser.add_argument(
    "--prefix", 
    type=str, 
    default=None, 
    help="The prefix for output files"
)

parser.add_argument(
    "--format",
    type=str,
    default="bigwig",
    help="Comma-separated list of the output formats. Possible values: bigwig, bedgraph.",
)

parser.add_argument(
    "--nproc",
    type=int,
    default=None,
    help="The number of processes. By default, use all available cores.",
)

parser.add_argument(
    "--chunksize", default=1e6, help="The number of pixels per processed chunk."
)


def get_dist_margs(clr_path, lo, hi, dist_bins, weight_name, ignore_diags):
    clr = cooler.Cooler(clr_path)

    bins = clr.bins()[:]
    chunk = clr.pixels()[lo:hi]
    res = clr.binsize

    chunk = chunk[chunk["bin2_id"] - chunk["bin1_id"] >= ignore_diags]

    chunk = cooler.annotate(chunk, bins)
    chunk["balanced"] = np.nan_to_num(
        chunk["count"] * chunk[f"{weight_name}1"] * chunk[f"{weight_name}2"]
    )
    chunk = chunk[chunk.chrom1 == chunk.chrom2]

    del clr

    return _dist_margs(chunk, dist_bins, res)


def _dist_margs(chunk, dist_bins, res):
    min_bin_id = chunk["bin1_id"].values[0]

    dists = (chunk["bin2_id"] - chunk["bin1_id"]) * res
    dist_bin_id = np.searchsorted(dist_bins, dists, "right")
    n_dist_bins = len(dist_bins)
    margs_down_loc = np.bincount(
        (chunk["bin1_id"].values - min_bin_id) * n_dist_bins + dist_bin_id,
        weights=np.nan_to_num(chunk["balanced"].values),
    )

    margs_up_loc = np.bincount(
        (chunk["bin2_id"].values - min_bin_id) * n_dist_bins + dist_bin_id,
        weights=np.nan_to_num(chunk["balanced"].values),
    )

    return min_bin_id, margs_down_loc, margs_up_loc


def drop_resolution(clrname):
    name_parts = clrname.split(".")
    if name_parts[-1] in ["cool", "mcool"]:
        name_parts = name_parts[:-1]
    if name_parts[-1].isnumeric():
        name_parts = name_parts[:-1]
    return ".".join(name_parts)


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    mp.freeze_support()

    chunksize = int(float(args.chunksize))
    clr = cooler.Cooler(args.COOL_URI)
    bins = clr.bins()[:]
    n_pixels = clr.pixels().shape[0]

    # dist_bins contain *right* bins edges; 0 is implied as the left edge of the first bin.
    dist_bins = np.array([int(float(i)) for i in args.dist_bins.split(",")]).astype(np.int64)
    dist_bins = np.r_[dist_bins, np.iinfo(dist_bins.dtype).max]

    weight_name = args.clr_weight_name
    ignore_diags = args.ignore_diags
    formats = args.format.split(",")

    n_dist_bins = len(dist_bins)

    chunk_spans = np.r_[np.arange(0, n_pixels, chunksize), n_pixels]

    nproc = mp.cpu_count() if args.nproc is None else args.nproc

    if nproc == 1:
        mapfunc = itertools.starmap
    else:
        pool = mp.Pool(nproc)
        mapfunc = pool.starmap
    logging.info(f"Calculating marginals for {args.COOL_URI}, weight name {weight_name}, ignore diags {ignore_diags}; using {nproc} processes")
    
    out = mapfunc(
        get_dist_margs,
        [
            (args.COOL_URI, lo, hi, dist_bins, weight_name, ignore_diags)
            for lo, hi in zip(chunk_spans[:-1], chunk_spans[1:])
        ],
    )

    margs_up = np.zeros(len(bins) * n_dist_bins + 1)
    margs_down = np.zeros(len(bins) * n_dist_bins + 1)

    for min_bin_id, margs_down_loc, margs_up_loc in out:
        margs_down[
            min_bin_id * n_dist_bins : min_bin_id * n_dist_bins + len(margs_down_loc)
        ] += margs_down_loc

        margs_up[
            min_bin_id * n_dist_bins : min_bin_id * n_dist_bins + len(margs_up_loc)
        ] += margs_up_loc

    margs_down = margs_down[:-1].reshape((len(bins), n_dist_bins)).T
    margs_up = margs_up[:-1].reshape((len(bins), n_dist_bins)).T

    out_folder = pathlib.Path(args.outfolder)
    clr_name = pathlib.Path(args.COOL_URI.split(":")[0]).name
    clr_name = drop_resolution(clr_name)

    prefix = clr_name if args.prefix is None else args.prefix
    res = clr.binsize

    for dist_bin_id in range(n_dist_bins-1):
        lo = np.r_[0, dist_bins][dist_bin_id]
        hi = np.r_[0, dist_bins][dist_bin_id + 1]

        for dir_str, margs in [
            ("up", margs_up),
            ("down", margs_down),
            ("both", margs_up + margs_down),
        ]:
            out_df = bins[["chrom", "start", "end"]].copy()
            out_df["marg"] = margs[dist_bin_id]
            out_df["marg"] = out_df["marg"].mask(bins[weight_name].isnull(), np.nan)

            if "bigwig" in formats:
                file_name = f"{prefix}.{res}.cross.{dir_str}.{lo}-{hi}.bw"
                out_path = (out_folder / file_name).resolve().as_posix()
                logging.info(f"Write output into {out_path}")
                bioframe.to_bigwig(
                    out_df,
                    chromsizes=clr.chromsizes.astype(int).to_dict(),
                    outpath=out_path,
                    engine='pybigtools'
                )

            if "bedgraph" in formats:
                file_name = f"{prefix}.{res}.cross.{dir_str}.{lo}-{hi}.bg.gz"
                out_path = (out_folder / file_name).resolve().as_posix()
                logging.info(f"Write output into {out_path}")
                out_df.to_csv(
                    out_path,
                    sep="\t",
                    index=False,
                    header=False,
                )
