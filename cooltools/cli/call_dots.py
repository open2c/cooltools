import os.path as op
from scipy.stats import poisson
import pandas as pd
import numpy as np
import cooler

import click
from . import cli
from .. import dotfinder


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str, #click.Path(exists=True, dir_okay=False),
    nargs=1)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
@click.option(
    '--expected-name',
    help="Name of value column in EXPECTED_PATH",
    type=str,
    default='balanced.avg',
    show_default=True,
    )
@click.option(
    '--nproc', '-n',
    help="Number of processes to split the work between."
         "[default: 1, i.e. no process pool]",
    default=1,
    type=int)
@click.option(
    '--max-loci-separation',
    help='Limit loci separation for dot-calling, i.e., do not call dots for'
         ' loci that are further than max_loci_separation that basepairs apart.'
         ' [current implementation is not ready to tackle max_loci_separation>3Mb].',
    type=int,
    default=2000000,
    show_default=True,
    )
@click.option(
    '--max-nans-tolerated',
    help='Maximum number of NaNs tolerated in a footprint of every used filter.',
    type=int,
    default=1,
    show_default=True,
    )
@click.option(
    '--tile-size',
    help='Tile size for the Hi-C heatmap tiling.'
         ' Typically on order of several mega-bases, and >= max_loci_separation.',
    type=int,
    default=6000000,
    show_default=True,
    )
@click.option(
    "--fdr",
    help="False discovery rate (FDR) to control in the multiple"
         " hypothesis testing BH-FDR procedure.",
    type=float,
    default=0.02,
    show_default=True)
@click.option(
    '--dots-clustering-radius',
    help='Radius for clustering dots that have been called too close to each other.'
         'Typically on order of 40 kilo-bases, and >= binsize.',
    type=int,
    default=39000,
    show_default=True,
    )
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--output-scores", "-s",
    help="Specify a pandas HDF5 table file where to dump"
         " all processed pixels before they get"
         " preprocessed in a BEDPE-like format.",
    type=str,
    required=False)
@click.option(
    "--output-calls", "-o",
    help="Specify output file name where to store"
         " the results of dot-calling, in a BEDPE-like format.",
    type=str)
def call_dots(
        cool_path,
        expected_path,
        expected_name,
        nproc,
        max_loci_separation,
        max_nans_tolerated,
        tile_size,
        fdr,
        dots_clustering_radius,
        verbose,
        output_scores,
        output_calls):
    """
    Call dots on a Hi-C heatmap that are not larger than max_loci_separation.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    EXPECTED_PATH : The paths to a tsv-like file with expected signal.

    Analysis will be performed for chromosomes referred to in EXPECTED_PATH, and
    therefore these chromosomes must be a subset of chromosomes referred to in
    COOL_PATH. Also chromosomes refered to in EXPECTED_PATH must be non-trivial,
    i.e., contain not-NaN signal. Thus, make sure to prune your EXPECTED_PATH
    before applying this script.

    COOL_PATH and EXPECTED_PATH must be binned at the same resolution.

    EXPECTED_PATH must contain at least the following columns for cis contacts:
    'chrom', 'diag', 'n_valid', value_name. value_name is controlled using
    options. Header must be present in a file.

    """
    clr = cooler.Cooler(cool_path)

    expected_columns = ['chrom', 'diag', 'n_valid', expected_name]
    expected_index = ['chrom', 'diag']
    expected_dtypes = {
        'chrom': np.str,
        'diag': np.int64,
        'n_valid': np.int64,
        expected_name: np.float64
    }
    expected = pd.read_table(
        expected_path,
        usecols=expected_columns,
        index_col=expected_index,
        dtype=expected_dtypes,
        comment=None,
        verbose=verbose)


    # Input validation
    # unique list of chroms mentioned in expected_path
    # do simple column-name validation for now
    get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
    expected_chroms = get_exp_chroms(expected)
    if not set(expected_chroms).issubset(clr.chromnames):
        raise ValueError(
            "Chromosomes in {} must be subset of ".format(expected_path) +
            "chromosomes in cooler {}".format(cool_path))
    # check number of bins
    # compute # of bins by comparing matching indexes
    get_exp_bins = lambda df, ref_chroms: (
        df.index.get_level_values("chrom").isin(ref_chroms).sum())
    expected_bins = get_exp_bins(expected, expected_chroms)
    cool_bins = clr.bins()[:]["chrom"].isin(expected_chroms).sum()
    if not (expected_bins == cool_bins):
        raise ValueError(
            "Number of bins is not matching: ",
            "{} in {}, and {} in {} for chromosomes {}".format(
                expected_bins,
                expected_path,
                cool_bins,
                cool_path,
                expected_chroms))
    if verbose:
        print("{} and {} passed cross-compatibility checks.".format(
            cool_path, expected_path))


    # Prepare some parameters.
    binsize = clr.binsize
    loci_separation_bins = int(max_loci_separation / binsize)
    tile_size_bins = int(tile_size / binsize)
    balance_factor = 1.0  #clr._load_attrs("bins/weight")["scale"]

    # clustering would deal with bases-units for now, so supress this for now
    # clustering_radius_bins = int(dots_clustering_radius/binsize)

    # kernels
    # 'upright' is a symmetrical inversion of "lowleft", not needed.
    ktypes = ['donut', 'vertical', 'horizontal', 'lowleft']

    # kernel parameters depend on the cooler resolution
    # TODO: rename w, p to wid, pix probably, or _w, _p to avoid naming conflicts
    if binsize > 28000:
        # > 30 kb - is probably too much ...
        raise ValueError(
            "Provided cooler '{}' has resolution {} bases, "
            "which is too coarse for analysis.".format(cool_path, binsize))
    elif binsize >= 18000:
        # ~ 20-25 kb:
        w, p = 3, 1
    elif binsize >= 8000:
        # ~ 10 kb
        w, p = 5, 2
    elif binsize >= 4000:
        # ~5 kb
        w, p = 7, 4
    else:
        # < 5 kb - is probably too fine ...
        raise ValueError(
            "Provided cooler {} has resolution {} bases, "
            "which is too fine for analysis.".format(cool_path, binsize))

    kernels = {k: dotfinder.get_kernel(w, p, k) for k in ktypes}

    if verbose:
        print("Kernels parameters are set as w,p={},{} "
              "for the cooler with {} bp resolution.".format(w,p,binsize))

    # list of tile coordinate ranges
    tiles = list(
        dotfinder.heatmap_tiles_generator_diag(
            clr,
            expected_chroms,
            w,
            tile_size_bins,
            loci_separation_bins
        )
    )

    # scoring_step(clr, expected, expected_name, tiles, kernels,
    #              max_nans_tolerated, loci_separation_bins, output_scores,
    #              nproc, verbose)


    # 1. Calculate genome-wide histograms of scores.
    # creating logspace l(ambda)bins with base=2^(1/3), for lambda-chunking
    # the first bin must be (-inf,1]
    # the last bin must be (2^(HiCCUPS_W1_MAX_INDX/3),+inf)
    nlchunks = dotfinder.HiCCUPS_W1_MAX_INDX
    base = 2 ** (1/3)
    ledges = np.concatenate((
        [-np.inf],
        np.logspace(0, nlchunks - 1, num=nlchunks, base=base, dtype=np.float),
        [np.inf]))
    # gw_hist for each kernel contains a histogram of raw pixel intensities
    # for every lambda-chunk (one per column) in a row-wise order, i.e.
    # each column is a histogram for each chunk ...
    gw_hist = dotfinder.scoring_and_histogramming_step(
        clr, expected, expected_name,
        tiles, kernels, ledges, max_nans_tolerated,
        loci_separation_bins, None, nproc, verbose)

    if output_scores is not None:
        for k in kernels:
            gw_hist[k].to_csv(
                "{}.{}.hist.txt".format(output_scores, k),
                sep='\t',
                header=True,
                index=False,
                compression=None)


    # 2. Determine the FDR thresholds.
    # threshold_df : dict
    #   each threshold_df[k] is a Series indexed by la_exp intervals
    #   (IntervalIndex) and it is all we need to extract "good" pixels from
    #   each chunk ...
    # qvalues : dict
    #   A dictionary with keys being kernel names and values pandas.DataFrames
    #   storing q-values: each column corresponds to a lambda-chunk,
    #   while rows correspond to observed pixels values.
    threshold_df, qvalues = dotfinder.determine_thresholds(
        kernels, ledges, gw_hist, fdr)


    # 3. Filter using FDR thresholds calculated in the histogramming step
    filtered_pix = dotfinder.scoring_and_extraction_step(
        clr, expected, expected_name,
        tiles, kernels, ledges, threshold_df, max_nans_tolerated,
        balance_factor, loci_separation_bins, output_calls,
        nproc, verbose)
    # Extract q-values using l-chunks and IntervalIndex.
    # we'll do it in an ugly but workign fashion, by simply
    # iteration over pairs of obs, la_exp and extracting needed qvals
    # one after another ...
    if verbose:
        print("Extracting q-values ...")
    for k in kernels:
        x = "la_exp." + k
        filtered_pix[x  + ".qval"] = [
            qvalues[k].loc[o, e] for o, e
                in filtered_pix[["obs.raw", x + ".value"]].itertuples(index=False)
        ]


    # 4. Post-processing
    if verbose:
        print("Subsequent clustering and thresholding steps are not production-ready")

    # 4a. clustering
    centroids = dotfinder.clustering_step_local(
        filtered_pix, expected_chroms, dots_clustering_radius, verbose)

    # 4b. filter by enrichment and qval
    out = dotfinder.thresholding_step(centroids)


    # Final result
    if output_calls is not None:

        final_output = op.join(
            op.dirname(output_calls),
            "final_" + op.basename(output_calls))

        out.to_csv(
            final_output,
            sep='\t',
            header=True,
            index=False,
            compression=None)
