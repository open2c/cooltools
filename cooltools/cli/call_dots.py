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

    # read expected and make preparations for validation,
    # that's what we expect as column names:
    expected_columns = ['chrom', 'diag', 'n_valid', expected_name]
    # what would become a MultiIndex:
    expected_index = ['chrom', 'diag']
    # expected dtype as a rudimentary form of validation:
    expected_dtype = {
        'chrom': np.str,
        'diag': np.int64,
        'n_valid': np.int64,
        expected_name: np.float64
    }
    # unique list of chroms mentioned in expected_path:
    get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
    # compute # of bins by comparing matching indexes:
    get_exp_bins = lambda df, ref_chroms: (
        df.index.get_level_values("chrom").isin(ref_chroms).sum())
    # use 'usecols' as a rudimentary form of validation,
    # and dtype. Keep 'comment' and 'verbose' - explicit,
    # as we may use them later:
    expected = pd.read_table(
        expected_path,
        usecols=expected_columns,
        index_col=expected_index,
        dtype=expected_dtype,
        comment=None,
        verbose=verbose)

    #############################################
    # CROSS-VALIDATE COOLER and EXPECTED:
    #############################################
    # EXPECTED vs COOLER:
    # chromosomes to deal with
    # are by default extracted
    # from the expected-file:
    expected_chroms = get_exp_chroms(expected)
    # do simple column-name validation for now:
    if not set(expected_chroms).issubset(clr.chromnames):
        raise ValueError(
            "Chromosomes in {} must be subset of ".format(expected_path) +
            "chromosomes in cooler {}".format(cool_path))
    # check number of bins:
    expected_bins = get_exp_bins(expected, expected_chroms)
    cool_bins   = clr.bins()[:]["chrom"].isin(expected_chroms).sum()
    if not (expected_bins == cool_bins):
        raise ValueError(
            "Number of bins is not matching:",
            " {} in {}, and {} in {} for chromosomes {}".format(expected_bins,
                                                                expected_path,
                                                                cool_bins,
                                                                cool_path,
                                                                expected_chroms))
    if verbose:
        print("{} and {} passed cross-compatibility checks.".format(
            cool_path, expected_path))

    # prepare some parameters:
    # turn them from nucleotides dims to bins, etc.:
    binsize = clr.binsize
    loci_separation_bins = int(max_loci_separation/binsize)
    tile_size_bins = int(tile_size/binsize)
    # # clustering would deal with bases-units for now, so
    # # supress this for now:
    # clustering_radius_bins = int(dots_clustering_radius/binsize)
    balance_factor = 1.0 #clr._load_attrs("bins/weight")["scale"]

    ktypes = ['donut', 'vertical', 'horizontal', 'lowleft']
    # 'upright' is a symmetrical inversion of "lowleft", not needed.

    # define kernel parameteres based on the cooler resolution:
    if binsize > 28000:
        # > 30 kb - is probably too much ...
        raise ValueError("Provided cooler {} has resolution {} bases,\
                         which is too low for analysis.".format(cool_path, binsize))
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
        raise ValueError("Provided cooler {} has resolution {} bases, \
                        which is too fine for analysis.".format(cool_path, binsize))
    # rename w, p to wid, pix probably, or _w, _p to avoid naming conflicts ...
    if verbose:
        print("Kernels parameters are set as w,p={},{}"
              " for the cooler with {} bp resolution.".format(w,p,binsize))

    kernels = {k: dotfinder.get_kernel(w,p,k) for k in ktypes}

    # creating logspace l(ambda)bins with base=2^(1/3), for lambda-chunking:
    nlchunks = dotfinder.HiCCUPS_W1_MAX_INDX
    base = 2**(1/3.0)
    ledges = np.concatenate(([-np.inf,],
                            np.logspace(0,
                                        nlchunks-1,
                                        num=nlchunks,
                                        base=base,
                                        dtype=np.float),
                            [np.inf,]))
    # the first bin must be (-inf,1]
    # the last bin must be (2^(HiCCUPS_W1_MAX_INDX/3),+inf):

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

    # ######################
    # # scoring only yields
    # # a HUGE list of both
    # # good and bad pixels
    # # (dot-like, and not)
    # ######################
    # scoring_step(clr, expected, expected_name, tiles, kernels,
    #              max_nans_tolerated, loci_separation_bins, output_scores,
    #              nproc, verbose)

    ################################
    # calculates genome-wide histogram (gw_hist):
    ################################
    gw_hist = dotfinder.scoring_and_histogramming_step(
        clr, expected, expected_name, tiles,
        kernels, ledges, max_nans_tolerated,
        loci_separation_bins, None, nproc,
        verbose)
    # gw_hist for each kernel contains a histogram of
    # raw pixel intensities for every lambda-chunk (one per column)
    # in a row-wise order, i.e. each column is a histogram
    # for each chunk ...

    if output_scores is not None:
        for k in kernels:
            gw_hist[k].to_csv(
                "{}.{}.hist.txt".format(output_scores,k),
                sep='\t',
                header=True,
                index=False,
                compression=None)

    ##############
    # prepare to determine the FDR threshold ...
    ##############


    # Reverse Cumulative Sum of a histogram ...
    rcs_hist = {}
    rcs_Poisson = {}
    qvalues = {}
    threshold_df = {}
    for k in kernels:
        # generate a reverse cumulative histogram for each kernel,
        #  such that 0th raw contains total # of pixels in each lambda-chunk:
        rcs_hist[k] = gw_hist[k].iloc[::-1].cumsum(axis=0).iloc[::-1]
        # now for every kernel-type k - create rcsPoisson,
        # a unit Poisson distribution for every lambda-chunk
        # using upper boundary of each lambda-chunk as the expected:
        rcs_Poisson[k] = pd.DataFrame()
        for mu, column in zip(ledges[1:-1], gw_hist[k].columns):
            # poisson.sf = 1 - poisson.cdf, but more precise
            # poisson.sf(-1,mu) == 1.0, i.e. is equivalent to the
            # poisson.pmf(gw_hist[k].index,mu)[::-1].cumsum()[::-1]
            # i.e., the way unitPoissonPMF is generated in HiCCUPS:
            renorm_factors = rcs_hist[k].loc[0,column]
            rcs_Poisson[k][column] = renorm_factors * poisson.sf(gw_hist[k].index-1, mu)
        # once we have both RCS hist and the Poisson:
        # now compare rcs_hist and re-normalized rcs_Poisson
        # to infer FDR thresolds for every lambda-chunk:
        fdr_diff = fdr * rcs_hist[k] - rcs_Poisson[k]
        # determine the threshold by checking the value at which
        # 'fdr_diff' first turns positive:
        threshold_df[k] = fdr_diff.where(fdr_diff>0).apply(lambda col: col.first_valid_index())
        # q-values ...
        # roughly speaking, qvalues[k] =  rcs_Poisson[k]/rcs_hist[k]
        # bear in mind some issues with lots of NaNs and Infs after
        # such a brave operation ...
        qvalues[k] = rcs_Poisson[k] / rcs_hist[k]
        # fill NaNs with the "unreachably" high value:
        very_high_value = len(rcs_hist[k])
        threshold_df[k] = threshold_df[k].fillna(very_high_value).astype(np.integer)

    #################
    # this way threshold_df's index is
    # a Categorical, where each element is
    # an IntervalIndex, which we can and should
    # use in the downstream analysis:

    ############################################
    # TODO: add q-values calculations !!!
    ############################################

    ##################################################################
    # each threshold_df[k] is a Series indexed by la_exp intervals
    # and it is all we need to extract "good" pixels from each chunk ...
    ##################################################################

    ###################
    # 'gw_hist' needs to be
    # processed and corresponding
    # FDR thresholds must be calculated
    # for every kernel-type
    # also q-vals for every 'obs.raw' value
    # and for every kernel-type must be stored
    # somewhere-somehow - the 'lognorm' thing
    # from HiCCUPS that would be ...
    ###################

    ###################
    # right after that
    # we'd have a scoring_and_filtering step
    # where the filtering part
    # would use FDR thresholds 'threshold_df'
    # calculated in the histogramming step ...
    ###################

    filtered_pix = dotfinder.scoring_and_extraction_step(
        clr, expected, expected_name, tiles, kernels,
        ledges, threshold_df, max_nans_tolerated,
        balance_factor, loci_separation_bins, output_calls,
        nproc, verbose)

    if verbose:
        print("preparing to extract needed q-values ...")

    # attempting to extract q-values using l-chunks and IntervalIndex:
    # we'll do it in an ugly but workign fashion, by simply
    # iteration over pairs of obs, la_exp and extracting needed qvals
    # one after another ...
    for k in kernels:
        filtered_pix["la_exp."+k+".qval"] = \
            [ qvalues[k].loc[o,e] for o,e \
                 in filtered_pix[["obs.raw","la_exp."+k+".value"]].itertuples(index=False) ]
    # qvalues : dict
    #   A dictionary with keys being kernel names and values pandas.DataFrame-s
    #   storing q-values: each column corresponds to a lambda-chunk,
    #   while rows correspond to observed pixels values.


    ######################################
    # post processing starts from here on:
    # it includes:
    # 0. remove low MAPQ reads (done externally ?!?)
    # 1. clustering
    # 2. filter pixels by FDR
    # 3. merge different resolutions. (external script)
    ######################################

    if verbose:
        print("Subsequent clustering and thresholding steps are not production-ready")

    # (1):
    centroids = dotfinder.clustering_step_local(filtered_pix, expected_chroms,
                                      dots_clustering_radius, verbose)
    # (2):
    out = dotfinder.thresholding_step(centroids)
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

    # (3):
    # Call dots for different resolutions individually and then use external methods
    # to merge the calls
