import os.path as op
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
    type=str,  # click.Path(exists=True, dir_okay=False),
    nargs=1,
)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1,
)
@click.option(
    "--expected-name",
    help="Name of value column in EXPECTED_PATH",
    type=str,
    default="balanced.avg",
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
    "-p", "--nproc",
    help="Number of processes to split the work between."
    " [default: 1, i.e. no process pool]",
    default=1,
    type=int,
)
@click.option(
    "--max-loci-separation",
    help="Limit loci separation for dot-calling, i.e., do not call dots for"
    " loci that are further than max_loci_separation basepair apart."
    " 2-20MB is reasonable and would capture most of CTCF-dots.",
    type=int,
    default=2000000,
    show_default=True,
)
@click.option(
    "--max-nans-tolerated",
    help="Maximum number of NaNs tolerated in a footprint of every used filter."
    " Must be controlled with caution, as large max-nans-tolerated, might lead to"
    ' pixels scored in the padding area of the tiles to "penetrate" to the list'
    " of scored pixels for the statistical testing. [max-nans-tolerated <= 2*w ]",
    type=int,
    default=1,
    show_default=True,
)
@click.option(
    "--tile-size",
    help="Tile size for the Hi-C heatmap tiling."
    " Typically on order of several mega-bases, and <= max_loci_separation.",
    type=int,
    default=6000000,
    show_default=True,
)
@click.option(
    "--kernel-width",
    help="Outer half-width of the convolution kernel in pixels"
    " e.g. outer size (w) of the 'donut' kernel, with the 2*w+1"
    " overall footprint of the 'donut'.",
    type=int,
)
@click.option(
    "--kernel-peak",
    help="Inner half-width of the convolution kernel in pixels"
    " e.g. inner size (p) of the 'donut' kernel, with the 2*p+1"
    " overall footprint of the punch-hole.",
    type=int,
)
@click.option(
    "--num-lambda-chunks",
    help="Number of log-spaced bins to divide your adjusted expected"
    " between. Same as HiCCUPS_W1_MAX_INDX in the original HiCCUPS.",
    type=int,
    default=45,
    show_default=True,
)
@click.option(
    "--fdr",
    help="False discovery rate (FDR) to control in the multiple"
    " hypothesis testing BH-FDR procedure.",
    type=float,
    default=0.02,
    show_default=True,
)
@click.option(
    "--dots-clustering-radius",
    help="Radius for clustering dots that have been called too close to each other."
    "Typically on order of 40 kilo-bases, and >= binsize.",
    type=int,
    default=39000,
    show_default=True,
)
@click.option(
    "-v", "--verbose",
    help="Enable verbose output",
    is_flag=True,
    default=False
)
@click.option(
    "-s", "--output-scores",
    help="At the moment it is a redundant option that"
    " does nothing. Reserve it for a better dump"
    " of convolved scores.",
    type=str,
    required=False,
)
@click.option(
    "--output-hists",
    help="Specify output file name to store"
    " lambda-chunked histograms. [Not implemented yet]",
    type=str,
    required=False,
)
@click.option(
    "-o", "--output-calls",
    help="Specify output file name where to store"
    " the results of dot-calling, in a BEDPE format."
    " Pre-processed dots are stored in that file."
    " Post-processed dots are stored in the .postproc one.",
    type=str,
)
@click.option(
    "--score-dump-mode",
    help="Specify file format for the dump of convolved scores."
    " This dump is used for the downstream processing and"
    " is read twice. Now 'parquet' is the only supported"
    " format. 'cooler' and 'hdf' in the future.",
    type=str,
    default="parquet",
    show_default=True,
)
@click.option(
    "--temp-dir",
    help="Create temporary files in specified directory.",
    type=str,
    default=".",
    show_default=True,
)
@click.option(
    "--no-delete-temp",
    help="Do not delete temporary files when finished.",
    is_flag=True,
    default=False,
)
def call_dots(
    cool_path,
    expected_path,
    expected_name,
    weight_name,
    nproc,
    max_loci_separation,
    max_nans_tolerated,
    tile_size,
    kernel_width,
    kernel_peak,
    num_lambda_chunks,
    fdr,
    dots_clustering_radius,
    verbose,
    output_scores,
    output_hists,
    output_calls,
    score_dump_mode,
    temp_dir,
    no_delete_temp,
):
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

    expected_columns = ["chrom", "diag", "n_valid", expected_name]
    expected_index = ["chrom", "diag"]
    expected_dtypes = {
        "chrom": np.str,
        "diag": np.int64,
        "n_valid": np.int64,
        expected_name: np.float64,
    }
    expected = pd.read_table(
        expected_path,
        usecols=expected_columns,
        dtype=expected_dtypes,
        comment=None,
        verbose=verbose,
    )
    expected.set_index(expected_index, inplace=True)

    # Input validation
    # unique list of chroms mentioned in expected_path
    # do simple column-name validation for now
    get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
    expected_chroms = get_exp_chroms(expected)
    if not set(expected_chroms).issubset(clr.chromnames):
        raise ValueError(
            "Chromosomes in {} must be subset of ".format(expected_path)
            + "chromosomes in cooler {}".format(cool_path)
        )
    # check number of bins
    # compute # of bins by comparing matching indexes
    get_exp_bins = lambda df, ref_chroms: (
        df.index.get_level_values("chrom").isin(ref_chroms).sum()
    )
    expected_bins = get_exp_bins(expected, expected_chroms)
    cool_bins = clr.bins()[:]["chrom"].isin(expected_chroms).sum()
    if not (expected_bins == cool_bins):
        raise ValueError(
            "Number of bins is not matching: ",
            "{} in {}, and {} in {} for chromosomes {}".format(
                expected_bins, expected_path, cool_bins, cool_path, expected_chroms
            ),
        )
    if verbose:
        print(
            "{} and {} passed cross-compatibility checks.".format(
                cool_path, expected_path
            )
        )

    # Prepare some parameters.
    binsize = clr.binsize
    loci_separation_bins = int(max_loci_separation / binsize)
    tile_size_bins = int(tile_size / binsize)
    balance_factor = 1.0  # clr._load_attrs("bins/weight")["scale"]

    # clustering would deal with bases-units for now, so supress this for now
    # clustering_radius_bins = int(dots_clustering_radius/binsize)

    # kernels
    # 'upright' is a symmetrical inversion of "lowleft", not needed.
    ktypes = ["donut", "vertical", "horizontal", "lowleft"]

    if (kernel_width is None) or (kernel_peak is None):
        w, p = dotfinder.recommend_kernel_params(binsize)
        print(
            "Using kernel parameters w={}, p={} recommended for binsize {}".format(
                w, p, binsize
            )
        )
    else:
        w, p = kernel_width, kernel_peak
        # add some sanity check for w,p:
        assert w > p, "Wrong inner/outer kernel parameters w={}, p={}".format(w, p)
        print("Using kernel parameters w={}, p={} provided by user".format(w, p))

    # once kernel parameters are setup check max_nans_tolerated
    # to make sure kernel footprints overlaping 1 side with the
    # NaNs filled row/column are not "allowed"
    # this requires dynamic adjustment for the "shrinking donut"
    assert max_nans_tolerated <= 2 * w, "Too many NaNs allowed!"
    # may lead to scoring the same pixel twice, - i.e. duplicates.

    # generate standard kernels - consider providing custom ones
    kernels = {k: dotfinder.get_kernel(w, p, k) for k in ktypes}

    # list of tile coordinate ranges
    tiles = list(
        dotfinder.heatmap_tiles_generator_diag(
            clr, expected_chroms, w, tile_size_bins, loci_separation_bins
        )
    )

    # lambda-chunking edges ...
    assert dotfinder.HiCCUPS_W1_MAX_INDX <= num_lambda_chunks <= 50
    base = 2 ** (1 / 3)
    ledges = np.concatenate(
        (
            [-np.inf],
            np.logspace(
                0,
                num_lambda_chunks - 1,
                num=num_lambda_chunks,
                base=base,
                dtype=np.float,
            ),
            [np.inf],
        )
    )

    # 1. Calculate genome-wide histograms of scores.
    gw_hist = dotfinder.scoring_and_histogramming_step(
        clr,
        expected,
        expected_name,
        weight_name,
        tiles,
        kernels,
        ledges,
        max_nans_tolerated,
        loci_separation_bins,
        nproc,
        verbose,
    )

    if verbose:
        print("Done building histograms ...")

    # 2. Determine the FDR thresholds.
    threshold_df, qvalues = dotfinder.determine_thresholds(
        kernels, ledges, gw_hist, fdr
    )

    # 3. Filter using FDR thresholds calculated in the histogramming step
    filtered_pixels = dotfinder.scoring_and_extraction_step(
        clr,
        expected,
        expected_name,
        weight_name,
        tiles,
        kernels,
        ledges,
        threshold_df,
        max_nans_tolerated,
        balance_factor,
        loci_separation_bins,
        output_calls,
        nproc,
        verbose,
    )

    # 4. Post-processing
    if verbose:
        print(
            "Begin post-processing of {} filtered pixels".format(len(filtered_pixels))
        )
        print("preparing to extract needed q-values ...")

    filtered_pixels_qvals = dotfinder.annotate_pixels_with_qvalues(
        filtered_pixels, qvalues, kernels
    )
    # 4a. clustering
    ########################################################################
    # Clustering has to be done using annotated DataFrame of filtered pixels
    # why ? - because - clustering has to be done chromosome by chromosome !
    ########################################################################
    filtered_pixels_annotated = cooler.annotate(filtered_pixels_qvals, clr.bins()[:])
    centroids = dotfinder.clustering_step(
        filtered_pixels_annotated, expected_chroms, dots_clustering_radius, verbose
    )

    # 4b. filter by enrichment and qval
    postprocessed_calls = dotfinder.thresholding_step(centroids)

    # Final-postprocessed result
    if output_calls is not None:

        postprocessed_fname = op.join(
            op.dirname(output_calls), op.basename(output_calls) + ".postproc"
        )

        postprocessed_calls.to_csv(
            postprocessed_fname, sep="\t", header=True, index=False, compression=None
        )
