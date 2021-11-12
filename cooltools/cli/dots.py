import os.path as op
from functools import partial
import pandas as pd
import numpy as np
import cooler
import bioframe
import logging

import click
from . import cli
from .. import api


from ..lib.common import assign_regions, \
                        read_expected, \
                        read_viewframe, \
                        make_cooler_view

from .util import validate_csv

logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str,
    nargs=1,
)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=str,
    callback=partial(validate_csv, default_column="balanced.avg"),
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file with the definition of viewframe (regions)"
    " used in the calculation of EXPECTED_PATH. Dot-calling will be"
    " performed for these regions independently e.g. chromosome arms."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=False, dir_okay=False),
    default=None,
    show_default=True,
)
@click.option(
    "--clr-weight-name",
    help="Use cooler balancing weight with this name.",
    type=str,
    default="weight",
    show_default=True,
)
@click.option(
    "-p",
    "--nproc",
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
    " between. Same as HiCCUPS_W1_MAX_INDX (40) in the original HiCCUPS.",
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
    "-v", "--verbose", help="Enable verbose output", is_flag=True, default=False
)
@click.option(
    "-o",
    "--out-prefix",
    help="Specify prefix for the output file, to store results of dot-calling:"
    " all enriched pixels as prefix + '.enriched.tsv',"
    " and post-processed dots (clustered,filtered) as prefix + '.postproc.bedpe'",
    type=str,
    required=True,
)
def dots(
    cool_path,
    expected_path,
    view,
    clr_weight_name,
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
    out_prefix,
):
    """
    Call dots on a Hi-C heatmap that are not larger than max_loci_separation.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    EXPECTED_PATH : The paths to a tsv-like file with expected signal,
    including a header. Use the '::' syntax to specify a column name.

    Analysis will be performed for chromosomes referred to in EXPECTED_PATH, and
    therefore these chromosomes must be a subset of chromosomes referred to in
    COOL_PATH. Also chromosomes refered to in EXPECTED_PATH must be non-trivial,
    i.e., contain not-NaN signal. Thus, make sure to prune your EXPECTED_PATH
    before applying this script.

    COOL_PATH and EXPECTED_PATH must be binned at the same resolution.

    EXPECTED_PATH must contain at least the following columns for cis contacts:
    'region1/2', 'dist', 'n_valid', value_name. value_name is controlled using
    options. Header must be present in a file.

    """
    clr = cooler.Cooler(cool_path)
    expected_path, expected_value_col = expected_path

    #### Generate viewframes ####
    # 1:cooler_view_df. Generate viewframe from clr.chromsizes:
    cooler_view_df = make_cooler_view(clr)

    # 2:view_df. Define global view for calculating calling dots
    # use input "view" BED file or all chromosomes :
    if view is None:
        view_df = cooler_view_df
    else:
        view_df = read_viewframe(view, clr, check_sorting=True)

    #### Read expected: ####
    expected_summary_cols = [expected_value_col, ]
    expected = read_expected(
        expected_path,
        contact_type="cis",
        expected_value_cols=expected_summary_cols,
        verify_view=view_df,
        verify_cooler=clr,
    )
    # add checks to make sure cis-expected is symmetric

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
        w, p = api.dotfinder.recommend_kernel_params(binsize)
        logging.info(f"Using kernel parameters w={w}, p={p} recommended for binsize {binsize}")
    else:
        w, p = kernel_width, kernel_peak
        # add some sanity check for w,p:
        if not w > p:
            raise ValueError(f"Wrong inner/outer kernel parameters w={w}, p={p}")
        logging.info(f"Using kernel parameters w={w}, p={p} provided by user")

    # once kernel parameters are setup check max_nans_tolerated
    # to make sure kernel footprints overlaping 1 side with the
    # NaNs filled row/column are not "allowed"
    # this requires dynamic adjustment for the "shrinking donut"
    if not max_nans_tolerated <= 2 * w:
        raise ValueError("Too many NaNs allowed!")
    # may lead to scoring the same pixel twice, - i.e. duplicates.

    # generate standard kernels - consider providing custom ones
    kernels = {k: api.dotfinder.get_kernel(w, p, k) for k in ktypes}

    # list of tile coordinate ranges
    tiles = list(
        api.dotfinder.heatmap_tiles_generator_diag(
            clr, view_df, w, tile_size_bins, loci_separation_bins
        )
    )

    # lambda-chunking edges ...
    if not 40 <= num_lambda_chunks <= 50:
        raise ValueError("Incompatible num_lambda_chunks")
    base = 2 ** (1 / 3)
    ledges = np.concatenate(
        (
            [-np.inf],
            np.logspace(
                0,
                num_lambda_chunks - 1,
                num=num_lambda_chunks,
                base=base,
                dtype=np.float64,
            ),
            [np.inf],
        )
    )

    # 1. Calculate genome-wide histograms of scores.
    gw_hist = api.dotfinder.scoring_and_histogramming_step(
        clr,
        expected.set_index(["region1","region2","dist"]),
        expected_value_col,
        clr_weight_name,
        tiles,
        kernels,
        ledges,
        max_nans_tolerated,
        loci_separation_bins,
        nproc,
        verbose,
    )

    if verbose:
        logging.info("Done building histograms ...")

    # 2. Determine the FDR thresholds.
    threshold_df, qvalues = api.dotfinder.determine_thresholds(
        kernels, ledges, gw_hist, fdr
    )

    # 3. Filter using FDR thresholds calculated in the histogramming step
    filtered_pixels = api.dotfinder.scoring_and_extraction_step(
        clr,
        expected.set_index(["region1","region2","dist"]),
        expected_value_col,
        clr_weight_name,
        tiles,
        kernels,
        ledges,
        threshold_df,
        max_nans_tolerated,
        balance_factor,
        loci_separation_bins,
        op.join(op.dirname(out_prefix), op.basename(out_prefix) + ".enriched.tsv"),
        nproc,
        verbose,
        bin1_id_name="bin1_id",
        bin2_id_name="bin2_id",
    )

    # 4. Post-processing
    if verbose:
        logging.info(f"Begin post-processing of {len(filtered_pixels)} filtered pixels")
        logging.info("preparing to extract needed q-values ...")

    filtered_pixels_qvals = api.dotfinder.annotate_pixels_with_qvalues(
        filtered_pixels, qvalues, kernels
    )
    # 4a. clustering
    ########################################################################
    # Clustering has to be done using annotated DataFrame of filtered pixels
    # why ? - because - clustering has to be done independently for every region!
    ########################################################################
    filtered_pixels_annotated = cooler.annotate(filtered_pixels_qvals, clr.bins()[:])
    filtered_pixels_annotated = assign_regions(filtered_pixels_annotated, view_df)
    # consider reseting index here
    centroids = api.dotfinder.clustering_step(
        filtered_pixels_annotated,
        view_df["name"],
        dots_clustering_radius,
        verbose,
    )

    # 4b. filter by enrichment and qval
    postprocessed_calls = api.dotfinder.thresholding_step(centroids)

    # Final-postprocessed result
    if out_prefix is not None:

        postprocessed_fname = op.join(
            op.dirname(out_prefix), op.basename(out_prefix) + ".postproc.bedpe"
        )

        postprocessed_calls.to_csv(
            postprocessed_fname, sep="\t", header=True, index=False, compression=None
        )
