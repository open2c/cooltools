import os.path as op
import pandas as pd
import numpy as np
import cooler
import bioframe

import click
from . import cli
from ..lib.common import assign_regions
from .. import dotfinder
from . import util


@cli.command()
@click.argument(
    "cool_path", metavar="COOL_PATH", type=str, nargs=1,
)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1,
)
@click.option(
    "--view",
    "--regions",
    help="Path to a BED file with the definition of viewframe (regions)"
    " used in the calculation of EXPECTED_PATH. Dot-calling will be"
    " performed for these regions independently e.g. chromosome arms."
    " When not provided regions will be interpreted from `region` column"
    " of EXPECTED_PATH (UCSC formatted, or full chromosome names)."
    " Note that '--regions' is the deprecated name of the option. Use '--view' instead. ",
    type=click.Path(exists=False, dir_okay=False),
    default=None,
    show_default=True,
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
def call_dots(
    cool_path,
    expected_path,
    view,
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
    out_prefix,
):
    """
    Call dots on a Hi-C heatmap that are not larger than max_loci_separation.

    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    EXPECTED_PATH : The paths to a tsv-like file with expected cis-expected.

    Analysis will be performed for chromosomes referred to in EXPECTED_PATH, and
    therefore these chromosomes must be a subset of chromosomes referred to in
    COOL_PATH. Also chromosomes refered to in EXPECTED_PATH must be non-trivial,
    i.e., contain not-NaN signal. Thus, make sure to prune your EXPECTED_PATH
    before applying this script.

    COOL_PATH and EXPECTED_PATH must be binned at the same resolution.

    EXPECTED_PATH must contain at least the following columns for cis contacts:
    'region', 'diag', 'n_valid', value_name. value_name is controlled using
    options. Header must be present in a file.

    """
    clr = cooler.Cooler(cool_path)

    # preliminary SCHEMA for cis-expected
    region_column_name = "region"
    expected_columns = [region_column_name, "diag", "n_valid", expected_name]
    expected_dtypes = {
        region_column_name: np.str,
        "diag": np.int64,
        "n_valid": np.int64,
        expected_name: np.float64,
    }

    try:
        expected = pd.read_table(
            expected_path,
            usecols=expected_columns,
            dtype=expected_dtypes,
            comment=None,
            verbose=verbose,
        )
    except ValueError as e:
        raise ValueError(
            "input expected does not match the schema\n"
            "tab-separated expected file must have a header as well"
        )
    expected_index = [
        region_column_name,
        "diag",
    ]
    expected.set_index(expected_index, inplace=True)
    # end of SCHEMA for cis-expected

    # Create a viewframe from regions in expected table:
    try:
        # Construct DataFrame of names of expected that will be parsed
        # by bioframe.from_ucsc_string_list:
        uniq_regions = pd.DataFrame(
            {"name": expected.index.get_level_values(region_column_name).unique()}
        )
        expected_regions_df = bioframe.make_viewframe(
            uniq_regions, check_bounds=clr.chromsizes
        )
    except ValueError as e:
        print(e)
        raise ValueError(
            "Cannot interpret regions from EXPECTED_PATH\n"
            "specify regions definitions using --view option."
        )

    # Create view_df,
    # use expected regions by default:
    if view is None:
        view_df = expected_regions_df
    # or use custom view if file provided:
    else:
        # Read view dataframe:
        try:
            view_df = bioframe.read_table(view, schema="bed4", index_col=False)
        except Exception:
            view_df = bioframe.read_table(view, schema="bed3", index_col=False)
        # Convert view dataframe to viewframe:
        try:
            view_df = bioframe.make_viewframe(
                view_df, check_bounds=clr.chromsizes
            )
        except ValueError as e:
            raise RuntimeError(
                "View table is incorrect, please, comply with the format. "
            ) from e

        assert bioframe.is_contained(
            view_df, expected_regions_df
        ), "View and expected are for different regions"
    # Verify appropriate columns order (required for heatmap_tiles_generator_diag):
    view_df = view_df[["chrom", "start", "end", "name"]]

    # check number of bins per region in cooler and expected table
    # compute # of bins by comparing matching indexes
    try:
        for region_name, group in expected.reset_index().groupby(region_column_name):
            n_diags = group.shape[0]
            region = view_df.set_index("name").loc[region_name]
            lo, hi = clr.extent(region)
            assert n_diags == (hi - lo)
    except AssertionError:
        raise ValueError(
            "Region shape mismatch between expected and cooler. "
            "Are they using the same resolution?"
        )
    # All the checks have passed:
    if verbose:
        print(
            "{} and {} passed cross-compatibility checks.".format(
                cool_path, expected_path
            )
        )

    # by now we have a usable region_table and expected for most scenarios

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
        print(f"Using kernel parameters w={w}, p={p} recommended for binsize {binsize}")
    else:
        w, p = kernel_width, kernel_peak
        # add some sanity check for w,p:
        assert w > p, f"Wrong inner/outer kernel parameters w={w}, p={p}"
        print(f"Using kernel parameters w={w}, p={p} provided by user")

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
            clr, view_df, w, tile_size_bins, loci_separation_bins
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
        op.join(op.dirname(out_prefix), op.basename(out_prefix) + ".enriched.tsv"),
        nproc,
        verbose,
        bin1_id_name="bin1_id",
        bin2_id_name="bin2_id",
    )

    # 4. Post-processing
    if verbose:
        print(f"Begin post-processing of {len(filtered_pixels)} filtered pixels")
        print("preparing to extract needed q-values ...")

    filtered_pixels_qvals = dotfinder.annotate_pixels_with_qvalues(
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
    centroids = dotfinder.clustering_step(
        filtered_pixels_annotated,
        expected_regions_df["name"],
        dots_clustering_radius,
        verbose,
    )

    # 4b. filter by enrichment and qval
    postprocessed_calls = dotfinder.thresholding_step(centroids)

    # Final-postprocessed result
    if out_prefix is not None:

        postprocessed_fname = op.join(
            op.dirname(out_prefix), op.basename(out_prefix) + ".postproc.bedpe"
        )

        postprocessed_calls.to_csv(
            postprocessed_fname, sep="\t", header=True, index=False, compression=None
        )
