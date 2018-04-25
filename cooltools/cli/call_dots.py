import click

import cooler

from . import cli
from .. import snipping
# LazyToeplitz
from ..import loopify
# square_matrix_tiling, \
# get_adjusted_expected_tile_some_nans, \
# get_qvals, \
# clust_2D_pixels

import pandas as pd
import numpy as np

import multiprocessing as mp

from scipy.linalg import toeplitz
from scipy.stats import poisson

from functools import partial


# from bioframe import parse_humanized, parse_region_string
# # from scipy.ndimage import convolve
# # from scipy.linalg import toeplitz
# from copy import copy, deepcopy


def get_kernel(w,p,ktype):
    """
    Return typical kernels given size parameteres w, p
    and kernel type.
    
    Parameters
    ----------
    w : int
        Outer kernel size (actually half of it).
    p : int
        Inner kernel size (half of it).
    ktype : str
        Name of the kernel type, could be one of the 
        following: 'donut', 'vertical', 'horizontal',
        'lowleft', 'upright'.
        
    Returns
    -------
    kernel : ndarray
        A square matrix of int type filled with 1 and 0,
        according to the kernel type.
    """
    width = 2*w+1
    kernel = np.ones((width,width),dtype=np.int)
    # mesh grid:
    y,x = np.ogrid[-w:w+1, -w:w+1]

    if ktype == 'donut':
        # mask inner pXp square:
        mask = ((((-p)<=x)&(x<=p))&
                (((-p)<=y)&(y<=p)) )
        # mask vertical and horizontal
        # lines of width 1 pixel:
        mask += (x==0)|(y==0)
        # they are all 0:
        kernel[mask] = 0
    elif ktype == 'vertical':
        # mask outside of vertical line
        # of width 3:
        mask = (((-1>x)|(x>1))&((y>=-w)))
        # mask inner pXp square:
        mask += (((-p<=x)&(x<=p))&
                ((-p<=y)&(y<=p)) )
        # kernel masked:
        kernel[mask] = 0
    elif ktype == 'horizontal':
        # mask outside of horizontal line
        # of width 3:
        mask = (((-1>y)|(y>1))&((x>=-w)))
        # mask inner pXp square:
        mask += (((-p<=x)&(x<=p))&
                ((-p<=y)&(y<=p)) )
        # kernel masked:
        kernel[mask] = 0
    elif ktype == 'lowleft':
        # mask inner pXp square:
        mask = (((x>=-p))&
                ((y<=p)) )
        mask += (x>=0)
        mask += (y<=0)
        # kernel masked:
        kernel[mask] = 0
    elif ktype == 'upright':
        # mask inner pXp square:
        mask = (((x>=-p))&
                ((y<=p)) )
        mask += (x>=0)
        mask += (y<=0)
        # reflect that mask to
        # make it upper-right:
        mask = mask[::-1,::-1]
        # kernel masked:
        kernel[mask] = 0
    else:
        print("Only 'donut' kernel has been"
            "determined so far.")
        raise
        raise ValueError("Kernel-type {} has not been implemented yet".format(ktype))
    # 
    return kernel



def heatmap_tiles_generator_diag(clr, chroms, pad_size, tile_size, band_to_cover):
    """
    A generator yielding heatmap tiles that are needed
    to cover the requested band_to_cover around diagonal.
    Each tile is "padded" with pad_size edge to allow
    proper kernel-convolution of pixels close to boundary.
    
    
    Parameters
    ----------
    clr : cooler
        Cooler object to use to extract chromosome extents.
    chroms : iterable
        Iterable of chromosomes to process
    pad_size : int
        Size of padding around each tile.
        Typically the outer size of the kernel.
    tile_size : int
        Size of the heatmap tile.
    band_to_cover : int
        Size of the diagonal band to be covered by
        the generated tiles.
        Typically correspond to the max_loci_separation
        for called dots.
        
    Returns
    -------
    tile : tuple
        Generator of tuples of three, which contain
        chromosome name, row index of the tile,
        column index of the tile (chrom, tilei, tilej).
    """

    for chrom in chroms:
        chr_start, chr_stop = clr.extent(chrom)
        for tilei, tilej in loopify.square_matrix_tiling(chr_start,
                                                        chr_stop,
                                                        tile_size,
                                                        pad_size):
            # check if a given tile intersects with 
            # with the diagonal band of interest ...
            diag_from = tilej[0] - tilei[1]
            diag_to   = tilej[1] - tilei[0]
            #
            band_from = 0
            band_to   = band_to_cover
            # we are using this >2*padding trick to exclude
            # tiles from the lower triangle from calculations ...
            if (min(band_to,diag_to) - max(band_from,diag_from)) > 2*pad_size:
                yield chrom, tilei, tilej





##############################################################################
##############################################################################
##############################################################################
# # for every tile, described as (chrom, tile, tile) ...
def get_results_per_chrom_tile(tile_cij,
                            clr,
                            cis_exp,
                            exp_v_name,
                            bal_v_name,
                            kernels,
                            nans_tolerated,
                            band_to_cover,
                            verbose):
    """
    The main working function that given a tile of
    a heatmap, applies kernels to perform convolution
    to calculate locally-adjusted expected and then
    calculates a p-value for every meaningfull pixel
    against these l.a. expected values.
    
    
    Parameters
    ----------
    tile_cij : tuple
        Tuple of 3: chromosome name, tile span row-wise,
        tile span column-wise: (chrom, tile_i, tile_j),
        where tile_i = (start_i, end_i), and tile_j
         = (start_j, end_j).
    clr : cooler
        Cooler object to use to extract Hi-C heatmap data.
    cis_exp : pandas.DataFrame
        DataFrame with 1 dimensional expected, indexed with
        'chrom' and 'diag'.
    exp_v_name : str
        Name of a value column in expected DataFrame
    bal_v_name : str
        Name of a value column with balancing weights in
        a cooler.bins() DataFrame. Typically 'weight'.
    kernels : dict
        A dictionary with keys being kernels names and
        values being ndarrays representing those kernels.
    nans_tolerated : int
        Number of NaNs tolerated in a footprint of every kernel.
    band_to_cover : int
        Results would be stored only for pixels connecting
        loci closer than 'band_to_cover'.
    verbose : bool
        Enable verbose output.
        
    Returns
    -------
    res_df : pandas.DataFrame
        results: annotated pixels with calculated locally
        adjusted expected for every kernels, observed,
        precalculated pvalues, number of NaNs in footprint
        of every kernels, all of that in a form of an annotated
        pixels DataFrame for eligible pixels of a given tile.
    """

    # unpack tile's coordinates:
    chrom, tilei, tilej = tile_cij
    # we have to do it for every tile, because
    # chrom is not known apriori (maybe move outside):
    lazy_exp = snipping.LazyToeplitz(cis_exp.loc[chrom][exp_v_name].values)
    # let's keep i,j-part explicit here:
    origin = (tilei[0], tilej[0])
    # RAW observed matrix slice:
    observed = clr.matrix(balance=False)[slice(*tilei), slice(*tilej)]
    # expected as a rectangular tile :
    expected = lazy_exp[slice(*tilei), slice(*tilej)]
    # slice of balance_weight for row-span and column-span :
    bal_weight_i = clr.bins()[slice(*tilei)][bal_v_name].values
    bal_weight_j = clr.bins()[slice(*tilej)][bal_v_name].values
    # that's the main working function from loopify:
    res = loopify.get_adjusted_expected_tile_some_nans(origin = origin,
                                                     observed = observed,
                                                     expected = expected,
                                                     bal_weight = (bal_weight_i,bal_weight_j),
                                                     kernels = kernels,
                                                     verbose = verbose)
    # post-processing filters:
    # (1) exclude pixels that connect loci further than 'band_to_cover' apart: 
    is_inside_band = (res["bin1_id"] > (res["bin2_id"]-band_to_cover))
    # (2) identify pixels that pass number of NaNs compliance test for ALL kernels:
    does_comply_nans = np.all(res[["la_exp."+k+".nnans" for k in kernels]] < nans_tolerated, axis=1)
    # so, selecting inside band and nNaNs compliant results:
    # ( drop dropping index maybe ??? ) ...
    res_df = res[is_inside_band & does_comply_nans].reset_index(drop=True)
    # do Poisson tests:
    get_pval = lambda la_exp : 1.0 - poisson.cdf(res_df["obs.raw"], la_exp)
    for k in kernels:
        res_df["la_exp."+k+".pval"] = get_pval( res_df["la_exp."+k+".value"] )
    # that's all we can do for a given tile ...
    return cooler.annotate(res_df.reset_index(drop=True), clr.bins()[:])

















@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
# options ...
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
    "--output",
    help="Specify output file name where to store"
         " the results of dot-calling, in a BEDPE-like format.",
    type=str)



######################################
# TO BE DELETED ...
######################################
# params from jupyter notebook ...
# b = the_c.binsize
# band_2Mb = 2e+6
# band_idx = int(band_2Mb/b)
# nans_tolerated = 1
# tile_size = int(6e6/b)
# verbosity = False
# clust_radius=39000
# threshold_cluster = round(clust_radius/float(b))
# chromosomes = list(the_c.chromnames[:-2])
# # try to deal with chromosomes by checking NaNs in tiles ...
# kernels = dict( (kt, get_kernel(w,p,kt)) for kt in ktypes )
# # just use all kernels for now
# alpha=0.02






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
            output):
    """
    Call dots on a Hi-C heatmap that are not larger
    than max_loci_separation.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    EXPECTED_PATH : The paths to a tsv-like file with expected signal.

    Analysis will be performed for chromosomes
    referred to in EXPECTED_PATH, and therefore these
    chromosomes must be a subset of chromosomes
    referred to in COOL_PATH.
    Also chromosomes refered to in EXPECTED_PATH must
    be non-trivial, i.e., contain not-NaN signal.
    Thus, make sure to prune your EXPECTED_PATH before
    applying this script.


    COOL_PATH and EXPECTED_PATH must
    be binned at the same resolution.

    EXPECTED_PATH must contain at least the
    following columns for cis contacts:
    'chrom', 'diag', 'n_valid', value_name.
    value_name is controlled using options.
    Header must be present in a file.


    """


    # load cooler file to process:
    c = cooler.Cooler(cool_path)


    # read expected and make preparations for validation,
    # that's what we expect as column names:
    expected_columns = ['chrom', 'diag', 'n_valid', expected_name]
    # what would become a MultiIndex:
    expected_index = ['chrom', 'diag']
    # expected dtype as a rudimentary form of validation:
    expected_dtype = {'chrom':np.str, 'diag':np.int64, 'n_valid':np.int64, expected_name:np.float64}
    # unique list of chroms mentioned in expected_path:
    get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
    # compute # of bins by comparing matching indexes:
    get_exp_bins   = lambda df,ref_chroms: df.index.get_level_values("chrom").isin(ref_chroms).sum()
    # use 'usecols' as a rudimentary form of validation,
    # and dtype. Keep 'comment' and 'verbose' - explicit,
    # as we may use them later:
    expected = pd.read_table(
                            expected_path,
                            usecols  = expected_columns,
                            index_col= expected_index,
                            dtype    = expected_dtype,
                            comment  = None,
                            verbose  = verbose)


    #############################################
    # CROSS-VALIDATE COOLER and EXPECTED:
    #############################################
    # EXPECTED vs COOLER:
    # chromosomes to deal with 
    # are by default extracted
    # from the expected-file:
    expected_chroms = get_exp_chroms(expected)
    # do simple column-name validation for now:
    if not set(expected_chroms).issubset(c.chromnames):
        raise ValueError("Chromosomes in {} must be subset of chromosomes in cooler {}".format(expected_path,
                                                                                               cool_path))
    # check number of bins:
    expected_bins = get_exp_bins(expected, expected_chroms)
    cool_bins   = c.bins()[:]["chrom"].isin(expected_chroms).sum()
    if not (expected_bins==cool_bins):
        raise ValueError("Number of bins is not matching:",
                " {} in {}, and {} in {} for chromosomes {}".format(expected_bins,
                                                                   expected_path,
                                                                   cool_bins,
                                                                   cool_path,
                                                                   expected_chroms))
    #############################################
    # CROSS-VALIDATION IS COMPLETE.
    #############################################
    # COOLER and EXPECTED seems cross-compatible.
    if verbose:
        print("{} and {} passed cross-compatibility checks.".format(cool_path,
                                                                    expected_path))


    # prepare some parameters:
    # turn them from nucleotides dims to bins, etc.:
    binsize = c.binsize
    # 
    loci_separation_bins   = int(max_loci_separation/binsize)
    tile_size_bins         = int(tile_size/binsize)
    # # clustering would deal with bases-units for now, so
    # # supress this for now:
    # clustering_radius_bins = int(dots_clustering_radius/binsize)
    # 
    ktypes = ['donut', 'vertical', 'horizontal', 'lowleft', 'upright']
    #
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
    # rename w, p to wid, pix probably, or _w, _p to avoid na,ing conflicts ...

    # estimate number of tiles to be processed:
    num_tiles_approx = int(c.chromsizes[expected_chroms].sum()/(loci_separation_bins * binsize))

    # add very_verbose to supress output
    # from convolution of every tile:
    very_verbose = False
    # pre-fill 'get_results_per_chrom_tile' function with context parameteres:
    get_res_chrom_tile = partial(get_results_per_chrom_tile,
                                    clr = c,
                                    cis_exp = expected,
                                    exp_v_name = expected_name,
                                    bal_v_name = 'weight',
                                    kernels = {k: get_kernel(w,p,k) for k in ktypes},
                                    nans_tolerated = max_nans_tolerated,
                                    band_to_cover = loci_separation_bins,
                                    verbose = very_verbose)
    # main working "loop":
    if nproc > 1:
        with mp.Pool(nproc) as p:
            if verbose:
                print("creating a Pool of {} workers to tackle ~{} tiles".format(nproc, num_tiles_approx))
            # for every tile, described as (chrom, tile, tile) ...
            res_list = p.map(get_res_chrom_tile,
                             heatmap_tiles_generator_diag(c,
                                                          expected_chroms,
                                                          w,
                                                          tile_size_bins,
                                                          loci_separation_bins),
                             chunksize=int(num_tiles_approx/nproc))
    else:
        # serial implementation:
        if verbose:
            print("Serial implementation.")
        # for every tile, described as (chrom, tile, tile) ...
        res_list = \
            [get_res_chrom_tile(tij) for tij in heatmap_tiles_generator_diag(c,
                                                                             expected_chroms,
                                                                             w,
                                                                             tile_size_bins,
                                                                             loci_separation_bins)]


    if verbose:
        print("Convolution part is over!")
    # concatenate the results ...
    res_df = pd.concat(res_list, ignore_index=True)
    # 
    if verbose:
        print("major concat is over!")



    # do Benjamin-Hochberg FDR multiple hypothesis tests
    # genome-wide:
    for k in ktypes:
        res_df["la_exp."+k+".qval"] = loopify.get_qvals( res_df["la_exp."+k+".pval"] )

    # combine results of all tests:
    res_df['comply_fdr'] = np.all(res_df[["la_exp."+k+".qval" for k in ktypes]] <= fdr, axis=1)

    ######################################
    # post processing starts from here on:
    # it includes:
    # 0. remove low MAPQ reads ?!
    # 1. clustering
    # 2. filter pixels by FDR
    # 3. merge different resolutions.
    ######################################

    # (1)
    # now clustering would be needed ...
    # clustering is still per chromosome basis...
    # 
    # using different bin12_id_names since all
    # pixels are annotated at this point.
    pixel_clust_list = []
    for chrom  in expected_chroms:
        # probably generate one big DataFrame with clustering
        # information only and then just merge it with the
        # existing 'res_df'-DataFrame.
        # should we use groupby instead of 'res_df['chrom12']==chrom' ?!
        # to be tested ...
        pixel_clust = loopify.clust_2D_pixels(res_df[res_df['comply_fdr'] & \
                                      (res_df['chrom1']==chrom) & \
                                      (res_df['chrom2']==chrom)],
                                threshold_cluster = dots_clustering_radius,
                                bin1_id_name      = 'start1',
                                bin2_id_name      = 'start2')
        pixel_clust_list.append(pixel_clust)

    if verbose:
        print("Clustering is over!")
    # concatenate clustering results ...
    # indexing information persists here ...
    pixel_clust_df = pd.concat(pixel_clust_list, ignore_index=False)
    # now merge pixel_clust_df and res_df DataFrame ...
    # # and merge (index-wise) with the main DataFrame:
    res_df_comply =  res_df[res_df['comply_fdr']].merge(
                                            pixel_clust_df,
                                            how='left',
                                            left_index=True,
                                            right_index=True) 

    # report only centroids with highest Observed:
    res_chrom_clust_group = res_df_comply.groupby(["chrom1","chrom2","c_label"])
    res_df_centroids      = res_df_comply.loc[res_chrom_clust_group["obs.raw"].idxmax()]
    ###################################
    # everyhting works up until here ...
    # great stuff !!!
    ###################################


    # (2)
    # filter by FDR, enrichment etc:
    enrichment_factor_1 = 1.5
    enrichment_factor_2 = 1.75
    enrichment_factor_3 = 2.0
    FDR_orphan_threshold = 0.02
    # 
    enrichment_fdr_comply = \
        (res_df_centroids["obs.raw"] > enrichment_factor_2*res_df_centroids["la_exp.lowleft.value"]) & \
        (res_df_centroids["obs.raw"] > enrichment_factor_2*res_df_centroids["la_exp.donut.value"]) & \
        (res_df_centroids["obs.raw"] > enrichment_factor_1*res_df_centroids["la_exp.vertical.value"]) & \
        (res_df_centroids["obs.raw"] > enrichment_factor_1*res_df_centroids["la_exp.horizontal.value"]) & \
        ( (res_df_centroids["obs.raw"] > enrichment_factor_3*res_df_centroids["la_exp.lowleft.value"]) | \
            (res_df_centroids["obs.raw"] > enrichment_factor_3*res_df_centroids["la_exp.donut.value"]) ) & \
        ( (res_df_centroids["c_size"] > 1) | \
            ((res_df_centroids["la_exp.lowleft.qval"] + \
             res_df_centroids["la_exp.donut.qval"] + \
             res_df_centroids["la_exp.vertical.qval"] + \
             res_df_centroids["la_exp.horizontal.qval"]) <= FDR_orphan_threshold) )
    # use "enrichment_fdr_comply" to filter out 
    # non-satisfying pixels:
    res_df_output = res_df_centroids[enrichment_fdr_comply]


    # # # columns up for grabs, take whatever you need:
    # chrom1
    # start1
    # end1
    # weight1
    # chrom2
    # start2
    # end2
    # weight2
    # la_exp.donut.value
    # la_exp.donut.nnans
    # la_exp.vertical.value
    # la_exp.vertical.nnans
    # la_exp.horizontal.value
    # la_exp.horizontal.nnans
    # la_exp.lowleft.value
    # la_exp.lowleft.nnans
    # la_exp.upright.value
    # la_exp.upright.nnans
    # exp.raw
    # obs.raw
    # la_exp.donut.pval
    # la_exp.vertical.pval
    # la_exp.horizontal.pval
    # la_exp.lowleft.pval
    # la_exp.upright.pval
    # la_exp.donut.qval
    # la_exp.vertical.qval
    # la_exp.horizontal.qval
    # la_exp.lowleft.qval
    # la_exp.upright.qval
    # comply_fdr
    # cstart1
    # cstart2
    # c_label
    # c_size


    # # tentaive output coplumns list:
    columns_for_output = \
            ['chrom1',
            'start1',
            'end1',
            'chrom2',
            'start2',
            'end2',
            'la_exp.donut.value',
            'la_exp.vertical.value',
            'la_exp.horizontal.value',
            'la_exp.lowleft.value',
            'la_exp.upright.value',
            'exp.raw',
            'obs.raw',
            'la_exp.donut.qval',
            'la_exp.vertical.qval',
            'la_exp.horizontal.qval',
            'la_exp.lowleft.qval',
            'la_exp.upright.qval',
            'cstart1',
            'cstart2',
            'c_label',
            'c_size']


    ##############################
    # OUTPUT:
    ##############################
    if output is not None:
        res_df_output[columns_for_output].to_csv(
                                        output,
                                        sep='\t',
                                        header=True,
                                        index=False,
                                        compression=None)

