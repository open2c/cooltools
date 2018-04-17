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












    ##############################
    # OUTPUT AND PLOTTING:
    ##############################
    if output is not None:
        np.savez(
            file = output+".saddledump", # .npz auto-added
            saddledata = saddledata,
            binedges = binedges,
            digitized = digitized)

    # compute naive compartment strength
    # if requested by the user:    
    if compute_strength:
        strength = get_compartment_strength(saddledata, fraction_for_strength)
        print("Comparment strength = {}".format(strength),
            "\ncalculated using {} of saddledata bins".format(fraction_for_strength),
            "to compute median intra- and inter-compartmental signal")


    if savefig is not None:
        try:
            import matplotlib as mpl
            # savefig only for now:
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            pal = sns.color_palette('muted')
        except ImportError:
            print("Install matplotlib and seaborn to use ", file=sys.stderr)
            sys.exit(1)


