import click

import cooler

from . import cli
from .. import saddle


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cooltools import saddle
from scipy.linalg import toeplitz

from bioframe import parse_humanized, parse_region_string
import seaborn as sns
pal = sns.color_palette('muted')

# inspired by:
# saddles.py by @nvictus
# https://github.com/nandankita/labUtilityTools


################################################
# helper getter functions ...
################################################
def make_cis_obsexp_fetcher(clr, expected, name):
    g = expected.groupby('chrom')
    return lambda chrom, _: (
            clr.matrix().fetch(chrom) / 
                toeplitz(g.get_group(chrom)[name].values)
        )
# old one with a single trans_expected for an entire genome:
def OLD_make_trans_obsexp_fetcher(clr, trans_avg):
    return lambda chrom1, chrom2: (
            clr.matrix().fetch(chrom1, chrom2) / trans_avg
        )
# NEW one with a trans_expected for every pair of chroms:
def NEW_make_trans_obsexp_fetcher(clr, trans_exp, name):
    denominator_exp = lambda chrom1, chrom2: trans_exp.loc[chrom1, chrom2][name] if (chrom1, chrom2) in trans_exp.index else trans_exp.loc[chrom2, chrom1][name]
    return lambda chrom1, chrom2: (
            clr.matrix().fetch(chrom1, chrom2) / denominator_exp(chrom1, chrom2)
        # .loc is the right way to get [chrom1,chrom2] value from MultiIndex df:
        # https://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-indexing-with-hierarchical-index
        )
# Note on the chrom1, chrom2 ordering in trans_exp:
# trans_exp.loc['chr2','chr10'] - is valid value
# trans_exp.loc['chr10','chr2'] - is NOT !!!!
# go figure ...
# 
# following works:
# ('chr2','chr1') in trans_exp.index
# 
def make_track_fetcher(df, name):
    g = df.groupby('chrom')
    return lambda chrom: g.get_group(chrom)[name]
# the mask ...
track_mask_fetcher = lambda chrom: (
    (track.groupby('chrom')
          .get_group(chrom)[track_name]
          .notnull()) &
    (track.groupby('chrom')
          .get_group(chrom)[track_name] != 0)
).values






# def saddleplot(
#         clr,
#         track,
#         track_name,
#         contact_type='cis',
#         expected=None,
#         expected_name=None,
#         chromosomes=None,
#         n_bins=50,
#         prange=(0.25, 99.75),
#         by_percentile=False,
#         color=pal[2],
#         fig_kws=None,
#         heatmap_kws=None, 
#         margin_kws=None):
#     """
#     Make a saddle plot for an arbitrary signal track on the genomic bins of 
#     a contact matrix.
#     Parameters
#     ----------
#     track_name : str
#         Name of value column in ``track``.
#     expected_name : str, optional
#         Required if ``contact_type`` is 'cis'. Name of value column in the
#         ``expected`` data frame.
#     chromosomes : list of str, optional
#         Restricted set of chromosomes to use. Default is to use all chromosomes
#         in the ``track`` dataframe.
#     bin_range_percentile : float, optional
#         ...
#     heatmap_kws : dict, optional
#         Extra keywords to pass to ``imshow`` for saddle heatmap.
#     margin_kws : dict, optional
#         Extra keywords to pass to ``hist`` for left and top margins.



@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=str,
    nargs=1)
#     track : pandas.DataFrame
#         bedGraph-like data frame ('chrom', 'start', 'end', <name>).
@click.argument(
    "track_path",
    metavar="TRACK_PATH",
    type=str,
    nargs=1)
#     expected : pandas.DataFrame or number, optional
#         Expected contact frequency. If ``contact_type`` is 'cis', then this is
#         a data frame of ('chrom', 'distance', <name>). If ``contact_type`` is 
#         'trans', then this is a single precomputed number equal to the average 
#         trans contact frequency.
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=str,
    nargs=1)
# options ...
@click.option(
    # optional
    "--n-bins",
    help="Number of bins for digitizing the signal track.",
    type=int,
    default=50,
    show_default=True)
@click.option(
    '--contact-type',
    help="Type of the contacts to aggregate"
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True,
    )
@click.option(
    # optional
    "--by-percentile",
    help="Whether to bin the signal track by percentile"
         " rather than by value.",
    is_flag=True,
    default=False)


# @click.option(
#     "--tol",
#     help="Threshold value of variance of the marginals for the algorithm to "
#          "converge.",
#     type=float,
#     default=1e-5,
#     show_default=True)

def compute_saddle(
            cool_path,
            track_path,
            expected_path,
            n_bins,
            contact_type,
            by_percentile):
    """
    Calculate saddle statistics and
    generate saddle plots.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    TRACK_PATH : The paths to bedGraph-file with a binned compartment track (eigenvalues).

    EXPECTED_PATH : The paths to a csv-like file with expected signal.

    """

    # load cooler file to process:
    c = cooler.Cooler(cool_path)

    # neccessary ingredients ...
    cis_exp = pd.read_table("PTB2539-NT.200000.cis.expected",index_col=[0,1])
    trans_exp = pd.read_table("PTB2539-NT.200000.trans.expected",index_col=[0,1])


    # read BED-file :
    track = pd.read_csv("Hap1-WT-NT_Eigen_values.trackfile",  sep='\t',  names=['chrom', 'start', 'end', 'eigen'])
    # or do tht as an alternative:
    # ...
    from bioframe.io import formats
    from bioframe.schemas import SCHEMAS
    formats.read_table("Hap1-WT-NT_Eigen_values.trackfile", schema=SCHEMAS["bed4"])


    # digitize the track (compartment track)
    # no matter the contact type ...
    digitized, binedges = saddle.digitize_track(
        bins=50,
        get_track = track_fetcher,
        get_mask  = track_mask_fetcher,
        chromosomes = track['chrom'].unique(),
        prange = (1.0, 99.75),
        by_percentile = False
    )



    # get_matrix,
    # get_digitized,
    # chromosomes,
    # contact_type,
    # verbose=False

    # playing with OBS/EXP a bit : ...

    obsexp_func = make_cis_obsexp_fetcher(c, cis_exp, 'balanced.avg')


    t_obsexp_func = NEW_make_trans_obsexp_fetcher(c, trans_exp, 'balanced.avg')

    # plt.imshow(np.log(obsexp_func("chr20","chr21")),cmap="YlOrRd")
    plt.imshow(np.log(t_obsexp_func("chr17","chr18")),cmap="YlOrRd")


    # Aggregate contacts
    sum_, count = saddle.make_saddle(
        t_obsexp_func, 
        lambda chrom: digitized[chrom],
        track['chrom'].unique(),
        contact_type='trans',
        verbose=True)


    # ###########################
    # do the plotting, if requested ...
    contact_type = 'trans'

    fig = saddle.saddleplot(
        binedges,
        digitized,
        saddledata,
    #     contact_type, 
        color=pal[2],
        cbar_label='log10 (contact frequency / mean) in {}'.format(contact_type),
        fig_kws=None,
        heatmap_kws=None, 
        margin_kws=None)






    # # execute EITHER cis OR trans (not both):
    # if contact_type == 'cis':
    #     # list of regions in a format (chrom,start,stop),
    #     # when start,stop ommited, process entire chrom:
    #     regions = [ (chrom,) for chrom in c.chromnames ]
    #     expected_result = expected.cis_expected(clr=c,
    #                                    regions=regions,
    #                                    field='balanced',
    #                                    chunksize=chunksize,
    #                                    use_dask=use_dask,
    #                                    ignore_diags=drop_diags)
    # elif contact_type == 'trans':
    #     # process for all chromosomes:
    #     chromosomes = c.chromnames
    #     expected_result = expected.trans_expected(clr=c,
    #                                      chromosomes=chromosomes,
    #                                      chunksize=chunksize,
    #                                      use_dask=use_dask)
    # # output to stdout,
    # # just like in diamond_insulation:
    # print(expected_result.to_csv(sep='\t', index=True, na_rep='nan'))
