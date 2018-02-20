import click

import cooler

from . import cli
from .. import saddle


import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
# from bioframe import parse_humanized, parse_region_string






# inspired by:
# saddles.py by @nvictus
# https://github.com/nandankita/labUtilityTools


################################################
# fetcher functions ...
################################################
def make_cis_obsexp_fetcher(clr, expected, name):
    """
    Given a cooler object 'clr',
    cis-expected dataframe 'expected',
    and a column-name in cis-expected 'name',
    this function yields a function that
    returns slice of OBS/EXP for a given 
    chromosome.
    """
    return lambda chrom, _: (
            clr.matrix().fetch(chrom) / 
                toeplitz(expected.loc[chrom][name].values)
        )

def make_trans_obsexp_fetcher(clr, expected, name):
    """
    Given a cooler object 'clr',
    trans-expected dataframe 'expected',
    and a column-name in trans-expected 'name',
    this function yields a function that
    returns slice of OBS/EXP for a given 
    pair of chromosomes.

    'expected' must have a MultiIndex with
    chrom1 and chrom2.
    """
    def _fetch_trans_exp(chrom1, chrom2):
        """
        get trans-expected from 'expected'
        dealing with (chrom1, chrom2) flipping.
        """
        if (chrom1, chrom2) in expected.index:
            return expected.loc[chrom1, chrom2][name]
        elif (chrom2, chrom1) in expected.index:
            return expected.loc[chrom2, chrom1][name]
        # .loc is the right way to get [chrom1,chrom2] value from MultiIndex df:
        # https://pandas.pydata.org/pandas-docs/stable/advanced.html#advanced-indexing-with-hierarchical-index
        else:
            raise KeyError("trans-exp index is missing a pair of chromosomes: {}, {}".format(chrom1,chrom2))
    # returning OBS/EXP(chrom1,chrom2) function:
    return lambda chrom1, chrom2: (
            clr.matrix().fetch(chrom1, chrom2) / _fetch_trans_exp(chrom1, chrom2)
        )
#####################################################################
# OLD make_trans_obsexp_fetcher WHEN trans_exp WAS A SINGLE VALUE:
#####################################################################
# # old one with a single trans_expected for an entire genome:
#####################################################################
# def OLD_make_trans_obsexp_fetcher(clr, trans_avg):
#     return lambda chrom1, chrom2: (
#             clr.matrix().fetch(chrom1, chrom2) / trans_avg
#         )
#####################################################################



def make_track_fetcher(df, name):
    """
    Given a BedGraph-like dataframe 'df'
    and a 'name' of column with values
    this function yields a function
    that returns slice of 'name'-data
    for a given chromosome.
    """
    g = df.groupby('chrom')
    return lambda chrom: g.get_group(chrom)[name]
# the mask ...
def make_track_mask_fetcher(df, name):
    """
    Given a BedGraph-like dataframe 'df'
    and a 'name' of column with values
    this function yields a function
    that returns slice of indeces of
    notnull and non-zero 'name'-data
    for a given chromosome.
    """
    fetch_track = make_track_fetcher(df, name)
    return lambda chrom: (
            fetch_track(chrom).notnull() &
            (fetch_track(chrom) != 0)
        ).values
##########################
# # rewritten from this:
##########################
# track_mask_fetcher = lambda chrom: (
#     (track.groupby('chrom')
#           .get_group(chrom)[track_name]
#           .notnull()) &
#     (track.groupby('chrom')
#           .get_group(chrom)[track_name] != 0)
# ).values




#########################################
# strength ?!?!?!?!
#########################################
# def compute_strength(saddledata):
#     leng_S=len(saddledata)
#     percent=int(len(saddledata)*0.2)
#     BB=saddledata[0:percent,0:percent]
#     AA=saddledata[leng_S-percent:leng_S,leng_S-percent:leng_S]
#     numerator = np.concatenate((AA, BB), axis=0)
#     medianNum= np.median(numerator)    
#     BA = saddledata[0:percent,leng_S-percent:leng_S]
#     AB = saddledata[leng_S-percent:leng_S,0:percent]
#     denominator = np.concatenate((BA, AB), axis=0)
#     medianDen= np.median(denominator)
#     compStrength=(medianNum/medianDen)
#     return compStrength
########################################


# saddleplot(
#         chromosomes=None,
#         fig_kws=None,
#         heatmap_kws=None, 
#         margin_kws=None):
#     """
#     Parameters
#     ----------
#     chromosomes : list of str, optional
#         Restricted set of chromosomes to use. Default is to use all chromosomes
#         in the ``track`` dataframe.
#     heatmap_kws : dict, optional
#         Extra keywords to pass to ``imshow`` for saddle heatmap.
#     margin_kws : dict, optional
#         Extra keywords to pass to ``hist`` for left and top margins.
# click.Path(exists=False, file_okay=True, dir_okay=True, writable=False, readable=True, resolve_path=False)



@cli.command()
@click.argument(
    "cool_path",
    metavar="COOL_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
#     track : pandas.DataFrame
#         bedGraph-like data frame ('chrom', 'start', 'end', <name>).
@click.argument(
    "track_path",
    metavar="TRACK_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
#     expected : pandas.DataFrame or number, optional
#         Expected contact frequency. If ``contact_type`` is 'cis', then this is
#         a data frame of ('chrom', 'distance', <name>). If ``contact_type`` is 
#         'trans', then this is a single precomputed number equal to the average 
#         trans contact frequency.
@click.argument(
    "expected_path",
    metavar="EXPECTED_PATH",
    type=click.Path(exists=True, dir_okay=False),
    nargs=1)
# options ...
@click.option(
    '--track-name',
    help="Name of value column in TRACK_PATH",
    type=str,
    default='eigen',
    show_default=True,
    )
@click.option(
    '--expected-name',
    help="Name of value column in EXPECTED_PATH",
    type=str,
    default='balanced.avg',
    show_default=True,
    )
@click.option(
    # optional
    "--n-bins",
    help="Number of bins for digitizing the signal track.",
    type=int,
    default=50,
    show_default=True)
@click.option(
    '--contact-type',
    help="Type of the contacts to aggregate",
    type=click.Choice(['cis', 'trans']),
    default='cis',
    show_default=True)
@click.option(
    # prange : pair of floats
    '--prange',
    help="The percentile of the genome-wide range of the track values used to"
         " generate bins. E.g., if `prange`=(2. 98) the lower bin would"
         " start at the 2-nd percentile and the upper bin would end at the 98-th"
         " percentile of the genome-wide signal."
         " Use to prevent the extreme track values from exploding the bin range.",
     nargs=2,
     default=(2.0, 98.0),
     type=float,
     show_default=True)
@click.option(
    # optional
    "--by-percentile",
    help="Whether to bin the signal track by percentile"
         " rather than by value.",
    is_flag=True,
    default=False)
@click.option(
    "--verbose", "-v",
    help="Enable verbose output",
    is_flag=True,
    default=False)
@click.option(
    "--savefig",
    help="Save the saddle-plot to a file. "
         "If not specified - no image is generated"
         "The figure format is deduced from the extension of the file, "
         "the supported formats are png, jpg, svg, pdf, ps and eps.",
    type=str)
@click.option(
    "--output",
    help="Save the saddle-plot data to a csv file.",
    type=str)
# TODO:
# maybe attach binedges as index/column names ...?!?!?!? ...
# just for output
#
# add flag to calculate compartment strength ...
# update help ...
# add validations ...

def compute_saddle(
            cool_path,
            track_path,
            track_name,
            expected_path,
            expected_name,
            n_bins,
            contact_type,
            prange,
            by_percentile,
            verbose,
            savefig):
    """
    Calculate saddle statistics and
    generate saddle plots.

    Make a saddle plot for an arbitrary signal track on the genomic bins of 
    a contact matrix.

    CLI version of compute_saddle allows 
    one to specify percentile range for
    track digitization and switching between
    value/percentile binning.

    There is no interface however for
    uploading custom track bins.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    TRACK_PATH : The paths to bedGraph-file with a binned compartment track (eigenvalues).

    EXPECTED_PATH : The paths to a csv-like file with expected signal.

    """


    # load cooler file to process:
    c = cooler.Cooler(cool_path)


    # read expected and validate it:
    if contact_type == "cis":
        expected_columns = ['chrom', 'diag', 'n_valid', expected_name]
        expected_index = ['chrom', 'diag']
        expected_dtype = {'chrom':np.str, 'diag':np.int64, 'n_valid':np.int64, expected_name:np.float64}
    else contact_type == "trans":
        expected_columns = ['chrom1', 'chrom2', 'n_valid', expected_name]
        expected_index = ['chrom1', 'chrom2']
        expected_dtype = {'chrom1':np.str, 'chrom2':np.str, 'n_valid':np.int64, expected_name:np.float64}
    # that's what we expect as columns names
    # use 'usecols' as a rudimentary form of validation:
    expected = pd.read_table(
                            expected_path,
                            usecols  = expected_columns,
                            index_col= expected_index,
                            dtype    = expected_dtype,
                            comment  = None,
                            verbose  = False)


    # read BED-file :
    track_columns = ['chrom', 'start', 'end', track_name]
    track_dtype = {'chrom':np.str, 'start':np.int64, 'end':np.int64, track_name:np.float64}
    track = pd.read_table(
                    track_path,
                    usecols  = [0,1,2,3],
                    names    = track_columns,
                    dtype    = track_dtype,
                    comment  = None,
                    verbose  = False)
    # # or do tht as an alternative:
    # # ...
    # from bioframe.io import formats
    # from bioframe.schemas import SCHEMAS
    # formats.read_table(track_path, schema=SCHEMAS["bed4"])


    # chromosomes to deal with 
    # are by default extracted
    # from the BedGraph track-file:
    track_chroms = track['chrom'].unique()
    # ADD VALIDATIONS:
    # We might want to try this eventually:
    # https://github.com/TMiguelT/PandasSchema
    # do simple column-name validation for now:
    if not set(track_chroms).issubset(c.chromnames):
        raise ValueError("Chromosomes in {} must be subset of chromosomes in cooler {}".format(track_path,
                                                                                               cool_path))

    # check number of bins at least:
    track_bins = len(track)
    cool_bins   = c.bins()[:]["chrom"].isin(track_chroms).sum()
    if not (track_bins==clr_bins):
        raise ValueError("Number of bins is not matching:",
                " {} in {}, and {} in {} for chromosomes {}".format(track_bins,
                                                                   track_path,
                                                                   cool_bins,
                                                                   cool_path,
                                                                   track_chroms))




    # digitize the track (compartment track)
    # no matter the contact type ...
    track_fetcher = make_track_fetcher(track, track_name)
    track_mask_fetcher = make_track_mask_fetcher(track, track_name)
    digitized, binedges = saddle.digitize_track(
                                    bins = n_bins,
                                    get_track = track_fetcher,
                                    get_mask  = track_mask_fetcher,
                                    chromosomes = track_chroms,
                                    prange = prange,
                                    by_percentile = by_percentile)
    # digitized fetcher, yielding per chrom slice:
    digitized_fetcher = lambda chrom: digitized[chrom]

    # playing with OBS/EXP a bit : ...
    if contact_type == "cis":
        obsexp_func = make_cis_obsexp_fetcher(c, expected, expected_name)
    elif contact_type == "trans":
        obsexp_func = make_trans_obsexp_fetcher(c, expected, expected_name)

    # Aggregate contacts
    sum_, count = saddle.make_saddle(
        get_matrix = obsexp_func, 
        get_digitized = digitized_fetcher,
        chromosomes = track_chroms,
        contact_type = contact_type,
        verbose = verbose)

    saddledata = sum_/count
    # # output to stdout,
    # # just like in diamond_insulation:
    if output is not None:
        pd.DataFrame(saddledata).to_csv(output,
                                        sep='\t',
                                        index=True,
                                        header=True,
                                        na_rep='nan')
        # binDf=pd.DataFrame(binedges)
        # binDf.to_csv("binedges.txt", sep='\t')
    else:
        print(pd.DataFrame(saddledata).to_csv(output,
                                        sep='\t',
                                        index=True,
                                        header=True,
                                        na_rep='nan'))
    



    # ###########################
    # do the plotting, if requested ...
    # from cooler show cli:
    # if savefig or something like that:
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
        ###############################
        #
        ###############################
        fig = saddle.saddleplot(
            binedges = binedges,
            digitized = digitized,
            saddledata = saddledata,
            color = pal[2],
            cbar_label = 'log10 (contact frequency / mean) in {}'.format(contact_type),
            fig_kws = None,
            heatmap_kws = None, 
            margin_kws = None)
        ######
        plt.savefig(savefig, dpi=None)
        # else:
        #     interactive(plt.gca(), c, row_chrom, col_chrom, field, balanced, scale)

