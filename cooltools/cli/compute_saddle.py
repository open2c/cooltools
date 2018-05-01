import click

import cooler

from . import cli
from .. import saddle

import pandas as pd
import numpy as np
from scipy.linalg import toeplitz


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




# TODO:
# add flag to calculate compartment strength ...
#########################################
# strength ?!?!?!?!
#########################################
def get_compartment_strength(saddledata, fraction):
    """
    Naive compartment strength calculator.
    """
    # assuming square shape:
    n_bins,n_bins = saddledata.shape
    # fraction must be < 0.5:
    if fraction >= 0.5:
        raise ValueError("Fraction for compartment strength calculations must be <0.5.")
    # # of bins to take for strenght computation:
    bins_for_strength = int(n_bins*fraction)
    # intra- (BB):
    intra_BB = saddledata[0:bins_for_strength,\
                        0:bins_for_strength]
    # intra- (AA):
    intra_AA = saddledata[n_bins-bins_for_strength:n_bins,\
                        n_bins-bins_for_strength:n_bins]
    intra = np.concatenate((intra_AA, intra_BB), axis=0)
    intra_median = np.median(intra)
    # inter- (BA):
    inter_BA = saddledata[0:bins_for_strength,\
                        n_bins-bins_for_strength:n_bins]
    # inter- (AB):
    inter_AB = saddledata[n_bins-bins_for_strength:n_bins,\
                        0:bins_for_strength]
    inter = np.concatenate((inter_BA, inter_AB), axis=0)
    inter_median = np.median(inter)
    # returning intra-/inter- ratrio as a
    # measure of compartment strength:
    return intra_median / inter_median
########################################


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
    help="Number of bins for digitizing track values.",
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
    # vrange : pair of floats
    '--vrange',
    help="Low and high values used for binning genome-wide track values, e.g."
         " if `vrange`=(-0.05, 0.05), `n-bins` equidistant bins would be generated"
         " between -0.05 and 0.05, dismissing `--prange` and `--by-percentile` options."
         " Use to prevent the extreme track values from exploding the bin range and"
         " to ensure consistent bins across several runs of `compute_saddle` command"
         " using different track files.",
    nargs=2,
    type=float)
@click.option(
    '--vmin',
    help="Low value of the saddleplot colorbar, e.g."
         " if `vmin`= -0.5, then all the saddledata <=-0.5"
         " would be depicted with the darkest color."
    type=float,
    default=-1)
@click.option(
    '--vmax',
    help="High value of the saddleplot colorbar, e.g."
         " if `vmax`= 0.5, then all the saddledata >=0.5,"
         " would be depicted with the brightest color."
    type=float,
    default=1)
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
    "--compute-strength",
    help="Compute compartment strength"
    " as a ratio between median intra-"
    " and inter-compartmental signal."
    " Result is printed to stdout.",
    is_flag=True,
    default=False)
@click.option(
    "--fraction-for-strength",
    help="Control what fraction of inter-"
    " and intra-compartmental 'saddledata'"
    " to use for calculation of strength.",
    default=0.2,
    type=float,
    show_default=True)
@click.option(
    "--savefig",
    help="Save the saddle-plot to a file. "
         " If not specified - no image is generated"
         " The figure format is deduced from the extension of the file, "
         " the supported formats are png, jpg, svg, pdf, ps and eps.",
    type=str)
@click.option(
    "--output",
    help="Dump 'saddledata', 'binedges' and 'digitized'"
         " track in a numpy-specific .npz container."
         " Use numpy.load to load these arrays"
         " into dict-like object."
         " Note: use .item() method to extract"
         " 'digitized' dict from ndarray wrapper:\n"
         " >>>npz = np.load(\"saved.npz\")\n"
         " >>>saddledata = npz[\"saddledata\"].item()",
    type=str)

def compute_saddle(
            cool_path,
            track_path,
            track_name,
            expected_path,
            expected_name,
            n_bins,
            contact_type,
            prange,
            vrange,
            vmin,
            vmax,
            by_percentile,
            verbose,
            compute_strength,
            fraction_for_strength,
            savefig,
            output):
    """
    Calculate saddle statistics and generate
    saddle plots for an arbitrary signal track
    on the genomic bins of a contact matrix.
    
    COOL_PATH : The paths to a .cool file with a balanced Hi-C map.

    TRACK_PATH : The paths to bedGraph-file with a binned compartment track (eigenvalues).

    EXPECTED_PATH : The paths to a tsv-like file with expected signal.

    Analysis will be performed for chromosomes
    referred to in TRACK_PATH, and therefore these
    chromosomes must be a subset of chromosomes
    referred to in COOL_PATH and EXPECTED_PATH.

    COOL_PATH, TRACK_PATH and EXPECTED_PATH must
    be binned at the same resolution (expect for 
    EXPECTED_PATH in case of trans contact type).

    EXPECTED_PATH must contain at least the
    following columns for cis contacts:
    'chrom', 'diag', 'n_valid', value_name
    and the following columns for trans contacts:
    'chrom1', 'chrom2', 'n_valid', value_name
    value_name is controlled using options.
    Header must be present in a file.



    """


    # load cooler file to process:
    c = cooler.Cooler(cool_path)


    # read expected and make preparations for validation,
    # it's contact_type dependent:
    if contact_type == "cis":
        # that's what we expect as column names:
        expected_columns = ['chrom', 'diag', 'n_valid', expected_name]
        # what would become a MultiIndex:
        expected_index = ['chrom', 'diag']
        # expected dtype as a rudimentary form of validation:
        expected_dtype = {'chrom':np.str, 'diag':np.int64, 'n_valid':np.int64, expected_name:np.float64}
        # unique list of chroms mentioned in expected_path:
        get_exp_chroms = lambda df: df.index.get_level_values("chrom").unique()
        # compute # of bins by comparing matching indexes:
        get_exp_bins   = lambda df,ref_chroms,_: df.index.get_level_values("chrom").isin(ref_chroms).sum()
    elif contact_type == "trans":
        # that's what we expect as column names:
        expected_columns = ['chrom1', 'chrom2', 'n_valid', expected_name]
        # what would become a MultiIndex:
        expected_index = ['chrom1', 'chrom2']
        # expected dtype as a rudimentary form of validation:
        expected_dtype = {'chrom1':np.str, 'chrom2':np.str, 'n_valid':np.int64, expected_name:np.float64}
        # unique list of chroms mentioned in expected_path:
        get_exp_chroms = lambda df: np.union1d(df.index.get_level_values("chrom1").unique(),
                                           df.index.get_level_values("chrom2").unique())
        # no way to get bins from trans-expected, so just get the number:
        get_exp_bins   = lambda _1,_2,correct_bins: correct_bins
    else:
        raise ValueError("Incorrect contact_type: {}, ".format(contact_type),
            "Should have been caught by click.")
    # use 'usecols' as a rudimentary form of validation,
    # and dtype. Keep 'comment' and 'verbose' - explicit,
    # as we may use them later:
    expected = pd.read_table(
                            expected_path,
                            usecols  = expected_columns,
                            index_col= expected_index,
                            dtype    = expected_dtype,
                            comment  = None,
                            verbose  = False)

    # read BED-file :
    track_columns = ['chrom', 'start', 'end', track_name]
    # specify dtype as a rudimentary form of validation:
    track_dtype = {'chrom':np.str, 'start':np.int64, 'end':np.int64, track_name:np.float64}
    track = pd.read_table(
                    track_path,
                    usecols  = [0,1,2,3],
                    names    = track_columns,
                    dtype    = track_dtype,
                    comment  = None,
                    verbose  = False)
    # ######################################
    # # potentially switch to bioframe in the future
    # # for DataFrame validations etc:
    # from bioframe.io import formats
    # from bioframe.schemas import SCHEMAS
    # formats.read_table(track_path, schema=SCHEMAS["bedGraph"])
    # ######################################

    #############################################
    # CROSS-VALIDATE COOLER, EXPECTED AND TRACK:
    #############################################
    # TRACK vs COOLER:
    # chromosomes to deal with 
    # are by default extracted
    # from the BedGraph track-file:
    track_chroms = track['chrom'].unique()
    # We might want to try this eventually:
    # https://github.com/TMiguelT/PandasSchema
    # do simple column-name validation for now:
    if not set(track_chroms).issubset(c.chromnames):
        raise ValueError("Chromosomes in {} must be subset of chromosomes in cooler {}".format(track_path,
                                                                                               cool_path))
    # check number of bins:
    track_bins = len(track)
    cool_bins   = c.bins()[:]["chrom"].isin(track_chroms).sum()
    if not (track_bins==cool_bins):
        raise ValueError("Number of bins is not matching:",
                " {} in {}, and {} in {} for chromosomes {}".format(track_bins,
                                                                   track_path,
                                                                   cool_bins,
                                                                   cool_path,
                                                                   track_chroms))
    # EXPECTED vs TRACK:
    # validate expected a bit as well:
    expected_chroms = get_exp_chroms(expected)
    # do simple column-name validation for now:
    if not set(track_chroms).issubset(expected_chroms):
        raise ValueError("Chromosomes in {} must be subset of chromosomes in expected {}".format(track_path,
                                                                                               expected_path))
    # and again bins are supposed to match up:
    # only for cis though ...
    expected_bins = get_exp_bins(expected,track_chroms,track_bins)
    if not (track_bins==expected_bins):
        raise ValueError("Number of bins is not matching:",
                " {} in {}, and {} in {} for chromosomes {}".format(track_bins,
                                                                   track_path,
                                                                   expected_bins,
                                                                   expected_path,
                                                                   track_chroms))
    #############################################
    # CROSS-VALIDATION IS COMPLETE.
    #############################################
    # COOLER, TRACK and EXPECTED seems cross-compatible:

    # define OBS/EXP getter functions,
    # it's contact_type dependent:
    if contact_type == "cis":
        obsexp_func = make_cis_obsexp_fetcher(c, expected, expected_name)
    elif contact_type == "trans":
        obsexp_func = make_trans_obsexp_fetcher(c, expected, expected_name)



    # digitize the track (i.e., compartment track)
    # no matter the contact_type ...
    track_fetcher = make_track_fetcher(track, track_name)
    track_mask_fetcher = make_track_mask_fetcher(track, track_name)
    if vrange:
        # if a manual vrange option provided,
        # use saddle.digitize_track's option
        # 'bins' that overwrites both prange
        # and 'by_percentile':
        bins = np.linspace(*vrange, n_bins + 1)
        # this is equivalent to the explicit:
        prange = None
        by_percentile = False
    else:
        # otherwise use prange and by_percentile:
        bins = n_bins
    digitized, binedges = saddle.digitize_track(
                                    bins = bins,
                                    get_track = track_fetcher,
                                    get_mask  = track_mask_fetcher,
                                    chromosomes = track_chroms,
                                    prange = prange,
                                    by_percentile = by_percentile)
    # digitized fetcher, yielding per chrom slice:
    digitized_fetcher = lambda chrom: digitized[chrom]


    # Aggregate contacts (implicit contact_type dependency):
    sum_, count = saddle.make_saddle(
        get_matrix = obsexp_func, 
        get_digitized = digitized_fetcher,
        chromosomes = track_chroms,
        contact_type = contact_type,
        verbose = verbose)
    # actual saddleplot data:
    saddledata = sum_/count

    ##############################
    # OUTPUT AND PLOTTING:
    ##############################
    # no stdout output, since there are several
    # distinct data-structures that needs to be saved,
    # let's pack them in a single container:
    if output is not None:
        # pack saddledata, binedges, and digitized 
        # in a single .npz container,
        # use it as a dump to redraw a saddleplot.
        np.savez(
            file = output+".saddledump", # .npz auto-added
            saddledata = saddledata,
            binedges = binedges,
            digitized = digitized)
        ###############################
        # DRAFT IMPLEMENTATION OF
        # A BedGraph-like OUTPUT.
        # KEEP FOR FUTURE REFERENCES.
        ###############################
        # # output digitized track, in bedGraph-like form
        # # with 3 additional columns: digitized.value, value_start, value_stop
        # # where digitized.value is an index of a digitized bin that a given
        # # value was assigned to, and (value_start, value_stop) are bounds of that bin.
        # track_copy = track.copy()
        # # left edge of the value bin:
        # get_bin_start = lambda idx: binedges[idx-1] if idx>0 else -np.inf
        # # right edge of the value bin:
        # get_bin_stop  = lambda idx: binedges[idx] if idx<len(binedges) else np.inf
        # # name of the digitized value:
        # digitized_name = "digitized."+track_name
        # track_copy[digitized_name] = \
        #         np.concatenate([
        #             digitized_fetcher(chrom) for chrom in track_copy['chrom'].drop_duplicates()
        #                        ]) 
        # track_copy[track_name+"_start"] = track_copy[digitized_name].apply(get_bin_start)
        # track_copy[track_name+"_stop"] = track_copy[digitized_name].apply(get_bin_stop)
        # track_copy.to_csv(output+".digitized.tsv",
        #             sep="\t",
        #             index=False,
        #             header=True,
        #             na_rep='nan')
    ####################################################
    # # no stdout output, since there are several
    # # distinct data-structures that needs to be saved,
    ####################################################
    # else:
    #     print(pd.DataFrame(saddledata).to_csv(
    #                                     sep='\t',
    #                                     index=False,
    #                                     header=False,
    #                                     na_rep='nan'))


    # compute naive compartment strength
    # if requested by the user:    
    if compute_strength:
        strength = get_compartment_strength(saddledata, fraction_for_strength)
        print("Comparment strength = {}".format(strength),
            "\ncalculated using {} of saddledata bins".format(fraction_for_strength),
            "to compute median intra- and inter-compartmental signal")


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
        heatmap_kws = dict(vmin=vmin, vmax=vmax)
        fig = saddle.saddleplot(
            binedges = binedges,
            digitized = digitized,
            saddledata = saddledata,
            color = pal[2],
            cbar_label = 'log10 (contact frequency / mean) in {}'.format(contact_type),
            fig_kws = None,
            heatmap_kws = heatmap_kws, 
            margin_kws = None)
        ######
        plt.savefig(savefig, dpi=None)
        # # No interactive plotting for now.
        # else:
        #     interactive(plt.gca(), c, row_chrom, col_chrom, field, balanced, scale)

