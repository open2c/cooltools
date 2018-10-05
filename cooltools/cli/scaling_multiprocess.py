import pypairix
import pandas as pd
import numpy as np
import cytoolz as toolz
import itertools
import cooltools
from cooltools import expected
from cooltools.lib import numutils
from multiprocessing import Pool
import click

column_name = ["readid","chrom1", "pos1", "chrom2", "pos2", "strand1", "strand2", "pair_type"]

def process_individual_chromosome(input_array):
    chrom, chr_sizes, distbins, dmin, dmax, pairs_file, chunksize  = input_array
    
    #Can grab start end for splitted chromosomes
    if len(chr_sizes.columns) == 1:
        start = 0
        end = chr_sizes.loc[chrom].values[0]
    else:
        start, end  = chr_sizes.loc[chrom].values

    region1 = (start, end)
    region2 = region1

    tb=pypairix.open(pairs_file)
    querystr='{}:{}-{}'.format(chrom, start, end)
    it = tb.querys(querystr)
    

    obs = np.zeros(int(distbins.shape[0]) -1)
    for chunk in toolz.partition_all(chunksize, it):
   
        df =pd.DataFrame(list(chunk),\
                                 columns=column_name,\
                                 dtype=np.int64)
        df = df[ 
                (df['pos1'] >= region1[0]) & 
                (df['pos1'] < region1[1]) & 
                (df['pos2'] >= region2[0]) & 
                (df['pos2'] < region2[1])]

        dists = (df['pos2'] - df['pos1']).values


        obs += np.histogram(
                        dists[(dists >= dmin) & (dists < dmax)],
                        bins=distbins)[0]

    area = expected.contact_areas(distbins, region1=(start, end), region2=(start, end))
    return chrom, obs/area

@click.command()
@click.argument(
    "pairs_path",
    metavar="PAIRS_PATH",
    type=str,
    nargs=1)
@click.argument(
    "chrom_sizes",
    metavar="Chrom_Sizes_Path",
    type=str,
    nargs=1)
@click.option(
    '--nproc', '-p',
    help="Number of processes to split the work between."
         "[default: 1, i.e. no process pool]",
    default=1,
    type=int)
@click.option(
    "--chunksize",
    help="Control the number reads of per dataframe at a time.",
    type=int,
    default=int(1000000),
    show_default=True)
@click.option(
    "--nbins", "-n",
    help="Number of bins",
    type=int,
    default=50,
    show_default=True)
@click.option(
    "--dmin", 
    help="startpoint scaling plot",
    type=int,
    default=1000,
    show_default=True)
@click.option(
    "--dmax", 
    help="Endpoint scaling plot",
    type=int,
    default=10000000,
    show_default=True)
@click.option(
    "--out-prefix", "-o",
    help="Dump 'scaling data' arrays in a numpy-specific "
         ".npz container. Use numpy.load to load these arrays into a "
         "dict-like object.")

def scaling_script(pairs_path, chrom_sizes, dmin, dmax, nbins, nproc, chunksize, out_prefix):

    chr_sizes = pd.read_table(chrom_sizes, 
                       header=None, index_col=0)
    chromosomes=list(chr_sizes.index) #chromosomes=list(chr_sizes[:-3]) for autosomale chr only
    
    distbins = numutils.logbins(dmin, dmax, N=nbins)
    output = {}
    input_list = []

    for chrom in chromosomes:
        input_list.append((chrom, chr_sizes, distbins, dmin, dmax, pairs_path, chunksize))
    
    with Pool(nproc) as p:
        results = p.map(process_individual_chromosome, input_list)
        
    for i in results:
        output[i[0]] = i[1]

    
    output["Avarage"] = np.nanmean(list(output.values()), axis=0)
    output["Distbins"] = distbins
    
    np.savez(out_prefix + ".scalingdata", **output)
    
if __name__ == '__main__':
    scaling_script()
   
