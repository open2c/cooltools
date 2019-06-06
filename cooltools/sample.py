import numpy as np
import pandas as pd

import cooler, cooler.tools, cooler.io


def sample_pixels_approx(pixels, frac):
    pixels['count'] = np.random.binomial(pixels['count'], frac)
    mask = pixels['count'] > 0
    if issubclass(type(pixels), pd.DataFrame):
        pixels = pixels[mask]
    elif issubclass(type(pixels), dict):
        pixels = {k:arr[mask] for k, arr in pixels.items()}
    
    return pixels


def sample_pixels_exact(pixels, count):
    count_cumsum = np.cumsum(np.asarray(pixels['count']))
    contact_indices = np.random.choice(count_cumsum[-1], size=count, replace=False)

    # testing:
    # contact_indices = np.arange(count_cumsum[-1])
    new_counts = np.bincount(
        np.searchsorted(count_cumsum, contact_indices, side='right'), 
            minlength=count_cumsum.shape[0])
    # assert (pixels['count'] != new_counts).sum() == 0 

    pixels['count'] = new_counts
    mask = pixels['count']>0
    
    if issubclass(type(pixels), pd.DataFrame):
        pixels = pixels[mask]
    elif issubclass(type(pixels), dict):
        pixels = {k:arr[mask] for k, arr in pixels.items()}
    
    return pixels


def _extract_pixel_chunk(chunk):
    return chunk['pixels']


def sample_cooler(clr, out_clr_path, count=None, frac=None, exact=False, map_func=map, chunksize=int(1e7)):
    """
    Pick a random subset of contacts from a Hi-C map.
    
    Parameters
    ----------
    clr : cooler.Cooler or str
        A Cooler or a path/URI to a Cooler with input data.
        
    out_clr_path : str
        A path/URI to the output.
    
    count : float
        The target number of contacts in the sample. 
        Mutually exclusive with `frac`.
        
    frac : float
        The target sample size as a fraction of contacts in the original dataset.
        Mutually exclusive with `count`.
    
    exact : bool
        If True, the resulting sample size will exactly match the target value.
        Exact sampling will load the whole pixel table into memory!
        If False, binomial sampling will be used instead and the sample size will be
        randomly distributed around the target value.
        
    map_func : function
        A map implementation.
        
    chunksize : int
        The number of pixels loaded and processed per step of computation.
    
    """
    if issubclass(type(clr), str):
        clr = cooler.Cooler(clr)

    if count is not None and frac is None:
        frac = count / clr.info['sum']
    elif count is None and frac is not None:
        count = np.round(frac * clr.info['sum'])
    else:
        raise ValueError('Either frac or tot_count must be specified!')
        
    if frac >= 1.0:
        raise ValueError('The number of contacts in a sample cannot exceed that in the original dataset.')
        
    if exact:
        pixels = sample_pixels_exact(clr.pixels()[:], count)
        cooler.create_cooler(out_clr_path, clr.bins()[:], pixels)
        
    else:
        iter_chunks = (
            cooler.tools.split(clr, include_bins=False, map=map_func, chunksize=chunksize)
            .pipe(_extract_pixel_chunk)
            .pipe(sample_pixels, frac=frac)
            .__iter__()
        )

        cooler.create_cooler(out_clr_path, clr.bins()[:], iter_chunks)
