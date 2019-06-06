import numpy as np
import pandas as pd

import cooler, cooler.tools, cooler.io


def sample_pixels(pixels, frac):
    pixels['count'] = np.random.binomial(pixels['count'], frac)
    mask = pixels['count'] > 0
    if issubclass(type(pixels), pd.DataFrame):
        pixels = pixels[mask]
    elif issubclass(type(pixels), dict):
        pixels = {k:arr[mask] for k, arr in pixels.items()}
    
    return pixels


def _extract_pixel_chunk(chunk):
    return chunk['pixels']


def sample_cooler(clr, out_clr_path, count=None, frac=None, map_func=map, chunksize=int(1e7)):
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
        The resulting sample size will not match it precisely. 
        Mutually exclusive with `frac`.
        
    frac : float
        The target sample size as a fraction of contacts in the original dataset.
        Mutually exclusive with `count`.
        
    map_func : function
        A map implementation.
        
    chunksize : int
        The number of pixels loaded and processed per step of computation.
    
    """
    if issubclass(type(clr), str):
        clr = cooler.Cooler(clr)

    if count is not None and frac is None:
        frac = count / clr.info['sum']
        if frac >= 1.0:
            raise ValueError('The target number of contacts must be ')
    elif count is None and frac is None:
        raise ValueError('Either frac or tot_count must be specified!')
    elif count is not None and frac is not None:
        raise ValueError('Either frac or tot_count must be specified!')
        
    iter_chunks = (
        cooler.tools.split(clr, include_bins=False, map=map_func, chunksize=chunksize)
        .pipe(_extract_pixel_chunk)
        .pipe(sample_pixels, frac=frac)
        .__iter__()
    )

    cooler.create_cooler(out_clr_path, clr.bins()[:], iter_chunks)
