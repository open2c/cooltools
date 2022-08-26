import numpy as np
import pandas as pd

import cooler
import cooler.tools
from .coverage import coverage


def sample_pixels_approx(pixels, frac):
    pixels["count"] = np.random.binomial(pixels["count"], frac)
    mask = pixels["count"] > 0

    if issubclass(type(pixels), pd.DataFrame):
        pixels = pixels[mask]
    elif issubclass(type(pixels), dict):
        pixels = {k: arr[mask] for k, arr in pixels.items()}
    return pixels


def sample_pixels_exact(pixels, count):
    cumcount = np.cumsum(np.asarray(pixels["count"]))
    total = cumcount[-1]
    n_pixels = cumcount.shape[0]

    # sample a given number of distinct contacts
    random_contacts = np.random.choice(total, size=count, replace=False)

    # find where those contacts live in the cumcount array
    loc = np.searchsorted(cumcount, random_contacts, side="right")

    # re-bin those locations to get new counts
    new_counts = np.bincount(loc, minlength=n_pixels)

    pixels["count"] = new_counts
    mask = pixels["count"] > 0
    if issubclass(type(pixels), pd.DataFrame):
        pixels = pixels[mask]
    elif issubclass(type(pixels), dict):
        pixels = {k: arr[mask] for k, arr in pixels.items()}
    return pixels


def _extract_pixel_chunk(chunk):
    return chunk["pixels"]


def sample(
    clr,
    out_clr_path,
    count=None,
    cis_count=None,
    frac=None,
    exact=False,
    map_func=map,
    chunksize=int(1e7),
):
    """
    Pick a random subset of contacts from a Hi-C map.

    Parameters
    ----------
    clr : cooler.Cooler or str
        A Cooler or a path/URI to a Cooler with input data.

    out_clr_path : str
        A path/URI to the output.

    count : int
        The target number of contacts in the sample.
        Mutually exclusive with `cis_count` and `frac`.

    cis_count : int
        The target number of cis contacts in the sample.
        Mutually exclusive with `count` and `frac`.

    frac : float
        The target sample size as a fraction of contacts in the original
        dataset. Mutually exclusive with `count` and `cis_count`.

    exact : bool
        If True, the resulting sample size will exactly match the target value.
        Exact sampling will load the whole pixel table into memory!
        If False, binomial sampling will be used instead and the sample size
        will be randomly distributed around the target value.

    map_func : function
        A map implementation.

    chunksize : int
        The number of pixels loaded and processed per step of computation.

    """
    if issubclass(type(clr), str):
        clr = cooler.Cooler(clr)

    if frac is not None and count is None and cis_count is None:
        pass
    elif frac is None and count is not None and cis_count is None:
        frac = count / clr.info["sum"]
    elif frac is None and count is None and cis_count is not None:
        # note division by two, since coverage() counts each side separately
        cis_total = clr.info.get("cis", np.sum(coverage(clr)[0] // 2, dtype=int))
        frac = cis_count / cis_total
    else:
        raise ValueError(
            "Please specify exactly one argument among `count`, `cis_count`"
            " and `frac`"
        )

    if frac >= 1.0:
        raise ValueError(
            "The number of contacts in a sample cannot exceed "
            "that in the original dataset."
        )

    if exact:
        count = np.round(frac * clr.info["sum"]).astype(int)
        pixels = sample_pixels_exact(clr.pixels()[:], count)
        cooler.create_cooler(out_clr_path, clr.bins()[:], pixels, ordered=True)

    else:
        pipeline = (
            cooler.tools.split(
                clr, include_bins=False, map=map_func, chunksize=chunksize
            )
            .pipe(_extract_pixel_chunk)
            .pipe(sample_pixels_approx, frac=frac)
        )

        cooler.create_cooler(
            out_clr_path,
            clr.bins()[:][["chrom", "start", "end"]],
            iter(pipeline),
            ordered=True,
        )
