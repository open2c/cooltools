import cooler
import os.path as op

import cooltools.snipping
import numpy as np

def test_snipper(request):
    """
    Test the snipping on matrix:
    """
    # Read cool file and create regions out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))

    regions = [(chrom, 0, clr.chromsizes[chrom]) for chrom in clr.chromnames]

    # Example region with windows, two regions from annotated genomic regions:
    windows = cooltools.snipping.make_bin_aligned_windows(1_000_000,
                                                        ['chr10', 'chr10'],
                                                        [10_000_000, 20_000_000],
                                                        flank_bp=2_000_000)
    
    windows = cooltools.snipping.assign_regions(windows, regions).reset_index(drop=True)

    snipper = cooltools.snipping.CoolerSnipper(clr)
    stack = cooltools.snipping.pileup(
            windows,
            snipper.select,
            snipper.snip,
            map=map)

    # Check that the size of snips is OK and there are two of them:
    assert stack.shape == (5, 5, 2)

    # Example region with windows, second window comes from unannotated genomic region:
    windows = cooltools.snipping.make_bin_aligned_windows(1_000_000,
                                                        ['chr10', 'chr10'],
                                                        [10_000_000, 150_000_000],
                                                        flank_bp=2_000_000)
    
    windows = cooltools.snipping.assign_regions(windows, regions).reset_index(drop=True)

    snipper = cooltools.snipping.CoolerSnipper(clr)
    stack = cooltools.snipping.pileup(
            windows,
            snipper.select,
            snipper.snip,
            map=map)

    assert stack.shape == (5, 5, 2)
    assert np.all(np.isfinite(stack[:, :, 0]))
    assert np.all(np.isnan(stack[:, :, 1]))
