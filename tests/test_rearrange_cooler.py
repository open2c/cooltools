import cooler
import bioframe
import os.path as op

import numpy as np

from cooltools.api.rearrange import rearrange_cooler
from pandas.testing import assert_frame_equal


def test_rearrange_cooler(request):
    # Read cool file and create view_df out of it:
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.10000kb.cool"))
    orig_view = bioframe.make_viewframe(clr.chromsizes)

    # I.
    # Check that with the same view, nothing changes
    rearrange_cooler(clr, orig_view, "test_not_reordered.cool")
    new_clr = cooler.Cooler("test_not_reordered.cool")
    assert_frame_equal(new_clr.pixels()[:], clr.pixels()[:])
    assert_frame_equal(new_clr.bins()[:], clr.bins()[:])
    assert_frame_equal(new_clr.bins()[:], clr.bins()[:])

    # II.
    # Check that when just getting one chrom, all is as expected
    new_view = orig_view.iloc[:1, :]
    rearrange_cooler(clr, new_view, "test_chrom1_reordered.cool")
    new_clr = cooler.Cooler("test_chrom1_reordered.cool")
    old_bins = clr.bins()[:].query('chrom=="chr1"')
    old_bins["chrom"] = old_bins["chrom"].astype(str)
    new_bins = new_clr.bins()[:]
    new_bins["chrom"] = new_bins["chrom"].astype(str)
    assert_frame_equal(old_bins, new_bins)

    old_pixels = clr.matrix(as_pixels=True).fetch("chr1").drop(columns=["balanced"])
    new_pixels = new_clr.pixels()[:]
    assert_frame_equal(old_pixels, new_pixels)
    assert_frame_equal(clr.chroms()[:1], new_clr.chroms()[:])

    # III.
    # Check that when just getting one chrom and inverting it, all is as expected
    inverted_view = new_view.copy()
    inverted_view["strand"] = "-"
    rearrange_cooler(clr, inverted_view, "test_chrom1_reordered_inverted.cool")
    inverted_clr = cooler.Cooler("test_chrom1_reordered_inverted.cool")
    inverted_bins = inverted_clr.bins()[:]
    inverted_bins[["end", "start"]] = (
        inverted_bins.iloc[-1]["end"] - inverted_bins[["start", "end"]]
    )
    inverted_bins = inverted_bins.iloc[::-1].reset_index(drop=True)
    inverted_bins["chrom"] = inverted_bins["chrom"].astype(str)
    assert_frame_equal(new_bins, inverted_bins)
    inverted_pixels = inverted_clr.pixels()[:]
    inverted_pixels[["bin1_id", "bin2_id"]] = np.sort(
        inverted_bins.index[-1] - inverted_pixels[["bin1_id", "bin2_id"]]
    )
    inverted_pixels = inverted_pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(
        drop=True
    )
    assert_frame_equal(new_clr.pixels()[:], inverted_pixels)
    assert_frame_equal(new_clr.chroms()[:1], inverted_clr.chroms()[:])

    # III.
    # Check that when taking two chromosomes in a different order and inverting one,
    # all is ax espected

    reorder_invert_view = (
        orig_view.iloc[1::-1].assign(strand=["+", "-"]).reset_index(drop=True)
    )
    rearrange_cooler(clr, reorder_invert_view, "test_chr2chr1_reordered_inverted.cool")
    reordered_inverted_clr = cooler.Cooler("test_chr2chr1_reordered_inverted.cool")

    # compare chr2 bins
    old_bins_chr2 = clr.bins().fetch("chr2").reset_index(drop=True)
    old_bins_chr2["chrom"] = old_bins_chr2["chrom"].astype(str)
    reordered_inverted_bins_chr2 = reordered_inverted_clr.bins().fetch("chr2")
    reordered_inverted_bins_chr2["chrom"] = reordered_inverted_bins_chr2[
        "chrom"
    ].astype(str)
    assert_frame_equal(old_bins_chr2, reordered_inverted_bins_chr2)
    # compare chr2 pixels
    old_pixels_chr2 = (
        clr.pixels()
        .fetch("chr2")
        .query(f'bin2_id<={clr.bins().fetch("chr2").index[-1]}')
        .reset_index(drop=True)
    )
    reordered_inverted_pixels_chr2 = (
        reordered_inverted_clr.pixels()
        .fetch("chr2")
        .query(f'bin2_id<={reordered_inverted_clr.bins().fetch("chr2").index[-1]}')
        .reset_index(drop=True)
    )
    reordered_inverted_pixels_chr2[["bin1_id", "bin2_id"]] += (
        clr.bins().fetch("chr1").index[-1] + 1
    )
    assert_frame_equal(old_pixels_chr2, reordered_inverted_pixels_chr2)
    # Compare chr1 bins
    old_bins_chr1 = clr.bins().fetch("chr1")
    old_bins_chr1["chrom"] = old_bins_chr1["chrom"].astype(str)

    reordered_inverted_bins_chr1 = reordered_inverted_clr.bins().fetch("chr1")
    reordered_inverted_bins_chr1[["end", "start"]] = (
        reordered_inverted_bins_chr1.iloc[-1]["end"]
        - reordered_inverted_bins_chr1[["start", "end"]]
    )
    reordered_inverted_bins_chr1.index = (
        reordered_inverted_bins_chr1.index[::-1] - old_bins_chr1.index[-1]
    )
    reordered_inverted_bins_chr1 = reordered_inverted_bins_chr1.iloc[::-1]
    reordered_inverted_bins_chr1["chrom"] = reordered_inverted_bins_chr1[
        "chrom"
    ].astype(str)

    assert_frame_equal(old_bins_chr1, reordered_inverted_bins_chr1)
    # Compare chr1 pixels
    old_pixels_chr1 = (
        clr.pixels()
        .fetch("chr1")
        .query(f'bin2_id<={clr.bins().fetch("chr1").index[-1]}')
        .reset_index(drop=True)
    )
    reordered_inverted_pixels_chr1 = reordered_inverted_clr.pixels().fetch("chr1")

    reordered_inverted_pixels_chr1[["bin1_id", "bin2_id"]] = np.sort(
        reordered_inverted_bins_chr1.index[-1]
        - reordered_inverted_pixels_chr1[["bin1_id", "bin2_id"]]
        + reordered_inverted_bins_chr2.index[-1]
        + 1
    )
    reordered_inverted_pixels_chr1 = inverted_pixels.sort_values(["bin1_id", "bin2_id"])
    assert_frame_equal(old_pixels_chr1, reordered_inverted_pixels_chr1)
    # Compare trans matrix (easier than pixels)
    old_trans_m = clr.matrix().fetch("chr1", "chr2")
    reordered_inverted_trans_m = reordered_inverted_clr.matrix().fetch("chr1", "chr2")[
        ::-1, :
    ]
    assert np.array_equal(old_trans_m, reordered_inverted_trans_m, equal_nan=True)
