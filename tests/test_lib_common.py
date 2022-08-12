import os.path as op
import pandas as pd
import cooler
import cooltools
import pytest

# TODO: tests for
# assign_supports, or assign_regions, or deprecate & remove both
# assign_regions_to_bins
# make_cooler_view
# view_from_track


def test_align_track_with_cooler(request, tmpdir):

    clr_file = op.join(request.fspath.dirname, "data/sin_eigs_mat.cool")
    clr = cooler.Cooler(clr_file)

    # valid track with three entries that can all be aligned
    track = pd.DataFrame(
        [
            ["chr1", 990, 995, 22],
            ["chr2", 20, 30, -1],
            ["chr3", 0, 10, 0.1],
        ],
        columns=["chrom", "start", "end", "value"],
    )
    assert (
        ~cooltools.lib.align_track_with_cooler(track, clr)["value"].isna()
    ).sum() == 3

    # not a track, is not sorted
    track = pd.DataFrame(
        [["chr3", 0, 10, 0.1], ["chr2", 20, 30, -1], ["chr2", 0, 10, 21]],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        cooltools.lib.align_track_with_cooler(track, clr)

    # not a track, is overlapping
    track = pd.DataFrame(
        [
            ["chr1", 990, 1000, 22],
            ["chr2", 5, 15, 0.1],
            ["chr2", 20, 30, -1],
        ],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        cooltools.lib.align_track_with_cooler(track, clr)

    # bin size mismatch
    track = pd.DataFrame(
        [["chr1", 990, 995, 22], ["chr2", 20, 25, -1], ["chr3", 0, 5, 0.1]],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        cooltools.lib.align_track_with_cooler(track, clr)

    # clr_weight_name mismatch
    track = pd.DataFrame(
        [
            ["chr1", 990, 995, 22],
            ["chr2", 20, 30, -1],
            ["chr3", 0, 10, 0.1],
        ],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        cooltools.lib.align_track_with_cooler(
            track, clr, clr_weight_name="invalid_weight_name"
        )

    # regions with no assigned values
    track = pd.DataFrame(
        [["chr1", 0, 10, 0.1], ["chr1", 20, 30, -1], ["chr1", 990, 995, 22]],
        columns=["chrom", "start", "end", "value"],
    )
    with pytest.raises(ValueError):
        cooltools.lib.align_track_with_cooler(track, clr)

    # using a restricted view only considers chr1, avoids valueError from no assigned values
    view_df = cooltools.lib.make_cooler_view(clr)
    assert (
        ~cooltools.lib.align_track_with_cooler(track, clr, view_df=view_df[:1])[
            "value"
        ].isna()
    ).sum() == 3

    # testing mask_bad_bins option
    clr_file = op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool")
    clr = cooler.Cooler(clr_file)
    view_df = cooltools.lib.make_cooler_view(clr)[:1]

    track = pd.DataFrame(
        [["chr1", 0, 1000000, 1], ["chr1", 3000000, 4000000, 10]],
        columns=["chrom", "start", "end", "value"],
    )
    # without masking, both get assigned
    assert (
        cooltools.lib.align_track_with_cooler(
            track, clr, view_df=view_df, mask_clr_bad_bins=False
        )["value"].sum()
        == 11
    )

    # with masking, only the second value from the track gets assigned
    assert (
        cooltools.lib.align_track_with_cooler(
            track, clr, view_df=view_df, mask_clr_bad_bins=True
        )["value"].sum()
        == 10
    )
