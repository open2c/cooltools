import os.path as op
import pandas as pd
from cooltools.lib.io import read_expected_from_file, read_viewframe_from_file
from cooltools.lib import is_valid_expected
import bioframe
import pytest


def test_read_expected_from_file(request, tmpdir):

    expected_file = op.join(request.fspath.dirname, "data/CN.mm9.toy_expected.chromnamed.tsv")
    expected_df = read_expected_from_file(expected_file, expected_value_cols=["balanced.avg"])

    assert is_valid_expected(
        expected_df, "cis", expected_value_cols=["balanced.avg"]
    )

    # test for error when string in one row of "n_valid" column (supposed to be Int64 dtype):
    expected_df_wrongdtype = expected_df.copy()
    expected_df_wrongdtype["n_valid"] = expected_df_wrongdtype["n_valid"].astype(str)
    expected_df_wrongdtype.loc[0,"n_valid"] = "string"
    expected_df_wrongdtype.to_csv(op.join(tmpdir, "CN.mm9.toy_expected_wrongdtype.tsv"), 
                                  sep="\t", index=False)
    with pytest.raises(ValueError):
        read_expected_from_file(
            op.join(tmpdir, "CN.mm9.toy_expected_wrongdtype.tsv"),
            expected_value_cols=["balanced.avg"],
        )

    # test that read_expected from file works if chroms are mix of str and int
    expected_df_intchr = expected_df.copy()
    expected_df_intchr["region1"] = expected_df_intchr["region1"].str.replace('chr1','1')
    expected_df_intchr["region2"] = expected_df_intchr["region2"].str.replace('chr1','1')
    expected_df_intchr.to_csv(op.join(tmpdir, "CN.mm9.toy_expected_intchr.tsv"), 
                                  sep="\t", index=False)
    expected_df_intchr = read_expected_from_file(op.join(tmpdir, "CN.mm9.toy_expected_intchr.tsv"),
                                                 expected_value_cols=["balanced.avg"])
    assert is_valid_expected(
        expected_df_intchr, "cis", expected_value_cols=["balanced.avg"]
    )


def test_read_viewframe_from_file(request, tmpdir):

    # test viewframe with 4 columns - i.e. with unique names
    view_file_wnames = op.join(request.fspath.dirname, "data/CN.mm9.toy_regions.bed")
    view_df = read_viewframe_from_file(view_file_wnames, verify_cooler=None, check_sorting=False)
    assert bioframe.is_viewframe(view_df)

    # test viewframe with 3 columns - i.e. without unique names
    view_file_wonames = op.join(request.fspath.dirname, "data/CN.mm9.toy_features.bed")
    view_df = read_viewframe_from_file(view_file_wonames, verify_cooler=None, check_sorting=False)
    assert bioframe.is_viewframe(view_df)
    # for a 3 column viewframe, UCSC strings should assigned to names
    assert view_df["name"].apply(bioframe.is_complete_ucsc_string).all()
