import os.path as op
import pandas as pd
from cooltools.lib.io import read_expected_from_file
from cooltools.lib import is_valid_expected
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