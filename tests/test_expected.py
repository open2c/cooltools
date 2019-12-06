import os.path as op
import pandas as pd

import bioframe
import cooler
import cooltools.expected

chromsizes = bioframe.fetch_chromsizes("mm9")
chromosomes = list(chromsizes.index)
supports = [(chrom, 0, chromsizes[chrom]) for chrom in chromosomes]


def test_diagsum(request):
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    tables = cooltools.expected.diagsum(
        clr,
        supports,
        transforms={"balanced": lambda p: p["count"] * p["weight1"] * p["weight2"]},
        chunksize=10000000,
    )
    pd.concat(
        [tables[support] for support in supports],
        keys=[support[0] for support in supports],
        names=["chrom"],
    )


def test_blocksum(request):
    clr = cooler.Cooler(op.join(request.fspath.dirname, "data/CN.mm9.1000kb.cool"))
    records = cooltools.expected.blocksum_pairwise(
        clr,
        supports,
        transforms={"balanced": lambda p: p["count"] * p["weight1"] * p["weight2"]},
        chunksize=10000000,
    )
    pd.DataFrame(
        [
            {"chrom1": s1[0], "chrom2": s2[0], **rec}
            for (s1, s2), rec in records.items()
        ],
        columns=["chrom1", "chrom2", "n_valid", "count.sum", "balanced.sum"],
    )
