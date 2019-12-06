import subprocess
import pandas as pd
import numpy as np
import h5py


# make chromsizes
with open("./test.chrom.sizes", "w") as chromsizes:
    chromsizes.write("chr1\t1000\n")
    chromsizes.write("chr2\t2000\n")
    chromsizes.write("chr3\t3000")

BIN_SIZE = 10
# make bins
subprocess.check_output(
    f"cooltools genome binnify ./test.chrom.sizes {BIN_SIZE} > ./test.10.bins",
    shell=True,
)

# make Hi-C data
bins = pd.read_table("./test.10.bins", sep="\t")
EIG_PERIOD_BP = 500
EIG_AMPLITUDE = np.sqrt(0.5)
SCALING = -2
MAX_CIS_COUNTS = 1e8
MAX_TRANS_COUNTS = 1e5

bins["eig"] = EIG_AMPLITUDE * np.sin(bins.start * 2 * np.pi / EIG_PERIOD_BP)
bins["key"] = 0
pixels = pd.merge(bins, bins, on="key", how="outer", suffixes=("1", "2"))
pixels.drop("key", axis="columns", inplace=True)
pixels["count"] = np.nan

cis = pixels.chrom1 == pixels.chrom2
pixels.loc[cis, "count"] = pixels[cis].eval(
    "@MAX_CIS_COUNTS * ((abs(start1-start2)+@BIN_SIZE)**@SCALING) * (1.0+eig1*eig2)"
)
pixels.loc[~cis, "count"] = pixels[~cis].eval("@MAX_TRANS_COUNTS * (1.0+eig1*eig2)")

pixels["count"] = pixels["count"].astype(int)
pixels[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "count"]].to_csv(
    "./sin_eigs_mat.bg2.gz", sep="\t", index=False, header=None, compression="gzip"
)


# make a cooler
subprocess.check_output(
    "cooler load -f bg2 --count-as-float --tril-action drop "
    + f"./test.chrom.sizes:{BIN_SIZE} ./sin_eigs_mat.bg2.gz "
    + "./sin_eigs_mat.cool",
    shell=True,
)

# fake IC
f = h5py.File("./sin_eigs_mat.cool")
f["bins/weight"] = np.ones_like(f["bins/start"], dtype=float)
f["bins/weight"].attrs["ignore_diags"] = 2
f.close()
