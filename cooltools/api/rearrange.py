import bioframe as bf
import cooler
import numpy as np

from ..lib.checks import is_compatible_viewframe
import logging

logging.basicConfig(level=logging.INFO)


def _generate_adjusted_chunks(
    clr, binmapping, chunksize=1_000_000, orientation_col="strand", nproc=1
):
    """Generates chunks of pixels from the cooler and adjusts their bin IDs to follow the view."""
    npixels = clr.pixels().shape[0]
    chunks = np.append(np.arange(0, npixels, chunksize), npixels)
    chunks = list(zip(chunks[:-1], chunks[1:]))

    # Running this loop in parallel slows the function down
    for i0, i1 in chunks:
        chunk = clr.pixels()[i0:i1]
        chunk["bin1_id"] = chunk["bin1_id"].map(binmapping)
        chunk["bin2_id"] = chunk["bin2_id"].map(binmapping)
        chunk = chunk[(chunk["bin1_id"] != -1) & (chunk["bin2_id"] != -1)]
        if chunk.shape[0] > 0:
            chunk[["bin1_id", "bin2_id"]] = np.sort(
                chunk[["bin1_id", "bin2_id"]].astype(int)
            )
            yield chunk.reset_index(drop=True)
        logging.info(f"Processed {i1/npixels*100:.2f}% pixels")


def _adjust_start_end(chromdf):
    chromdf["end"] = chromdf["length"].cumsum()
    chromdf["start"] = chromdf["end"] - chromdf["length"]
    return chromdf


def _flip_bins(regdf):
    regdf = regdf.iloc[::-1].reset_index(drop=True)
    l = regdf["end"] - regdf["start"]
    regdf["start"] = regdf["end"].max() - regdf["end"]
    regdf["end"] = regdf["start"] + l
    return regdf


def rearrange_bins(
    bins_old, view_df, new_chrom_col="new_chrom", orientation_col="strand"
):
    """
    Rearranges the input `bins_old` based on the information in the `view_df` DataFrame.

    Parameters
    ----------
    bins_old : bintable
        The original bintable to rearrange.
    view_df : viewframe
        Viewframe with new order of genomic regions. Can have an additional column for
        the new chromosome name (`new_chrom_col`), and an additional column for the
        strand orientation (`orientation_col`, '-' will invert the region).
    new_chrom_col : str, optional
        Column name in the view_df specifying new chromosome name for each region,
        by default 'new_chrom'. If None, then the default chrom column names will be used.
    orientation_col : str, optional
        Column name in the view_df specifying strand orientation of each region,
        by default 'strand'. The values in this column can be "+" or "-".
        If None, then all will be assumed "+".


    Returns
    -------
    bins_new : bintable
        The rearranged bintagle with the new chromosome names and orientations.
    bin_mapping : dict
        Mapping of original bin IDs to new bin IDs
    """
    chromdict = dict(zip(view_df["name"].to_numpy(), view_df[new_chrom_col].to_numpy()))
    flipdict = dict(
        zip(view_df["name"].to_numpy(), (view_df[orientation_col] == "-").to_numpy())
    )
    bins_old = bins_old.reset_index(names=["old_id"])
    bins_subset = bf.assign_view(bins_old, view_df, drop_unassigned=False).dropna(
        subset=["view_region"]
    )
    bins_inverted = (
        bins_subset.groupby("view_region", group_keys=False)
        .apply(lambda x: _flip_bins(x) if flipdict[x.name] else x)
        .reset_index(drop=True)
    )
    bins_new = bf.sort_bedframe(
        bins_inverted,
        view_df=view_df,
        df_view_col="view_region",
    )
    bins_new["chrom"] = bins_new["view_region"].map(chromdict)
    bins_new["length"] = bins_new["end"] - bins_new["start"]
    bins_new = (
        bins_new.groupby("chrom", group_keys=False)
        .apply(_adjust_start_end)
        .drop(columns=["length", "view_region"])
    )
    logging.info("Rearranged bins")
    bin_mapping = {old_id: -1 for old_id in bins_old["old_id"].astype(int)}
    bin_mapping.update(
        {
            old_id: new_id
            for old_id, new_id in zip(bins_new["old_id"].astype(int), bins_new.index)
        }
    )
    logging.info("Created bin mapping")
    bins_new = bins_new.drop(columns=["old_id"])
    return bins_new, bin_mapping


def rearrange_cooler(
    clr,
    view_df,
    out_cooler,
    new_chrom_col="new_chrom",
    orientation_col="strand",
    assembly=None,
    chunksize=10_000_000,
    mode="w",
):
    """Reorder cooler following a genomic view.

    Parameters
    ----------
    clr : cooler.Cooler
        Cooler object
    view_df : viewframe
        Viewframe with new order of genomic regions. Can have an additional column for
        the new chromosome name (`new_chrom_col`), and an additional column for the
        strand orientation (`orientation_col`, '-' will invert the region).
    out_cooler : str
        File path to save the reordered data
    new_chrom_col : str, optional
        Column name in the view_df specifying new chromosome name for each region,
        by default 'new_chrom'. If None, then the default chrom column names will be used.
    orientation_col : str, optional
        Column name in the view_df specifying strand orientation of each region,
        by default 'strand'. The values in this column can be "+" or "-".
        If None, then all will be assumed "+".
    assembly : str, optional
        The name of the assembly for the new cooler. If None, uses the same as in the
        original cooler.
    chunksize : int, optional
        The number of pixels loaded and processed per step of computation.
    mode : str, optional
        (w)rite or (a)ppend to the output cooler file. Default: w
    """

    view_df = view_df.copy()
    try:
        _ = is_compatible_viewframe(
            view_df[["chrom", "start", "end", "name"]],
            clr,
            check_sorting=False,
            raise_errors=True,
        )
    except Exception as e:
        raise ValueError("view_df is not a valid viewframe or incompatible") from e

    if assembly is None:
        assembly = clr.info["genome-assembly"]

    # Add repeated entries for new chromosome names if they were not requested/absent:
    if new_chrom_col is None:
        new_chrom_col = "new_chrom"
        if new_chrom_col in view_df.columns:
            logging.warn(
                "new_chrom_col is not provided, but new_chrom column exists."
                " Pre-existing new_chrom column will not be used."
            )
            while new_chrom_col in view_df.columns:
                new_chrom_col = (
                    f"new_chrom_{np.random.randint(0, np.iinfo(np.int32).max)}"
                )
    if new_chrom_col not in view_df.columns:
        view_df.loc[:, new_chrom_col] = view_df["chrom"]

    # Add repeated entries for strand orientation of chromosomes if they were not requested/absent:
    if orientation_col is None:
        orientation_col = "strand"
        if orientation_col in view_df.columns:
            logging.warn(
                "orientation_col is not provided, but strand column exists."
                " Pre-existing strand column will not be used."
            )
        while orientation_col in view_df.columns:
            orientation_col = f"strand_{np.random.randint(0, np.iinfo(np.int32).max)}"
    if orientation_col not in view_df.columns:
        view_df.loc[:, orientation_col] = "+"

    if not np.all(
        view_df.groupby(new_chrom_col).apply(lambda x: np.all(np.diff(x.index) == 1))
    ):
        raise ValueError("New chromosomes are not consecutive")
    bins_old = clr.bins()[:]
    # Creating new bin table
    bins_new, bin_mapping = rearrange_bins(
        bins_old, view_df, new_chrom_col=new_chrom_col, orientation_col=orientation_col
    )
    logging.info("Creating a new cooler")
    cooler.create_cooler(
        out_cooler,
        bins_new,
        _generate_adjusted_chunks(
            clr,
            bin_mapping,
            chunksize=chunksize,
        ),
        assembly=assembly,
        mode=mode,
        mergebuf=int(1e9),
    )
    logging.info(f"Created a new cooler at {out_cooler}")
