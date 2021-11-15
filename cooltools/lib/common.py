import warnings
from itertools import tee, starmap
from operator import gt
from copy import copy

import numpy as np
import pandas as pd


# Test data download requirements:
import requests
import os
import hashlib
import bioframe

from . import schemas

URL_DATA = "https://raw.githubusercontent.com/open2c/cooltools/master/datasets/external_test_files.tsv"

def _is_sorted_ascending(iterable):
    # code copied from "more_itertools" package
    """Returns ``True`` if the items of iterable are in sorted order, and
    ``False`` otherwise.

    The function returns ``False`` after encountering the first out-of-order
    item. If there are no out-of-order items, the iterable is exhausted.
    """

    it0, it1 = tee(iterable) # duplicate the iterator
    next(it1, None) # skip 1st element in "it1" copy
    # check if all values in iterable are in ascending order
    # similar to all(array[:-1] < array[1:])
    _pairs_out_of_order = starmap(gt, zip(it0, it1) )
    # no pairs out of order returns True, i.e. iterator is sorted
    return not any(_pairs_out_of_order)


def assign_regions(features, supports):
    """
    DEPRECATED. Will be removed in the future versions and replaced with bioframe.overlap()
    For each feature in features dataframe assign the genomic region (support)
    that overlaps with it. In case if feature overlaps multiple supports, the
    region with largest overlap will be reported.
    """

    index_name = features.index.name  # Store the name of index
    features = (
        features.copy().reset_index().rename({
            'index' if index_name is None else index_name: 'native_order'
        }, axis=1)
    )  # Store the original features' order as a column with original index

    if "chrom" in features.columns:
        overlap = bioframe.overlap(
            features,
            supports,
            how="left",
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            keep_order=True,
            return_overlap=True,
            suffixes=('_1', '_2')
        )
        overlap_columns = overlap.columns  # To filter out duplicates later
        overlap["overlap_length"] = overlap["overlap_end"] - overlap["overlap_start"]
        # Filter out overlaps with multiple regions:
        overlap = (
            overlap.sort_values("overlap_length", ascending=False)
            .drop_duplicates(overlap_columns, keep="first")
            .sort_index()
        )
        # Copy single column with overlapping region name:
        features["region"] = overlap["name_2"]

    if "chrom1" in features.columns:
        for idx in ("1", "2"):
            overlap = bioframe.overlap(
                features,
                supports,
                how="left",
                cols1=[f"chrom{idx}", f"start{idx}", f"end{idx}"],
                cols2=[f"chrom", f"start", f"end"],
                keep_order=True,
                return_overlap=True,
                suffixes=('_1', '_2')
            )
            overlap_columns = overlap.columns  # To filter out duplicates later
            overlap[f"overlap_length{idx}"] = (
                overlap[f"overlap_end{idx}"] - overlap[f"overlap_start{idx}"]
            )
            # Filter out overlaps with multiple regions:
            overlap = (
                overlap.sort_values(f"overlap_length{idx}", ascending=False)
                .drop_duplicates(overlap_columns, keep="first")
                .sort_index()
            )
            # Copy single column with overlapping region name:
            features[f"region{idx}"] = overlap["name_2"]

        # Form a single column with region names where region1 == region2, and np.nan in other cases:
        features["region"] = np.where(
            features["region1"] == features["region2"], features["region1"], np.nan
        )
        features = features.drop(
            ["region1", "region2"], axis=1
        )  # Remove unnecessary columns

    features = features.set_index('native_order')  # Restore the original index
    features.index.name = index_name  # Restore original index title
    return features


def assign_supports(features, supports, labels=False, suffix=""):
    """
    Assign support regions to a table of genomic intervals.
    Obsolete, replaced by assign_regions now.

    Parameters
    ----------
    features : DataFrame
        Dataframe with columns `chrom`, `start`, `end`
        or `chrom1`, `start1`, `end1`, `chrom2`, `start2`, `end2`
    supports : array-like
        Support areas

    """
    features = features.copy()
    supp_col = pd.Series(index=features.index, data=np.nan)

    c = "chrom" + suffix
    s = "start" + suffix
    e = "end" + suffix
    for col in (c, s, e):
        if col not in features.columns:
            raise ValueError(
                'Column "{}" not found in features data frame.'.format(col)
            )

    for i, region in enumerate(supports):
        # single-region support
        if len(region) in [3, 4]:
            sel = (features[c] == region[0]) & (features[e] > region[1])
            if region[2] is not None:
                sel &= features[s] < region[2]
        # paired-region support
        elif len(region) == 2:
            region1, region2 = region
            sel1 = (features[c] == region1[0]) & (features[e] > region1[1])
            if region1[2] is not None:
                sel1 &= features[s] < region1[2]
            sel2 = (features[c] == region2[0]) & (features[e] > region2[1])
            if region2[2] is not None:
                sel2 &= features[s] < region2[2]
            sel = sel1 | sel2
        supp_col.loc[sel] = i

    if labels:
        supp_col = supp_col.map(lambda i: supports[int(i)], na_action="ignore")

    return supp_col


def assign_regions_to_bins(bin_ids, regions_span):
    regions_binsorted = (
        regions_span[(regions_span["bin_start"] >= 0) & (regions_span["bin_end"] >= 0)]
            .sort_values(["bin_start", "bin_end"])
            .reset_index()
    )

    bin_reg_idx_lo = regions_span["bin_start"].searchsorted(bin_ids, "right") - 1
    bin_reg_idx_hi = regions_span["bin_end"].searchsorted(bin_ids, "right")
    mask_assigned = (bin_reg_idx_lo == bin_reg_idx_hi) & (bin_reg_idx_lo >= 0)

    region_ids = pd.array([pd.NA] * len(bin_ids))
    region_ids[mask_assigned] = regions_span["name"][bin_reg_idx_lo[mask_assigned]]

    return region_ids


def download_file(url, local_filename=None):
    """
    Download single file and return its name.

    """

    if local_filename is None:
        local_filename = url.split('/')[-1]
    print('downloading:', url, 'as', local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def download_data(name="all",
                  cache=True,
                  data_dir=None,
                  ignore_checksum=False,
                  checksum_chunk=8192,
                  _url_info=URL_DATA):
    """
    Download the specified dataset or all available datasets ("all" option).
    Check available datasets with cooltools.print_available_datasets()

    By default, checksums are verified for downloaded and cached files.

    Parameters
    ----------
    name : str
        Name of the dataset. Default: "all"

    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required. Default: True

    data_dir : string, optional
        The directory where to cache data; default is defined by :func:`get_data_dir`.

    ignore_checksum : boolean, optional
        Ignore checksum of the file (not recommended). Default: False

    checksum_chunk : int, optional
        Sets the chunksize for calculation of md5sum of the file.
        Used only if ignore_checksum is False.

    Returns
    -------
    Path to the file if a single file was downloaded, or a list (for 'all' download).

    """

    available_datasets = _get_datasets_info(_url_info)
    available_keys = list(map(lambda x: x['key'], available_datasets))

    if name in available_keys:
        keys = [name]
    elif name == "all":
        keys = available_keys
    else:
        raise Exception(
            """Dataset {key} is not available. 
            Available datasets: {datasets}
            Use print_available_datasets() to see the details. 
            """.format(key=name, datasets=','.join(available_keys)))

    data_dir = get_data_dir(data_dir)

    assert len(keys) > 0  # Checking that test data file is parsed successfully
    downloaded = []
    for data in available_datasets:

        key, url, local_filename, original_checksum = data["key"], data["link"], data["filename"], data["checksum"]

        if key not in keys:
            continue

        file_path = os.path.join(data_dir, local_filename)

        if cache and os.path.exists(file_path):
            if not ignore_checksum:
                checksum = get_md5sum(file_path, chunksize=checksum_chunk)
                assert checksum==original_checksum, \
f"""The downloaded {key} file in {local_filename} differs from original test: {url}
Re-reun with cache=False """
            downloaded += [file_path]
            continue

        elif cache:
            print("Test dataset {} (file {}) is not in the cache directory {}".format(key, local_filename, data_dir))

        file_path = download_file(url, file_path)
        if not ignore_checksum:
            checksum = get_md5sum(file_path, chunksize=checksum_chunk)
            assert checksum == original_checksum, \
                f"Download of {key} to {local_filename} failed. File differs from original test: {url}"

        downloaded += [file_path]

    # No files, return empty string
    if len(downloaded) == 0:
        return ''
    # A single file downloaded, return its name
    elif len(downloaded) == 1:
        return downloaded[0]
    # Multiple files downloaded, return list
    else:
        return downloaded


def get_md5sum(file_path, chunksize=8192):
    """
    Load file by chunks and return md5sum of a file.
    File might be rather large, and we don't load it into memory as a whole.
    Name is excluded from calculation of hash.

    Parameters
    ----------
    file_path : str
        Location of the file

    chunksize : int
        Size of the chunk

    Returns
    -------
    String with md5sum hex hash

    """

    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            file_fragment = f.read(chunksize)
            if not file_fragment:
                break
            md5.update(file_fragment)
        return md5.hexdigest()


def get_data_dir(data_dir=None):
    """
    Return a path to cache directory for example datasets.

    This directory is then used by :func:`download_data`.

    By default, this function uses the location of cooltools as the home folder
    and stores the files under ./datasets/ path there.

    Parameters
    ----------
    data_dir : str, optional
        Location of the home for test data storage

    Returns
    -------
    String with path to the folder with test data

    Notes
    -------
    Designed following seaborn download test data examples:
    https://github.com/mwaskom/seaborn/blob/4dd57d63e23e4628d8b7f41cdefdf64f7fefcf74/seaborn/utils.py#L427

    """

    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets/'))

    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, '')


def _get_datasets_info(url=URL_DATA):
    """
    Reports all available datasets in URL_DATA

    Requires internet connection if url is a remote link.

    Returns
    -------

    A list of datasets, where each element is a dictionary with key, url, local_filename and checksum fields.

    """

    # Info file is remote
    if url.startswith("https://") or url.startswith("http://"):
        datasets_metadata = requests.get(url, stream=True).iter_lines()
    # Info file is local
    else:
        datasets_metadata = open(url, 'rb').readlines()

    header = []
    datasets_parsed = []
    for i, line_metadata in enumerate(datasets_metadata):
        # Read header:
        if line_metadata.decode("utf-8")[0] == "#":
            header = line_metadata.decode("utf-8")[2:].strip().split("\t")
        # Read the rest of the file:
        else:
            if len(header)==0:
                raise Exception(f"Data info file {url} is corrupted, no header denoted by '#'.")
            data_line = line_metadata.decode("utf-8").strip().split("\t")
            data = dict(zip(header, data_line))
            data.update({'index': i})
            datasets_parsed.append(dict(data))

    return datasets_parsed


def print_available_datasets(url=URL_DATA):
    """
    Prints all available test datasets in URL_DATA

    Requires internet connection if url is a remote link.

    """

    datasets_parsed = _get_datasets_info(url)
    for data in datasets_parsed:
        print("""{index}) {key} : {comment} 
\tDownloaded from {link} 
\tStored as {filename} 
\tOriginal md5sum: {checksum}
""".format(**data))


def make_cooler_view(clr, ucsc_names=False):
    """
    Generate a full chromosome viewframe
    using cooler's chromsizes

    Parameters
    ----------
    clr :  cooler
        cooler-object to extract chromsizes
    ucsc_names : bool
        Use full UCSC formatted names instead
        of short chromosome names.

    Returns
    -------
    cooler_view : viewframe
        full chromosome viewframe
    """
    cooler_view = bioframe.make_viewframe(clr.chromsizes)
    if ucsc_names:
        # UCSC formatted names
        return cooler_view
    else:
        # rename back to short chromnames
        cooler_view["name"] = cooler_view["chrom"]
        return cooler_view



def _is_expected(
        expected_df,
        contact_type="cis",
        expected_value_cols=["count.avg","balanced.avg"],
        raise_errors=False
    ):
    """
    Check if a expected_df looks like an expected
    DataFrame, i.e.:
     - has neccessary columns
     - there are no Nulls in regions1/2, diag
     - every trans region1/2 has a single value
     - every cis region1/2 has at least one value

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    contact_type : str
        'cis' or 'trans': run contact type specific checks
    expected_value_cols : list of str
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    _is_expected : bool
        True when expected_df passes the checks, False otherwise
    """

    if contact_type == "cis":
        expected_dtypes = copy(schemas.diag_expected_dtypes) # mutable copy
    elif contact_type == "trans":
        expected_dtypes = copy(schemas.block_expected_dtypes) # mutable copy
    else:
        raise ValueError(
            f"Incorrect contact_type: {contact_type}, only cis and trans are supported."
        )

    # that's what we expect as column names:
    expected_columns = [col for col in expected_dtypes]
    # separate "structural" columns: region1/2 and diag if "cis":
    grouping_columns = expected_columns[:-1]

    # add columns with values and their dtype (float64):
    for name in expected_value_cols:
        expected_columns.append(name)
        expected_dtypes[name] = "float"


    # try a bunch of assertions about expected
    try:
        # make sure expected is a DataFrame
        if not isinstance(expected_df, pd.DataFrame):
            raise ValueError(f"expected_df must be DataFrame, it is {type(expected_df)} instead")
        # make sure required columns are present and can be cast to the dtypes
        if set(expected_columns).issubset(expected_df.columns):
            try:
                expected_df = expected_df.astype(expected_dtypes)
            except Exception as e:
                raise ValueError(
                        "expected_df does not match the expected schema:\n"
                        f"columns {expected_columns} cannot be cast to required data types."
                    ) from e
        # raise special message for the old formatted expected_df :
        elif set(["region","chrom"]).intersection(expected_df.columns):
            warnings.warn(
                    "The expected dataframe appears to be in the old format."
                    "It should have `region1` and `region2` columns instead of `region` or `chrom`."
                    "Please recalculated your expected using current vestion of cooltools."
                )
            raise ValueError(
                "The expected dataframe appears to be in the old format."
                "It should have `region1` and `region2` columns instead of `region` or `chrom`."
                "Please recalculated your expected using current vestion of cooltools."
            )
        # does not look like expected at all :
        else:
            raise ValueError(
                "expected_df does not match the expected schema:\n"
                f"required columns {expected_columns} are missing"
            )

        # make sure there is no missing data in grouping columns
        if expected_df[grouping_columns].isna().any().any():
            raise ValueError(
                f"There are missing values in columns {grouping_columns}"
                )

        # make sure "grouping" columns are unique:
        if expected_df.duplicated(subset=grouping_columns).any():
            raise ValueError(
                f"Values in {grouping_columns} columns must be unique"
                )

        # make sure region1/2 groups have 1 value for trans contacts
        # and more than 1 values for cis contacts
        region1_col, region2_col = grouping_columns[:2]
        for (r1, r2), df in expected_df.groupby([region1_col, region2_col]):
            if contact_type == "trans":
                if len(df) != 1:
                    ValueError(
                        f"region {r1},{r2} has more than a single value.\n"
                        "It has to be single for trans-expected"
                        )
                if r1 == r2:
                    ValueError(
                        f"region {r1},{r2} is symmetric\n"
                        "trans expected is caluclated for asymmetric regions only"
                        )
            if contact_type == "cis":
                # generally there shoud be >1 values per region in cis-expected, but
                # tiny regions smaller than a binsize could have 1 value
                if len(df) < 1:
                    ValueError(
                        f"region {r1},{r2} has to have at least one values for cis-expected"
                        )

    except Exception as e:
        if raise_errors:
            raise e
        else:
            # does not look like proper expected
            return False
    else:
        # if no exceptions were raised, it looks like expected_df
        return True


def _is_compatible_cis_expected(
        expected_df,
        verify_view,
        verify_cooler=None,
        expected_value_cols=["count.avg","balanced.avg"],
        raise_errors=False,
    ):
    """
    Verify expected_df to make sure it is compatible
    with its view (viewframe) and cooler, i.e.:
        - regions1/2 are matching names from view
        - number of diagonals per region1/2 matches cooler

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    try:
        # make sure it looks like cis-expected in the first place
        try:
            _ = _is_expected(
                expected_df,
                "cis",
                expected_value_cols,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("expected_df does not look like cis-expected") from e

        # Check that view regions are named as in expected table.
        if not bioframe.is_cataloged(
            expected_df,
            verify_view,
            df_view_col="region1",
            view_name_col="name",
        ):
            raise ValueError(
                "View regions are not in the expected table. Provide expected table for the same regions"
            )

        # check if the number of diagonals is correct:
        if verify_cooler is not None:
            # check number of bins per region in cooler and expected table
            # compute # of bins by comparing matching indexes
            for (name1, name2), group in expected_df.groupby(["region1","region2"]):
                n_diags_expected = len(group)
                if name1 == name2:
                    region = verify_view.set_index("name").loc[name1]
                    lo, hi = verify_cooler.extent(region)
                    n_diags_cooler = hi - lo
                else:
                    region1 = verify_view.set_index("name").loc[name1]
                    region2 = verify_view.set_index("name").loc[name2]
                    lo1, hi1 = verify_cooler.extent(region1)
                    lo2, hi2 = verify_cooler.extent(region2)
                    if not _is_sorted_ascending([lo1, hi1, lo2, hi2]):
                        raise ValueError(f"Only upper right cis regions are supported, {name1}:{name2} is not")
                    # rectangle that is fully contained within upper-right part of the heatmap
                    n_diags_cooler = (hi1 - lo1) + (hi2 - lo2) - 1
            if n_diags_expected != n_diags_cooler:
                raise ValueError(
                    "Region shape mismatch between expected and cooler. "
                    "Are they using the same resolution?"
                )
    except Exception as e:
        if raise_errors:
            raise e
        else:
            # expected_df is not compatible
            return False
    else:
        return True


def _is_compatible_trans_expected(
        expected_df,
        verify_view,
        verify_cooler=None,
        expected_value_cols=["count.avg","balanced.avg"],
        raise_errors=False,
    ):
    """
    Verify expected_df to make sure it is compatible
    with its view (viewframe) and cooler, i.e.:
        - regions1/2 are matching names from view
        - number of diagonals per region1/2 matches cooler

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    try:
        # make sure it looks like trans-expected in the first place
        try:
            _ = _is_expected(
                expected_df,
                "trans",
                expected_value_cols,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError("expected_df does not look like trans-expected") from e

        # Check that view regions are named as in expected table.
        # Check region names:
        _all_expected_regions = expected_df[["region1", "region2"]].values.flatten()
        if not np.all(verify_view["name"].isin(_all_expected_regions)):
            raise ValueError(
                "View regions are not in the expected table. Provide expected table for the same regions"
            )

        # check if the number of diagonals is correct:
        if verify_cooler is not None:
            # check number of bins per region in cooler and expected table
            # compute # of bins by comparing matching indexes
            for (name1, name2), group in expected_df.groupby(["region1","region2"]):
                n_valid_expected = group["n_valid"].iat[0]  # extract single `n_valid` from group
                region1 = verify_view.set_index("name").loc[name1]
                region2 = verify_view.set_index("name").loc[name2]
                lo1, hi1 = verify_cooler.extent(region1)
                lo2, hi2 = verify_cooler.extent(region2)
                if not _is_sorted_ascending([lo1, hi1, lo2, hi2]):
                    raise ValueError(f"Only upper right trans regions are supported, {name1}:{name2} is not")
                # compare n_valid per trans block and make sure it make sense:
                n_valid_cooler = (hi1 - lo1) * (hi2 - lo2)
                if n_valid_cooler < n_valid_expected:
                    warnings.warn(
                        "trans expected was calculated for a cooler with higher resolution."
                        "make sure this is intentional."
                        )
                # consider adding a proper check here - which requires using balancing weights
    except Exception as e:
        if raise_errors:
            raise e
        else:
            # expected_df is not compatible
            return False
    else:
        return True


def is_compatible_expected(
        expected_df,
        contact_type,
        verify_view,
        verify_cooler=None,
        expected_value_cols=["count.avg","balanced.avg"],
        raise_errors=False,
    ):
    """
    Verify expected_df to make sure it is compatible

    Parameters
    ----------
    expected_df :  DataFrame
        expected DataFrame to be validated
    contact_type : str
        'cis' or 'trans': run contact type specific checks
    verify_view : viewframe
        Viewframe that defines regions in expected_df.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.
    expected_value_cols : list[str]
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    compatibility : bool
        Whether expected_df is compatible with view and cooler
    """

    if contact_type == "cis":
        return _is_compatible_cis_expected(
            expected_df,
            verify_view,
            verify_cooler=verify_cooler,
            expected_value_cols=expected_value_cols,
            raise_errors=raise_errors
        )
    elif contact_type == "trans":
        return _is_compatible_trans_expected(
            expected_df,
            verify_view,
            verify_cooler=verify_cooler,
            expected_value_cols=expected_value_cols,
            raise_errors=raise_errors
        )
    else:
        raise ValueError("contact_type can be only cis or trans")



def read_expected(
    fname,
    contact_type="cis",
    expected_value_cols=["count.avg","balanced.avg"],
    verify_view=None,
    verify_cooler=None,
    ):
    """
    Read an expected from a file.
    Expected must conform v1.0 format
    https://github.com/open2c/cooltools/issues/217

    Parameters
    ----------
    fname : str
        Path to a tsv file with expected
    contact_type : str
        cis and trans expected have different formats
    expected_value_cols : list of str
        Names of the column with the values of expected.
        Summaries and averaged values can be requested.
    verify_view : None or viewframe
        Viewframe that defines regions in expected. Used for
        verifications. All verifications are skipped if None.
    verify_cooler : None or cooler
        Cooler object to use when verifying if expected
        is compatible. No verifications is performed when None.

    Returns
    -------
    expected_df : pd.DataFrame
        DataFrame with the expected
    """

    # basic input check
    if contact_type not in ["cis", "trans"]:
        raise ValueError(f"contact_type can be only cis or trans, {contact_type} provided")


    try:
        expected_df = pd.read_table(fname)
        _ = _is_expected(expected_df,
                contact_type,
                expected_value_cols,
                raise_errors=True
                )
    except ValueError as e:
        raise ValueError(
            "Input expected does not match the schema\n"
            "It has to be a tab-separated file with a header"
        ) from e


    # validations against cooler and view_df
    if verify_view is not None:
        try:
            _ = is_compatible_expected(
                expected_df,
                contact_type,
                verify_view,
                verify_cooler,
                expected_value_cols,
                raise_errors=True,
            )
        except Exception as e:
            raise ValueError(
                "provided expected is not compatible with the specified view and/or cooler"
            ) from e

    return expected_df



def is_compatible_viewframe(
        view_df,
        verify_cooler,
        check_sorting=False,
        raise_errors=False
    ):
    """
    Check if view_df is a viewframe and if
    it is compatible with the provided cooler.

    Parameters
    ----------
    view_df :  DataFrame
        view_df DataFrame to be validated
    verify_cooler : cooler
        cooler object to use for verification
    check_sorting : bool
        Check is regions in view_df are sorted as in
        chromosomes in cooler.
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_compatible_viewframe : bool
        True when view_df is compatible, False otherwise
    """
    try:
        try:
            _ = bioframe.is_viewframe(view_df, raise_errors=True)
        except Exception as error_not_viewframe:
            try:
                _ = bioframe.make_viewframe(view_df)
            except Exception as error_cannot_make_viewframe:
                # view_df is not viewframe and cannot be easily converted
                raise ValueError(
                        "view_df is not a valid viewframe and cannot be recovered"
                    ) from error_cannot_make_viewframe
            else:
                # view_df is not viewframe, but can be converted - formatting issue ? name-column ?
                raise ValueError(
                        "view_df is not a valid viewframe, apply bioframe.make_viewframe to convert"
                    ) from error_not_viewframe

        # is view_df contained inside cooler-chromosomes ?
        cooler_view = make_cooler_view(verify_cooler)
        if not bioframe.is_contained(view_df, cooler_view):
            raise ValueError("View table is out of the bounds of chromosomes in cooler.")

        # is view_df sorted by coord and chrom order as in cooler ?
        if check_sorting:
            if not bioframe.is_sorted(view_df, cooler_view, df_view_col = "chrom"):
                raise ValueError(
                    "regions in the view_df must be sorted by coordinate"
                    " and chromosomes order as as in the verify_cooler."
                )

    except Exception as e:
        if raise_errors:
            raise ValueError("view_df is not compatible, or not a viewframe") from e
        else:
            # something went wrong: it's not a viewframe
            return False
    else:
        # no exceptions were raised: it's a compatible viewframe
        return True



def read_viewframe(
    view_fname,
    verify_cooler=None,
    check_sorting=False,
    ):
    """
    Read a BED file with regions that conforms
    a definition of a viewframe (non-overlaping, unique names, etc).

    Parameters
    ----------
    view_fname : str
        Path to a BED file with regions.
    verify_cooler : cooler | None
        cooler object to get chromsizes for bound checking
        No checks are done when None.
    check_sorting : bool
        Check is regions in view_df are sorted as in
        chromosomes in cooler.

    Returns
    -------
    view_df : pd.DataFrame
        DataFrame with the viewframe
    """

    # read BED file assuming bed4/3 formats (with names-columns and without):
    try:
        view_df = bioframe.read_table(view_fname, schema="bed4", index_col=False)
    except Exception as err_bed4:
        try:
            view_df = bioframe.read_table(view_fname, schema="bed3", index_col=False)
        except Exception as err_bed3:
            raise ValueError(f"{view_fname} is not a BED file with 3 or 4 columns") from err_bed4

    # Convert view dataframe to viewframe:
    try:
            view_df = bioframe.make_viewframe(view_df)
    except ValueError as e:
        raise ValueError(
            "View table is incorrect, please, comply with the format. "
        ) from e

    if verify_cooler is not None:
        try:
            _ = is_compatible_viewframe(
                    view_df,
                    verify_cooler,
                    check_sorting,
                    raise_errors=True
                )
        except Exception as e:
            raise ValueError("view_df is not compatible with the cooler") from e
        else:
            # view_df is compaible, returning
            return view_df
    else:
        # no cooler for checking, returning
        return view_df


def is_cooler_balanced(clr, clr_weight_name="weight", raise_errors=False):
    """
    Check if cooler is balanced, by checking
    if the requested weight column exist in the bin table.

    Parameters
    ----------
    clr : cooler
        cooler object to check
    clr_weight_name : str
        name of the weight column to check
    raise_errors : bool
        raise expection instead of returning False

    Returns
    -------
    is_balanced : bool
        True if weight column is present, False otherwise
    """

    if not isinstance(clr_weight_name, str):
        raise TypeError("clr_weight_name has to be str that specifies name of balancing weight in clr")

    if clr_weight_name in schemas.DIVISIVE_WEIGHTS_4DN:
        raise KeyError(
            f"clr_weight_name: {clr_weight_name} is reserved as divisive by 4DN"
            "cooltools supports multiplicative weights at this time."
        )

    # check if clr_weight_name is in cooler
    if clr_weight_name not in clr.bins().columns:
        if raise_errors:
            raise ValueError(f"specified balancing weight {clr_weight_name} is not available in cooler")
        else:
            return False
    else:
        return True
