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
from .checks import is_valid_expected, is_compatible_viewframe

URL_DATA = "https://raw.githubusercontent.com/open2c/cooltools/master/datasets/external_test_files.tsv"


def download_file(url, local_filename=None):
    """
    Download single file and return its name.

    """

    if local_filename is None:
        local_filename = url.split("/")[-1]
    print("downloading:", url, "as", local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def download_data(
    name="all",
    cache=True,
    data_dir=None,
    ignore_checksum=False,
    checksum_chunk=8192,
    _url_info=URL_DATA,
):
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
    available_keys = list(map(lambda x: x["key"], available_datasets))

    if name in available_keys:
        keys = [name]
    elif name == "all":
        keys = available_keys
    else:
        raise Exception(
            """Dataset {key} is not available. 
            Available datasets: {datasets}
            Use print_available_datasets() to see the details. 
            """.format(
                key=name, datasets=",".join(available_keys)
            )
        )

    data_dir = get_data_dir(data_dir)

    assert len(keys) > 0  # Checking that test data file is parsed successfully
    downloaded = []
    for data in available_datasets:

        key, url, local_filename, original_checksum = (
            data["key"],
            data["link"],
            data["filename"],
            data["checksum"],
        )

        if key not in keys:
            continue

        file_path = os.path.join(data_dir, local_filename)

        if cache and os.path.exists(file_path):
            if not ignore_checksum:
                checksum = get_md5sum(file_path, chunksize=checksum_chunk)
                assert (
                    checksum == original_checksum
                ), f"""The downloaded {key} file in {local_filename} differs from original test: {url}
Re-reun with cache=False """
            downloaded += [file_path]
            continue

        elif cache:
            print(
                "Test dataset {} (file {}) is not in the cache directory {}".format(
                    key, local_filename, data_dir
                )
            )

        file_path = download_file(url, file_path)
        if not ignore_checksum:
            checksum = get_md5sum(file_path, chunksize=checksum_chunk)
            assert (
                checksum == original_checksum
            ), f"Download of {key} to {local_filename} failed. File differs from original test: {url}"

        downloaded += [file_path]

    # No files, return empty string
    if len(downloaded) == 0:
        return ""
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
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "datasets/")
        )

    data_dir = os.path.expanduser(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return os.path.join(data_dir, "")


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
        datasets_metadata = open(url, "rb").readlines()

    header = []
    datasets_parsed = []
    for i, line_metadata in enumerate(datasets_metadata):
        # Read header:
        if line_metadata.decode("utf-8")[0] == "#":
            header = line_metadata.decode("utf-8")[2:].strip().split("\t")
        # Read the rest of the file:
        else:
            if len(header) == 0:
                raise Exception(
                    f"Data info file {url} is corrupted, no header denoted by '#'."
                )
            data_line = line_metadata.decode("utf-8").strip().split("\t")
            data = dict(zip(header, data_line))
            data.update({"index": i})
            datasets_parsed.append(dict(data))

    return datasets_parsed


def print_available_datasets(url=URL_DATA):
    """
    Prints all available test datasets in URL_DATA

    Requires internet connection if url is a remote link.

    """

    datasets_parsed = _get_datasets_info(url)
    for data in datasets_parsed:
        print(
            """{index}) {key} : {comment} 
\tDownloaded from {link} 
\tStored as {filename} 
\tOriginal md5sum: {checksum}
""".format(
                **data
            )
        )


def read_expected_from_file(
    fname,
    contact_type="cis",
    expected_value_cols=["count.avg", "balanced.avg"],
    verify_view=None,
    verify_cooler=None,
    raise_errors=True,
):
    """
    Read expected dataframe from a file.

    Expected must conform v1.0 format
    https://github.com/open2c/cooltools/issues/217

    dtypes are from https://github.com/open2c/cooltools/blob/master/cooltools/lib/schemas.py

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

    if contact_type == "cis":
        expected_dtypes = copy(schemas.diag_expected_dtypes)  # mutable copy
    elif contact_type == "trans":
        expected_dtypes = copy(schemas.block_expected_dtypes)  # mutable copy
    else:
        raise ValueError(
            f"Incorrect contact_type: {contact_type}, only cis and trans are supported."
        )

    try:
        expected_df = pd.read_table(fname, dtype=expected_dtypes)
        _ = is_valid_expected(
            expected_df,
            contact_type,
            verify_view=verify_view,
            verify_cooler=verify_cooler,
            expected_value_cols=expected_value_cols,
            raise_errors=raise_errors,
        )
    except ValueError as e:
        raise ValueError(
            "Input expected file does not match the schema\n"
            "Expected must be tab-separated file with a header"
        ) from e

    return expected_df


def read_viewframe_from_file(
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
            raise ValueError(
                f"{view_fname} is not a BED file with 3 or 4 columns"
            ) from err_bed4

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
                view_df, verify_cooler, check_sorting, raise_errors=True
            )
        except Exception as e:
            raise ValueError("view_df is not compatible with the cooler") from e
        else:
            # view_df is compaible, returning
            return view_df
    else:
        # no cooler for checking, returning
        return view_df
