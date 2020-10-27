import numpy as np
import pandas as pd
# Test data download requirements:
import requests
import os

def assign_supports(features, supports, labels=False, suffix=""):
    """
    Assign support regions to a table of genomic intervals.

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


def download_test_data(name="all", cache=True, data_home=None):
    """

    Parameters
    ----------
    name : str
        Name of the dataset.
    cache : boolean, optional
        If True, try to load from the local cache first, and save to the cache
        if a download is required.
    data_home : string, optional
        The directory where to cache data; default is defined by :func:`get_data_home`.

    Returns
    -------
    Path to the last downloaded file.

    """

    def download_file(url, local_filename=None):
        if local_filename is None:
            local_filename = url.split('/')[-1]
        print('downloading:', url, 'as', local_filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    available_datasets = _get_datasets_info()
    available_keys = map(lambda x: x[0], available_datasets)

    if name in available_keys:
        keys = [name]
    elif name=="all":
        keys = available_keys
    else:
        raise Exception(
            """Dataset {key} is not available. 
            Available datasets: {datasets}
            Use print_available_datasets() to see the details. 
            """.format(key=name, datasets=','.join(available_keys)))

    data_home = get_data_home(data_home)

    assert len(keys)>0 # Checking that test data file is parsed sucessfully
    for key, url, local_filename in available_datasets:
        if not key in keys:
            continue

        file_path = os.path.join(data_home, local_filename)
        if cache and os.path.exists(file_path):
            continue
        elif cache:
            print("Test dataset {} (file {}) is not in the cache directory {}".format(key, local_filename, data_home))
        downloaded = download_file(url, file_path)

    return downloaded


def get_data_home(data_home=None):
    """
    Returns a path to cache directory for example datasets.

    This directory is then used by :func:`download_test_data`.

    By default, this function uses the location of cooltools as the home folder
    and stores the files under ./tests/data/external/ path there.

    Parameters
    ----------
    data_home : str, optional
        Location of the home for test data storage

    Returns
    -------
    String with path to the folder with test data

    Notes
    -------
    Designed following seaborn download test data examples:
    https://github.com/mwaskom/seaborn/blob/4dd57d63e23e4628d8b7f41cdefdf64f7fefcf74/seaborn/utils.py#L427

    """

    if data_home is None:
        data_home = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..', 'tests/data/external/'))

    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)

    return data_home


def _get_datasets_info():
    """
    Reports all available datasets in https://github.com/open2c/cooltools/tests/data/external_test_files.tsv

    Requires an internet connection.

    Returns
    -------

    A list of datasets: [key, url, local_filename]

    """

    url = "https://github.com/agalitsyna/cooltools/tests/data/external_test_files.tsv"
    datasets_metadata = requests.get(url, stream=True).iter_lines()
    datasets = []
    for line_metadata in datasets_metadata:
        line_metadata = line_metadata.decode("utf-8")
        if not line_metadata[0]=="#":
            datasets.append([line_metadata.split()[0], line_metadata.split()[2], line_metadata.split()[1]])

    return datasets

def print_available_datasets():
    """
    Prints all available test datasets in https://github.com/open2c/cooltools/tests/data/external_test_files.tsv

    Requires an internet connection.

    """

    url = "https://github.com/agalitsyna/cooltools/tests/data/external_test_files.tsv"
    datasets_metadata = requests.get(url, stream=True).iter_lines()
    print("Available datasets:")
    for line_metadata in datasets_metadata:
        if not line_metadata.decode("utf-8")[0]=="#":
            print("{0} : {3} \n  Downloaded from {2} \n  Stored as {1}".format(*line_metadata.decode("utf-8").split("\t")))
