# cooltools

[![Pytest](https://github.com/open2c/cooltools/actions/workflows/pytest.yml/badge.svg)](https://github.com/open2c/cooltools/actions/workflows/pytest.yml)
[![Documentation Status](https://readthedocs.org/projects/cooltools/badge/?version=latest)](https://cooltools.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/82413481.svg)](https://zenodo.org/badge/latestdoi/82413481)

> tools for your .cools

Chromosome conformation capture technologies reveal the incredible complexity of genome folding. A growing number of labs and multiple consortia, including the 4D Nucleome, the International Nucleome Consortium, and ENCODE, are generating higher-resolution datasets to probe genome architecture across cell states, types, and organisms. Larger datasets increase the challenges at each step of computational analysis, from storage, to memory, to researchers’ time. The recently-introduced [***cooler***](https://github.com/open2c/cooler/tree/master/cooler) format readily handles storage of high-resolution datasets via a sparse data model.

***cooltools*** leverages this format to enable flexible and reproducible analysis of high-resolution data. ***cooltools*** provides a suite of computational tools with a paired python API and command line access, which facilitates workflows either on high-performance computing clusters or via custom analysis notebooks. As part of the [***Open2C*** ecosystem](https://open2c.github.io/), ***cooltools*** also provides detailed introductions to key concepts in Hi-C-data analysis with interactive notebook documentation.

If you use ***cooltools*** in your work, please cite ***cooltools*** via its zenodo [DOI:10.5281/zenodo.5214125](https://doi.org/10.5281/zenodo.5214125)

## Requirements

The following are required before installing cooltools:

* Python 3.7+
* `numpy`
* `cython`

## Installation

```sh
pip install cooltools
```

or install the latest version directly from github:

```
    $ pip install https://github.com/open2c/cooltools/archive/refs/heads/master.zip
``` 

See the [requirements.txt](https://github.com/open2c/cooltools/blob/master/requirements.txt) file for information on compatible dependencies, especially for [cooler](https://github.com/open2c/cooler/tree/master/cooler) and [bioframe](https://github.com/open2c/bioframe).


## Documentation and Tutorials

Documentation can be found here: https://cooltools.readthedocs.io/en/latest/.

Cooltools offers a number of tutorials to showcase analyses it enables using the [Open2c code ecosystem](https://github.com/open2c/). For users who are new to Hi-C analysis, we recommend going through example notebooks in the following order:

- [Visualization](https://cooltools.readthedocs.io/en/latest/notebooks/viz.html): how to load and visualize Hi-C data stored in coolers.
- [Contacts vs Distance](https://cooltools.readthedocs.io/en/latest/notebooks/contacts_vs_distance.html):  how to calculate contact frequency as a function of genomic distance, the most prominent feature in Hi-C maps.
- [Compartments and Saddles](https://cooltools.readthedocs.io/en/latest/notebooks/compartments_and_saddles.html):  how to extract eigenvectors and create saddleplots reflecting A/B compartments.
- [Insulation and Boundaries](https://cooltools.readthedocs.io/en/latest/notebooks/insulation_and_boundaries.html):  how to extract insulation profiles and call boundaries using insulation profile minima.
- [Pileups and Average Patterns](https://cooltools.readthedocs.io/en/latest/notebooks/pileup_CTCF.html): how to create avearge maps around genomic features like CTCF.

Note that these notebooks currently focus on mammalian interphase Hi-C analysis, but are readily extendible to other organisms and cellular contexts.

## Contributing

Our contributing guide can be found [here](https://github.com/open2c/cooltools/blob/master/CONTRIBUTING.md).

