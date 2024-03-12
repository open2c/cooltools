# cooltools: enabling high-resolution Hi-C analysis in Python


<img src="https://github.com/open2c/cooltools/blob/master/docs/figs/cooltools-logo-futura.png" width=15%> 

[![Pytest](https://github.com/open2c/cooltools/actions/workflows/pytest.yml/badge.svg)](https://github.com/open2c/cooltools/actions/workflows/pytest.yml)
[![Documentation Status](https://readthedocs.org/projects/cooltools/badge/?version=latest)](https://cooltools.readthedocs.io/en/latest/?badge=latest)
[![Latest Release PyPI](https://img.shields.io/pypi/v/cooltools?color=blue&label=PyPI%20package)](https://pypi.org/project/cooltools)
[![Latest Release Bioconda](https://img.shields.io/conda/vn/bioconda/cooltools?color=blue)](https://bioconda.github.io/recipes/cooltools/README.html)
[![DOI](https://zenodo.org/badge/82413481.svg)](https://zenodo.org/badge/latestdoi/82413481)

> tools for your .cools

Chromosome conformation capture technologies reveal the incredible complexity of genome folding. A growing number of labs and multiple consortia, including the 4D Nucleome, the International Nucleome Consortium, and ENCODE, are generating higher-resolution datasets to probe genome architecture across cell states, types, and organisms. Larger datasets increase the challenges at each step of computational analysis, from storage, to memory, to researchers’ time. The recently-introduced [***cooler***](https://github.com/open2c/cooler/tree/master/cooler) format readily handles storage of high-resolution datasets via a sparse data model.

***cooltools*** leverages this format to enable flexible and reproducible analysis of high-resolution data. ***cooltools*** provides a suite of computational tools with a paired python API and command line access, which facilitates workflows either on high-performance computing clusters or via custom analysis notebooks. As part of the [***Open2C*** ecosystem](https://open2c.github.io/), ***cooltools*** also provides detailed introductions to key concepts in Hi-C-data analysis with interactive notebook documentation. For more information, see the [preprint](https://doi.org/10.1101/2022.10.31.514564): https://doi.org/10.1101/2022.10.31.514564.

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

Cooltools offers a number of tutorials using the [Open2c code ecosystem](https://github.com/open2c/). For users who are new to Hi-C analysis, we recommend going through example notebooks in the following order:

- [Visualization](https://cooltools.readthedocs.io/en/latest/notebooks/viz.html): how to load and visualize Hi-C data stored in coolers.
- [Contacts vs Distance](https://cooltools.readthedocs.io/en/latest/notebooks/contacts_vs_distance.html):  how to calculate contact frequency as a function of genomic distance, the most prominent feature in Hi-C maps.
- [Compartments and Saddles](https://cooltools.readthedocs.io/en/latest/notebooks/compartments_and_saddles.html):  how to extract eigenvectors and create saddleplots reflecting A/B compartments.
- [Insulation and Boundaries](https://cooltools.readthedocs.io/en/latest/notebooks/insulation_and_boundaries.html):  how to extract insulation profiles and call boundaries using insulation profile minima.
- [Pileups and Average Patterns](https://cooltools.readthedocs.io/en/latest/notebooks/pileup_CTCF.html): how to create avearge maps around genomic features like CTCF.

For users interested in running analyses from the commmand line:
- [Command line interface](https://cooltools.readthedocs.io/en/latest/notebooks/command_line_interface.html): how to use the cooltools CLI.

Note that these notebooks currently focus on mammalian interphase Hi-C analysis, but are readily extendible to other organisms and cellular contexts. To clone notebooks for interactive analysis, visit https://github.com/open2c/open2c_examples. Docs for cooltools are built directly from these notebooks.

## Contributing
Cooltools welcomes contributions. The guiding principles for tools are that they are (i) as simple as possible, (ii) as interpretable as possible, (iii) should not involve visualization. The following applies for contributing new functionality to cooltools.

New functionality should:
- clearly define the problem 
- discuss alternative solutions
- provide a separate example (provided as a gist/notebook/etc) explaining its use cases on multiple datasets.
- be compatible with the latest versions of cooler and cooltools (e.g. should be able to be run on any cooler generated by the latest version of cooler)

New functionality should either:
- generalize or extend existing tool without impairing user experience, and be submitted as PR to the relevant tool
- or extract a distinct feature of genome organization, and be submitted as pull request to the sandbox

Vignettes, using existing tools in new ways, should be submitted as pull requests to open2c_vignettes as a distinct jupyter notebook, rather than to cooltools sandbox. The bar for contributions to this repository is minimal. We recommend each vignette to include package version information, and raise an error for other versions. If it makes sense, the example data available for download using cooltools can be used to allow an easy way to try out the analysis. Otherwise, the source of data can be specified for others to obtain it.

Practical aspects for contributing can be found in the guide [here](https://github.com/open2c/cooltools/blob/master/CONTRIBUTING.md).

## Citing `cooltools`

Open2C*, Nezar Abdennur*, Sameer Abraham, Geoffrey Fudenberg*, Ilya M. Flyamer*, Aleksandra A. Galitsyna*, Anton Goloborodko*, Maxim Imakaev, Betul A. Oksuz, and Sergey V. Venev*. “Cooltools: Enabling High-Resolution Hi-C Analysis in Python.” bioRxiv, November 1, 2022. https://doi.org/10.1101/2022.10.31.514564.
