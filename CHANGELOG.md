# Release notes

## [Upcoming release](https://github.com/open2c/cooltools/compare/v0.7.0...HEAD)

## [v0.7.0](https://github.com/open2c/cooltools/compare/v0.6.1...v0.7.0)

### New features
* Add pool decorator to functions for supporting multiprocess
* `expected_cis` now accepts unbalanced cool file too

### API changes
* `expected_cis`
  * output cvd table now also includes "dist_bp", "contact_frequency", and "n_valid" columns
  * now returns "count.avg.smoothed" and "count.avg.smoothed.agg", when `clr_weight_name=None, smooth=True, aggregate_smoothed=True`

### Maintenance
* Replaced np.int with int in adaptive_coarsegrain
* OE update in sandbox
* Cross score sandbox fixes
* Support for pandas 2

## [v0.6.1](https://github.com/open2c/cooltools/compare/v0.6.0...v0.6.1)

### Maintenance
* Bug fix in CLI pileup

## [v0.6.0](https://github.com/open2c/cooltools/compare/v0.5.4...v0.6.0)

### New features
* New function/tool `rearrange_cooler` to reorder/subset/flip regions of the genome in a cooler
* New test dataset for micro-C from hESCs

### API changes
* snipping: reorder the axes of the output snipper array to (snippet_idx, i, j).

### Maintenance

* snipping: fix spurious nan->0 conversion of bad bins in on-diagonal pileups
* snipping: fix snipping without provided view
* snipping: fix for storing the stack in a file
* virtual4c: fix for the case when viewpoint has no contacts
* fix: Fix numba deprecation warnings by adding `nopython=True`
* Other small bugfixes

## [v0.5.4](https://github.com/open2c/cooltools/compare/v0.5.3...v0.5.4)

### Maintenance

* Updated import statements and requirements to use cooler 0.9.

## [v0.5.3](https://github.com/open2c/cooltools/compare/v0.5.2...v0.5.3)

### Maintenance

* Improvements for `read_expected_from_file`
* Bug fix for dot caller 0/0 occurrences
* Remove `cytoolz` dependency
* Pin cooler <0.9 until compatibility

## [v0.5.2](https://github.com/open2c/cooltools/compare/v0.5.1...v0.5.2)

### API changes
* remove custom bad_bins from expected & eigdecomp [#336](https://github.com/open2c/cooltools/pull/336)
* coverage can store total cis counts in the cooler, and sampling can use cis counts [#332](https://github.com/open2c/cooltools/pull/322)
* can now calculate coverge for balanced data [#385](https://github.com/open2c/cooltools/pull/385)
* new drop_track_na argument for align_track_with_cooler, allows calcultions that that missing data in tracks as absent [#360](https://github.com/open2c/cooltools/pull/360)
* multi-thread insulation by chromosome (TODO: by chunk)
* Virtual 4C tool [#378](https://github.com/open2c/cooltools/pull/378)

### CLI changes
* CLI tool for `coverage()`

### Documentation
* snipping documentation
* dots tutorial
* CLI tutorial

### Maintenance
* Dropped support for Python 3.7 (due to Pandas compatability issues)
* Added support for Python 3.10

* Minor bugfixes and compatibility updates
  * Pandas compatibility, pinned above 1.5.1
  * bioframe compatability
  * scikit-learn, pinned above >=1.1.2
  * saddle binedges, value limits [#361](https://github.com/open2c/cooltools/pull/361)
  * pileup CLI bugfix for reading features


### Other 
* [Code of conduct](https://github.com/open2c/cooltools/blob/master/CODE_OF_CONDUCT.md)

## [v0.5.1](https://github.com/open2c/cooltools/compare/v0.5.0...v0.5.1)
### API changes
* cooltools.dots is the new user-facing function for calling dots

### Maintenance
* Compatibility with pandas 1.4
* Strict dictinary typing for new numba versions
* Update to bioframe 0.3.3

## [v0.5.0](https://github.com/open2c/cooltools/compare/v0.4.0...v0.5.0)

**NOTE: THIS RELEASE BREAKS BACKWARDS COMPATIBILITY!**

This release addresses two major issues:
* Integration with bioframe [viewframes](https://bioframe.readthedocs.io/en/latest/guide-intervalops.html#genomic-views) defined as of bioframe v0.3.
* Synchronization of the CLI and Python API

Additionally, [the documentation](https://cooltools.readthedocs.io/en/latest/) has been greatly improved and now includes detailed tutorials that show how to use the `cooltools` API in conjunction with other Open2C libraries. These tutorials are automatically re-built from notebooks copied from https://github.com/open2c/open2c_examples repository.

### API changes
* More clear separation of top-level user-facing functions and low-level API.
  * Most standard analyses can be performed using just the user-facing functions which are imported into the top-level namespace. Some of them are new or heavily modified from earlier versions.
    * `cooltools.expected_cis` and `cooltools.expected_trans` for average by-diagonal contact frequency in intra-chromosomal data and in inter-chromosomal data, respectively
    * `cooltools.eigs_cis` and `cooltools.eigs_trans` for eigenvectors (compartment profiles) of cis and trans data, repectively
    * `cooltools.digitize` and `cooltools.saddle` can be used together for creation of 2D summary tables of Hi-C interactions in relation to a digitized genomic track, such as eigenvectors
    * `cooltools.insulation` for insulation score and annotation of insulating boundaries
    * `cooltools.directionality` for directionality index
    * `cooltools.pileup` for average signal at 1D or 2D genomic features, including APA
    * `cooltools.coverage` for calculation of per-bin sequencing depth
    * `cooltools.sample` for random downsampling of cooler files

    * For non-standard analyses that require custom algorithms, a lower level API is available under `cooltools.api`

* Most functions now take an optional `view_df` argument. A pandas dataframe defining a genomic view (https://bioframe.readthedocs.io/en/latest/guide-technical-notes.html) can be provided to limit the analyses to regions included in the view. If not provided, the analysis is performed on whole chromosomes according to what’s stored in the cooler.
* All functions apart from `coverage` now take a `clr_weight_name` argument to specify how the desired balancing weight column is named. Providing a `None` value allows one to use unbalanced data (except the `eigs_cis`, `eigs_trans` methods, since eigendecomposition is only defined for balanced Hi-C data).
* The output of `expected-cis` function has changed: it now contains `region1` and `region2` columns (with identical values in case of within-region expected). Additionally, it now allows smoothing of the result to avoid noisy values at long distances (enabled by default and result saved in additional columns of the dataframe)
* The new `cooltools.insulation` method includes a thresholding step to detect strong boundaries, using either the Li or the Otsu method (from `skimage.thresholding`), or a fixed float value. The result of thresholding for each window size is stored as a boolean in a new column `is_boundary_{window}`.
* New subpackage `sandbox` for experimental codes that are either candidates for merging into cooltools or candidates for removal. No documentation and tests are expected, proceed at your own risk.
* New subpackage `lib` for auxiliary modules

### CLI changes
* CLI tools are renamed with prefixes dropped (e.g. `diamond-insulation` is now `insulation`), to align with names of user-facing API functions.
* The CLI tool for expected has been split in two for intra- and inter-chromosomal data (`expected-cis` and `expected-trans`, repectively). 
* Similarly, the compartment profile calculation is now separate for cis and trans (`eigs-cis` and `eigs-trans`).
* New CLI tool `cooltools pileup` for creation of average features based on Hi-C data. It takes a .bed- or .bedpe-style file to create average on-diagonal or off-diagonal pileups, respectively.

### Maintenance
Support for Python 3.6 dropped


## [v0.4.0](https://github.com/open2c/cooltools/compare/v0.3.2...v0.4.0)

Date: 2021-04-06

Maintenance
* Make saddle strength work with NaNs
* Add output option to diamond-insulation
* Upgrade bioframe dependency
* Parallelize random sampling
* Various compatibility fixes to expected, saddle and snipping and elsewhere to work with standard formats for "expected" and "regions": https://github.com/open2c/cooltools/issues/217

New features
* New dataset download API
* New functionality for smoothing P(s) and derivatives (API is not yet stable): `logbin_expected`, `interpolate_expected`

## [v0.3.2](https://github.com/open2c/cooltools/compare/v0.3.0...v0.3.2)

Date: 2020-05-05

Updates and bug fixes
* Error checking for vmin/vmax in compute-saddle
* Various updates and fixes to expected and dot-caller code

Project health
* Added docs on RTD, tutorial notebooks, code formatting, linting, and contribution guidelines.


## [v0.3.0](https://github.com/open2c/cooltools/compare/v0.2.0...v0.3.0)

Date: 2019-11-04

* Several library utilities added: `plotting.gridspec_inches`, `adaptive_coarsegrain`, singleton interpolation, and colormaps.

* New tools: `cooltools sample` for random downsampling, `cooltools coverage` for marginalization.

Improvements to saddle functions:

* `compute-saddle` now saves saddledata without transformation, and the `scale` argument (with options `log` or `linear`) now only determines how the saddle is plotted. Consequently, `saddleplot` function now expects untransformed `saddledata`, and plots it directly or with log-scaling of the colormap. (https://github.com/open2c/cooltools/pull/105)
* Added `saddle.mask_bad_bins` method to filter bins in a track based on Hi-C bin-level filtering - improves saddle and histograms when using ChIP-seq and similar tracks. It is automatically applied in the CLI interface. Shouldn't affect the results when using eigenvectors calculated from the same data.
* `make_saddle` Python function and `compute-saddle` CLI now allow setting min and max distance to use for calculating saddles.

## [v0.2.0](https://github.com/open2c/cooltools/compare/v0.1.0...v0.2.0)

Date: 2019-05-02

* New tagged release for DCIC. Many updates, including more memory-efficient insulation score calling. Next release should include docs.


## [v0.1.0](https://github.com/open2c/cooltools/releases/tag/v0.1.0)

Date: 2018-05-07

* First official release
