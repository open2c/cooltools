# Release notes

## [Upcoming release](https://github.com/open2c/cooltools/compare/v0.4.0...HEAD)

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
