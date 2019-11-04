# Release notes

## [Unreleased](https://github.com/mirnylab/cooltools/compare/v0.2.0...HEAD)

### Improvements to saddle functions
* Added `saddle.mask_bad_bins` method to filter bins in a track based on Hi-C bin-level filtering - improves saddle and histograms when using ChIP-seq and similar tracks. It is automatically applied in the CLI interface. Shouldn't affect the results when using eigenvectors calculated from the same data.
* `make_saddle` Python function and `compute-saddle` CLI now allow setting min and max distance to use for calculating saddles.
* `compute-saddle` now saves saddledata without transformation, and the `scale` argument (with options `log` or `linear`) now only determines how the saddle is plotted. Consequently, `saddleplot` function now expects untransformed `saddledata`, and plots it directly or with log-scaling of the colormap. (https://github.com/mirnylab/cooltools/pull/105)

## [v0.2.0](https://github.com/mirnylab/cooltools/compare/v0.1.0...v0.2.0)

* Date: 2019-05-02

New tagged release for DCIC. Many updates, including more memory-efficient insulation score calling. Next release should include docs.


## [v0.1.0](https://github.com/mirnylab/cooltools//releases/tag/v0.1.0)

* Date: 2018-05-07

First official release
