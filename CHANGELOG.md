# Release notes

## [v0.3.0](https://github.com/mirnylab/cooltools/compare/v0.2.0...HEAD)

Date: 2019-11-04

* Several library utilities added: `plotting.gridspec_inches`, `adaptive_coarsegrain`, singleton interpolation, and colormaps.

* New tools: `cooltools sample` for random downsampling, `cooltools coverage` for marginalization.

* `compute-saddle` now saves saddledata without transformation, and the `scale` argument (with options `log` or `linear`) now only determines how the saddle is plotted. Consequently, `saddleplot` function now expects untransformed `saddledata`, and plots it directly or with log-scaling of the colormap. (https://github.com/mirnylab/cooltools/pull/105)


## [v0.2.0](https://github.com/mirnylab/cooltools/compare/v0.1.0...v0.2.0)

Date: 2019-05-02

* New tagged release for DCIC. Many updates, including more memory-efficient insulation score calling. Next release should include docs.


## [v0.1.0](https://github.com/mirnylab/cooltools//releases/tag/v0.1.0)

Date: 2018-05-07

* First official release
