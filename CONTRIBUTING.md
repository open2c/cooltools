# Contributing

## General guidelines

If you haven't contributed to open-source before, we recommend you read [this excellent guide by GitHub on how to contribute to open source](https://opensource.guide/how-to-contribute). The guide is long, so you can gloss over things you're familiar with.

If you're not already familiar with it, we follow the [fork and pull model](https://help.github.com/articles/about-collaborative-development-models) on GitHub. Also, check out this recommended [git workflow](https://www.asmeurer.com/git-workflow/).

As a rough guide for cooltools:
- contributors should preferably work on their forks and submit pull requests to the main branch
- core maintainers can work on feature branches in the main fork and then submit pull requests to the main branch
- core maintainers can push directly to the main branch if it's urgently needed 


## Contributing Code

This project has a number of requirements for all code contributed.

* We follow the [PEP-8 style](https://www.python.org/dev/peps/pep-0008/) convention.
* We use [flake8](http://flake8.pycqa.org/en/latest/) to automatically lint the code and maintain code style. You can use a code formatter like [black](https://github.com/psf/black) or [autopep8](https://github.com/hhatto/autopep8) to help keep the linter happy.
* We use [Numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
* User-facing API changes or new features should have documentation added.

Ideally, provide full test coverage for new code submitted in PRs.


## Setting up Your Development Environment

For setting up an isolated virtual environment for development, we recommend using [conda](https://docs.conda.io/en/latest/miniconda.html). After forking and cloning the repository, install in "editable" (i.e. development) mode using the `-e` option:

```sh
$ git clone https://github.com/open2c/cooltools.git
$ cd cooltools
$ pip install -e .
```

Editable mode installs the package by creating a "link" to your working (repo) directory.


## Unit Tests

It is best if all new functionality and/or bug fixes have unit tests added with each use-case.

We use [pytest](https://docs.pytest.org/en/latest) as our unit testing framework with the `pytest-cov` extension to check code coverage and `pytest-flake8` to check code style. You don't need to configure these extensions yourself.
This automatically checks code style and functionality, and prints code coverage, even though it doesn't fail on low coverage. 

Once you've configured your environment, you can just `cd` to the root of your repository and run

```sh
$ pytest
```

Unit tests are automatically run on Travis CI for pull requests.


## Coverage

The `pytest` script automatically reports coverage, both on the terminal for missing line numbers, and in annotated HTML form in `htmlcov/index.html`.


## Documentation

If a feature is stable and relatively finalized, it is time to add it to the documentation. If you are adding any private/public functions, it is best to add docstrings, to aid in reviewing code and also for the API reference.

We use [Numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html>) and [Sphinx](http://www.sphinx-doc.org/en/stable) to document this library. Sphinx, in turn, uses [reStructuredText](http://www.sphinx-doc.org/en/stable/rest.html) as its markup language for adding code.

We use the [Sphinx Autosummary extension](http://www.sphinx-doc.org/en/stable/ext/autosummary.html) to generate API references. You may want to look at `docs/api.rst` to see how these files look and where to add new functions, classes or modules.

We also use the [nbsphinx extension](https://nbsphinx.readthedocs.io/en/0.5.0/) to render tutorial pages from Jupyter notebooks.

To build the documentation:

```sh
$ make docs
```

After this, you can find an HTML version of the documentation in `docs/_build/html/index.html`.

Documentation from `master` and tagged releases is automatically built and hosted thanks to [readthedocs](https://readthedocs.org/).


## Acknowledgement

If you've contributed significantly and would like your authorship to be included in subsequent uploads to [Zenodo](https://zenodo.org), please make a separate PR to add your name and affiliation to the `.zenodo.json` file.

---

This document was modified from the [guidelines from the sparse project](https://github.com/pydata/sparse/blob/master/docs/contributing.rst).
