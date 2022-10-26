# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from unittest.mock import Mock
MOCK_MODULES = [
    'cooltools.io.fastsavetxt',
    'cooltools.lib._numutils',
    'cooler',
    'cooler.core',
    'cooler.tools',
    'cython',
    'dask',
    'h5py',
    'matplotlib',
    'matplotlib.cm',
    'matplotlib.pyplot',
    'numba',
    # 'numpy',
    'pandas',
    'scipy',
    'scipy.interpolate',
    'scipy.linalg',
    'scipy.sparse',
    'scipy.sparse.linalg',
    'scipy.ndimage',
    'scipy.ndimage.filters',
    'scipy.ndimage.interpolation',
    'scipy.signal',
    'scipy.stats',
    'sklearn',
    'sklearn.cluster',
    'skimage',
    'skimage.filters',
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = Mock()


# -- Project information -----------------------------------------------------

project = 'cooltools'
copyright = '2020, cooltoolers'
author = 'cooltoolers'


# -- General configuration ---------------------------------------------------

# Apparently readthedocs looks for contents.rst by default if this isn't set.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_click.ext',
    'recommonmark',
    'nbsphinx',
    'sphinx_rtd_theme'
]

# Extension configuration
napoleon_google_docstring = False
# napoleon_use_param = False
# napoleon_use_ivar = True
napoleon_use_rtype = False

# Notebook prolog and epilog
nbsphinx_prolog = """"""
nbsphinx_epilog = r"""
----
{% set docname = env.doc2path(env.docname, base='docs') %}

This page was generated with nbsphinx_ from `{{ docname }}`__

__ https://github.com/open2c/cooltools/blob/master{{ env.config.release }}/{{ docname }}

.. _nbsphinx: https://nbsphinx.readthedocs.io/

"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Style overrides ----------------------------------------------------------
# Place CSS in _static directory
# def setup(app):
#     app.add_stylesheet('theme_overrides.css')


# Pull jupyter notebooks from the open2c_examples repo
def setup(app):
    from subprocess import run

    if os.path.isdir('notebooks'):
        cmd = 'cd notebooks && git pull'
    else:
        cmd = 'git clone https://github.com/open2c/open2c_examples.git notebooks'

    print("Updating Open2C examples...")
    run(cmd, check=True, shell=True)
