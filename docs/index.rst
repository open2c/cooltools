.. cooltools documentation master file, created by
   sphinx-quickstart on Wed Jun 12 16:42:43 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :caption: Overview
   :hidden:
   :maxdepth: 2

   self

Getting started
***************

The tools for your *.cool*\ s

Chromosome conformation capture technologies reveal the incredible complexity of genome folding. A growing number of labs and multiple consortia, including the 4D Nucleome, the International Nucleome Consortium, and ENCODE, are generating higher-resolution datasets to probe genome architecture across cell states, types, and organisms. Larger datasets increase the challenges at each step of computational analysis, from storage, to memory, to researchers’ time. The recently-introduced `cooler <https://github.com/open2c/cooler/tree/master/cooler>`_ format readily handles storage of high-resolution datasets via a sparse data model.

**cooltools** leverages this format to enable flexible and reproducible analysis of high-resolution data. **cooltools** provides a suite of computational tools with a paired python API and command line access, which facilitates workflows either on high-performance computing clusters or via custom analysis notebooks. As part of the `Open2C ecosystem <https://open2c.github.io/>`_, **cooltools** also provides detailed introductions to key concepts in Hi-C-data analysis with interactive notebook documentation.

If you use **cooltools** in your work, please cite **cooltools** via its zenodo `DOI 10.5281/zenodo.5214125 <https://doi.org/10.5281/zenodo.5214125>`_


Installation
============

Requirements
------------

- Python 3.7+
- Scientific Python packages

Install using pip
-----------------

Compile and install `cooltools` and its Python dependencies from
PyPI using pip:

.. code-block:: bash

    $ pip install cooltools

or install the latest version directly from github:

.. code-block:: bash

    $ pip install https://github.com/open2c/cooltools/archive/refs/heads/master.zip


Install the development version
-------------------------------

Finally, you can install the latest development version of `cooltools` from
github. First, make a local clone of the github repository:

.. code-block:: bash

    $ git clone https://github.com/open2c/cooltools

Then, you can compile and install `cooltools` in
`development mode <https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode>`_,
which installs the package without moving it to a system folder and thus allows
immediate live-testing any changes in the python code.

.. code-block:: bash

    $ cd cooltools
    $ pip install -e ./


.. toctree::
  :maxdepth: 2
  :caption: Tutorials
  :titlesonly:

  ./notebooks/viz.ipynb
  ./notebooks/contacts_vs_distance.ipynb
  ./notebooks/compartments_and_saddles.ipynb
  ./notebooks/insulation_and_boundaries.ipynb
  ./notebooks/dots.ipynb
  ./notebooks/pileup_CTCF.ipynb

Note that these notebooks currently focus on mammalian interphase Hi-C analysis, but are readily extendible to other organisms and cellular contexts. To clone and work interactively with these notebooks, visit: https://github.com/open2c/open2c_examples.


.. toctree::
  :maxdepth: 1
  :caption: Reference

  cli
  cooltools
  releases

