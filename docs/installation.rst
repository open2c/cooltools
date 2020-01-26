Installation
============

Requirements
------------

- Python 3.x
- Scientific Python packages

Install using pip
-----------------

Compile and install `cooltools` and its Python dependencies from
PyPI using pip:

.. code-block:: bash

    $ pip install cooltools

or install the latest version directly from github:

.. code-block:: bash

    $ pip install git+https://github.com/mirnylab/cooltools.git


Install the development version
-------------------------------

Finally, you can install the latest development version of `cooltools` from
github. First, make a local clone of the github repository:

.. code-block:: bash

    $ git clone https://github.com/mirnylab/cooltools 

Then, you can compile and install `cooltools` in 
`the development mode <https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode>`_, 
which installs the package without moving it to a system folder and thus allows
immediate live-testing any changes in the python code.

.. code-block:: bash

    $ cd cooltools
    $ pip install -e ./
