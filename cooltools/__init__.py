# -*- coding: utf-8 -*-
"""
Cool tools
~~~~~~~~~~

The tools for your .cool's.

:author: Cooltools developers
:license: MIT

"""
import logging

__version__ = "0.4.0rc1"

from . import io
from . import lib
from .lib import numutils, download_data, print_available_datasets, get_data_dir, download_file, get_md5sum
