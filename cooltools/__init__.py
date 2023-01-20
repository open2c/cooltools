# -*- coding: utf-8 -*-
"""
Cool tools
~~~~~~~~~~

The tools for your .cool's.

:author: Cooltools developers
:license: MIT

"""
import logging

__version__ = "0.5.3"

from . import lib

from .lib import (
    numutils,
    download_data,
    print_available_datasets,
    get_data_dir,
    download_file,
    get_md5sum,
)

from .api.expected import expected_cis, expected_trans
from .api.coverage import coverage
from .api.eigdecomp import eigs_cis, eigs_trans
from .api.saddle import digitize, saddle
from .api.sample import sample
from .api.snipping import pileup
from .api.directionality import directionality
from .api.insulation import insulation
from .api.dotfinder import dots
from .api.virtual4c import virtual4c
