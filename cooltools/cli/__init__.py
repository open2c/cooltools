# -*- coding: utf-8 -*-
from __future__ import division, print_function
import click
import sys
from .. import __version__


# Monkey patch
click.core._verify_python3_env = lambda: None


CONTEXT_SETTINGS = {
    'help_option_names': ['-h', '--help'],
}


@click.version_option(version=__version__)
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    '--debug/--no-debug',
    help="Verbose logging",
    default=False)
@click.option(
    '-pm', '--post-mortem',
    help="Post mortem debugging",
    is_flag=True,
    default=False)
def cli(debug, post_mortem):
    """
    Type -h or --help after any subcommand for more information.

    """
    if debug:
    	pass
        #logger.setLevel(logging.DEBUG)

    if post_mortem:
        import traceback
        try:
            import ipdb as pdb
        except ImportError:
            import pdb
        def _excepthook(exc_type, value, tb):
            traceback.print_exception(exc_type, value, tb)
            print()
            pdb.pm()
        sys.excepthook = _excepthook


from . import (
    dump_cworld,
    diamond_insulation,
    compute_expected,
    compute_saddle,
    call_dots,
    call_compartments,
    genome,
    random_sample
)
