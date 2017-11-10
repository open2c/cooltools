#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import re

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

classifiers = """\
    Development Status :: 4 - Beta
    Operating System :: OS Independent
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
"""


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text

def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read('cooltools', '__init__.py'),
        re.MULTILINE).group(1)
    return version


def get_long_description():
    return _read('README.md')


install_requires = [
    'numpy',
    'cython',
    'click',
    'cooler>=0.6',
]


extensions = [
    Extension(
        "cooltools.io.fastsavetxt", ["cooltools/io/fastsavetxt.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "cooltools.num._numutils_cy", ["cooltools/num/_numutils_cy.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        "cooltools.num.kernels", ["cooltools/num/kernels.pyx"],
        include_dirs=[np.get_include()]
    )
]

packages = find_packages()
setup(
    name='cooltools',
    author='Mirny Lab',
    author_email='espresso@mit.edu',
    version=get_version(),
    license='BSD3',
    description='Analysis tools for genomic interaction data stored in .cool format',
    long_description=get_long_description(),
    keywords=['genomics', 'bioinformatics', 'Hi-C', 'contact', 'matrix', 'format', 'hdf5'],
    url='https://github.com/mirnylab/cooltools',
    packages=find_packages(),
    ext_modules = cythonize(extensions),
    zip_safe=False,
    classifiers=[s.strip() for s in classifiers.split('\n') if s],
    include_dirs=[np.get_include()],

    install_requires=install_requires,
    entry_points={
        'console_scripts': [
             'cooltools = cooltools.cli:cli',
        ]
    }

)
