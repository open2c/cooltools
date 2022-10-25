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
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
"""


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read("cooltools", "__init__.py"),
        re.MULTILINE,
    ).group(1)
    return version


def get_long_description():
    return _read("README.md")


def get_requirements(path):
    content = _read(path)
    return [
        req
        for req in content.split("\n")
        if req != "" and not (req.startswith("#") or req.startswith("-"))
    ]


setup_requires = [
    "cython",
    "numpy",
]


install_requires = get_requirements("requirements.txt")


extensions = [
    Extension(
        "cooltools.lib._numutils",
        ["cooltools/lib/_numutils.pyx"],
        include_dirs=[np.get_include()],
    ),
]


packages = find_packages()


setup(
    name="cooltools",
    author="Open2C",
    author_email="open.chromosome.collective@gmail.com",
    version=get_version(),
    license="MIT",
    description="Analysis tools for genomic interaction data stored in .cool format",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    keywords=["genomics", "bioinformatics", "Hi-C", "analysis", "cooler"],
    url="https://github.com/open2c/cooltools",
    zip_safe=False,
    classifiers=[s.strip() for s in classifiers.split("\n") if s],
    python_requires=">=3.7.1",  # same as pandas
    packages=packages,
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    setup_requires=setup_requires,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "cooltools = cooltools.cli:cli",
        ]
    },
)
