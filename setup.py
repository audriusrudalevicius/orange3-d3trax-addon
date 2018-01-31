#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "Orange3 d3trax Add-on"

VERSION = "0.0.3"

DESCRIPTION = "Add-on containing new algos and data processors"
LONG_DESCRIPTION = open(path.join(path.dirname(__file__), 'README.md')).read()

LICENSE = "BSD"

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
)

PACKAGES = find_packages()

PACKAGE_DATA = {
    'orangecontrib.d3trax': ['tutorials/*.ows'],
    'orangecontrib.d3trax.widgets': ['icons/*'],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    'Orange3', 'catboost>=0.5.2.1', 'numpy', 'pandas', 'dedupe', 'AnyQt', 'PyQt5'
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange3.addon': (
        'd3trax = orangecontrib.d3trax',
    ),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/d3trax/widgets/__init__.py
        'd3trax = orangecontrib.d3trax.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.d3trax.widgets:WIDGET_HELP_PATH',)
}

NAMESPACE_PACKAGES = ["orangecontrib"]

def _discover_tests():
    import unittest
    return unittest.defaultTestLoader.discover('orangecontrib.d3trax',
                                               pattern='test_*.py',
                                               top_level_dir='.')

TEST_SUITE = "setup._discover_tests"


AUTHOR = 'Audrius Rudalevicius'


if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=NAMESPACE_PACKAGES,
        test_suite=TEST_SUITE,
        author=AUTHOR,
        include_package_data=True,
        zip_safe=False,
    )
