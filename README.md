Various ML tools
---

[![Build Status](https://travis-ci.org/audriusrudalevicius/orange3-d3trax-addon.svg?branch=dev)](https://travis-ci.org/audriusrudalevicius/orange3-d3trax-addon)


Installation
------------

To install the add-on, run

    pip install .

To register this add-on with Orange, but keep the code in the development directory (do not copy it to 
Python's site-packages directory), run

    pip install -e .

Documentation / widget help can be built by running

    make html htmlhelp

from the doc directory.

Usage
-----

After the installation, the widget from this add-on is registered with Orange. To run Orange from the terminal,
use

    python -m Orange.canvas

