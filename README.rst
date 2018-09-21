=============================================================
ReynirCorrect: A spelling and grammar corrector for Icelandic
=============================================================

********
Overview
********

**ReynirCorrector** is a Python 3.x package for
**checking and correcting spelling and grammar** in Icelandic text.

ReynirCorrector uses the `Reynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

*******
Example
*******

>>> from reynir_correct import Corrector

*************
Prerequisites
*************

This package runs on CPython 3.4 or newer, and on PyPy 3.5 or newer.

************
Installation
************

To install this package::

    $ pip3 install reynir-correct   # or pip install reynir-correct if Python3 is your default

If you want to be able to edit the source, do like so (assuming you have **git** installed)::

    $ git clone https://github.com/vthorsteinsson/ReynirCorrect
    $ cd ReynirCorrect
    $ # [ Activate your virtualenv here if you have one ]
    $ python setup.py develop

The package source code is now in ``ReynirCorrect/src/reynir_correct``.

*****
Tests
*****

To run the built-in tests, install `pytest <https://docs.pytest.org/en/latest/>`_, ``cd`` to your
``ReynirCorrect`` subdirectory (and optionally activate your virtualenv), then run::

    $ python -m pytest

