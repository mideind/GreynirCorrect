=============================================================
ReynirCorrect: A spelling and grammar corrector for Icelandic
=============================================================

.. start-badges

.. image:: https://travis-ci.org/vthorsteinsson/ReynirCorrect.svg?branch=master
    :target: https://travis-ci.org/vthorsteinsson/ReynirCorrect

.. end-badges

********
Overview
********

**ReynirCorrector** is a Python 3.x package for
**checking and correcting spelling and grammar** in Icelandic text.

ReynirCorrector uses the `Reynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

******
Status
******

**This code is under active development and has not reached Alpha status.**

*******
Example
*******

To tokenize text with token-level correction (the text is not parsed,
so grammar checking is done):

>>> from reynir_correct import tokenize
>>> g = tokenize("Af gefnu tilefni fékk hann vilja sýnum "
>>>		"framgengt við hana í auknu mæli.")
>>> for tok in g:
>>>     print("{0:10} {1}".format(tok.txt or "", tok.error_description))

Output::

	Að         Orðasambandið 'Af gefnu tilefni' var leiðrétt í 'að gefnu tilefni'
	gefnu
	tilefni
	fékk
	hann
	vilja      Orðasambandið 'vilja sýnum framgengt' var leiðrétt í 'vilja sínum framgengt'
	sínum
	framgengt
	við
	hana
	í          Orðasambandið 'í auknu mæli' var leiðrétt í 'í auknum mæli'
	auknum
	mæli
	.

To get a list of spelling and grammar annotations for a sentence:

>>> from reynir_correct import check
>>> toklist, annotations = check("Mér dreymdi kysu af gefnu tilefni.")

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

