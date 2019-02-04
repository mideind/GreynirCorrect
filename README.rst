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

**ReynirCorrect** is a Python 3.x package for
**checking and correcting spelling and grammar** in Icelandic text.

ReynirCorrect uses the `Reynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

Token-level correction
----------------------

ReynirCorrect can tokenize text and return a corrected token list.
This catches token-level errors, such as spelling errors and erroneous
phrases, but not grammatical errors.

Full grammar analysis
---------------------

ReynirCorrect can also analyze text grammatically by attempting to parse
it, after token-level correction, using Reynir's context-free grammar for
Icelandic. The analysis returns a set of annotations (errors and suggestions)
that apply to spans (consecutive tokens) within sentences in the resulting
token list.

******
Status
******

**This code is under active development and has Alpha status.**

*******
Example
*******

To tokenize text with token-level correction (the text is not parsed in this case,
so no grammar checking is done):

>>> from reynir_correct import tokenize
>>> g = tokenize("Af gefnu tilefni fékk fékk daninn vilja sýnum "
>>>     "framgengt við hana í auknu mæli.")
>>> for tok in g:
>>>     print("{0:10} {1}".format(tok.txt or "", tok.error_description))

Output::

	Að         Orðasambandið 'Af gefnu tilefni' var leiðrétt í 'að gefnu tilefni'
	gefnu
	tilefni
	fékk       Endurtekið orð ('fékk') var fellt burt
	Daninn     Orð á að byrja á hástaf: 'daninn'
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

>>> from reynir_correct import check_single
>>> sent = check_single("Páli, vini mínum, langaði að horfa á sjónvarpið.")
>>> for annotation in sent.annotations:
>>>     print("{0}".format(annotation))

Output::

	000-004: E003   Frumlag sagnarinnar 'að langa' á að vera í þolfalli en ekki í þágufalli

	                [ The subject of the verb 'að langa/to want' should be in the
	                  accusative case, not the dative case ]

Note that the ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the indices of the first and last
tokens to which the annotation applies. ``E003`` is an error code.

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

