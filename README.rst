=============================================================
ReynirCorrect: A spelling and grammar corrector for Icelandic
=============================================================

.. image:: https://travis-ci.com/mideind/ReynirCorrect.svg?branch=master
    :target: https://travis-ci.com/mideind/ReynirCorrect

********
Overview
********

**ReynirCorrect** is a Python 3.x package for
**checking and correcting spelling and grammar** in Icelandic text.

ReynirCorrect uses the `Greynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

Token-level correction
----------------------

ReynirCorrect can tokenize text and return a corrected token list.
This catches token-level errors, such as spelling errors and erroneous
phrases, but not grammatical errors.

Full grammar analysis
---------------------

ReynirCorrect can also analyze text grammatically by attempting to parse
it, after token-level correction. The parsing is done according to Reynir's
context-free grammar for Icelandic, augmented with additional production
rules for common grammatical errors. The analysis returns a set of annotations
(errors and suggestions) that apply to spans (consecutive tokens) within
sentences in the resulting token list.

******
Status
******

This code is under development and has early Beta status. It will eventually
become the foundation of a spelling and grammar checker that will be open
to the public on the `Greynir.is <https://greynir.is>`_ website.
Of course it will also be available as an open-source Python package
that can be installed using ``pip``.

***************************
Deep vs. shallow correction
***************************

ReynirCorrect can do both deep and shallow correction.

*Deep* correction shows both the proposed corrections (if they are available), 
and error messages to guide the user.

*Shallow* correction corrects the input file automatically but doesn't give any error messages 
or information about the errors. Only clear errors are corrected at this stage. 
No grammar errors are corrected.

*******
Example
*******

To tokenize text with deep token-level correction (the text is not parsed
in this case, so no grammar checking is done):

.. code-block:: python

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

.. code-block:: python

   >>> from reynir_correct import check_single
   >>> sent = check_single("Páli, vini mínum, langaði að horfa á sjónvarpið.")
   >>> for annotation in sent.annotations:
   >>>     print("{0}".format(annotation))

Output::

   000-004: E003  Frumlag sagnarinnar 'að langa' á að vera í þolfalli en ekki í þágufalli

                  [ The subject of the verb 'að langa/to want' should be in the
                     accusative case, not the dative case ]

Note that the ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the indices of the first
and last tokens to which the annotation applies. ``E003`` is an error code.

*************
Prerequisites
*************

This package runs on CPython 3.5 or newer, and on PyPy 3.5 or newer.

************
Installation
************

To install this package:

.. code-block:: console

   $ pip3 install reynir-correct   # or pip install reynir-correct if Python3 is your default

If you want to be able to edit the source, do like so
(assuming you have **git** installed):

.. code-block:: console

   $ git clone https://github.com/mideind/ReynirCorrect
   $ cd ReynirCorrect
   $ # [ Activate your virtualenv here if you have one ]
   $ python setup.py develop

The package source code is now in ``ReynirCorrect/src/reynir_correct``.


*********************
The command line tool
*********************

After installation, the corrector can be invoked directly from the command line:

.. code-block:: console

   $ correct input.txt output.txt

Input and output files are encoded in UTF-8. If the files are not
given explicitly, ``stdin`` and ``stdout`` are used for input and output,
respectively.

Empty lines in the input are treated as sentence boundaries.

By default, the output consists of one sentence per line, where each
line ends with a single newline character (ASCII LF, ``chr(10)``, ``"\n"``).
Within each line, tokens are separated by spaces.

The following (mutually exclusive) options can be specified
on the command line:

+-------------------+---------------------------------------------------+
| | ``--csv``       | Deep tokenization. Output token objects in CSV    |
|                   | format, one per line. Sentences are separated by  |
|                   | lines containing ``0,"",""``                      |
+-------------------+---------------------------------------------------+
| | ``--json``      | Deep tokenization. Output token objects in JSON   |
|                   | format, one per line.                             |
+-------------------+---------------------------------------------------+
| | ``--normalize`` | Normalize punctuation, causing e.g. quotes to be  |
|                   | output in Icelandic form and hyphens to be        |
|                   | regularized. This option is only applicable to    |
|                   | shallow tokenization.                             |
+-------------------+---------------------------------------------------+

Type ``correct -h`` to get a short help message.

*******
Example
*******

.. code-block:: console

   $ echo "Atvinuleysi jókst um 3%" | correct
   Atvinnuleysi jókst um 3%

   $ echo "Barnið vil grænann lit" | correct --csv
   6,"Barnið",""
   6,"vil",""
   6,"grænan",""
   6,"lit",""
   0,"",""

   $ echo "Pakkin er fyrir hestin" | correct --json
   {"k":"BEGIN SENT"}
   {"k":"WORD","t":"Pakkinn"}
   {"k":"WORD","t":"er"}
   {"k":"WORD","t":"fyrir"}
   {"k":"WORD","t":"hestinn"}
   {"k":"END SENT"}


*****
Tests
*****

To run the built-in tests, install `pytest <https://docs.pytest.org/en/latest/>`_,
``cd`` to your ``ReynirCorrect`` subdirectory (and optionally activate your
virtualenv), then run:

.. code-block:: console

   $ python -m pytest

