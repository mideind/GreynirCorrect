==============================================================
GreynirCorrect: A spelling and grammar corrector for Icelandic
==============================================================

.. image:: https://travis-ci.com/mideind/GreynirCorrect.svg?branch=master
    :target: https://travis-ci.com/mideind/GreynirCorrect

.. _overview:

********
Overview
********

**GreynirCorrect** is a Python 3 (>= 3.5) package and command line tool for
**checking and correcting spelling and grammar** in Icelandic text.

GreynirCorrect relies on the `Greynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

GreynirCorrect is documented in detail `here <https://yfirlestur.is/doc/>`__.

The software has three main modes of operation, described below.

Token-level correction
----------------------

GreynirCorrect can tokenize text and return an automatically corrected token stream.
This catches token-level errors, such as spelling errors and erroneous
phrases, but not grammatical errors. Token-level correction is relatively fast.

Full grammar analysis
---------------------

GreynirCorrect can analyze text grammatically by attempting to parse
it, after token-level correction. The parsing is done according to Greynir's
context-free grammar for Icelandic, augmented with additional production
rules for common grammatical errors. The analysis returns a set of annotations
(errors and suggestions) that apply to spans (consecutive tokens) within
sentences in the resulting token list. Full grammar analysis is considerably
slower than token-level correction.

Command-line tool
-----------------

GreynirCorrect can be invoked as a command-line tool
to perform token-level correction. The command is ``correct infile.txt outfile.txt``.
The command-line tool is further documented below.

.. _examples:

********
Examples
********

To perform token-level correction from Python code:

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

To perform full spelling and grammar analysis of a sentence from Python code:

.. code-block:: python

   from reynir_correct import check_single
   sent = check_single("Páli, vini mínum, langaði að horfa á sjónnvarpið.")
   for annotation in sent.annotations:
       print("{0}".format(annotation))

Output::

   000-004: P_WRONG_CASE_þgf_þf Á líklega að vera 'Pál, vin minn' / [Pál , vin minn]
   009-009: S004   Orðið 'sjónnvarpið' var leiðrétt í 'sjónvarpið'

.. code-block:: python

   sent.tidy_text

Output::

   'Páli, vini mínum, langaði að horfa á sjónvarpið.'

Note that the ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the indices of the first
and last tokens to which the annotation applies.
``P_WRONG_CASE_þgf_þf`` and ``S004`` are error codes.

.. _prerequisites:

*************
Prerequisites
*************

GreynirCorrect runs on CPython 3.5 or newer, and on PyPy 3.5 or newer. It has
been tested on Linux, MacOS and Windows. The
`PyPi package <https://pypi.org/project/reynir-correct/>`_
includes binary wheels for common environments, but if the setup on your OS
requires compilation from sources, you may need

.. code-block:: bash

   $ sudo apt-get install python3-dev

...or something to similar effect to enable this.

.. _installation:

************
Installation
************

To install this package (assuming you have Python 3 with ``pip`` installed):

.. code-block:: bash

   $ pip install reynir-correct

If you want to be able to edit the source, do like so
(assuming you have ``git`` installed):

.. code-block:: bash

   $ git clone https://github.com/mideind/GreynirCorrect
   $ cd GreynirCorrect
   $ # [ Activate your virtualenv here if you have one ]
   $ pip install -e .

The package source code is now in ``GreynirCorrect/src/reynir_correct``.

.. _commandline:

*********************
The command line tool
*********************

After installation, the corrector can be invoked directly from the command line:

.. code-block:: bash

   $ correct input.txt output.txt

...or:

.. code-block:: bash

   $ echo "Þinngið samþikkti tilöguna" | correct
   Þingið samþykkti tillöguna

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
| | ``--csv``       | Output token objects in CSV                       |
|                   | format, one per line. Sentences are separated by  |
|                   | lines containing ``0,"",""``                      |
+-------------------+---------------------------------------------------+
| | ``--json``      | Output token objects in JSON format, one per line.|
+-------------------+---------------------------------------------------+
| | ``--normalize`` | Normalize punctuation, causing e.g. quotes to be  |
|                   | output in Icelandic form and hyphens to be        |
|                   | regularized.                                      |
+-------------------+---------------------------------------------------+

The CSV and JSON formats are identical to those documented for the
`Tokenizer package <https://github.com/mideind/Tokenizer>`_.

Type ``correct -h`` to get a short help message.


Command Line Examples
---------------------

.. code-block:: bash

   $ echo "Atvinuleysi jógst um 3%" | correct
   Atvinnuleysi jókst um 3%

.. code-block:: bash

   $ echo "Barnið vil grænann lit" | correct --csv
   6,"Barnið",""
   6,"vil",""
   6,"grænan",""
   6,"lit",""
   0,"",""

Note how *vil* is not corrected, as it is a valid and common word, and
the ``correct`` command does not perform grammar checking.

.. code-block:: bash

   $ echo "Pakkin er fyrir hestin" | correct --json
   {"k":"BEGIN SENT"}
   {"k":"WORD","t":"Pakkinn"}
   {"k":"WORD","t":"er"}
   {"k":"WORD","t":"fyrir"}
   {"k":"WORD","t":"hestinn"}
   {"k":"END SENT"}

.. _tests:

*****
Tests
*****

To run the built-in tests, install `pytest <https://docs.pytest.org/en/latest/>`_,
``cd`` to your ``GreynirCorrect`` subdirectory (and optionally activate your
virtualenv), then run:

.. code-block:: bash

   $ python -m pytest

.. _license:

*********************
Copyright and License
*********************

.. image:: https://github.com/mideind/ReynirPackage/raw/master/doc/_static/MideindLogoVert100.png?raw=true
   :target: https://mideind.is
   :align: right
   :alt: Miðeind ehf.

**Copyright © 2020 Miðeind ehf.**

GreynirCorrect's original author is *Vilhjálmur Þorsteinsson*.

Parts of this software are developed under the auspices of the
Icelandic Government's 5-year Language Technology Programme for Icelandic,
which is described
`here <https://www.stjornarradid.is/lisalib/getfile.aspx?itemid=56f6368e-54f0-11e7-941a-005056bc530c>`__
(English version `here <https://clarin.is/media/uploads/mlt-en.pdf>`__).

This software is licensed under the *MIT License*:

   *Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without restriction,
   including without limitation the rights to use, copy, modify, merge,
   publish, distribute, sublicense, and/or sell copies of the Software,
   and to permit persons to whom the Software is furnished to do so,
   subject to the following conditions:*

   *The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.*

   *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
   TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*

