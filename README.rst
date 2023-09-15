
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/python-3.8-blue.svg
    :target: https://www.python.org/downloads/release/python-380/
.. image:: https://img.shields.io/pypi/v/reynir-correct
    :target: https://pypi.org/project/reynir-correct/
.. image:: https://shields.io/github/v/release/mideind/GreynirCorrect?display_name=tag
    :target: https://github.com/mideind/GreynirCorrect/releases
.. image:: https://github.com/mideind/GreynirCorrect/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/mideind/GreynirCorrect/actions?query=workflow%3A%22Python+package%22

==============================================================
GreynirCorrect: Spelling and grammar correction for Icelandic
==============================================================

********
Overview
********

**GreynirCorrect** is a Python 3 (>= 3.8) package and command line tool for
**checking and correcting spelling and grammar** in Icelandic text.

GreynirCorrect relies on the `Greynir <https://pypi.org/project/reynir/>`__ package,
by the same authors, to tokenize and parse text.

GreynirCorrect is documented in detail `here <https://yfirlestur.is/doc/>`__.

The software has three main modes of operation, described below.

As a fourth alternative, you can call the JSON REST API
of `Yfirlestur.is <https://yfirlestur.is>`__
to apply the GreynirCorrect spelling and grammar engine to your text,
as `documented here <https://github.com/mideind/Yfirlestur#https-api>`__.

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
to perform token-level correction and, optionally, grammar analysis.
The command is ``correct infile.txt outfile.txt``.
The command-line tool is further documented below.

********
Examples
********

To perform token-level correction from Python code:

.. code-block:: python

   >>> from reynir_correct import tokenize
   >>> g = tokenize("Af gefnu tilefni fékk fékk daninn vilja sýnum "
   >>>     "framgengt í auknu mæli.")
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

The ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the 0-based indices of the first
and last tokens to which the annotation applies.
The ``annotation.start_char`` and ``annotation.end_char`` properties
contain the indices of the first and last character to which the
annotation applies, within the original input string.

``P_WRONG_CASE_þgf_þf`` and ``S004`` are error codes.

For more detailed, low-level control, the ``check_errors()`` function
supports options and can produce various types of output:

.. code-block:: python

   from reynir_correct import check_errors
   x = "Páli, vini mínum, langaði að horfa á sjónnvarpið."
   options = { "input": x, "annotations": True, "format": "text" }
   s = check_errors(**options)
   for i in s.split("\n"):
      print(i)

Output::

   Pál, vin minn, langaði að horfa á sjónvarpið.
   000-004: P_WRONG_CASE_þgf_þf Á líklega að vera 'Pál, vin minn' | 'Páli, vini mínum,' -> 'Pál, vin minn' | None
   009-009: S004   Orðið 'sjónnvarpið' var leiðrétt í 'sjónvarpið' | 'sjónnvarpið' -> 'sjónvarpið' | None


The following options can be specified:

+-----------------------------------+--------------------------------------------------+-----------------+
| | Option                          | Description                                      | Default value   |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``input``                       | Defines the input. Can be a string or an         | ``sys.stdin``   |
|                                   | iterable of strings, such as a file object.      |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``all_errors``                  | Defines the level of correction.                 | ``True``        |
| | (alias ``grammar``)             | If False, only token-level annotation is         |                 |
|                                   | carried out. If True, sentence-level             |                 |
|                                   | annotation is carried out.                       |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``annotate_unparsed_sentences`` | If True, sentences that cannot be parsed         | ``True``        |
|                                   | are annotated in their entirety as errors.       |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``generate_suggestion_list``    | If True, annotations can in certain              | ``False``       |
|                                   | cases contain a list of possible corrections,    |                 |
|                                   | for the user to pick from.                       |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``suppress_suggestions``        | If True, more farfetched automatically           | ``False``       |
|                                   | suggested corrections are suppressed.            |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``ignore_wordlist``             | The value is a set of strings to whitelist.      | ``set()``       |
|                                   | Each string is a word that should not be         |                 |
|                                   | marked as an error or corrected. The comparison  |                 |
|                                   | is case-sensitive.                               |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``one_sent``                    | The input contains a single sentence only.       | ``False``       |
|                                   | Sentence splitting should not be attempted.      |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``ignore_rules``                | A set of error codes that should be ignored      | ``set()``       |
|                                   | in the annotation process.                       |                 |
+-----------------------------------+--------------------------------------------------+-----------------+
| | ``tov_config``                  | Path to an additional configuration file that    | ``False``       |
|                                   | may be provided for correcting custom            |                 |
|                                   | tone-of-voice issues.                            |                 |
+-----------------------------------+--------------------------------------------------+-----------------+

An overview of error codes is available `here <https://github.com/mideind/GreynirCorrect/blob/master/doc/errorcodes.rst>`__.

*************
Prerequisites
*************

GreynirCorrect runs on CPython 3.8 or newer, and on PyPy 3.8 or newer. It has
been tested on Linux, macOS and Windows. The
`PyPi package <https://pypi.org/project/reynir-correct/>`_
includes binary wheels for common environments, but if the setup on your OS
requires compilation from sources, you may need

.. code-block:: bash

   $ sudo apt-get install python3-dev

...or something to similar effect to enable this.

************
Installation
************

To install this package (assuming you have Python >= 3.8 with ``pip`` installed):

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
| | ``--grammar``   | Output whole-sentence annotations, including      |
|                   | corrections and suggestions for spelling and      |
|                   | grammar. Each sentence in the input is output as  |
|                   | a text line containing a JSON object, terminated  |
|                   | by a newline.                                     |
+-------------------+---------------------------------------------------+

The CSV and JSON formats of token objects are identical to those documented
for the `Tokenizer package <https://github.com/mideind/Tokenizer>`__.

The JSON format of whole-sentence annotations is identical to the one documented for
the `Yfirlestur.is HTTPS REST API <https://github.com/mideind/Yfirlestur#https-api>`__.

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
the ``correct`` command does not perform grammar checking by default.


.. code-block:: bash

   $ echo "Pakkin er fyrir hestin" | correct --json
   {"k":"BEGIN SENT"}
   {"k":"WORD","t":"Pakkinn"}
   {"k":"WORD","t":"er"}
   {"k":"WORD","t":"fyrir"}
   {"k":"WORD","t":"hestinn"}
   {"k":"END SENT"}

To perform whole-sentence grammar checking and annotation as well as spell checking,
use the ``--grammar`` option:


.. code-block:: bash

   $ echo "Ég kláraði verkefnið þrátt fyrir að ég var þreittur." | correct --grammar
   {
      "original":"Ég kláraði verkefnið þrátt fyrir að ég var þreittur.",
      "corrected":"Ég kláraði verkefnið þrátt fyrir að ég var þreyttur.",
      "tokens":[
         {"k":6,"x":"Ég","o":"Ég"},
         {"k":6,"x":"kláraði","o":" kláraði"},
         {"k":6,"x":"verkefnið","o":" verkefnið"},
         {"k":6,"x":"þrátt fyrir","o":" þrátt fyrir"},
         {"k":6,"x":"að","o":" að"},
         {"k":6,"x":"ég","o":" ég"},
         {"k":6,"x":"var","o":" var"},
         {"k":6,"x":"þreyttur","o":" þreittur"},
         {"k":1,"x":".","o":"."}
      ],
      "annotations":[
         {
            "start":6,
            "end":6,
            "start_char":35,
            "end_char":37,
            "code":"P_MOOD_ACK",
            "text":"Hér er réttara að nota viðtengingarhátt
               sagnarinnar 'vera', þ.e. 'væri'.",
            "detail":"Í viðurkenningarsetningum á borð við 'Z'
               í dæminu 'X gerði Y þrátt fyrir að Z' á sögnin að vera
               í viðtengingarhætti fremur en framsöguhætti.",
            "suggest":"væri"
         },
         {
            "start":7,
            "end":7,
            "start_char":38,
            "end_char":41,
            "code":"S004",
            "text":"Orðið 'þreittur' var leiðrétt í 'þreyttur'",
            "detail":"",
            "suggest":"þreyttur"
         }
      ]
   }


The output has been formatted for legibility - each input sentence is actually
represented by a JSON object in a single line of text, terminated by newline.

Note that the ``corrected`` field only includes token-level spelling correction
(in this case *þreittur* ``->`` *þreyttur*), but no grammar corrections.
The grammar corrections are found in the ``annotations`` list.
To apply corrections and suggestions from the annotations,
replace source text or tokens (as identified by the ``start`` and ``end``,
or ``start_char`` and ``end_char`` properties) with the ``suggest`` field, if present.

*****
Tests
*****

To run the built-in tests, install `pytest <https://docs.pytest.org/en/latest/>`_,
``cd`` to your ``GreynirCorrect`` subdirectory (and optionally activate your
virtualenv), then run:

.. code-block:: bash

   $ python -m pytest

****************
Acknowledgements
****************

Parts of this software are developed under the auspices of the
Icelandic Government's 5-year Language Technology Programme for Icelandic,
which is managed by Almannarómur and described
`here <https://www.stjornarradid.is/lisalib/getfile.aspx?itemid=56f6368e-54f0-11e7-941a-005056bc530c>`__
(English version `here <https://clarin.is/media/uploads/mlt-en.pdf>`__).

*********************
Copyright and License
*********************

.. image:: https://github.com/mideind/GreynirPackage/raw/master/doc/_static/MideindLogoVert100.png?raw=true
   :target: https://mideind.is
   :align: right
   :alt: Miðeind ehf.

**Copyright © 2023 Miðeind ehf.**

GreynirCorrect's original author is *Vilhjálmur Þorsteinsson*.

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

----

GreynirCorrect indirectly embeds the `Database of Icelandic Morphology <https://bin.arnastofnun.is>`_
(`Beygingarlýsing íslensks nútímamáls <https://bin.arnastofnun.is>`_), abbreviated BÍN,
along with directly using
`Ritmyndir <https://bin.arnastofnun.is/DMII/LTdata/comp-format/nonstand-form/>`_,
a collection of non-standard word forms.
Miðeind does not claim any endorsement by the BÍN authors or copyright holders.

The BÍN source data are publicly available under the
`CC BY-SA 4.0 license <https://creativecommons.org/licenses/by-sa/4.0/>`_, as further
detailed `here in English <https://bin.arnastofnun.is/DMII/LTdata/conditions/>`_
and `here in Icelandic <https://bin.arnastofnun.is/gogn/mimisbrunnur/>`_.

In accordance with the BÍN license terms, credit is hereby given as follows:

*Beygingarlýsing íslensks nútímamáls. Stofnun Árna Magnússonar í íslenskum fræðum.*
*Höfundur og ritstjóri Kristín Bjarnadóttir.*
