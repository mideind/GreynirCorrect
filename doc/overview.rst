.. _overview:

Overview
========

**GreynirCorrect** is a Python 3 (>= 3.5) package and command line tool for
**checking and correcting spelling and grammar** in Icelandic text.

GreynirCorrect relies on the `Greynir <https://pypi.org/project/reynir/>`_ package,
by the same authors, to tokenize and parse text.

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

Examples
--------

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

   >>>> from reynir_correct import check_single
   >>>> sent = check_single("Páli, vini mínum, langaði að horfa á sjónnvarpið.")
   >>>> for annotation in sent.annotations:
   >>>>     print("{0}".format(annotation))

Output::

   000-004: P_WRONG_CASE_þgf_þf Á líklega að vera 'Pál, vin minn' / [Pál , vin minn]
   009-009: S004   Orðið 'sjónnvarpið' var leiðrétt í 'sjónvarpið'

.. code-block:: python

   >>>> sent.tidy_text

Output::

   'Páli, vini mínum, langaði að horfa á sjónvarpið.'

Note that the ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the indices of the first
and last tokens to which the annotation applies.
``P_WRONG_CASE_þgf_þf`` and ``S004`` are error codes.

