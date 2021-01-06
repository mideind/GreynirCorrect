.. _overview:

Overview
========

GreynirCorrect can be used in three modes, depending on your requirements.

*   You can use it as a Python package to proofread text at the
    **token (word) level**, correcting and annotating individual
    errors in the token stream.

*   It can also perform more extensive grammar
    checking at the **sentence level**, returning a list of
    annotations for each sentence.

*   Finally, it can be used as a **command-line tool**, at the token level,
    to consume an input file stream (or *stdin*) and write to a corrected
    output file stream (or *stdout*).

These modes are described below. The :ref:`Reference` section contains
further detail.

Token-level correction
----------------------

GreynirCorrect can tokenize text and return an automatically corrected token stream.
This catches token-level errors, such as spelling errors and erroneous
fixed phrases (*?að ýmsu leiti* → *að ýmsu leyti*), but not grammatical errors.
The returned token objects are annotated with explanations of each correction
or suggestion.

Token-level correction is relatively fast.

Full grammar analysis
---------------------

GreynirCorrect can analyze text grammatically by attempting to parse
each sentence in turn, after token-level correction. The parsing is done according
to Greynir's context-free grammar for Icelandic, augmented with additional production
rules for common grammatical errors (*?Manninum á verkstæðinu vantaði hamar* → *Manninn
á verkstæðinu vantaði hamar*). The analysis returns each sentence along with
a set of annotations (errors and suggestions) that apply to spans
(consecutive tokens) within the sentence, in addition to individual token annotations.

Full grammar analysis of sentences is slower than token-level correction.

Command-line tool
-----------------

GreynirCorrect can be invoked as a command-line tool
to perform token-level correction. The command is ``correct infile.txt outfile.txt``,
where ``infile.txt`` and ``outfile.txt`` are the input and output filenames,
respectively.

The command-line tool is further documented :ref:`here <commandline>`.

Examples
--------

To perform token-level correction from Python code:

.. code-block:: python

    from reynir_correct import tokenize
    g = tokenize("Af gefnu tilefni fékk fékk daninn vilja sýnum "
        "framgengt í auknu mæli.")
    for tok in g:
        print("{0:10} {1}".format(tok.txt or "", tok.error_description))

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

    >>> sent.tidy_text

Output::

    'Páli, vini mínum, langaði að horfa á sjónvarpið.'

Note that the ``annotation.start`` and ``annotation.end`` properties
(here ``start`` is 0 and ``end`` is 4) contain the indices of the first
and last tokens to which the annotation applies.
``P_WRONG_CASE_þgf_þf`` and ``S004`` are error codes.

