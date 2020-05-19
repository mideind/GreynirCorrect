.. _reference:

Reference
=========

This section describes the following functions and classes:

* :py:func:`tokenize()`
* :py:func:`check_single()`
* :py:func:`check()`
* :py:func:`check_with_stats()`
* :py:class:`CorrectToken`
* :py:class:`Annotation`

Importing GreynirCorrect
------------------------

After installing the ``reynir-correct`` package (see :ref:`installation`),
import it using:

.. code-block:: python

   import reynir_correct as grc

If you only want to do token-level checking, the simplest method
is to import only the :py:func:`tokenize()` function (documented below):

.. code-block:: python

    from reynir_correct import tokenize

Similarly, if you only want straightforward checking on single
sentences, you can import only the :py:func:`check_single()` method
(documented below):

.. code-block:: python

    from reynir_correct import check_single


The tokenize() function
-----------------------

.. py:function:: tokenize(text: Union[str, Iterable[str]], **options) -> Iterator[CorrectToken]

    Consumes a text stream and returns a generator of instances of
    :py:class:`CorrectToken`, corrected or annotated as the case may be.

    :param text: A text string, or an iterator of strings.

    :param options: Tokenizer options can be passed via keyword arguments,
        as in ``g = tokenize(text, convert_numbers=True)``. See
        the documentation for the `Tokenizer <https://github.com/mideind/Tokenizer>`_
        package for further information.

        Two boolean flags directly affect the correction process.
        Setting ``only_ci=True`` tells the checker to look only for
        context-independent errors. Setting ``apply_suggestions=True``
        makes the checker more aggressive in turning suggestions into
        corrections.

    :return: A generator of tokens, where each token is an instance
        of the :py:class:`CorrectToken` class.

    Example::

        from reynir_correct import tokenize
        g = tokenize("Maðurin borðaði aldrey Danskan hammborgara")
        for t in g:
            if t.txt:
                print(f"{t.txt:12} {t.error_code:8} {t.error_description}")

    Output::

        Maðurinn     S004   Orðið 'Maðurin' var leiðrétt í 'Maðurinn'
        borðaði
        aldrei       S001   Orðið 'aldrey' var leiðrétt í 'aldrei'
        danskan      Z001   Orð á að byrja á lágstaf: 'Danskan'
        hamborgara   S004   Orðið 'hammborgara' var leiðrétt í 'hamborgara'


The check_single() function
---------------------------

.. py:function:: check_single(sentence: str) -> _Sentence

    Analyzes the spelling and grammar of a sentence, returning an
    instance of the :py:class:`_Sentence` class. The :py:class:`_Sentence`
    class is described in the
    `Greynir documentation <https://greynir.is/doc/reference.html#_Sentence>`__.
    GreynirCorrect adds the ``annotations`` property to the :py:class:`_Sentence`
    object, which returns a list of :py:class:`Annotation` instances applying
    to the sentence.

    :param sentence: The sentence to analyze, as a string. If the string
        contains more than one sentence, only the first one is analyzed.

    :return: A :py:class:`_Sentence` object.

    Example::

        s = check_single("Ég dreimi um að leita af mindinni")
        for a in s.annotations:
            print(a)

    Output (showing token span, error code, error description and
    suggested replacement)::

        000-000: P_WRONG_CASE_nf_þf Á líklega að vera 'Mig' / [Mig]
        001-001: S004   Orðið 'dreimi' var leiðrétt í 'dreymi'
        004-005: P001   'leita af' á sennilega að vera 'leita að' / [um að leita að myndinni]
        006-006: S004   Orðið 'mindinni' var leiðrétt í 'myndinni'


The check() function
--------------------

.. py:function:: check(text: str, *, split_paragraphs: bool=False) -> Iterable[_Paragraph]

    Returns a generator of checked paragraphs of text
    (instances of the :py:class:`_Paragraph` class),
    with each of those being a generator of checked
    sentences with annotations. Sentences are parsed and checked
    "on demand", just before being returned from the generator.

    :param text: The text to analyze, as a string. It may contain
        multiple paragraphs and sentences.

    :param split_paragraphs: If set to ``True``, the text will be
        split into paragraphs at each newline.

    :return: A generator of :py:class:`_Paragraph` instances.


The check_with_stats() function
-------------------------------

.. py:function:: check_with_stats(text: str, *, split_paragraphs: bool=False) -> Dict

    Returns a dictionary with the results of a grammar and spelling check on the
    given text. This is a synchronous call, i.e. it does not return until
    the entire text has been processed.

    :param text: The text to analyze, as a string. It may contain multiple
        paragraphs and sentences.

    :param split_paragraphs: If set to ``True``, the text is automatically
        split into paragraphs between empty lines.

    :return: A dictionary with the following keys and values:

        * ``paragraphs``: A list of lists of :py:class:`_Sentence` objects,
          each having the ``annotations`` property containing a list of
          :py:class:`Annotation` objects.

        * ``num_tokens``: The total number of tokens processed.

        * ``num_sentences``: The number of sentences found in the text.

        * ``num_parsed``: The number of sentences that were successfully parsed.

        * ``ambiguity``: A ``float`` weighted average of the ambiguity of the parsed
          sentences. Ambiguity is defined as the *n*-th root of the number
          of possible parse trees for the sentence, where *n* is the number
          of tokens in the sentence.

        * ``parse_time``: A ``float`` with the wall clock time, in seconds,
          spent on tokenizing and parsing the sentences.


The CorrectToken class
----------------------

.. py:class:: CorrectToken

    The :py:class:`CorrectToken` class replaces the default ``tokenizer.Tok``
    named tuple normally returned by the
    `Tokenizer <https://github.com/mideind/Tokenizer>`_. By way of duck typing,
    it replicates the ``kind``, ``txt`` and ``val`` properties of the ``Tok``
    tuple. It then adds a number of properties to access error codes and
    annotations on the token, as described here:

    .. py:attribute:: error_description(self) -> str

        Returns the description of the error associated with the token, or
        an empty string if there is no error.

    .. py:attribute:: error_code(self) -> str

        Returns the code of the error associated with the token, or
        an empty string if there is no error.

    .. py:attribute:: error_suggestion(self) -> str

        Returns the text of a suggested replacement for the text of this
        token, or an empty string if there is no error.

    .. py:attribute:: error_span(self) -> int

        Returns the number of consecutive tokens, starting with this one,
        that are affected by the same error. In most cases this is 1,
        meaning that there are no additional affected tokens.


The Annotation class
--------------------

.. py:class:: Annotation

    The :py:class:`Annotation` class represents an annotation of a
    token span within a sentence. An annotation describes a correction
    that has already been applied to the sentence, or a suggested
    correction.

    .. py:method:: __str__(self) -> str

        Returns a string representation of the annotation. This is intended
        mainly for debugging and development purposes.

    .. py:attribute:: start(self) -> int

        Returns the index of the first token to which the annotation
        applies. Token indices are 0-based.

    .. py:attribute:: end(self) -> int

        Returns the index of the last token to which the annotation
        applies. Token indices are 0-based.

    .. py:attribute:: code(self) -> str

        Returns an error or warning code for the annotation.
        If the code ends with ``"/w"``, it is a warning.

    .. py:attribute:: text(self) -> str

        Returns a brief, human-readable description of the annotation.

    .. py:attribute:: detail(self) -> str

        Returns a more detailed, human-readable description of the annotation.

    .. py:attribute:: suggest(self) -> str

        Returns a suggested replacement for the text within the token
        span to which the annotation applies. This only applies for
        suggested corrections, i.e. if the correction has not been already
        applied to the sentence.


The _Paragraph class
--------------------

.. py:class:: _Paragraph

    The :py:class:`_Paragraph` class is described in the
    `Greynir documentation <https://greynir.is/doc/reference.html#_Paragraph>`__.


The _Sentence class
--------------------

.. py:class:: _Sentence

    The :py:class:`_Sentence` class is described in the
    `Greynir documentation <https://greynir.is/doc/reference.html#_Sentence>`__.

