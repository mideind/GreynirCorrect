.. _reference:

Reference
=========

Importing GreynirCorrect
------------------------

After installing the ``reynir-correct`` package (see :ref:`installation`),
import it using:

.. code-block:: python

   import reynir_correct as grc

Alternatively, use the following code to initialize an instance of
the :py:class:`GreynirCorrect` class:

.. code-block:: python

    from reynir_correct import GreynirCorrect
    grc = GreynirCorrect()

If you only want to do token-level checking, the simplest method
is to import only the tokenize() method (documented below):

.. code-block:: python

    from reynir_correct import tokenize

Similarly, if you only want straightforward checking on single
sentences, you can import only the check_single() method
(documented below):

.. code-block:: python

    from reynir_correct import check_single

The GreynirCorrect class
------------------------

.. py:class:: GreynirCorrect


The ``tokenize()`` function
---------------------------

    .. py:method:: tokenize(text, **options)

        Consumes a text stream and returns a generator of corrected or
        annotated tokens.

        :param text: A text string, or an iterator of strings.

        :param options: Tokenizer options can be passed via keyword arguments,
            as in ``grc = GreynirCorrect(convert_numbers=True)``. See
            the documentation for the `Tokenizer <https://github.com/mideind/Tokenizer>`_
            package for further information.

            Two boolean flags directly affect the correction process.
            Setting ``only_ci=True`` tells the checker to look only for
            context-independent errors. Setting ``apply_suggestions=True``
            makes the checker more aggressive in turning suggestions into
            corrections.

        :return: A generator of tokens, where each token is an instance
            of the CorrectToken class.


