.. _commandline:

Command Line Tool
=================

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

