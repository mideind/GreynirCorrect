.. _customization:

Custom configuration
====================

Tone-of-voice issues
--------------------

The user can customize GreynirCorrect by providing a configuration file 
with additional issues to check for. The configuration file should be in the 
same format as the [GreynirCorrect.conf](src/reynir_correct/config/GreynirCorrect.conf)
default configuration file, with any of the needed sections from that file,
defined with a header in square brackets.

The user can then pass in the path to the configuration file using 
the ``--tov-config`` command line argument. The config is loaded in addition to the
default configuration file.

Two additional section headers can be added to the configuration file: 

``[tone_of_voice_words]`` and ``[tone_of_voice_patterns]``.

### Tone-of-voice words
The issues defined ucan be tone-of-voice issues, such as words that should be avoided
in a particular use case, with suggestions for replacements.

An example of a tone-of-voice section is as follows:

.. code-block:: python
    [tone_of_voice_words]
    kúrs_kk áfangi_kk "Við kjósum frekar orðið 'áfangi' í textum sem fjalla um framhaldsskólanám."
    labba_so ganga_so "Það er talmálslegt að nota 'labba', við tölum frekar um að 'ganga'."

This only works for single words, in whitespace-separated columns within each line. The format is as follows:

1. Word + '_' + category
2. Optional replacement word + '_' + category. There can be multiple replacement words,
   separated by tight forward slashes '/'
3. Optional explanatory comment, enclosed in double quotes.

Note that lines can be continued by ending them with a backslash '\',
which is especially useful for long explanatory comments.


Other sections
--------------
Other sections can be added to the configuration file, as long as they
are the same as the sections in the default configuration file, such as
``[capitalization_errors]`` or ``[multiword_errors]``.

More complex issues
-------------------
In case the user wants to check for more complex issues, such as multiword phrases
or grammatical errors which require pattern matching, the user can add a separate 
python module where these issues are handled. This module is provided in a file path 
in the configuration file, in the section ``[tone_of_voice_patterns]`` and loaded
dynamically by the ``checker.py`` module.

.. code-block:: python
    [tone_of_voice_patterns]
    file_path = "/path/to/pattern_module.py"

This however requires understanding of the syntactic patterns used in patterns.py, and 
the Greynir sentence trees.
