.. _errorcodes:

Error codes
=================

New error codes are frequently added, as the tool is still a work in progress.
Error codes are documented in the code as soon as they are added.

Errors codes can have ``_w`` attached, which indicates the annotation should only be reported as a warning.

For categories retrieved from Ritmyndir, see `here <https://bin.arnastofnun.is/gogn/storasnid/ritmyndir/>`__.
Other categories currently covered are:


+------------+----------------------------------------------------------------------------------------------+
| | Error code                          | Description                                                       |
+------------+----------------------------------------------------------------------------------------------+
|| ``A001``  |  Abbreviation corrected.                                                                     |
+------------+----------------------------------------------------------------------------------------------+
|| ``A002``  |  Token corrected as an acronym.                                                              |
+------------+----------------------------------------------------------------------------------------------+
|| ``C001``  |  Duplicated word removed. Should be corrected.                                               |
+------------+----------------------------------------------------------------------------------------------+
|| ``C002``  |  Wrongly compounded words split up. Should be corrected.                                     |
+------------+----------------------------------------------------------------------------------------------+
|| ``C003``  |  Wrongly split compounds united. Should be corrected.                                        |
+------------+----------------------------------------------------------------------------------------------+
|| ``C004``  |  Duplicated word marked as a possible error. Should be pointed out but not deleted.          |
+------------+----------------------------------------------------------------------------------------------+
|| ``C005``  |  Possible split compound, depends on meaning/PoS chosen by parser.                           |
+------------+----------------------------------------------------------------------------------------------+
|| ``C006``  |  A part of a compound word is wrong.                                                         |
+------------+----------------------------------------------------------------------------------------------+
|| ``C007``  |  A multiword compound such as "skóla-og frístundasvið" with merged first parts split up.     |
+------------+----------------------------------------------------------------------------------------------+
|| ``E001``  |  Unable to parse sentence. This means that the sentence does not conform to GreynirCorrect's |
||           |  built-in grammar rules. Either the sentence is grammatically deficient, or it is correct    |
||           |  but the grammar doesn't cover its structure.                                                |
+------------+----------------------------------------------------------------------------------------------+
|| ``E004``  |  The sentence is probably not in Icelandic.                                                  |
+------------+----------------------------------------------------------------------------------------------+
|| ``N001``  |  Wrong quotation marks.                                                                      |
+------------+----------------------------------------------------------------------------------------------+
|| ``N002``  |  Three periods should be an ellipsis. A warning is given.                                    |
+------------+----------------------------------------------------------------------------------------------+
|| ``N003``  |  Informal combination of punctuation marks (??!!). A warning is given.                       |
+------------+----------------------------------------------------------------------------------------------+
|| ``P_xxx`` |  Phrase error codes. Description for each type is provided within error annotations.         |
+------------+----------------------------------------------------------------------------------------------+
|| ``S001``  |  Common errors that cannot be interpreted as other words. Should be corrected.               |
+------------+----------------------------------------------------------------------------------------------+
|| ``S002``  |  Less common errors that cannot be interpreted as other words.                               |
||           |  Corrections should possibly only be suggested.                                              |
+------------+----------------------------------------------------------------------------------------------+
|| ``S003``  |  Erroneously formed words. Should be corrected.                                              |
+------------+----------------------------------------------------------------------------------------------+
|| ``S004``  |  Rare word, a more common one has been substituted.                                          |
+------------+----------------------------------------------------------------------------------------------+
|| ``S005``  |  Annotation was lost due to subsequent merging of tokens, a generic message is given.        |
+------------+----------------------------------------------------------------------------------------------+
|| ``T001``  |  Taboo word usage warning, with suggested replacement.                                       |
+------------+----------------------------------------------------------------------------------------------+
|| ``U001``  |  Unknown word. Nothing more is known. Cannot be corrected, only pointed out.                 |
+------------+----------------------------------------------------------------------------------------------+
|| ``W001``  |  Spelling suggestion. Replacement suggested.                                                 |
+------------+----------------------------------------------------------------------------------------------+
|| ``W002``  |  Spelling suggestion. A list of suggestions is given.                                        |
+------------+----------------------------------------------------------------------------------------------+
|| ``Y001``  |  Style warning for word.                                                                     |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z001``  |  Word should begin with a lowercase letter.                                                  |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z002``  |  Word should begin with an uppercase letter.                                                 |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z003``  |  Month name should begin with a lowercase letter.                                            |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z004``  |  Numbers should be written in lowercase ('24 milljónir').                                    |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z005``  |  Amounts should be written in lowercase ('24 milljónir króna').                              |
+------------+----------------------------------------------------------------------------------------------+
|| ``Z006``  | Acronyms should be written in uppercase ('RÚV').                                             |
+------------+----------------------------------------------------------------------------------------------+

