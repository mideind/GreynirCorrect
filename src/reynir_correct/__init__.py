# type: ignore
"""

    Greynir: Natural language processing for Icelandic

    Copyright (C) 2022 Miðeind ehf.

    This software is licensed under the MIT License:

        Permission is hereby granted, free of charge, to any person
        obtaining a copy of this software and associated documentation
        files (the "Software"), to deal in the Software without restriction,
        including without limitation the rights to use, copy, modify, merge,
        publish, distribute, sublicense, and/or sell copies of the Software,
        and to permit persons to whom the Software is furnished to do so,
        subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
        MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
        CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
        SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

# Expose the reynir-correct API

from reynir import Greynir, Paragraph, Sentence, correct_spaces, mark_paragraphs
from tokenizer import detokenize

# Annotations
from .annotation import Annotation

# Grammar checking
from .checker import AnnotatedSentence, GreynirCorrect

# Token-level correction
from .errtokenizer import Correct_TOK, CorrectionPipeline, CorrectToken
from .readability import FleschKincaidFeedback, FleschKincaidScorer, RareWordsFinder
from .settings import Settings
from .version import __version__
from .wrappers import CorrectedSentence, CorrectionResult, GreynirCorrectAPI, ParseResultStats, check_errors

__author__ = "Miðeind ehf"
__copyright__ = "(C) 2023 Miðeind ehf."

__all__ = (
    "Greynir",
    "correct_spaces",
    "mark_paragraphs",
    "Sentence",
    "Paragraph",
    "detokenize",
    "Settings",
    "ParseResultStats",
    "FleschKincaidScorer",
    "FleschKincaidFeedback",
    "RareWordsFinder",
    "CorrectionPipeline",
    "Correct_TOK",
    "CorrectToken",
    "GreynirCorrect",
    "GreynirCorrectAPI",
    "CorrectionResult",
    "CorrectedSentence",
    "check_errors",
    "AnnotatedSentence",
    "Annotation",
    "__version__",
    "__author__",
    "__copyright__",
)
