# type: ignore
"""

    Greynir: Natural language processing for Icelandic

    Copyright (C) 2021 Miðeind ehf.

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

from reynir import Greynir, correct_spaces, mark_paragraphs, Sentence, Paragraph
from tokenizer import detokenize
from .settings import Settings

# Token-level correction
from .errtokenizer import (
    CorrectionPipeline,
    tokenize,
    Correct_TOK,
)

# Grammar checking
from .checker import (
    GreynirCorrect,
    check,
    check_single,
    check_with_stats,
    check_with_custom_parser,
    AnnotatedSentence,
)

from .version import __version__

# Annotations
from .annotation import Annotation


__author__ = u"Miðeind ehf"
__copyright__ = "(C) 2021 Miðeind ehf."


Settings.read("config/GreynirCorrect.conf")
