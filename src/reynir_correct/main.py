#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Spelling and grammar checking module

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


    This is an executable program wrapper (main module) for the GreynirCorrect
    package. It can be used to invoke the corrector from the command line,
    or via fork() or exec(), with the command 'correct'. The main() function
    of this module is registered as a console_script entry point in setup.py.

"""

from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    Iterator,
    Iterable,
    Dict,
    Any,
    Union,
    cast,
)

import sys
import argparse
import json
from functools import partial
from typing_extensions import TypedDict

from tokenizer import detokenize, normalized_text_from_tokens
from tokenizer.definitions import AmountTuple, NumberTuple

from .errtokenizer import TOK, CorrectToken, Error
from .errtokenizer import tokenize as errtokenize
from .annotation import Annotation
from .checker import check_tokens
from .wrappers import check_errors


class AnnTokenDict(TypedDict, total=False):

    """Type of the token dictionaries returned from check_grammar()"""

    # Token kind
    k: int
    # Token text
    x: str
    # Original text of token
    o: str
    # Character offset of token, indexed from the start of the checked text
    i: int


class AnnDict(TypedDict):

    """A single annotation, as returned by the Yfirlestur.is API"""

    start: int
    end: int
    start_char: int
    end_char: int
    code: str
    text: str
    detail: Optional[str]
    suggest: Optional[str]


class AnnResultDict(TypedDict):

    """The annotation result for a sentence"""

    original: str
    corrected: str
    annotations: List[AnnDict]
    tokens: List[AnnTokenDict]


# File types for UTF-8 encoded text files
ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

# Configure our JSON dump function
json_dumps = partial(json.dumps, ensure_ascii=False, separators=(",", ":"))

# Define the command line arguments

parser = argparse.ArgumentParser(description="Corrects Icelandic text")

parser.add_argument(
    "infile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to correct",
)
parser.add_argument(
    "outfile",
    nargs="?",
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file",
)

parser.add_argument(
    "--suppress_suggestions",
    "-sss",
    action="store_true",
    help="Suppress more agressive error suggestions",
)

parser.add_argument("--spaced", help="Separate tokens with spaces", action="store_true")

# Determines the output format
parser.add_argument(
    "--format",
    nargs="?",
    type=str,
    default="text",
    help="Determine output format.\ntext: Corrected text only\ncsv: One token per line in CSV format\njson: One token per line in JSON format\nm2: M2 format, GEC standard",
)

# Determines whether we supply only token-level annotations or also sentence-level annotations
parser.add_argument(
    "--all_errors",
    "-a",
    help="Annotate both grammar and spelling errors",
    action="store_true",
)


def gen(f: Iterator[str]) -> Iterable[str]:
    """Generate the lines of text in the input file"""
    yield from f


def quote(s: str) -> str:
    """Return the string s within double quotes, and with any contained
    backslashes and double quotes escaped with a backslash"""
    if not s:
        return '""'
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def val(
    t: CorrectToken, quote_word: bool = False
) -> Union[None, str, float, Tuple[Any, ...], Sequence[Any]]:
    """Return the value part of the token t"""
    if t.val is None:
        return None
    if t.kind in {TOK.WORD, TOK.PERSON, TOK.ENTITY}:
        # No need to return list of meanings
        return None
    if t.kind in {TOK.PERCENT, TOK.NUMBER, TOK.CURRENCY}:
        return cast(NumberTuple, t.val)[0]
    if t.kind == TOK.AMOUNT:
        num, iso, _, _ = cast(AmountTuple, t.val)
        if quote_word:
            # Format as "1234.56|USD"
            return '"{0}|{1}"'.format(num, iso)
        return num, iso
    if t.kind == TOK.S_BEGIN:
        return None
    if t.kind == TOK.PUNCTUATION:
        punct = t.punctuation
        return quote(punct) if quote_word else punct
    if quote_word and t.kind in {
        TOK.DATE,
        TOK.TIME,
        TOK.DATEABS,
        TOK.DATEREL,
        TOK.TIMESTAMP,
        TOK.TIMESTAMPABS,
        TOK.TIMESTAMPREL,
        TOK.TELNO,
        TOK.NUMWLETTER,
        TOK.MEASUREMENT,
    }:
        # Return a |-delimited list of numbers
        return quote("|".join(str(v) for v in cast(Iterable[Any], t.val)))
    if quote_word and isinstance(t.val, str):
        return quote(t.val)
    return t.val


def main() -> None:
    """Main function, called when the 'correct' command is invoked"""

    args = parser.parse_args()

    # By default, no options apply
    options: Dict[str, bool] = {}
    if args.all_errors:
        # Needed to get the correct original form for token-level annotations
        options["suggest_everything"] = True
    print(check_errors(args, **options), file=args.outfile)


if __name__ == "__main__":
    main()
