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

from typing import List, Sequence, Tuple, Iterator, Iterable, Dict, Any, Union, cast

import sys
import argparse
import json
from functools import partial

from tokenizer import Tok, detokenize, normalized_text_from_tokens
from tokenizer.definitions import AmountTuple, NumberTuple
from .errtokenizer import TOK, CorrectToken, Error, tokenize


# File types for UTF-8 encoded text files
ReadFile = argparse.FileType('r', encoding="utf-8")
WriteFile = argparse.FileType('w', encoding="utf-8")

# Define the command line arguments

parser = argparse.ArgumentParser(description="Corrects Icelandic text")

parser.add_argument(
    'infile',
    nargs='?',
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to correct",
)
parser.add_argument(
    'outfile',
    nargs='?',
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file"
)

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--csv",
    help="Output one token per line in CSV format", action="store_true"
)
group.add_argument(
    "--json",
    help="Output one token per line in JSON format", action="store_true"
)
group.add_argument(
    "--spaced",
    help="Separate tokens with spaces", action="store_true"
)


def main() -> None:
    """ Main function, called when the correct command is invoked """

    args = parser.parse_args()

    # By default, no options apply
    options: Dict[str, bool] = {}
    if not (args.csv or args.json):
        # If executing a plain ('shallow') correct,
        # apply most suggestions to the text
        options["apply_suggestions"] = True

    def quote(s: str) -> str:
        """ Return the string s within double quotes, and with any contained
            backslashes and double quotes escaped with a backslash """
        if not s:
            return "\"\""
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\""

    def gen(f: Iterator[str]) -> Iterable[str]:
        """ Generate the lines of text in the input file """
        yield from f

    def val(t: CorrectToken, quote_word: bool=False) -> Union[None, str, float, Tuple[Any, ...], Sequence[Any]]:
        """ Return the value part of the token t """
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
                return "\"{0}|{1}\"".format(num, iso)
            return num, iso
        if t.kind == TOK.S_BEGIN:
            return None
        if t.kind == TOK.PUNCTUATION:
            punct = t.punctuation
            return quote(punct) if quote_word else punct
        if quote_word and t.kind in {
            TOK.DATE, TOK.TIME, TOK.DATEABS, TOK.DATEREL, TOK.TIMESTAMP,
            TOK.TIMESTAMPABS, TOK.TIMESTAMPREL, TOK.TELNO, TOK.NUMWLETTER,
            TOK.MEASUREMENT
        }:
            # Return a |-delimited list of numbers
            return quote("|".join(str(v) for v in cast(Iterable[Any], t.val)))
        if quote_word and isinstance(t.val, str):
            return quote(t.val)
        return t.val

    # Function to convert a token list to output text
    if args.spaced:
        to_text = normalized_text_from_tokens
    else:
        to_text = partial(detokenize, normalize=True)

    # Configure our JSON dump function
    json_dumps = partial(json.dumps, ensure_ascii=False, separators=(',', ':'))

    # Initialize sentence accumulator list
    curr_sent: List[CorrectToken] = []

    for t in tokenize(gen(args.infile), **options):
        if args.csv:
            # Output the tokens in CSV format, one line per token
            if t.txt:
                print(
                    "{0},{1},{2},{3}"
                    .format(
                        t.kind,
                        quote(t.txt),
                        val(t, quote_word=True) or "\"\"",
                        quote(str(t.error) if t.error else "")
                    ),
                    file=args.outfile
                )
            elif t.kind == TOK.S_END:
                # Indicate end of sentence
                print("0,\"\",\"\"", file=args.outfile)
        elif args.json:
            # Output the tokens in JSON format, one line per token
            d: Dict[str, Any] = dict(k=TOK.descr[t.kind])
            if t.txt is not None:
                d["t"] = t.txt
            v = val(t)
            if t.kind not in {TOK.WORD, TOK.PERSON, TOK.ENTITY} and v is not None:
                d["v"] = v
            if isinstance(t.error, Error):
                d["e"] = t.error.to_dict()
            print(json_dumps(d), file=args.outfile)
        else:
            # Normal shallow parse, one line per sentence,
            # tokens separated by spaces
            if t.kind in TOK.END:
                # End of sentence/paragraph
                if curr_sent:
                    print(to_text(cast(Iterable[Tok], curr_sent)), file=args.outfile)
                    curr_sent = []
            else:
                curr_sent.append(t)

    if curr_sent:
        print(to_text(cast(Iterable[Tok], curr_sent)), file=args.outfile)


if __name__ == "__main__":
    main()
