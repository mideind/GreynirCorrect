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
parser.add_argument("--text", help="Output corrected text only", action="store_true")

group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--csv", help="Output one token per line in CSV format", action="store_true"
)
group.add_argument(
    "--json", help="Output one token per line in JSON format", action="store_true"
)
group.add_argument("--spaced", help="Separate tokens with spaces", action="store_true")
group.add_argument(
    "--grammar",
    help="Annotate grammar and spelling errors; output JSON",
    action="store_true",
)

parser.add_argument(
    "-suppress_suggestions",
    "--sss",
    action="store_true",
    help="Suppress more agressive error suggestions",
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


def check_spelling(args: argparse.Namespace, **options: Any) -> None:
    # Initialize sentence accumulator list
    curr_sent: List[CorrectToken] = []

    # Function to convert a token list to output text
    if args.spaced:
        to_text = normalized_text_from_tokens
    else:
        to_text = partial(detokenize, normalize=True)

    for t in errtokenize(gen(args.infile), **options):
        if args.csv:
            # Output the tokens in CSV format, one line per token
            if t.txt:
                print(
                    "{0},{1},{2},{3}".format(
                        t.kind,
                        quote(t.txt),
                        val(t, quote_word=True) or '""',
                        quote(str(t.error) if t.error else ""),
                    ),
                    file=args.outfile,
                )
            elif t.kind == TOK.S_END:
                # Indicate end of sentence
                print('0,"",""', file=args.outfile)
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
                    print(to_text(curr_sent), file=args.outfile)
                    curr_sent = []
            else:
                curr_sent.append(t)

    if curr_sent:
        print(to_text(curr_sent), file=args.outfile)


def check_grammar(args: argparse.Namespace, **options: Any) -> None:
    """Do a full spelling and grammar check of the source text"""

    def sentence_stream() -> Iterator[List[CorrectToken]]:
        """Yield a stream of sentence token lists from the source text"""
        # Initialize sentence accumulator list
        curr_sent: List[CorrectToken] = []
        for t in errtokenize(gen(args.infile), **options):
            # Normal shallow parse, one line per sentence,
            # tokens separated by spaces
            # Note this uses the tokenize function in errtokenizer.py
            # instead of the one in Tokenizer
            curr_sent.append(t)
            if t.kind in TOK.END:
                # End of sentence/paragraph
                yield curr_sent
                curr_sent = []
        if curr_sent:
            yield curr_sent

    offset = 0
    for toklist in sentence_stream():
        len_tokens = len(toklist)
        # Invoke the spelling and grammar checker on the token list
        sent = check_tokens(toklist)

        if sent is None:
            # Should not happen?
            continue

        tokens: List[AnnTokenDict]
        if sent.tree is None:
            # Not parsed: use the raw token list
            tokens = [
                AnnTokenDict(k=d.kind, x=d.txt, o=d.original or d.txt)
                for d in sent.tokens
            ]
        else:
            # Successfully parsed: use the text from the terminals (where available)
            # since we have more info there, for instance on em/en dashes.
            # Create a map of token indices to corresponding terminal text
            assert sent.terminals is not None
            token_map = {t.index: t.text for t in sent.terminals}
            tokens = [
                AnnTokenDict(
                    k=d.kind, x=token_map.get(ix, d.txt), o=d.original or d.txt
                )
                for ix, d in enumerate(sent.tokens)
            ]

        # Maintain token character offsets, accumulated over the entire source text
        token_offsets: Dict[int, int] = dict()
        for ix, t in enumerate(toklist):
            token_offsets[ix] = offset
            offset += len(t.original or t.txt or "")

        # Create a normalized form of the sentence
        cleaned = detokenize(toklist, normalize=True)
        # Extract the annotation list (defensive programming here)
        a: List[Annotation] = getattr(sent, "annotations", cast(List[Annotation], []))
        # Sort in ascending order by token start index, and then by end index
        # (more narrow/specific annotations before broader ones)
        a.sort(key=lambda ann: (ann.start, ann.end))

        # Convert the annotations to a standard format before encoding in JSON
        annotations: List[AnnDict] = [
            AnnDict(
                # Start token index of this annotation
                start=ann.start,
                # End token index (inclusive)
                end=ann.end,
                # Character offset of the start of the annotation in the original text
                start_char=token_offsets[ann.start],
                # Character offset of the end of the annotation in the original text
                # (inclusive, i.e. the offset of the last character)
                end_char=(
                    token_offsets[ann.end + 1] if ann.end + 1 < len_tokens else offset
                )
                - 1,
                code=ann.code,
                text=ann.text,
                detail=ann.detail or "",
                suggest=ann.suggest or "",
            )
            for ann in a
        ]
        if args.text:
            arev = a
            arev.sort(key=lambda ann: (ann.start, ann.end), reverse=True)
            if sent.tree is None:
                # No need to do more, no grammar errors have been checked
                print(cleaned, file=args.outfile)
            else:
                # We know we have a sentence tree, can use that
                cleantoklist: List[CorrectToken] = toklist
                for xann in arev:
                    if xann.suggest is None:
                        # Nothing to correct with, nothing we can do
                        continue
                    cleantoklist[xann.start + 1].txt = xann.suggest
                    if xann.end > xann.start:
                        # Annotation spans many tokens
                        # "Okkur börnunum langar í fisk"
                        # Only case is one ann, many toks in toklist
                        # Give the first token the correct value
                        # Delete the other tokens
                        del cleantoklist[xann.start + 2 : xann.end + 2]
                doubleclean = detokenize(cleantoklist, normalize=True)
                print(doubleclean, file=args.outfile)
        else:
            # Create final dictionary for JSON encoding
            ard = AnnResultDict(
                original=cleaned,
                corrected=sent.tidy_text,
                tokens=tokens,
                annotations=annotations,
            )

            print(json_dumps(ard), file=args.outfile)


def main() -> None:
    """Main function, called when the 'correct' command is invoked"""

    args = parser.parse_args()

    # By default, no options apply
    options: Dict[str, bool] = {}
    if not (args.csv or args.json or args.grammar):
        # If executing a plain ('shallow') correct,
        # apply most suggestions to the text
        options["apply_suggestions"] = True
    if args.grammar:
        # Check grammar, output a text or JSON object for each sentence
        check_grammar(args, **options)
    else:
        # Check spelling, output text or token objects in JSON or CSV form
        check_spelling(args, **options)


if __name__ == "__main__":
    main()
