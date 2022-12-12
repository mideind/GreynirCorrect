"""

    Greynir: Natural language processing for Icelandic

    Wrapper functions module

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


    This module exposes functions to return corrected strings given an input text.
    The following options are defined:

    input:  Defines the input. Can be a string or an iterable of strings, such
            as a file object.
    format: Defines the output format. String.
            text: Output is returned as a corrected version of the input.
            json: Output is returned as a JSON string.
            csv:  Output is returned in a csv format.
            m2:   Output is returned in the M2 format, see https://github.com/nusnlp/m2scorer
                  The output is as follows:
                  S <tokenized system output for sentence 1>
                  A <token start offset> <token end offset>|||<error type>|||<correction1>||<correction2||..||correctionN|||<required>|||<comment>|||<annotator id>
    all_errors: Defines the level of correction. If False, only token-level annotation is carried out.
                If True, sentence-level annotation is carried out.
    annotate_unparsed_sentences: If True, sentences that cannot be parsed are annotated as errors as a whole.
    annotations: If True, can all error annotations are returned at the end of the output. Works with format text.
    generate_suggestion_list: If True, the annotation can in certain cases contain a list of possible corrections, for the user to pick from.
    suppress_suggestions: If True, more farfetched automatically retrieved corrections are rejected and no error is added.
    ignore_wordlist: The value is a set of strings, a whitelist. Each string is a word that should not be marked as an error or corrected.
    one_sent: Defines input as containing only one sentence.
    ignore_rules: A list of error codes that should be ignored in the annotation process.
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
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from .classifier import SentenceClassifier

import sys
import argparse
import json
from functools import partial
from typing_extensions import TypedDict

from tokenizer import detokenize, text_from_tokens, normalized_text_from_tokens, TOK
from tokenizer.definitions import AmountTuple, NumberTuple

from .errtokenizer import CorrectToken, Error
from .errtokenizer import tokenize as errtokenize
from .annotation import Annotation
from .checker import check_tokens


class AnnTokenDict(TypedDict, total=False):

    """Type of the token dictionaries returned from check_errors()"""

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


TokenSumType = List[Union[List[CorrectToken], CorrectToken]]


# File types for UTF-8 encoded text files
ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

# Configure our JSON dump function
json_dumps = partial(json.dumps, ensure_ascii=False, separators=(",", ":"))

# Define the command line arguments


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


def check_errors(**options: Any) -> str:
    """Return a string in the chosen format and correction level
    using the spelling and grammar checker"""
    input = options.get("input", None)
    if isinstance(input, str):
        options["input"] = [input]
    if options.get("all_errors", True):
        return check_grammar(**options)
    else:
        return check_spelling(**options)


def check_spelling(**options: Any) -> str:
    # Initialize sentence accumulator list
    # Function to convert a token list to output text
    format = options.get("format", "json")
    if options.get("spaced", False):
        if options.get("normalize", False):
            to_text = normalized_text_from_tokens
        else:
            to_text = text_from_tokens
    else:
        to_text = partial(detokenize, normalize=True)
    toks = sentence_stream(**options)
    unisum: List[str] = []
    allsum: List[str] = []
    annlist: List[str] = []
    annotations = options.get("annotations", False)
    print_all = options.get("print_all", False)
    for toklist in toks:
        if format == "text":
            txt = to_text(toklist)
            if annotations:
                for t in toklist:
                    if t.error:
                        annlist.append(str(t.error))
                if annlist and not print_all:
                    txt = txt + "\n" + "\n".join(annlist)
                    annlist = []
            unisum.append(txt)
            continue
        for t in toklist:
            if format == "csv":
                if t.txt:
                    allsum.append(
                        "{0},{1},{2},{3}".format(
                            t.kind,
                            quote(t.txt),
                            val(t, quote_word=True) or '""',
                            quote(str(t.error) if t.error else ""),
                        )
                    )
                elif t.kind == TOK.S_END:
                    # Indicate end of sentence
                    allsum.append('0,"",""')
            elif format == "json":
                # Output the tokens in JSON format, one line per token
                d: Dict[str, Any] = dict(k=TOK.descr[t.kind])
                if t.txt is not None:
                    d["t"] = t.txt
                v = val(t)
                if t.kind not in {TOK.WORD, TOK.PERSON, TOK.ENTITY} and v is not None:
                    d["v"] = v
                if isinstance(t.error, Error):
                    d["e"] = t.error.to_dict()
                allsum.append(json_dumps(d))
        if allsum:
            unisum.extend(allsum)
            allsum = []
    if print_all:
        # We want the annotations at the bottom
        unistr = " ".join(unisum)
        if annlist:
            unistr = unistr + "\n" + "\n".join(annlist)
    else:
        unistr = "\n".join(unisum)
    return unistr


def test_spelling(**options: Any) -> Tuple[str, TokenSumType]:
    # Initialize sentence accumulator list
    # Function to convert a token list to output text
    if options.get("spaced", False):
        if options.get("normalize", False):
            to_text = normalized_text_from_tokens
        else:
            to_text = text_from_tokens
    else:
        to_text = partial(detokenize, normalize=True)
    toks = sentence_stream(**options)
    unisum: List[str] = []
    toksum: TokenSumType = []
    annlist: List[str] = []
    print_all = options.get("print_all", False)
    for toklist in toks:
        unisum.append(to_text(toklist))
        if print_all:
            toksum.extend(toklist)
        else:
            toksum.append(toklist)
        continue
    if print_all:
        # We want the annotations at the bottom
        unistr = " ".join(unisum)
        if annlist:
            unistr = unistr + "\n" + "\n".join(annlist)
    else:
        unistr = "\n".join(unisum)
    return unistr, toksum


def sentence_stream(**options: Any) -> Iterator[List[CorrectToken]]:
    """Yield a stream of sentence token lists from the source text"""
    # Initialize sentence accumulator list
    curr_sent: List[CorrectToken] = []
    gen = options.get("input", None)
    if gen is None:
        gen = sys.stdin
    for t in errtokenize(gen, **options):
        # Normal shallow parse, one line per sentence,
        # tokens separated by spaces
        curr_sent.append(t)
        if t.kind in TOK.END:
            # End of sentence/paragraph
            yield curr_sent
            curr_sent = []
    if curr_sent:
        yield curr_sent


def test_grammar(**options: Any) -> Tuple[str, TokenSumType]:
    """Do a full spelling and grammar check of the source text"""

    accumul: List[str] = []
    offset = 0
    alltoks: TokenSumType = []
    inneroptions: Dict[str, Union[str, bool]] = {}
    inneroptions["annotate_unparsed_sentences"] = options.get(
        "annotate_unparsed_sentences", True
    )
    inneroptions["ignore_rules"] = options.get("ignore_rules", set())
    annlist: List[str] = []
    for toklist in sentence_stream(**options):
        # Invoke the spelling and grammar checker on the token list
        # Only contains options relevant to the grammar check
        sent = check_tokens(toklist, **inneroptions)
        if sent is None:
            # Should not happen?
            continue

        # Maintain token character offsets, accumulated over the entire source text
        token_offsets: Dict[int, int] = dict()
        for ix, t in enumerate(toklist):
            token_offsets[ix] = offset
            offset += len(t.original or t.txt or "")

        # Extract the annotation list (defensive programming here)
        a: List[Annotation] = getattr(sent, "annotations", cast(List[Annotation], []))
        # Sort in ascending order by token start index, and then by end index
        # (more narrow/specific annotations before broader ones)
        a.sort(key=lambda ann: (ann.start, ann.end))

        arev = sorted(a, key=lambda ann: (ann.start, ann.end), reverse=True)
        cleantoklist: List[CorrectToken] = toklist[:]
        alltoks.extend(cleantoklist)
        for xann in arev:
            if xann.suggest is None:
                # Nothing to correct with, nothing we can do
                continue
            cleantoklist[xann.start + 1].txt = xann.suggest
            if xann.end > xann.start:
                # Annotation spans many tokens
                # "Okkur börnunum langar í fisk"
                # "Leita að kílómeter af féinu" → leita að kílómetri af fénu → leita að kílómetra af fénu
                # "dást af þeim" → "dást að þeim"
                # Single-token annotations for this span have already been handled
                # Only case is one ann, many toks in toklist
                # Give the first token the correct value
                # Delete the other tokens
                del cleantoklist[xann.start + 2 : xann.end + 2]
        txt = detokenize(cleantoklist, normalize=True)
        if options.get("annotations", False):
            for aann in a:
                annlist.append(str(aann))
            if annlist and not options.get("print_all", False):
                txt = txt + "\n" + "\n".join(annlist)
                annlist = []
        accumul.append(txt)

    accumstr = "\n".join(accumul)

    return accumstr, alltoks


def check_grammar(**options: Any) -> str:
    """Do a full spelling and grammar check of the source text"""

    inneroptions: Dict[str, Union[str, bool]] = {}
    inneroptions["annotate_unparsed_sentences"] = options.get(
        "annotate_unparsed_sentences", True
    )
    inneroptions["ignore_rules"] = options.get("ignore_rules", set())

    sentence_results: List[Dict[str, Any]] = []

    sentence_classifier: Optional[SentenceClassifier] = None

    for raw_tokens in sentence_stream(**options):
        original_sentence = "".join([t.original or t.txt for t in raw_tokens])

        if options.get("sentence_prefilter", False):
            # Only construct the classifier model if we need it
            from .classifier import SentenceClassifier

            if sentence_classifier is None:
                sentence_classifier = SentenceClassifier()

            if not sentence_classifier.classify(original_sentence):
                # Skip the full parse if we think the sentence is probably correct

                # Remove the metatokens (e.g. sentence-begin) from the token stream
                # to be consistent with what the parser returns.
                # TODO: use TOK.META_BEGIN when that PR is ready
                tokens_no_meta = [t for t in raw_tokens if t.kind < TOK.S_SPLIT]
                nice_tokens = [
                    AnnTokenDict(k=d.kind, x=d.txt, o=d.original or d.txt)
                    for d in tokens_no_meta
                ]
                sentence_results.append(
                    {
                        "original": original_sentence,
                        "corrected": original_sentence,
                        "partially_corrected": original_sentence,
                        "annotations": [],
                        "tokens": nice_tokens,
                    }
                )
                continue

        annotated_sentence = check_tokens(raw_tokens, **inneroptions)
        if annotated_sentence is None:
            # This should not happen, but we check to be sure, and to satisfy mypy
            # TODO: Should we rather raise an exception instead of silently discarding the sentence?
            continue

        # Extract the annotation list (defensive programming here)
        annotations: List[Annotation] = getattr(
            annotated_sentence, "annotations", cast(List[Annotation], [])
        )
        # Sort in ascending order by token start index, and then by end index
        # (more narrow/specific annotations before broader ones)
        annotations.sort(key=lambda ann: (ann.start, ann.end))

        # Generate a sentence with only spelling corrections applied
        partially_corrected_sentence = detokenize(annotated_sentence.tokens)

        # Generate a sentence with all corrections applied
        full_correction_toks = annotated_sentence.tokens[:]
        for ann in annotations[::-1]:
            if ann.suggest is None:
                # Nothing to correct with, nothing we can do
                continue
            full_correction_toks[ann.start].txt = ann.suggest
            if ann.end > ann.start:
                # Annotation spans many tokens
                # "Okkur börnunum langar í fisk"
                # "Leita að kílómeter af féinu" → leita að kílómetri af fénu → leita að kílómetra af fénu
                # "dást af þeim" → "dást að þeim"
                # Single-token annotations for this span have already been handled
                # Only case is one ann, many toks in toklist
                # Give the first token the correct value
                # Delete the other tokens
                del full_correction_toks[ann.start + 1 : ann.end + 1]
        fully_corrected_sentence = detokenize(full_correction_toks)

        # Make a nice token list
        tokens: List[AnnTokenDict]
        if annotated_sentence.tree is None:
            # Not parsed: use the raw token list
            tokens = [
                AnnTokenDict(k=d.kind, x=d.txt, o=d.original or d.txt)
                for d in annotated_sentence.tokens
            ]
        else:
            # Successfully parsed: use the text from the terminals (where available)
            # since we have more info there, for instance on em/en dashes.
            # Create a map of token indices to corresponding terminal text
            assert annotated_sentence.terminals is not None
            token_map = {t.index: t.text for t in annotated_sentence.terminals}
            tokens = [
                AnnTokenDict(
                    k=d.kind, x=token_map.get(ix, d.txt), o=d.original or d.txt
                )
                for ix, d in enumerate(annotated_sentence.tokens)
            ]

        sentence_results.append(
            {
                "original": original_sentence,
                "partially_corrected": partially_corrected_sentence,
                "corrected": fully_corrected_sentence,
                "annotations": annotations,
                "tokens": tokens,
            }
        )

    extra_text_options = {
        "annotations": options.get("annotations", False),
        "print_all": options.get("print_all", False),
    }
    return format_output(
        sentence_results, options.get("format", "json"), extra_text_options
    )


def format_output(
    sentence_results: List[Dict[str, Any]],
    format_type: str,
    extra_text_options: Dict[str, Any],
) -> str:
    """
    Format grammar analysis results in the given format.

    `sentence_results` is a list of individual sentences and their analysis
    `format_type` is the output format to use, one of 'text', 'json', 'csv', 'm2'
    `extra_text_options` takes extra options for the text format. Ignored for other formats.
    """

    if format_type == "text":
        return format_text(sentence_results, extra_text_options)
    elif format_type == "json":
        return format_json(sentence_results)
    elif format_type == "csv":
        return format_csv(sentence_results)
    elif format_type == "m2":
        return format_m2(sentence_results)

    raise Exception(f"Tried to format with invalid format: {format_type}")


def format_text(
    sentence_results: List[Dict[str, Any]], extra_options: Dict[str, Any]
) -> str:
    output: List[str] = []
    annlist: List[str] = []
    for result in sentence_results:
        txt = result["corrected"]

        if extra_options["annotations"]:
            for aann in result["annotations"]:
                annlist.append(str(aann))
            if annlist and not extra_options["print_all"]:
                txt = txt + "\n" + "\n".join(annlist)
                annlist = []
        output.append(txt)

    return "\n".join(output)


def format_json(sentence_results: List[Dict[str, Any]]) -> str:
    formatted_sentences: List[str] = []

    offset = 0
    for result in sentence_results:
        tokens = result["tokens"]

        token_offsets: List[int] = []
        for t in tokens:
            token_offsets.append(offset)
            offset += len(t["o"] or t["x"] or "")
        # Add a past-the-end offset to make later calculations more convenient
        token_offsets.append(offset)

        # Convert the annotations to a standard format before encoding in JSON
        formatted_annotations: List[AnnDict] = [
            AnnDict(
                # Start token index of this annotation
                start=ann.start,
                # End token index (inclusive)
                end=ann.end,
                # Character offset of the start of the annotation in the original text
                start_char=token_offsets[ann.start],
                # Character offset of the end of the annotation in the original text
                # (inclusive, i.e. the offset of the last character)
                end_char=token_offsets[ann.end + 1] - 1,
                code=ann.code,
                text=ann.text,
                detail=ann.detail or "",
                suggest=ann.suggest or "",
            )
            for ann in result["annotations"]
        ]
        ard = AnnResultDict(
            original=result["original"],
            corrected=result["corrected"],
            tokens=tokens,
            annotations=formatted_annotations,
        )

        formatted_sentences.append(json.dumps(ard))

    return "\n".join(formatted_sentences)


def format_csv(sentence_results: List[Dict[str, Any]]) -> str:
    accumul: List[str] = []
    for result in sentence_results:
        for ann in result["annotations"]:
            accumul.append(
                "{},{},{},{},{},{}".format(
                    ann.code,
                    ann.original,
                    ann.suggest,
                    ann.start,
                    ann.end,
                    ann.suggestlist,
                )
            )
    return "\n".join(accumul)


def format_m2(sentence_results: List[Dict[str, Any]]) -> str:
    accumul: List[str] = []
    for result in sentence_results:
        accumul.append("S {0}".format(" ".join([t["x"] for t in result["tokens"]])))
        for ann in result["annotations"]:
            accumul.append(
                "A {0} {1}|||{2}|||{3}|||REQUIRED|||-NONE-|||0".format(
                    ann.start, ann.end, ann.code, ann.suggest
                )
            )
        accumul.append("")
    return "\n".join(accumul)
