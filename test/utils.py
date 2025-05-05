from typing import Iterable, List, Optional, Tuple

from functools import partial

from tokenizer import detokenize, normalized_text_from_tokens, text_from_tokens

import reynir_correct
from reynir_correct.errtokenizer import CorrectToken
from reynir_correct.wrappers import GreynirCorrectAPI


def check_sentence(
    api: GreynirCorrectAPI,
    s: str,
    annotations: Optional[List[Tuple[int, int, str]]],
    is_foreign: bool = False,
    ignore_warnings: bool = False,
) -> None:
    """Check whether a given single sentence gets the
    specified annotations when checked"""

    def check_sent(sent: reynir_correct.CorrectedSentence) -> None:
        assert sent is not None
        if not sent.parsed and not is_foreign:
            # If the sentence should not parse, call
            # check_sentence with annotations=None
            assert annotations is None
            return
        assert annotations is not None
        if not is_foreign:
            assert sent.parsed
        # Compile a list of error annotations, omitting warnings
        assert sent.annotations is not None
        sent_errors = [a for a in sent.annotations if not a.code.endswith("/w")]
        if not annotations:
            # This sentence is not supposed to have any annotations
            if ignore_warnings:
                assert len(sent_errors) == 0
            else:
                assert len(sent.annotations) == 0
            return
        if ignore_warnings:
            assert len(sent_errors) == len(annotations)
            for a, (start, end, code) in zip(sent_errors, annotations):
                assert a.start == start
                assert a.end == end
                assert a.code == code
        else:
            assert len(sent.annotations) == len(annotations)
            for a, (start, end, code) in zip(sent.annotations, annotations):
                assert a.start == start, f"Mismatch between ({a.start}, {a.end}, {a.code}) and ({start}, {end}, {code})"
                assert a.end == end
                assert a.code == code

    result = api.correct(s)
    for sent in result.sentences:
        assert isinstance(sent, reynir_correct.CorrectedSentence)
        check_sent(sent)
    # Test presevation of original token text
    assert "".join(tok.original or "" for sent in result.sentences for tok in sent.tokens) == s


def correct_spelling_format(
    text: Iterable[str],
    api: GreynirCorrectAPI,
    spaced: bool = False,
    normalize: bool = False,
) -> Tuple[str, List[CorrectToken]]:
    """Do a full spelling check of the source text. This is only used for testing"""
    assert not api.do_grammar_check, "Grammar checking is enabled"
    # Initialize sentence accumulator list
    # Function to convert a token list to output text
    if spaced:
        if normalize:
            to_text = normalized_text_from_tokens
        else:
            to_text = text_from_tokens
    else:
        to_text = partial(detokenize, normalize=True)
    result = api.correct(text)
    return "\n".join(to_text(sent.tokens) for sent in result.sentences), [
        tok for sent in result.sentences for tok in sent.tokens
    ]


def correct_grammar_format(
    text: Iterable[str],
    api: GreynirCorrectAPI,
    print_annotations: bool = False,
    print_all: bool = False,
    ignore_rules: Optional[frozenset[str]] = None,
    suppress_suggestions: bool = False,
) -> Tuple[str, List[CorrectToken]]:
    """Do a full spelling and grammar check of the source text. This is only used for testing"""
    assert api.do_grammar_check, "Grammar checking is disabled"
    accumul: List[str] = []
    alltoks: List[CorrectToken] = []

    correction_result = api.correct(text, ignore_rules=ignore_rules, suppress_suggestions=suppress_suggestions)

    for sent in correction_result.sentences:
        a = sent.annotations
        if a is None:
            # This should not happen
            raise Exception("Annotations not set in sentence which was supposedly parsed")
        # Sort in ascending order by token start index, and then by end index
        # (more narrow/specific annotations before broader ones)
        a.sort(key=lambda ann: (ann.start, ann.end))

        arev = sorted(a, key=lambda ann: (ann.start, ann.end), reverse=True)
        cleantoklist: List[CorrectToken] = sent.tokens[:]
        alltoks.extend(cleantoklist)
        for xann in arev:
            if xann.suggest is None:
                # Nothing to correct with, nothing we can do
                continue
            cleantoklist[xann.start].txt = xann.suggest
            if xann.end > xann.start:
                # Annotation spans many tokens
                # "Okkur börnunum langar í fisk"
                # "Leita að kílómeter af féinu" → leita að kílómetri af fénu → leita að kílómetra af fénu
                # "dást af þeim" → "dást að þeim"
                # Single-token annotations for this span have already been handled
                # Only case is one ann, many toks in toklist
                # Give the first token the correct value
                # Delete the other tokens
                del cleantoklist[xann.start + 1 : xann.end + 1]
        txt = detokenize(cleantoklist, normalize=True)
        if print_annotations and not print_all:
            txt = txt + "\n" + "\n".join(str(ann) for ann in a)
        accumul.append(txt)

    accumstr = "\n".join(accumul)

    # !!!! SEE HERE !!!! WARNING !!!
    # For backwards compatibility with a bunch of tests we add a beginning-of-sentence token
    # and an end-of-sentence token to the token list. Both tokens are in fact invalid.
    # They are invalid because they are not a part of the results and should not be used.
    # Some tests verify that there are indeed no errors on these tokens.
    alltoks = [CorrectToken(-10, "", -10)] + alltoks + [CorrectToken(-20, "", -20)]
    return accumstr, alltoks
