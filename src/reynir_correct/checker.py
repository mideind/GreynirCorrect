"""

    Reynir: Natural language processing for Icelandic

    Spelling and grammar checking module

    Copyright(C) 2018 Miðeind ehf.

        This program is free software: you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


    This module exposes functions to check spelling and grammar for
    text strings.

"""

from .errtokenizer import tokenize as tokenize_and_correct
from reynir import Reynir, correct_spaces
from reynir.fastparser import ParseForestNavigator


class Annotation:

    """ An annotation of a span of a token list for a sentence """

    def __init__(self, start, end, text, code):
        assert isinstance(start, int)
        self._start = start
        assert isinstance(end, int)
        self._end = end
        self._text = text
        self._code = code

    @property
    def start(self):
        """ The index of the first token to which the annotation applies """
        return self._start

    @property
    def end(self):
        """ The index of the last token to which the annotation applies """
        return self._end

    @property
    def text(self):
        """ A description of the annotation """
        return self._text

    @property
    def code(self):
        """ A code for the annotation type, usually an error or warning code """
        return self._code


class ErrorFinder(ParseForestNavigator):

    """ Utility class to find nonterminals in parse trees that are
        tagged as errors in the grammar """

    def __init__(self, ann, toklist):
        super().__init__(visit_all=True)
        # Annotation list
        self._ann = ann
        self._toklist = toklist

    def _visit_nonterminal(self, level, node):
        """ Entering a nonterminal node """
        if node.is_interior or node.nonterminal.is_optional:
            pass
        elif node.nonterminal.has_tag("error"):
            # This node has a nonterminal that is tagged with $tag(error)
            # in the grammar file (Reynir.grammar)
            txt = correct_spaces(
                " ".join(t.txt for t in self._toklist[node.start:node.end] if t.txt)
            )
            self._ann.append(
                # E002: Probable grammatical error
                # !!! TODO: add further info and guidance to the text field
                Annotation(
                    start=node.start,
                    end=node.end-1,
                    text="'{0}' er líklega málfræðilega rangt (regla '{1}')"
                        .format(txt, node.nonterminal.name),
                    code="E002"
                )
            )
        return None


class ReynirCorrect(Reynir):

    """ Parser augmented with error correction """

    def __init__(self):
        super().__init__()

    def tokenize(self, text):
        """ Use the correcting tokenizer instead of the normal one """
        return tokenize_and_correct(text)


def check_single(sentence):
    """ Check and annotate a single sentence, given in plain text """
    r = ReynirCorrect()
    sent = r.parse_single(sentence)
    # Generate annotations
    ann = []
    # First, add token-level annotations
    for ix, t in enumerate(sent.tokens):
        if t.error_code:
            ann.append(
                Annotation(
                    start=ix,
                    end=ix + t.error_span - 1,
                    text=t.error_description,
                    code=t.error_code
                )
            )
    # Then: if the sentence couldn't be parsed,
    # put an annotation on it as a whole
    if sent.deep_tree is None:
        ann.append(
            # E001: Unable to parse sentence
            Annotation(
                start=0,
                end=len(sent.tokens)-1,
                text="Ekki tókst að þátta setninguna",
                code="E001"
            )
        )
    else:
        # Successfully parsed:
        # Add error rules from the grammar
        ErrorFinder(ann, sent.tokens).go(sent.deep_tree)
    # Sort the annotations by their start token index,
    # and then by decreasing span length
    ann.sort(key=lambda a: (a.start, -a.end))
    # Add an attribute to the returned sent object
    sent.annotations = ann
    return sent

