"""

    Reynir: Natural language processing for Icelandic

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

"""

from .errtokenizer import tokenize as tokenize_and_correct
from reynir import Reynir, correct_spaces
from reynir.fastparser import ParseForestNavigator


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
                dict(
                    start=node.start,
                    end=node.end-1,
                    text="'{0}' er líklega málfræðilega rangt (regla '{1}')"
                        .format(txt, node.nonterminal.name),  # !!! TODO
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
                dict(
                    start=ix,
                    end=ix + t.error_span - 1,
                    text=t.error_description,
                    code=t.error_code
                )
            )
    # Then: if the sentence couldn't be parsed,
    # put an annotation on it as a whole
    if sent.tree is None:
        ann.append(
            dict(
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
    # Add an attribute to the returned sent object
    sent.annotations = ann
    return sent

