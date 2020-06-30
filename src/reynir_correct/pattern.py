"""

    Greynir: Natural language processing for Icelandic

    Sentence tree pattern matching module

    Copyright (C) 2020 Miðeind ehf.

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


    This module contains the PatternMatcher class, which implements
    functionality to look for questionable grammatical patterns in parse
    trees. These are typically not grammatical errors per se, but rather
    incorrect usage, e.g., attaching a wrong/uncommon preposition
    to a verb. The canonical example is "Ég leitaði af kettinum"
    ("I searched off the cat") which should very likely have been
    "Ég leitaði að kettinum" ("I searched for the cat"). The first form
    is grammatically correct but the meaning is borderline nonsensical.

"""

from typing import List, Tuple, Set, Callable, Dict, Optional, Union
from threading import Lock
from functools import partial

from reynir import _Sentence
from reynir.simpletree import SimpleTree
from reynir.settings import VerbObjects

from .annotation import Annotation


# The types involved in pattern processing
CheckFunction = Callable[[SimpleTree], bool]
ContextType = Dict[str, Union[str, CheckFunction]]
AnnotationFunction = Callable[["PatternMatcher", SimpleTree], None]
PatternTuple = Tuple[str, str, AnnotationFunction, Optional[ContextType]]


class PatternMatcher:

    """ Class to match parse trees with patterns to find probable usage errors """

    # The patterns to be matched are created when the
    # first class instance is initialized.

    # Each entry in the patterns list is a tuple of four values:

    # * Trigger lemma, which must be present in the sentence for the pattern
    #   to be applied. This is an optimization only, to save unnecessary matching.
    #   If the trigger is falsy (None, ""), it is not applied and the sentence will be
    #   checked regardless of its content.
    # * Match pattern expression, to be passed to match_pattern()
    # * Annotation function, called for each match
    # * Context dictionary to be passed to match_pattern()

    PATTERNS = []  # type: List[PatternTuple]

    _LOCK = Lock()

    ctx_af = None  # type: ContextType
    ctx_verb_01 = None  # type: ContextType
    ctx_verb_02 = None  # type: ContextType


    def __init__(self, ann: List[Annotation], sent: _Sentence) -> None:
        # Annotation list
        self._ann = ann
        # The original sentence object
        self._sent = sent
        # Token list
        self._tokens = sent.tokens
        # Terminal node list
        self._terminal_nodes = sent.terminal_nodes
        # Avoid race conditions in multi-threaded scenarios
        with self._LOCK:
            if not self.PATTERNS:
                # First instance: create the class-wide pattern list
                self.create_patterns()

    def wrong_preposition_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { \"af\" }")
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        text = "'{0} af' á sennilega að vera '{0} að'".format(vp.tidy_text)
        detail = (
            "Sögnin '{0}' tekur yfirleitt með sér "
            "forsetninguna 'að', ekki 'af'.".format(vp.tidy_text)
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P001",
                text=text,
                detail=detail,
                suggest=suggest,
            )
        )

    def wrong_preposition_vitni_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'verða vitni af' á sennilega að vera 'verða vitni að'"
        detail = (
            "Í samhenginu 'verða vitni að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        if match.tidy_text.count(" af ") == 1:
            # Only one way to substitute af -> að: do it
            suggest = match.tidy_text.replace(" af ", " að ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P001",
                text=text,
                detail=detail,
                suggest=suggest,
            )
        )

    def wrong_verb_use(self, match: SimpleTree, correct_verb: str) -> None:
        """ Annotate wrong verbs being used with nouns,
            for instance 'byði hnekki' where the verb should
            be 'bíða' -> 'biði hnekki' instead of 'bjóða' """
        vp = match.first_match("VP > { %verb }", self.ctx_verb_01)
        verb = next(ch for ch in vp.children if ch.tcat == "so").own_lemma_mm
        np = match.first_match("NP >> { %noun }", self.ctx_verb_01)
        start, end = min(vp.span[0], np.span[0]), max(vp.span[1], np.span[1])
        # noun = next(ch for ch in np.leaves if ch.tcat == "no").own_lemma
        text = (
            "Hér á líklega að vera sögnin '{0}' í stað '{1}'."
            .format(correct_verb, verb)
        )
        detail = (
            "Í samhenginu '{0}' er rétt að nota sögnina '{1}' í stað '{2}'."
            .format(match.tidy_text, correct_verb, verb)
        )
        suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P002",
                text=text,
                detail=detail,
                suggest=suggest,
            )
        )

    @classmethod
    def create_patterns(cls) -> None:
        """ Initialize the list of patterns and handling functions """
        p = cls.PATTERNS

        # Access the dictionary of verb+preposition attachment errors
        # from the settings (actually from the reynir settings),
        # read from config/Verbs.conf
        prep_errors = VerbObjects.PREPOSITIONS_ERRORS
        # Build a set of verbs with common af/að errors
        verbs_af = set()
        for verb, d in prep_errors.items():
            if "_" in verb:
                # At this point, we're not interested in composites ('birgja_sig' etc.)
                continue
            spec = d.get("af")
            if spec is not None:
                # Decipher the error specification from config/Verbs.conf
                a = spec.split(", ")
                if len(a) >= 2 and a[0] == "PP":
                    a = a[1].split()
                    if len(a) == 2 and a[0] == "/að" and a[1] == "þgf":
                        # Found a verb that has a correction from 'af' to 'að'
                        verbs_af.add(verb)
        if verbs_af:
            # Create matching patterns with a context that catches the af/að verbs.

            # The following context dictionary defines a resolver function for the
            # '%verb' macro. This function returns True if the potentially matching
            # tree node refers to a lemma that is one of the að/af verbs, and that verb
            # does not have arguments (i.e. it doesn't have 1 or 2 arguments; it may
            # have none or 0). An example is a 'so_0_gm_fh_nt' terminal matching the
            # verb 'leita'.

            # Note that we use the own_lemma_mm property instead of own_lemma. This
            # means that the lambda condition matches middle voice stem forms,
            # such as 'dást' instead of 'dá'.
            cls.ctx_af = {
                "verb": lambda tree: (
                    tree.own_lemma_mm in verbs_af and not (set(tree.variants) & {"1", "2"})
                )
            }
            # Catch sentences such as 'Jón leitaði af kettinum'
            p.append((
                "af",  # Trigger lemma for this pattern
                "VP > { VP >> { %verb } PP >> { P > { \"af\" } } }",
                cls.wrong_preposition_af,
                cls.ctx_af
            ))
            # Catch sentences such as 'Vissulega er hægt að brosa af þessu',
            # 'Friðgeir var leitandi af kettinum í allan dag'
            p.append((
                "af",  # Trigger lemma for this pattern
                ". > { (NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { \"af\" } } }",
                cls.wrong_preposition_af,
                cls.ctx_af
            ))

            # Catch "Þetta er mesta vitleysa sem ég hef orðið vitni af"
            p.append((
                "vitni",  # Trigger lemma for this pattern
                "VP > { VP >> [ .* ('verða' | 'vera') .* \"vitni\" ] "
                "ADVP > \"af\" }",
                cls.wrong_preposition_vitni_af,
                None
            ))
            # Catch "Hún varð vitni af því þegar kúturinn sprakk"
            p.append((
                "vitni",  # Trigger lemma for this pattern
                "VP > { VP > [ .* ('verða' | 'vera') .* "
                "NP-PRD > { \"vitni\" PP > { P > { \"af\" } } } ] } ",
                cls.wrong_preposition_vitni_af,
                None
            ))

        # Verbs used wrongly with particular nouns
        def wrong_noun(nouns: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combinations
                of verbs and their noun objects """
            lemma = tree.own_lemma
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_01 = {
            "ósigur_þf", "hnekkir_þf", "álitshnekkir_þf",
            "afhroð_þf", "bani_þf", "færi_ef", "boð_ef", "átekt_ef"
        }
        cls.ctx_verb_01 = {"verb": "'bjóða'", "noun": partial(wrong_noun, NOUNS_01) }
        p.append((
            "bjóða",  # Trigger lemma for this pattern
            "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
            lambda self, match: self.wrong_verb_use(match, "bíða"),
            cls.ctx_verb_01
        ))

        NOUNS_02 = {
            "haus_þf", "þvottur_þf"
        }
        cls.ctx_verb_02 = {"verb": "'hegna'", "noun": partial(wrong_noun, NOUNS_02) }
        p.append((
            "hegna",  # Trigger lemma for this pattern
            "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
            lambda self, match: self.wrong_verb_use(match, "hengja"),
            cls.ctx_verb_02
        ))

    def go(self) -> None:
        """ Apply the patterns to the sentence """
        tree = None if self._sent is None else self._sent.tree
        if tree is None:
            # No tree: nothing to do
            return
        # Make a set of the lemmas in the sentence
        # (Note: these are ordinary lemmas, not middle voice lemmas, so be careful
        # not to use middle voice lemmas as trigger words)
        if not self._sent.lemmas:
            return
        lemmas = set(lemma.replace("-", "") for lemma in self._sent.lemmas)
        for trigger, pattern, func, context in self.PATTERNS:
            # We only do the expensive pattern matching if the trigger lemma
            # for a pattern rule (if given) is actually found in the sentence
            if trigger and trigger in lemmas:
                for match in tree.all_matches(pattern, context):
                    # Call the annotation function for this match
                    func(self, match)

