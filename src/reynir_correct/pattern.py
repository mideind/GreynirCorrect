"""

    Greynir: Natural language processing for Icelandic

    Sentence tree pattern matching module

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


    This module contains the PatternMatcher class, which implements
    functionality to look for questionable grammatical patterns in parse
    trees. These are typically not grammatical errors per se, but rather
    incorrect usage, e.g., attaching a wrong/uncommon preposition
    to a verb. The canonical example is "Ég leitaði af kettinum"
    ("I searched off the cat") which should very likely have been
    "Ég leitaði að kettinum" ("I searched for the cat"). The first form
    is grammatically correct but the meaning is borderline nonsensical.

"""

from typing import (
    Iterable,
    List,
    Mapping,
    Tuple,
    Set,
    FrozenSet,
    Callable,
    Dict,
    Optional,
    Union,
    cast,
)

from threading import Lock
from functools import partial
import os
import json

from islenska import Bin

from reynir import Sentence, NounPhrase
from reynir.simpletree import SimpleTree
from reynir.verbframe import VerbErrors
from reynir.matcher import ContextDict
from reynir.bintokenizer import ALL_CASES

from .annotation import Annotation


# The types involved in pattern processing
AnnotationFunction = Callable[["PatternMatcher", SimpleTree], None]
PatternTuple = Tuple[
    Union[str, Set[str], FrozenSet[str]], str, AnnotationFunction, Optional[ContextDict]
]

BIN = Bin()

# Variants not needed for lookup
SKIPVARS = frozenset(("op", "subj", "0", "1", "2"))


class IcelandicPlaces:

    """ Wraps a dictionary of Icelandic place names with their
        associated prepositions """

    # This is not strictly accurate as the correct prepositions
    # are based on convention, not rational rules. :/
    _SUFFIX2PREP: Mapping[str, str] = {
        "vík": "í",
        # "fjörður": "á",  # Skip this since 'í *firði' is also common
        "eyri": "á",
        "vogur": "í",
        "brekka": "í",
        "staðir": "á",
        # "höfn": "á",  # Skip this since 'í *höfn' is also common
        "eyjar": "í",
        "ey": "í",
        "nes": "á",
        "borg": "í",
    }
    _SUFFIXES = tuple(_SUFFIX2PREP.keys())

    ICELOC_PREP: Optional[Dict[str, str]] = None
    ICELOC_PREP_JSONPATH = os.path.join(
        os.path.dirname(__file__), "resources", "iceloc_prep.json"
    )

    @classmethod
    def _load_json(cls) -> None:
        """ Load the place name dictionary from a JSON file into memory """
        with open(cls.ICELOC_PREP_JSONPATH, encoding="utf-8") as f:
            cls.ICELOC_PREP = json.load(f)

    @classmethod
    def lookup_preposition(cls, place: str) -> Optional[str]:
        """ Look up the correct preposition to use with a placename,
            or None if the placename is not known """
        if cls.ICELOC_PREP is None:
            cls._load_json()
        assert cls.ICELOC_PREP is not None
        prep = cls.ICELOC_PREP.get(place)
        if prep is None and place.endswith(cls._SUFFIXES):
            # Not explicitly known: try to associate a preposition by suffix
            for suffix, prep in cls._SUFFIX2PREP.items():
                if place.endswith(suffix):
                    break
            else:
                # Should not get here
                assert False
                prep = None
        return prep

    @classmethod
    def includes(cls, place: str) -> bool:
        """ Return True if the given place is found in the dictionary """
        if cls.ICELOC_PREP is None:
            cls._load_json()
        assert cls.ICELOC_PREP is not None
        return place in cls.ICELOC_PREP


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

    PATTERNS: List[PatternTuple] = []

    _LOCK = Lock()

    ctx_af: ContextDict = cast(ContextDict, None)
    ctx_að: ContextDict = cast(ContextDict, None)
    ctx_noun_af: ContextDict = cast(ContextDict, None)
    ctx_noun_af_obj: ContextDict = cast(ContextDict, None)
    ctx_verb_01: ContextDict = cast(ContextDict, None)
    ctx_verb_02: ContextDict = cast(ContextDict, None)
    ctx_noun_að: ContextDict = cast(ContextDict, None)
    ctx_place_names: ContextDict = cast(ContextDict, None)
    ctx_uncertain_verbs: ContextDict = cast(ContextDict, None)
    ctx_confident_verbs: ContextDict = cast(ContextDict, None)
    ctx_dir_loc: ContextDict = cast(ContextDict, None)

    def __init__(self, ann: List[Annotation], sent: Sentence) -> None:
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

    def get_wordform(self, lemma: str, cat: str, variants: Iterable[str]) -> str:
        """ Get correct wordform from BinPackage, 
            given a set of variants """
        realvars: Union[Set[str], Iterable[str]]
        if cat == "so":
            # Get rid of irrelevant variants for verbs
            realvars = set(variants) - SKIPVARS
            if "lh" not in realvars:
                realvars -= ALL_CASES
        else:
            realvars = variants
        wordforms = BIN.lookup_variants(lemma, cat, tuple(realvars), lemma=lemma)
        if not wordforms:
            return ""
        # Can be many possible word forms; we want the first one in most cases
        return wordforms[0].bmynd

    def wrong_preposition_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        # Calculate the start and end token indices, spanning both phrases
        assert vp is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        text = "'{0} af' á sennilega að vera '{0} að'".format(vp.tidy_text)
        detail = (
            "Sögnin '{0}' tekur yfirleitt með sér "
            "forsetninguna 'að', ekki 'af'.".format(vp.tidy_text)
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_að)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        # Calculate the start and end token indices, spanning both phrases
        assert vp is not None
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Sögnin '{0}' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'.".format(vp.tidy_text)
        )
        suggest = match.substituted_text(pp_að, "af")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_spyrja_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'spyrja' }")
        assert vp is not None
        # Find the attached prepositional/adverbial phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp_af.span[0]), max(vp.span[1], pp_af.span[1])
        text = "'{0} af' á sennilega að vera '{0} að'".format(vp.tidy_text)
        detail = (
            "Í samhenginu 'að spyrja að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_vitni_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match(". >> { 'vitni' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'verða vitni af' á sennilega að vera 'verða vitni að'"
        detail = (
            "Í samhenginu 'verða vitni að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_grin_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'gera' }")
        np = match.first_match("NP >> { 'grín' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        if vp is None:
            start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        else:
            start, end = (
                min(vp.span[0], np.span[0], pp.span[0]),
                max(vp.span[1], np.span[1], pp.span[1]),
            )
        text = "'gera grín af' á sennilega að vera 'gera grín að'"
        detail = (
            "Í samhenginu 'gera grín að e-u' er notuð " "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_leida_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'leiða' }")
        np = match.first_match("NP >> { ( 'líkur'|'rök'|'rak' ) }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], np.span[0], pp.span[0]),
            max(vp.span[1], np.span[1], pp.span[1]),
        )
        text = "'leiða {0} af' á sennilega að vera 'leiða {0} að'".format(np.tidy_text)
        detail = (
            "Í samhenginu 'leiða {0} af e-u' er notuð "
            "forsetningin 'að', ekki 'af'.".format(np.tidy_text)
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_marka_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'marka' }")
        if vp is None:
            vp = match.first_match("NP > { 'markaður' }")
        np = match.first_match("NP >> { ( 'upphaf'|'upphafinn' ) }")
        if np is None:
            np = match.first_match("VP > { 'upphefja' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], np.span[0], pp.span[0]),
            max(vp.span[1], np.span[1], pp.span[1]),
        )
        text = "'marka upphaf af' á sennilega að vera 'marka upphaf að'"
        detail = (
            "Í samhenginu 'marka upphaf að e-u' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_leggja_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'leggja' }")
        if vp is None:
            vp = match.first_match("VP >> { 'leggja' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        # Find the offending nominal phrase
        np = match.first_match("NP >> { \"velli\" }")
        assert vp is not None
        assert pp is not None
        assert np is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], pp.span[0], np.span[0]),
            max(vp.span[-1], pp.span[-1], np.span[-1]),
        )
        text = "'leggja af velli' á sennilega að vera 'leggja að velli'"
        detail = (
            "Í samhenginu 'leggja einhvern að velli' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_utan_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending adverbial phrase
        advp = match.first_match("ADVP > { 'utan' }")
        if advp is None:
            advp = match.first_match("ADVP >> { 'utan' }")
        # Find the attached prepositional phrase
        pp = match.first_match('ADVP > { "af" }')
        assert advp is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(advp.span[0], pp.span[0]), max(advp.span[1], pp.span[1])
        text = "'utan af' á sennilega að vera 'utan að'"
        detail = (
            "Í samhenginu 'kunna eitthvað utan að' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_uppvis_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP >> { 'verða' }")
        # Find the attached nominal phrase
        np = match.first_match("NP >> { 'uppvís' }")
        # Find the attached prepositional phrase
        pp = match.first_match('PP > { "af" }')
        if pp is None:
            pp = match.first_match("ADVP > { 'af' }")
        assert vp is not None
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], np.span[0], pp.span[0]),
            max(vp.span[1], np.span[1], pp.span[1]),
        )
        text = "'uppvís af' á sennilega að vera 'uppvís að'"
        detail = (
            "Í samhenginu 'verða uppvís að einhverju' er notuð "
            "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_verða_af(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'verða' }")
        if vp is None:
            vp = match.first_match("VP >> { 'verða' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > 'af' ")
        if pp is None:
            pp = match.first_match("ADVP > 'af' ")
        # Find the attached nominal phrase
        np = match.first_match("NP > 'ósk' ")
        if np is None:
            np = match.first_match("NP >> 'ósk' ")
        assert vp is not None
        assert pp is not None
        assert np is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], pp.span[0], np.span[0]),
            max(pp.span[1], pp.span[1], np.span[1]),
        )
        text = "'af ósk' á sennilega að vera 'að ósk'"
        detail = (
            "Í samhenginu 'að verða að ósk' er notuð " "forsetningin 'að', ekki 'af'."
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_ahyggja_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        np = match.first_match("NP > { 'áhyggja' }")
        pp = match.first_match("PP > { \"að\" }")
        assert np is not None
        assert pp is not None
        start, end = min(np.span[0], pp.span[0]), max(np.span[-1], pp.span[-1])
        text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = (
            "Í samhenginu 'hafa áhyggjur af e-u' er notuð "
            "forsetningin 'af', ekki 'að'."
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
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_hluti_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'hluti að' á sennilega að vera 'hluti af'"
        detail = "Í samhenginu 'hluti af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að_mörkum(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending prepositional phrase
        pp = match.first_match('PP > { "að" "mörkum" }')
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp.span
        suggest = match.substituted_text(pp_að, "af")
        text = f"'{pp.tidy_text}' á sennilega að vera '{suggest}'"
        detail = (
            "Í samhenginu 'leggja e-ð af mörkum' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að_leiða(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        pp = match.first_match("P > { 'að' }")
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        suggest = match.substituted_text(pp_að, "af")
        text = f"'{match.tidy_text}' á sennilega að vera '{suggest}'"
        detail = (
            "Í samhenginu 'láta gott af sér leiða' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heiður_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'heiður' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert np is not None
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        suggest = match.substituted_text(pp_að, "af")
        text = f"'{match.tidy_text}' á sennilega að vera '{suggest}'"
        detail = (
            "Í samhenginu 'fá/hljóta heiðurinn af' er notuð "
            "forsetningin 'af', ekki 'að'."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_eiga_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verb phrase
        vp = match.first_match("VP > { 'eiga' }")
        if vp is None:
            vp = match.first_match("VP >> { 'eiga' }")
        # Find the nominal object
        np = match.first_match("( NP|ADVP )")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        if pp is None:
            pp = match.first_match('PP > { "að" }')
        assert vp is not None
        assert np is not None
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = (
            min(vp.span[0], np.span[0], pp.span[0]),
            max(vp.span[1], np.span[1], pp.span[1]),
        )
        suggest = match.substituted_text(pp_að, "af")
        text = f"'{match.tidy_text}' á sennilega að vera '{suggest}'"
        detail = (
            f"Orðasambandið '{match.tidy_text}' tekur yfirleitt með sér "
            f"forsetninguna 'af', ekki 'að'."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_vera_til_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        start, end = match.span
        text = "'til að' á sennilega að vera 'til af'"
        detail = (
            "Orðasambandið 'vera mikið/lítið til af e-u' innifelur "
            "yfirleitt forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_gagn_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        start, end = match.span
        text = "'gagn að' á sennilega að vera 'gagn af'"
        detail = (
            "Orðasambandið 'að hafa gagn af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_frettir_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending preposition
        pp = match.first_match("(P | ADVP) > { 'að' }")
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp.span[0], pp.span[1]
        text = "'að' á sennilega að vera 'af'"
        detail = (
            "Orðasambandið 'fréttir berast af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_stafa_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'stafa' }")
        assert vp is not None
        start, end = match.span
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(
                vp.tidy_text, vp.tidy_text.replace(" að ", " af ")
            )
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Orðasambandið 'að stafa af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_ólétt_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'óléttur' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = (
            "Orðasambandið 'að vera ólétt/ur af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heyra_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'heyra' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert vp is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(vp.span[0], pp.span[0]), max(vp.span[1], pp.span[1])
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(
                vp.tidy_text, vp.tidy_text.replace(" að ", " af ")
            )
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Orðasambandið 'að heyra af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_hafa_gaman_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'gaman' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert np is not None
        assert pp is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "'gaman að' á sennilega að vera 'gaman af'"
        detail = (
            "Orðasambandið 'að hafa gaman af e-u' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heillaður_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Calculate the start and end token indices, spanning both phrases
        start, end = match.span
        text = "'heillaður að' á sennilega að vera 'heillaður af'"
        detail = (
            "Í samhenginu 'heillaður af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_valinn_að(self, match: SimpleTree) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        vp = match.first_match("VP > { 'velja' }")
        if vp is None:
            vp = match.first_match("NP > { 'valinn' }")
        assert vp is not None
        start, end = match.span
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(vp.tidy_text, vp.tidy_text.replace(" að ", " af "))
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = (
            "Orðasambandið 'að vera valin/n af e-m' tekur yfirleitt með sér "
            "forsetninguna 'af', ekki 'að'."
        )
        if match.tidy_text.count(" að ") == 1:
            # Only one way to substitute að -> af: do it
            suggest = match.tidy_text.replace(" að ", " af ")
        else:
            # !!! TODO: More intelligent substitution to create a suggestion
            suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_að_use(self, match: SimpleTree, context: ContextDict) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending noun
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        assert np is not None
        assert pp is not None
        pp_að = pp.first_match('"að"')
        assert pp_að is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        suggest = match.substituted_text(pp_að, "af")
        text = "Hér á líklega að vera forsetningin 'af' í stað 'að'."
        detail = (
            f"Í samhenginu '{match.tidy_text}' er rétt að nota "
            f"forsetninguna 'af' í stað 'að'."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def check_pp_with_place(self, match: SimpleTree) -> None:
        """ Check whether the correct preposition is being used with a place name """
        place = match.NP.lemma
        correct_preposition = IcelandicPlaces.lookup_preposition(place)
        if correct_preposition is None:
            # This is not a known or likely place name
            return
        preposition = match.P.lemma
        if correct_preposition == preposition:
            # Correct: return
            return
        start, end = match.span
        suggest = match.substituted_text(match.P, correct_preposition)
        text = "Rétt er að rita '{0}'".format(suggest)
        detail = (
            "Ýmist eru notaðar forsetningarnar 'í' eða 'á' með nöfnum "
            "staða, bæja og borga. Í tilviki {0:ef} er notuð forsetningin '{1}'.".format(
                NounPhrase(place), correct_preposition
            )
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PLACE_PP",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_noun_with_verb(self, match: SimpleTree) -> None:
        """ Wrong noun used with a verb, for instance
            'bjóða e-m birginn' instead of 'byrginn' """
        # TODO: This code is provisional, intended as a placeholder for similar cases
        start, end = match.span
        text = "Mælt er með að rita 'bjóða e-m byrginn' í stað 'birginn'."
        detail = text
        tidy_text = match.tidy_text
        suggest = tidy_text.replace("birginn", "byrginn", 1)
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_NOUN_WITH_VERB",
                text=text,
                detail=detail,
                original="birginn",
                suggest=suggest,
            )
        )

    def wrong_verb_use(
        self, match: SimpleTree, correct_verb: str, context: ContextDict,
    ) -> None:
        """ Annotate wrong verbs being used with nouns,
            for instance 'byði hnekki' where the verb should
            be 'bíða' -> 'biði hnekki' instead of 'bjóða' """
        vp = match.first_match("VP > { %verb }", context)
        assert vp is not None
        verb = next(ch for ch in vp.children if ch.tcat == "so").own_lemma_mm
        np = match.first_match("NP >> { %noun }", context)
        assert np is not None
        start, end = min(vp.span[0], np.span[0]), max(vp.span[1], np.span[1])
        # noun = next(ch for ch in np.leaves if ch.tcat == "no").own_lemma
        text = "Hér á líklega að vera sögnin '{0}' í stað '{1}'.".format(
            correct_verb, verb
        )
        detail = "Í samhenginu '{0}' er rétt að nota sögnina '{1}' í stað '{2}'.".format(
            match.tidy_text, correct_verb, verb
        )
        suggest = ""
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_VERB_USE",
                text=text,
                detail=detail,
                original=verb,
                suggest=suggest,
            )
        )

    def wrong_af_use(self, match: SimpleTree, context: ContextDict) -> None:
        """ Handle a match of a suspect preposition pattern """
        # Find the offending nominal phrase
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'af' }")
        assert np is not None
        assert pp is not None
        pp_af = pp.first_match('"af"')
        assert pp_af is not None
        # Calculate the start and end token indices, spanning both phrases
        start, end = min(np.span[0], pp.span[0]), max(np.span[1], pp.span[1])
        text = "Hér á líklega að vera forsetningin 'að' í stað 'af'."
        detail = "Í samhenginu '{0}' er rétt að nota forsetninguna 'að' í stað 'af'.".format(
            match.tidy_text
        )
        suggest = match.substituted_text(pp_af, "að")
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    def vera_að(self, match: SimpleTree) -> None:
        """ 'vera að' in front of verb is unneccessary """
        # TODO don't match verbs that allow 'vera að'
        so = match.first_match("VP >> 'vera'")
        if so is None:
            return
        so = so.first_match("so")
        if so is None:
            return
        subj = match.first_match("NP-SUBJ")
        if subj is None:
            return
        works = False
        # TODO For now, only correct if the subject is a 1st or 2nd person pronoun or person name
        allowed: FrozenSet[str] = frozenset(["PERSON"])
        for x in subj.children:
            if x.is_terminal and x.kind != "PUNCTUATION":
                if x.kind in allowed:
                    works = True
                if x.cat and x.cat == "pfn":
                    if {"p1", "p2"} & frozenset(x.all_variants):
                        works = True
        if not works:
            return
        # nhm = match.first_match("TO > nhm").first_match("nhm")
        start, _ = so.span
        realso = match.first_match("IP-INF >> VP")
        if realso is None:
            return
        realso = realso.first_match("so_nh")
        if realso is None:
            return
        _, end = realso.span
        suggest = self.get_wordform(realso.lemma, realso.cat, so.all_variants)
        if not suggest:
            return
        text = (
            f"Mælt er með að sleppa '{so.tidy_text} að' og beygja frekar sögnina "
            f"'{realso.lemma}' svo hún verði '{suggest}'."
        )
        detail = (
            f"Skýrara er að nota beina ræðu ('Ég skil þetta ekki') fremur en "
            "svokallað dvalarhorf ('Ég er ekki að skilja þetta')."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_VeraAð",
                text=text,
                detail=detail,
                original="vera að",
            )
        )

    def dir_loc(self, match: SimpleTree) -> None:
        adv = match.first_match("( 'inn'|'út'|'upp' )")
        if adv is None:
            return
        pp = match.first_match(
            "PP > { P > { ( 'í'|'á'|'um' ) } NP > { ( no_þgf|no_þf|pfn_þgf|pfn_þf ) } }"
        )
        if pp is None:
            return
        start, end = min(adv.span[0], pp.span[0]), max(adv.span[1], pp.span[1])
        if adv.span < pp.span:
            if pp.tidy_text.startswith(adv.tidy_text):
                narrow_match = pp.tidy_text
            else:
                narrow_match = adv.tidy_text + " " + pp.tidy_text
        elif pp.span < adv.span:
            narrow_match = pp.tidy_text + " " + adv.tidy_text
        else:
            # Should not happen
            narrow_match = ""
            assert False
        correction = adv.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{adv.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            narrow_match, correction, adv.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=adv.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_comp(self, match: SimpleTree) -> None:
        p = match.first_match("P > ( 'inná'|'inní'|'útá'|'útí'|'uppá'|'uppí' ) ")
        if p is None:
            return
        start, end = match.span
        match_text = match.tidy_text
        tidy_text = p.tidy_text
        correction = tidy_text[:-1] + "i" + " " + tidy_text[-1]
        text = f"Hér á líklega að vera '{correction}' í stað '{tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota '{1}' í stað '{2}'.".format(
            match_text, correction, tidy_text
        )
        suggest = match_text.replace(tidy_text, correction)
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=match_text,
                suggest=suggest,
            )
        )

    def dir_loc_ut_um(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { ( 'út'|'útum' ) }")
        if advp is None:
            advp = match.first_match("( 'út'|'útum' )")
        pp = match.first_match("PP > { 'um' }")
        if pp is None:
            pp = match.first_match("NP > { 'um' }")
        if advp is None:
            return
        if pp is None:
            return
        start, end = min(advp.span[0], pp.span[0]), max(advp.span[1], pp.span[1])
        if advp.tidy_text == "útum":
            correction = "úti um"
        else:
            correction = advp.tidy_text + "i"
        if pp.tidy_text.startswith(advp.tidy_text):
            context = pp.tidy_text
        else:
            if pp.span < advp.span:
                context = pp.tidy_text + " " + advp.tidy_text
            else:
                context = advp.tidy_text + " " + pp.tidy_text
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            context, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_standa(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { 'upp' }")
        if advp is None:
            return
        start, end = match.span
        correction = advp.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_safna(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { 'inn' }")
        assert advp is not None
        start, end = match.span
        correction = advp.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_niður(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { 'niður' }")
        assert advp is not None
        start, end = match.span
        correction = "niðri"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_búð(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { 'út' }")
        assert advp is not None
        start, end = match.span
        correction = advp.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def dir_loc_læsa(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { 'inn' }")
        assert advp is not None
        start, end = match.span
        correction = advp.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=advp.tidy_text,
                suggest=correction,
            )
        )

    def mood_sub(self, kind: str, match: SimpleTree) -> None:
        """ Subjunctive mood, present tense, is used instead of indicative 
            in conditional ("COND"), purpose ("PURP"), relative ("REL")
            or temporal ("TEMP/w") subclauses """
        vp = match.first_match("VP > so_vh")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        start, end = so.span
        # Check if so is in a different subclause
        if "þt" in so.all_variants:
            return
        variants = set(so.all_variants) - {"vh"}
        variants.add("fh")
        suggest = self.get_wordform(so.lemma, so.cat, variants)
        if suggest == so.tidy_text:
            return
        if not suggest:
            return
        text = (
            f"Hér á líklega að nota framsöguhátt sagnarinnar '{so.tidy_text}', "
            f"þ.e. '{suggest}'."
        )
        detail = ""
        sent_kind = ""
        if kind == "COND":
            sent_kind = "skilyrðissetningum á borð við 'Z' í 'X gerir Y ef Z'"
        elif kind == "PURP":
            sent_kind = "tilgangssetningum á borð við 'Z' í 'X gerir Y til þess að Z'"
        elif kind == "TEMP/w":
            sent_kind = "tíðarsetningum á borð við 'Z' í 'X gerði Y áður en Z'"
        elif kind == "REL":
            detail = f"Í tilvísunarsetningum er aðeins framsöguháttur sagna tækur."
        else:
            assert False
        if not detail:
            detail = f"Í {sent_kind} er yfirleitt notaður framsöguháttur sagna."
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_MOOD_" + kind,
                text=text,
                detail=detail,
                original=so.tidy_text,
                suggest=suggest,
            )
        )

    def mood_ind(self, kind: str, match: SimpleTree) -> None:
        """ Indicative mood is used instead of subjunctive 
            in concessive or purpose subclauses """
        vp = match.first_match("VP > so_fh")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        start, end = so.span
        # Check if so is in a different subclause
        variants = set(so.all_variants) - {"fh"}
        variants.add("vh")
        suggest = self.get_wordform(so.lemma, so.cat, variants)
        if suggest == so.tidy_text:
            return
        if not suggest:
            return
        text = (
            f"Hér er réttara að nota viðtengingarhátt "
            f"sagnarinnar '{so.lemma}', þ.e. '{suggest}'."
        )
        if kind == "ACK":
            detail = (
                "Í viðurkenningarsetningum á borð við 'Z' í dæminu "
                "'X gerði Y þrátt fyrir að Z' á sögnin að vera í "
                "viðtengingarhætti fremur en framsöguhætti."
            )
        elif kind == "PURP":
            detail = (
                "Í tilgangssetningum á borð við 'Z' í dæminu "
                "'X gerði Y til þess að Z' á sögnin að vera í "
                "viðtengingarhætti fremur en framsöguhætti."
            )
        else:
            assert False
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_MOOD_" + kind,
                text=text,
                detail=detail,
                original=so.tidy_text,
                suggest=suggest,
            )
        )

    def doubledefinite(self, match: SimpleTree) -> None:
        """ A definite noun appears with a definite pronoun,
            e.g. 'þessi maðurinn' """
        no = match.first_match("no")
        if no is None:
            return
        fn = match.first_match("fn")
        if fn is None:
            return
        fn_lemma = fn.lemma
        if fn_lemma not in {"sá", "þessi"}:
            return
        start, end = match.span
        suggest = no.lemma
        variants = set(no.all_variants)
        variants.discard("gr")
        variants.discard(no.cat)  # all_variants for no_ terminals includes the gender
        variants.add("nogr")
        v = BIN.lookup_variants(no.lemma, no.cat, tuple(variants))
        if not v:
            return
        suggest = v[0].bmynd
        text = (
            f"Hér ætti annaðhvort að sleppa '{fn.tidy_text}' eða "
            f"breyta '{no.tidy_text}' í '{suggest}'."
        )
        detail = (
            "Hér er notuð tvöföld ákveðni, þ.e. ábendingarfornafn á undan "
            "nafnorði með greini. Það er ekki í samræmi við viðtekinn málstaðal."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DOUBLE_DEFINITE",
                text=text,
                detail=detail,
                original=match.tidy_text,
                suggest=suggest,
            )
        )

    @classmethod
    def add_pattern(cls, p: PatternTuple) -> None:
        """ Validates and adds a pattern to the class global pattern list """
        _, pattern, _, ctx = p
        if "%" in pattern:
            assert ctx is not None, "Missing context for pattern with %macro"
        else:
            assert ctx is None, "Unnecessary context given for pattern with no %macro"
        cls.PATTERNS.append(p)

    @classmethod
    def create_patterns(cls) -> None:
        """ Initialize the list of patterns and handling functions """

        # Access the dictionary of verb+preposition attachment errors
        # from the settings (actually from the reynir settings),
        # read from config/Verbs.conf
        prep_errors = VerbErrors.PREPOSITIONS_ERRORS

        # Build a set of verbs with common af/að errors
        verbs_af: Set[str] = set()
        verbs_að: Set[str] = set()

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
                continue

            spec = d.get("að")
            if spec is not None:
                # Decipher the error specification from config/Verbs.conf
                a = spec.split(", ")
                if len(a) >= 2 and a[0] == "PP":
                    a = a[1].split()
                    if len(a) == 2 and a[0] == "/af" and a[1] == "þgf":
                        # Found a verb that has a correction from 'að' to 'af'
                        verbs_að.add(verb)
                continue

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
                    tree.own_lemma_mm in verbs_af
                    and not (set(tree.variants) & {"1", "2"})
                )
            }
            # Catch sentences such as 'Jón leitaði af kettinum'
            cls.add_pattern(
                (
                    verbs_af,  # Trigger lemma for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "af" } } }',
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )
            # Catch sentences such as 'Vissulega er hægt að brosa af þessu',
            # 'Friðgeir var leitandi af kettinum í allan dag'
            cls.add_pattern(
                (
                    verbs_af,  # Trigger lemma for this pattern
                    '. > { (NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "af" } } }',
                    cls.wrong_preposition_af,
                    cls.ctx_af,
                )
            )
            # Catch "Það sem Jón spurði ekki af...", "Jón spyr (ekki) af því."
        #    cls.add_pattern(
        #        (
        #            "spyrja",  # Trigger lemma for this pattern
        #            "IP > { VP >> { 'spyrja' } ADVP > { 'af' } }",
        #            cls.wrong_preposition_af,
        #            cls.ctx_af,
        #        )
        #    )
            cls.add_pattern(
                (
                    "spyrja",  # Trigger lemma for this pattern
                    "VP > { VP > { 'spyrja' } ADVP > { \"af\" } }",
                    cls.wrong_preposition_spyrja_af,
                    None,
                )
            )
            # Catch "Jón spyr af því."
        #    cls.add_pattern(
        #        (
        #            "spyrja",  # Trigger lemma for this pattern
        #            "IP > { VP >> { 'spyrja' } PP > { 'af' } }",
        #            cls.wrong_preposition_af,
        #            None,
        #        )
        #    )
            # Catch "...vegna þess að dýr leita af öðrum smærri dýrum."
            cls.add_pattern(
                (
                    "leita",  # Trigger lemma for this pattern
                    "VP > { PP >> { 'leita' } PP > 'af' }",
                    cls.wrong_preposition_af,
                    None,
                )
            )

            # Catch "Þetta er mesta vitleysa sem ég hef orðið vitni af", "Hún varð vitni af því þegar kúturinn sprakk"
            cls.add_pattern(
                (
                    "vitni",  # Trigger lemma for this pattern
                    "VP > { VP > { ('verða'|'vera') } NP > { \"vitni\" } ADVP > \"af\" }",
                    cls.wrong_preposition_vitni_af,
                    None,
                )
            )
            # Catch "Hún gerði grín af því.", "Þetta er mesta vitleysa sem ég hef gert grín af.", "...og gerir grín af sjálfum sér."
            cls.add_pattern(
                (
                    "grín",  # Trigger lemma for this pattern
                    #"IP",
                    "VP > { NP > { 'grín' } ( PP|ADVP ) > { \"af\" } }",
                    cls.wrong_preposition_grin_af,
                    None,
                )
            )
            # Catch "Hann leiðir (ekki) líkur af því.", "Hann hefur aldrei leitt líkur af því."
            cls.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > { VP > { 'leiða' } NP > { ('líkur' | 'rök' | 'rak') } PP > { \"af\" } }",
                    cls.wrong_preposition_leida_af,
                    None,
                )
            )
            # Catch "Tíminn markar upphaf af því."
            cls.add_pattern(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP-OBJ > { 'upphaf' PP > { 'af' } } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar ekki upphaf af því."
            cls.add_pattern(
                (
                    frozenset(
                        ("upphafinn", "upphaf")
                    ),  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP > { ('upphafinn'|'upphaf') } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar upphaf af því."
            cls.add_pattern(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'marka' } NP-SUBJ > { 'upphaf' } } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
        #    cls.add_pattern(
        #        (
        #            "upphefja",  # Trigger lemma for this pattern
        #            "IP",
        #            cls.wrong_preposition_marka_af,
        #            None,
        #        )
        #    )
            # Catch "Það hefur ekki markað upphafið af því."
            cls.add_pattern(
                (
                    "upphefja",  # Trigger lemma for this pattern
                    "VP > { NP > { 'markaður' } VP > { 'upphefja' } PP > { 'af' } }",
                    cls.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Jón leggur hann (ekki) af velli.", "Jón hefur (ekki) lagt hann af velli."
            cls.add_pattern(
                (
                    frozenset(
                        ("völlur", "vell", "velli")
                    ),  # Trigger lemmas for this pattern
                    "VP > { VP > { 'leggja' } PP > { P > { \"af\" } NP > { \"velli\" } } }",
                    cls.wrong_preposition_leggja_af,
                    None,
                )
            )
            # Catch "Jón kann það (ekki) utan af."
            cls.add_pattern(
                (
                    "kunna",  # Trigger lemma for this pattern
                    "VP > { VP > { 'kunna' } ADVP > { 'utan' } ADVP > { 'af' } }",
                    cls.wrong_preposition_utan_af,
                    None,
                )
            )
            # Catch "Honum varð af ósk sinni."
            cls.add_pattern(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "(S-MAIN | IP) > { VP > { 'verða' } PP > { 'af' NP > { 'ósk' } } }",
                    cls.wrong_preposition_verða_af,
                    None,
                )
            )
            # Catch "...en varð ekki af ósk sinni."
            cls.add_pattern(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "IP > { VP > { VP > { 'verða' } PP > { P > { 'af' } NP > { 'ósk' } } } }",
                    cls.wrong_preposition_verða_af,
                    None,
                )
            )
            # Catch "Ég varð (ekki) uppvís af athæfinu.", "Hann hafði (ekki) orðið uppvís af því."
            cls.add_pattern(
                (
                    "uppvís",  # Trigger lemma for this pattern
                    "VP > { VP > { 'verða' } NP > { 'uppvís' } PP > { 'af' } }",
                    cls.wrong_preposition_uppvis_af,
                    None,
                )
            )

        if verbs_að:
            # Create matching patterns with a context that catches the að/af verbs.
            cls.ctx_að = {
                "verb": lambda tree: (
                    tree.own_lemma_mm in verbs_að
                    and not (set(tree.variants) & {"1", "2"})
                )
            }
            # Catch sentences such as 'Jón heillaðist að kettinum'
            cls.add_pattern(
                (
                    verbs_að,  # Trigger lemmas for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "að" } } }',
                    cls.wrong_preposition_að,
                    cls.ctx_að,
                )
            )
            # Catch sentences such as 'Vissulega er hægt að heillast að þessu'
            cls.add_pattern(
                (
                    verbs_að,  # Trigger lemma for this pattern
                    '(NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "að" } }',
                    cls.wrong_preposition_að,
                    cls.ctx_að,
                )
            )
            # Catch "Þetta er fallegasta kona sem ég hef orðið heillaður að"
            cls.add_pattern(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "VP > { VP > [ .* ('verða' | 'vera') ] NP-PRD > [ .* 'heilla' .* ADVP > { \"að\" } ] }",
                    cls.wrong_preposition_heillaður_að,
                    None,
                )
            )
            # Catch "Ég hef lengi verið heillaður að henni."
            cls.add_pattern(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "NP > { NP >> { 'heilla' } PP > { 'að' } }",
                    cls.wrong_preposition_heillaður_að,
                    None,
                )
            )
            # Catch "Ég er (ekki) hluti að heildinni.", "Við höfum öll verið hluti að heildinni."
            cls.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { NP > { 'hluti' } PP > { \"að\" } }",
                    #"VP > { VP > { 'vera' NP-PRD > { 'hluti' } } PP > { 'að' } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Þeir sögðu að ég hefði verið hluti að heildinni."
            cls.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "NP > { 'hluti' PP > { \"að\" } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Þeir sögðu að ég hefði verið hluti að heildinni."  # Two patterns to catch the same sentence due to variable parsing
            cls.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { CP >> { 'hluti' } PP > { \"að\" } }",
                    cls.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Ég hef (ekki) áhyggjur að honum.", "Ég hef áhyggjur að því að honum líði illa."
            cls.add_pattern(
                (
                    "áhyggja",  # Trigger lemma for this pattern
                    "VP > { NP > { 'áhyggja' } PP > { \"að\" } }",
                    #"VP > { VP >> { 'áhyggja' } PP > { 'að' } }",
                    cls.wrong_preposition_ahyggja_að,
                    None,
                )
            )
            # Catch "Ég hafði ekki lagt mikið að mörkum."
            cls.add_pattern(
                (
                    frozenset(("mörk", "mark")),  # Trigger lemmas for this pattern
                    "VP > { VP >> { 'leggja' } PP > { \"að\" \"mörkum\" } }",
                    cls.wrong_preposition_að_mörkum,
                    None,
                )
            )
            # Catch "Jón hefur látið gott að sér leiða."
            #cls.add_pattern(
            #    (
            #        "leiða",  # Trigger lemma for this pattern
            #        "VP > { VP > { 'láta' } PP > { P > \"að\" } VP > { 'leiða' } }",
            #        cls.wrong_preposition_að_leiða,
            #        None,
            #    )
            #)
            # Catch "Ég lét gott að mér leiða."
            cls.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > [ .* VP > { 'láta' } NP (\"að mér\"|\"að þér\"|\"að sér\") 'leiða']",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Ég lét (ekki) gott að mér leiða." (In case of different parse)
            cls.add_pattern(
                (
                    "leiður",  # Trigger lemma for this pattern
                    "VP > [ VP > [ .* 'láta' .* ] NP > [ .* \"gott\" .* ] PP > [ \"að\" NP > [ (\"mér\"|\"þér\"|\"sér\"|\"okkur\") ] \"leiða\" ] ]",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Hann lét (ekki) gott að sér leiða"
            cls.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > [ VP > [ .* 'láta' .* ] .* NP > [ .* \"gott\" .* ] PP > [ \"að\" NP > [ (\"mér\"|\"þér\"|\"sér\"|\"okkur\") ] ] VP > { 'leiða' } ]",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            cls.add_pattern(
                (
                    frozenset(("leiða", "leiður")),  # Trigger lemma for this pattern (probably a wrong parse)
                    "VP > [ .* 'láta' .* NP-OBJ > [ .* \"gott\" .* (\"að mér leiða\" | \"að sér leiða\" | \"að þér leiða\") ] ]",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            cls.add_pattern(
                (
                    frozenset(("leiða", "leiður")),  # Trigger lemma for this pattern (probably a wrong parse)
                    "VP > { IP-INF > { \"að\" \"láta\" } NP-PRD > { \"gott\" } PP > [ \"að\" ( \"mér\" | \"þér\" | \"sér\" ) \"leiða\" ] }",
                    cls.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Hún á/fær/hlýtur (ekki) heiðurinn að þessu.", "Hún hafði (ekki) fengið/hlotið heiðurinn að þessu." ÞA: Including 'eiga' here causes double annotation
            cls.add_pattern(
                (
                    "heiður",  # Trigger lemma for this pattern
                    (
                    "( "
                        "VP > [ VP-AUX? .* VP > { ( 'fá'|'hljóta' ) } .* NP-OBJ > { 'heiður' PP > { P > { 'að' } NP } } ] "
                    "| "
                        "VP > [ VP-AUX? .* VP > { ( 'fá'|'hljóta' ) } .* NP-OBJ > { 'heiður' } PP > { P > { 'að' } NP } ] "
                    ") "
                    ),
                    cls.wrong_preposition_heiður_að,
                    None,
                )
            )
            # Catch "Hún á (ekki) mikið/fullt/helling/gommu... að börnum."
            cls.add_pattern(
                (
                    "eiga",  # Trigger lemma for this pattern
                    (
                    "( "
                    "VP > [ VP-AUX? .* VP > { 'eiga' } .* NP-OBJ PP > { P > { 'að' } NP } ] "
                    "| "
                    "VP > [ VP-AUX? .* VP > { 'eiga' } .* NP-OBJ > { PP > { P > { 'að' } NP } } ] "
                    ") "
                    ),
                    cls.wrong_preposition_eiga_að,
                    None,
                )
            )
            # Catch "Hún á (ekki) lítið að börnum."
            cls.add_pattern(
                (
                    "eiga",  # Trigger lemma for this pattern
                    "VP > { VP > { 'eiga' } ADVP > { 'lítið' } PP > { P > { 'að' } NP } }",
                    cls.wrong_preposition_eiga_að,
                    None,
                )
            )
            # Catch "Það er (ekki) til mikið að þessu."
            cls.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } NP > { NP >> { 'til' } PP > { 'að' } } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Mikið er til að þessu."
            cls.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "( S|VP ) > { NP VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Ekki er mikið til að þessu."
            cls.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    cls.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Hún hefur (ekki) gagn að þessu.", "Hún hefur (ekki) haft gagn að þessu."
            cls.add_pattern(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "VP > { NP > { 'gagn' } PP > { \"að\" } }",
                    cls.wrong_preposition_gagn_að,
                    None,
                )
            )
            # Catch "Hvaða gagn hef ég að þessu?"
            cls.add_pattern(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "S > { NP > { 'gagn' } IP > { VP > { VP > { 'hafa' } PP > { 'að' } } } }",
                    cls.wrong_preposition_gagn_að,
                    None,
                )
            )
            # Catch "Fréttir bárust (ekki) að slysinu."
            cls.add_pattern(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "( IP|VP ) > { NP > { 'frétt' } VP > { PP > { P > { 'að' } } } }",
                    cls.wrong_preposition_frettir_að,
                    None,
                )
            )
            # Catch "Það bárust (ekki) fréttir að slysinu."
            cls.add_pattern(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "NP > { 'frétt' PP > { P > { 'að' } } }",
                    cls.wrong_preposition_frettir_að,
                    None,
                )
            )
            # Catch "Hætta stafar (ekki) að þessu.", "Hætta hefur (ekki) stafað að þessu."
            cls.add_pattern(
                (
                    "stafa",  # Trigger lemma for this pattern
                    "( VP|IP ) > { VP >> { 'stafa' } ( PP|ADVP ) > { 'að' } }",
                    cls.wrong_preposition_stafa_að,
                    None,
                )
            )
            # Catch "Hún er (ekki) ólétt að sínu þriðja barni.", "Hún hefur (ekki) verið ólétt að sínu þriðja barni."
            cls.add_pattern(
                (
                    "óléttur",  # Trigger lemma for this pattern
                    "VP > { NP > { 'óléttur' } PP > { 'að' } }",
                    cls.wrong_preposition_ólétt_að,
                    None,
                )
            )
            # Catch "Hún heyrði að lausa starfinu.", "Hún hefur (ekki) heyrt að lausa starfinu."
            cls.add_pattern(
                (
                    "heyra",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'heyra' } PP > { 'að' } }",
                    cls.wrong_preposition_heyra_að,
                    None,
                )
            )
            # Catch "Ég hef (ekki) gaman að henni.", "Ég hef aldrei haft gaman að henni."
            cls.add_pattern(
                (
                    "gaman",  # Trigger lemma for this pattern
                    "VP > { VP > { 'hafa' } NP > { 'gaman' } PP > { 'að' } }",
                    cls.wrong_preposition_hafa_gaman_að,
                    None,
                )
            )
            # Catch "Ég var valinn að henni.", "Ég hafði (ekki) verið valinn að henni."
            cls.add_pattern(
                (
                    "velja",  # Trigger lemma for this pattern
                    "VP > { VP > { 'velja' } PP > { 'að' } }",
                    #"NP-PRD > { NP-PRD > { 'velja' } PP > { 'að' } }",
                    cls.wrong_preposition_valinn_að,
                    None,
                )
            )
            # Catch "Ég var ekki valinn að henni.", "Þau voru sérstaklega valin að stjórninni."
            cls.add_pattern(
                (
                    "valinn",  # Trigger lemma for this pattern
                    "VP > { NP > { 'valinn' } PP > { 'að' } }",
                    cls.wrong_preposition_valinn_að,
                    None,
                )
            )

        # Verbs used wrongly with particular nouns
        def wrong_noun(nouns: FrozenSet[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combinations
                of verbs and their noun objects """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & ALL_CASES).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_01 = {
            "ósigur_þf",
            "hnekkir_þf",
            "álitshnekkir_þf",
            "afhroð_þf",
            "bani_þf",
            "færi_ef",
            "boð_ef",
            "átekt_ef",
        }
        # The macro %verb expands to "@'bjóða'" which matches terminals
        # whose corresponding token has a meaning with the 'bjóða' lemma.
        # The macro %noun is resolved by calling the function wrong_noun()
        # with the potentially matching tree node as an argument.
        cls.ctx_verb_01 = {"verb": "@'bjóða'", "noun": partial(wrong_noun, NOUNS_01)}
        cls.add_pattern(
            (
                "bjóða",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda self, match: self.wrong_verb_use(
                    match, "bíða", cls.ctx_verb_01,
                ),
                cls.ctx_verb_01,
            )
        )

        NOUNS_02 = {"haus_þgf", "þvottur_þgf"}
        cls.ctx_verb_02 = {"verb": "@'hegna'", "noun": partial(wrong_noun, NOUNS_02)}
        cls.add_pattern(
            (
                "hegna",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda self, match: self.wrong_verb_use(
                    match, "hengja", cls.ctx_verb_02,
                ),
                cls.ctx_verb_02,
            )
        )

        # 'af' incorrectly used with particular nouns
        def wrong_noun_af(nouns: FrozenSet[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combination
                with 'af' """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_AF: FrozenSet[str] = frozenset(
            ("beiðni_þgf", "siður_þgf", "tilefni_þgf", "fyrirmynd_þgf")
        )
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_af = {"noun": partial(wrong_noun_af, NOUNS_AF)}
        af_lemmas = set(n.split("_")[0] for n in NOUNS_AF)
        cls.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "PP > { P > { 'af' } NP > { %noun } }",
                lambda self, match: self.wrong_af_use(match, cls.ctx_noun_af),
                cls.ctx_noun_af,
            )
        )

        NOUNS_AF_OBJ: Set[str] = {
            "aðgangur_þf",
            "aðgangur_nf",
            "drag_nf",
            "drag_þf",
            "grunnur_nf",
            "grunnur_þf",
            "hugmynd_nf",
            "hugmynd_þf",
            "leit_nf",
            "leit_þf",
            "dauðaleit_nf",
            "dauðaleit_þf",
            "lykill_nf",
            "lykill_þf",
            "uppskrift_nf",
            "uppskrift_þf",
            # "grín_þf"
        }
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_af_obj = {"noun": partial(wrong_noun_af, NOUNS_AF_OBJ)}
        af_lemmas = set(n.split("_")[0] for n in NOUNS_AF_OBJ)
        cls.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "NP > { %noun PP > { P > { 'af' } } }",
                lambda self, match: self.wrong_af_use(match, cls.ctx_noun_af_obj),
                cls.ctx_noun_af_obj,
            )
        )
        cls.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "VP > { VP >> { %noun } PP > { P > { 'af' } } }",
                lambda self, match: self.wrong_af_use(match, cls.ctx_noun_af_obj),
                cls.ctx_noun_af_obj,
            )
        )
        cls.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "VP > { PP > { NP > %noun } PP > { 'af' } }",
                lambda self, match: self.wrong_af_use(match, cls.ctx_noun_af_obj),
                cls.ctx_noun_af_obj,
            )
        )

        def wrong_noun_að(nouns: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combination
                with 'að' """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_AÐ: Set[str] = {
            "tag_þgf",
            "togi_þgf",
            "kraftur_þgf",
            "hálfa_þgf",
            "hálfur_þgf",
        }
        # The macro %noun is resolved by calling the function wrong_noun_að()
        # with the potentially matching tree node as an argument.
        cls.ctx_noun_að = {"noun": partial(wrong_noun_að, NOUNS_AÐ)}
        að_lemmas = set(n.split("_")[0] for n in NOUNS_AÐ)
        cls.add_pattern(
            (
                að_lemmas,  # Trigger lemma for this pattern
                "PP > { P > { 'að' } NP > { %noun } }",
                lambda self, match: self.wrong_að_use(match, cls.ctx_noun_að),
                cls.ctx_noun_að,
            )
        )

        def maybe_place(tree: SimpleTree) -> bool:
            """ Context matching function for the %maybe_place macro.
                Returns True if the associated lemma is an uppercase
                word that might be a place name. """
            lemma = tree.lemma
            return lemma[0].isupper() if lemma else False

        # Check prepositions used with place names
        cls.ctx_place_names = {"maybe_place": maybe_place}
        cls.add_pattern(
            (
                frozenset(("á", "í")),  # Trigger lemmas for this pattern
                "PP > { P > ('á' | 'í') NP > %maybe_place }",
                lambda self, match: self.check_pp_with_place(match),
                cls.ctx_place_names,
            )
        )
        # Check use of 'bjóða e-m birginn' instead of 'bjóða e-m byrginn'
        # !!! TODO: This is a provisional placeholder for similar cases
        cls.add_pattern(
            (
                "birgir",  # Trigger lemma for this pattern
                "VP > [ VP > { 'bjóða' } .* NP-IOBJ .* NP-OBJ > { \"birginn\" } ]",
                cls.wrong_noun_with_verb,
                None,
            )
        )
        # Check use of "vera að" instead of a simple verb
        cls.add_pattern(
            (
                "vera",  # Trigger lemma for this pattern
                "IP > {VP > [VP > { @'vera' } (ADVP|NP-SUBJ)? IP-INF > {TO > nhm}]}",
                lambda self, match: self.vera_að(match),
                None,
            )
        )

        # Check mood in subclauses

        # Concessive clause - viðurkenningarsetning
        cls.add_pattern(
            (
                frozenset(
                    ("þrátt fyrir", "þrátt", "þó", "þótt")
                ),  # Trigger lemmas for this pattern
                "CP-ADV-ACK > { IP >> {VP > so_fh} }",
                lambda self, match: self.mood_ind("ACK", match),
                None,
            )
        )
        # Relative clause - tilvísunarsetning
        cls.add_pattern(
            (
                frozenset(("sem", "er")),  # Trigger lemmas for this pattern
                "CP-REL > { IP >> {VP > so_vh} }",
                lambda self, match: self.mood_sub("REL", match),
                None,
            )
        )
        # Temporal clause - tíðarsetning
        cls.add_pattern(
            (
                frozenset(
                    ("áður", "eftir", "þangað", "þegar")
                ),  # Trigger lemmas for this pattern
                "CP-ADV-TEMP > { IP >> {VP > so_vh} }",
                lambda self, match: self.mood_sub("TEMP/w", match),
                None,
            )
        )
        # Conditional clause - skilyrðissetning
        cls.add_pattern(
            (
                frozenset(("ef", "svo")),  # Trigger lemmas for this pattern
                "CP-ADV-COND > { IP >> {VP > so_vh} }",
                lambda self, match: self.mood_sub("COND", match),
                None,
            )
        )
        # Purpose clause - tilgangssetning
        cls.add_pattern(
            (
                frozenset(("til", "svo")),  # Trigger lemmas for this pattern
                "CP-ADV-PURP > { IP >> {VP > so_fh} }",
                lambda self, match: self.mood_ind("PURP", match),
                None,
            )
        )
        # Article errors; demonstrative pronouns and nouns with an article
        cls.add_pattern(
            (
                # TODO: The trigger lemmas originally included 'segja', is that correct?
                frozenset(("sá", "þessi")),  # Trigger lemmas for this pattern
                "NP > [.* fn .* no_gr]",
                lambda self, match: self.doubledefinite(match),
                None,
            )
        )

        # Check errors in dir4loc
        def dir4loc(verbs: Set[str], tree: SimpleTree) -> bool:
            """ Context matching function for the %noun macro in combination
                with 'að' """
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            return lemma in verbs

        VERBS: FrozenSet[str] = frozenset(("safna", "kaupa", "læsa", "geyma"))
        # The macro %verb is resolved by calling the function dir4loc()
        # with the potentially matching tree node as an argument.
        cls.ctx_dir_loc = {"verb": partial(dir4loc, VERBS)}
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP > { PP > { ADVP > { 'út' } P > { 'í' } NP > { 'búð' } } } }",
                lambda self, match: self.dir_loc(match),
                cls.ctx_dir_loc,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } ADVP > { 'saman' } PP > { ADVP > { 'inn' } P > { 'í' } NP } }",
                lambda self, match: self.dir_loc(match),
                cls.ctx_dir_loc,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP > { PP > { ADVP > { 'inn' } } } }",
                lambda self, match: self.dir_loc(match),
                cls.ctx_dir_loc,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } ADVP > { 'inn' } }",
                lambda self, match: self.dir_loc(match),
                cls.ctx_dir_loc,
            )
        )

        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* VP > { 'hafa' } .* ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "NP > [ ( no_nf|pfn_nf ) PP > [ ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > { IP >> [ .* ADVP > { 'út' } ] PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > [ .* VP > { VP > [ 'vera' ] IP >> { ADVP > [ 'út' ] } } .* PP > [ P > [ 'á' ] NP > { ( no_þgf|pfn_þgf ) } ] .* ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > [ VP > [ 'gera' ] NP > [ .* PP > { ADVP > { 'út' } P > [ 'í' ] NP } ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "útá",  # Trigger lemma for this pattern
                "PP > { P > { 'útá' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "útí",  # Trigger lemma for this pattern
                "PP > { P > { 'útí' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'inn' } PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } .* ]",
                #    "( PP|VP|IP ) > [ .* ADVP > { 'inn' } .* PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } .* ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > { IP >> [ .* ADVP > { 'inn' } ] PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "NP > { IP >> { VP > { 'vera' } ADVP > { 'inn' } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { 'verða' } ADVP > { 'inn' } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "geyma",  # Trigger lemma for this pattern
                "VP > { VP > { 'geyma' } ADVP > { 'inn' } PP }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inná",  # Trigger lemma for this pattern
                "PP > { P > { 'inná' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inní",  # Trigger lemma for this pattern
                "VP > { VP > [ .* ] NP > { PP > { P > { 'inní' } NP > { ( no_þgf|pfn_þgf ) } } } }",
                # "PP > { P > { 'inní' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > [ VP > { ( 'verða'|'vera' ) } .* ADVP > { 'inn' } PP > { P > { 'á' } } ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > [ VP > { 'vera' } .* ADVP > { 'inn' } PP > { P > { 'í' } } ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        # Catches "Ég hef upp á honum."
        cls.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ VP > { ('standa'|'hafa') } .* ADVP > { 'upp' } PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        # Catches "Það liggur í augum upp."
        cls.add_pattern(
            (
                "auga",  # Trigger lemma for this pattern
                "VP > [ VP > [ 'liggja' ] PP > [ P > { ( 'í'|'á' ) } NP > { 'auga' } ] ADVP > [ 'upp' ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "teningur",  # Trigger lemma for this pattern
                "PP > [ .* ADVP > { 'upp' } PP > { P > { ( 'í'|'á' ) } NP > { 'teningur' } } ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > [ IP >> { ADVP > { 'upp' } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )

        cls.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "VP > [ VP >> { VP > { VP > { 'hafa' } ADVP > { 'upp' } } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] .* ]",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "uppá",  # Trigger lemma for this pattern
                "VP > { VP > { 'taka' } NP > { PP > { P > { 'uppá' } NP > { ( no_þgf|pfn_þgf|no_þf|pfn_þf ) } } } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "uppí",  # Trigger lemma for this pattern
                "PP > { P > { 'uppí' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "uppí",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } PP > { P > { 'uppí' } NP > { ( no_þf|pfn_þf ) } } }",
                lambda self, match: self.dir_loc_comp(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "niður",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'niður' } PP > { P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) } ]",
                lambda self, match: self.dir_loc_niður(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "niður",  # Trigger lemma for this pattern
                "VP > [ VP > { 'vera' } .* PP > { ADVP > { 'niður' } P > { 'í' } NP } ]",
                lambda self, match: self.dir_loc_niður(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "verða",  # Trigger lemma for this pattern
                "VP > { VP > { 'verða' } NP > { ( pfn_þgf|abfn_þgf ) } NP > { 'út' 'um' } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "standa",  # Trigger lemma for this pattern
                "IP > { ADVP > { 'upp' } VP > { VP > { 'vera' } NP > { 'standa' } } }",
                lambda self, match: self.dir_loc_standa(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] NP > { PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } } }",
                lambda self, match: self.dir_loc_búð(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] NP > [ .* PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } ] }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP PP >> { NP > { PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } } } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] ADVP > [ 'út' ] PP > { P > [ 'um' ] } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' .* ] NP > { 'út' 'um' } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "útum",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } NP > { 'útum' } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "útum",  # Trigger lemma for this pattern
                "VP > { VP > { 'sækja' } PP > { 'um' } NP > { 'útum' } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' .* ] ADVP > { 'út' } PP > { P > { 'um' } NP } }",
                lambda self, match: self.dir_loc_ut_um(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > { 'gera' } NP > [ .* PP > { ADVP > { 'út' } P > { 'í' } } ] }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP >> { ADVP > { 'hér' } } PP > { ADVP > { 'inn' } } }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )
        cls.add_pattern(
            (
                "Skagi",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } PP >> { PP > { ADVP > { 'upp' } P > { 'á' } NP > { 'Skagi' } } } }",
                lambda self, match: self.dir_loc(match),
                None,
            )
        )

    def run(self) -> None:
        """ Apply the patterns to the sentence """
        tree = None if self._sent is None else self._sent.tree
        if tree is None:
            # No tree: nothing to do
            return
        # Make a set of the lemmas in the sentence.
        # Note that we collect the middle voice lemmas, such as 'dást' for the
        # verb 'dá'. This means that trigger lemmas should also be middle voice lemmas.
        lemmas_mm = self._sent.lemmas_mm
        if not lemmas_mm:
            return
        lemmas = set(lemma.replace("-", "") for lemma in lemmas_mm)

        def lemma_match(trigger: Union[str, FrozenSet[str], Set[str]]) -> bool:
            """ Returns True if any of the given trigger lemmas
                occur in the sentence """
            if not trigger:
                return True
            if isinstance(trigger, str):
                return trigger in lemmas
            return bool(lemmas & trigger)

        for trigger, pattern, func, context in self.PATTERNS:
            # We only do the expensive pattern matching if the trigger lemma
            # for a pattern rule (if given) is actually found in the sentence
            if lemma_match(trigger):
                for match in tree.all_matches(pattern, context):
                    # Call the annotation function for this match
                    func(self, match)
