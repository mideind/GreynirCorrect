"""

    Greynir: Natural language processing for Icelandic

    Sentence tree pattern matching module

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


    This module contains the PatternMatcher class, which implements
    functionality to look for questionable grammatical patterns in parse
    trees. These are typically not grammatical errors per se, but rather
    incorrect usage, e.g., attaching a wrong/uncommon preposition
    to a verb. The canonical example is "Ég leitaði af kettinum"
    ("I searched off the cat") which should very likely have been
    "Ég leitaði að kettinum" ("I searched for the cat"). The first form
    is grammatically correct but the meaning is borderline nonsensical.

"""

from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set, Tuple, Union, cast

import json
import os
from functools import partial

from islenska import Bin
from reynir import NounPhrase, Sentence
from reynir.bintokenizer import ALL_CASES
from reynir.matcher import ContextDict
from reynir.simpletree import SimpleTree
from reynir.verbframe import VerbErrors

from reynir_correct.errtokenizer import emulate_case

from .annotation import Annotation

# The types involved in pattern processing
AnnotationFunction = Callable[[SimpleTree], None]
PatternTuple = Tuple[Union[str, Set[str], FrozenSet[str]], str, AnnotationFunction, Optional[ContextDict]]

BIN = Bin()

# Variants not needed for lookup
SKIPVARS = frozenset(("op", "subj", "0", "1", "2"))


class IcelandicPlaces:

    """Wraps a dictionary of Icelandic place names with their
    associated prepositions"""

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
    ICELOC_PREP_JSONPATH = os.path.join(os.path.dirname(__file__), "resources", "iceloc_prep.json")

    @classmethod
    def _load_json(cls) -> None:
        """Load the place name dictionary from a JSON file into memory"""
        with open(cls.ICELOC_PREP_JSONPATH, encoding="utf-8") as f:
            cls.ICELOC_PREP = json.load(f)

    @classmethod
    def lookup_preposition(cls, place: str) -> Optional[str]:
        """Look up the correct preposition to use with a placename,
        or None if the placename is not known"""
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
        """Return True if the given place is found in the dictionary"""
        if cls.ICELOC_PREP is None:
            cls._load_json()
        assert cls.ICELOC_PREP is not None
        return place in cls.ICELOC_PREP


class PatternMatcher:

    """Class to match parse trees with patterns to find probable usage errors"""

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
    def __init__(self, ann: List[Annotation], sent: Sentence) -> None:
        # Annotation list
        self._ann = ann
        # The original sentence object
        self._sent = sent
        # Token list
        self._tokens = sent.tokens
        # Terminal node list
        self._terminal_nodes = sent.terminal_nodes

        self.PATTERNS: List[PatternTuple] = []

        self.ctx_af: ContextDict = cast(ContextDict, None)
        self.ctx_að: ContextDict = cast(ContextDict, None)
        self.ctx_noun_af: ContextDict = cast(ContextDict, None)
        self.ctx_noun_af_obj: ContextDict = cast(ContextDict, None)
        self.ctx_verb_01: ContextDict = cast(ContextDict, None)
        self.ctx_verb_02: ContextDict = cast(ContextDict, None)
        self.ctx_noun_að: ContextDict = cast(ContextDict, None)
        self.ctx_subjsing: ContextDict = cast(ContextDict, None)
        self.ctx_place_names: ContextDict = cast(ContextDict, None)
        self.ctx_uncertain_verbs: ContextDict = cast(ContextDict, None)
        self.ctx_confident_verbs: ContextDict = cast(ContextDict, None)
        self.ctx_dir_loc: ContextDict = cast(ContextDict, None)

        # First instance: create the class-wide pattern list
        self.create_patterns()

    @staticmethod
    def get_wordform(word: str, lemma: str, cat: str, variants: Iterable[str]) -> str:
        """Get correct wordform from BinPackage,
        given a set of variants"""
        realvars: Union[Set[str], Iterable[str]]
        if cat == "so":
            # Get rid of irrelevant variants for verbs
            realvars = set(variants) - SKIPVARS
            if "lhþt" not in realvars:
                # No need for cases if this is not LHÞT
                realvars -= ALL_CASES
        else:
            realvars = variants
        wordforms = BIN.lookup_variants(word, cat, tuple(realvars), lemma=lemma)
        if not wordforms:
            return ""
        # Can be many possible word forms; we want the first one in most cases
        return wordforms[0].bmynd

    def wrong_preposition_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_af)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_af)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if vp is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'{0} af' á sennilega að vera '{0} að'".format(vp.tidy_text)
        detail = "Sögnin '{0}' tekur yfirleitt með sér " "forsetninguna 'að', ekki 'af'.".format(vp.tidy_text)
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verb phrase
        vp = match.first_match("VP > { %verb }", self.ctx_að)
        if vp is None:
            vp = match.first_match("VP >> { %verb }", self.ctx_að)
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        # Calculate the start and end token indices, spanning both phrases
        if vp is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = "Sögnin '{0}' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'.".format(vp.tidy_text)
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_spyrja_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'spyrja' }")
        if vp is None:
            return
        # Find the attached prepositional/adverbial phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        if pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "Í '{0}' á 'af' sennilega að vera 'að'".format(vp.tidy_text)
        detail = "Í samhenginu 'að spyrja að e-u' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_vitni_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        np = match.first_match(". >> { 'vitni' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        if np is None or pp is None:
            return
        pp_af = pp.first_match('"að"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'verða vitni af' á sennilega að vera 'verða vitni að'"
        detail = "Í samhenginu 'verða vitni að e-u' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_grin_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'gera' }")
        np = match.first_match("NP >> { 'grín' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        if vp is None or np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'gera grín af' á sennilega að vera 'gera grín að'"
        detail = "Í samhenginu 'gera grín að e-u' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_leida_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal and nominal phrases
        vp = match.first_match("VP > { 'leiða' }")
        np = match.first_match("NP >> { ( 'líkur'|'rök'|'rak' ) }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        if vp is None or np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'leiða {0} af' á sennilega að vera 'leiða {0} að'".format(np.tidy_text)
        detail = "Í samhenginu 'leiða {0} af e-u' er notuð " "forsetningin 'að', ekki 'af'.".format(np.tidy_text)
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_marka_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
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
        if vp is None or np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'marka upphaf af' á sennilega að vera 'marka upphaf að'"
        detail = "Í samhenginu 'marka upphaf að e-u' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_leggja_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'leggja' }")
        if vp is None:
            vp = match.first_match("VP >> { 'leggja' }")
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "af" }')
        if pp is None:
            pp = match.first_match('ADVP > { "af" }')
        # Find the offending nominal phrase
        np = match.first_match('NP >> { "velli" }')
        if vp is None or pp is None or np is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'leggja af velli' á sennilega að vera 'leggja að velli'"
        detail = "Í samhenginu 'leggja einhvern að velli' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_utan_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending adverbial phrase
        advp = match.first_match("ADVP > { 'utan' }")
        if advp is None:
            advp = match.first_match("ADVP >> { 'utan' }")
        # Find the attached prepositional phrase
        pp = match.first_match('ADVP > { "af" }')
        if advp is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'utan af' á sennilega að vera 'utan að'"
        detail = "Í samhenginu 'kunna eitthvað utan að' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_uppvis_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal phrase
        vp = match.first_match("VP >> { 'verða' }")
        # Find the attached nominal phrase
        np = match.first_match("NP >> { 'uppvís' }")
        # Find the attached prepositional phrase
        pp = match.first_match('PP > { "af" }')
        if pp is None:
            pp = match.first_match("ADVP > { 'af' }")
        if vp is None or np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'uppvís af' á sennilega að vera 'uppvís að'"
        detail = "Í samhenginu 'verða uppvís að einhverju' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_verða_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
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
        if vp is None or pp is None or np is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'af ósk' á sennilega að vera 'að ósk'"
        detail = "Í samhenginu 'að verða að ósk' er notuð " "forsetningin 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def suggestion_complex(self, match: SimpleTree, lemma: str, prep: str) -> str:
        """Find the preposition to correct for the suggestion"""
        p_ter = match.first_match(f"'{lemma}'")
        if p_ter is None:
            return ""
        # The instance of the preposition which comes right after the phrase terminal is substituted
        all_m = match.all_matches(f"@'{prep}'")
        subtree = None
        for m in all_m:
            if m.span[0] > p_ter.span[-1]:
                subtree = m
                break
        if subtree is None:
            return ""
        suggest = ""
        if prep == "að":
            suggest = match.substituted_text(subtree, "af")
        elif prep == "af":
            suggest = match.substituted_text(subtree, "að")
        assert suggest != ""  # All cases should be handled above
        return suggest

    def wrong_preposition_ahyggja_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Calculate the start and end token indices, spanning both phrases
        np = match.first_match("NP > { 'áhyggja' }")
        pp = match.first_match('PP > { "að" }')
        if np is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'{0} að' á sennilega að vera '{0} af'".format(np.tidy_text)
        detail = "Í samhenginu 'hafa áhyggjur af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_hluti_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Calculate the start and end token indices, spanning both phrases
        np = match.first_match("NP > { 'hluti' }")
        pp = match.first_match("PP > { 'að' }")
        if np is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'hluti að' á sennilega að vera 'hluti af'"
        detail = "Í samhenginu 'hluti af e-u' er notuð forsetningin 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að_mörkum(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending prepositional phrase
        pp = match.first_match('PP > { "að" "mörkum" }')
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        suggest = self.suggestion_complex(match, "leggja", "að")
        text = f"'{pp_að.tidy_text}' á sennilega að vera '{suggest}'"
        detail = "Í samhenginu 'leggja e-ð af mörkum' er notuð " "forsetningin 'af', ekki 'að'."
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_að_leiða(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Calculate the start and end token indices, spanning both phrases
        pp = match.first_match("P > { 'að' }")
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        suggest = pp_að.substituted_text(pp_að, "af")
        whole = self.suggestion_complex(match, "láta", "að")
        text = f"'{match.tidy_text}' á sennilega að vera '{whole}'"
        detail = "Í samhenginu 'láta gott af sér leiða' er notuð " "forsetningin 'af', ekki 'að'."
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heiður_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'heiður' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        if np is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        whole = self.suggestion_complex(match, "heiður", "að")
        suggest = pp_að.substituted_text(pp_að, "af")
        text = f"'{match.tidy_text}' á sennilega að vera '{whole}'"
        detail = "Í samhenginu 'fá/hljóta heiðurinn af' er notuð " "forsetningin 'af', ekki 'að'."
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_eiga_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verb phrase
        vp = match.first_match("VP > { 'eiga' }")
        if vp is None:
            vp = match.first_match("VP >> { 'eiga' }")
        # Find the nominal object
        np = match.first_match("( NP|ADVP )")
        if np is None:
            return
        legal_lemmas = frozenset(("aðild", "frumkvæði", "hlut", "upptak"))
        if any(lemma in legal_lemmas for lemma in np.lemmas):
            # 'Eiga aðild/frumkvæði/hlut/upptök að e-u' is legal; do not complain
            return
        # Find the attached prepositional phrase
        pp = match.first_match('P > { "að" }')
        if pp is None:
            pp = match.first_match('PP > { "að" }')
        if vp is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        suggest = pp_að.substituted_text(pp_að, "af")
        whole = self.suggestion_complex(match, "eiga", "að")
        text = f"'{match.tidy_text}' á sennilega að vera '{whole}'"
        detail = f"Orðasambandið '{match.tidy_text}' tekur yfirleitt með sér " f"forsetninguna 'af', ekki 'að'."
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_vera_til_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        pp = match.first_match('P > { "að" }')
        if pp is None:
            pp = match.first_match('PP > { "að" }')
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'til að' á sennilega að vera 'til af'"
        detail = "Orðasambandið 'vera mikið/lítið til af e-u' innifelur " "yfirleitt forsetninguna 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_gagn_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        pp = match.first_match('P > { "að" }')
        if pp is None:
            pp = match.first_match('PP > { "að" }')
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'gagn að' á sennilega að vera 'gagn af'"
        detail = "Orðasambandið 'að hafa gagn af e-u' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_frettir_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending preposition
        pp = match.first_match('P > { "að" }')
        if pp is None:
            pp = match.first_match('PP > { "að" }')
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        # Calculate the start and end token indices, spanning both phrases
        text = "'að' á sennilega að vera 'af'"
        detail = "Orðasambandið 'fréttir berast af e-u' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_stafa_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'stafa' }")
        if vp is None:
            return
        pp = vp.first_match('P > { "að" }')
        if pp is None:
            pp = vp.first_match('PP > { "að" }')
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        suggest = pp_að.substituted_text(pp_að, "af")
        whole = self.suggestion_complex(match, "stafa", "að")
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(match.tidy_text, whole)
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = "Orðasambandið 'að stafa af e-u' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        if suggest == pp_að.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_ólétt_af(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'óléttur' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'af' }")
        if np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "'{0} af' á sennilega að vera '{0} að'".format(np.tidy_text)
        detail = "Orðasambandið 'að vera ólétt/ur að e-u' tekur yfirleitt með sér " "forsetninguna 'að', ekki 'af'."
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heyra_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending verbal phrase
        vp = match.first_match("VP > { 'heyra' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        if vp is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        suggest = pp_að.substituted_text(pp_að, "af")
        whole = self.suggestion_complex(match, "heyra", "að")
        text = "'{0}' á sennilega að vera '{1}'".format(match.tidy_text, whole)
        detail = "Orðasambandið 'að heyra af e-u' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_hafa_gaman_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        np = match.first_match("NP > { 'gaman' }")
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        if np is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        text = "'gaman að' á sennilega að vera 'gaman af'"
        detail = "Orðasambandið 'að hafa gaman af e-u' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_heillaður_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Calculate the start and end token indices, spanning both phrases
        pp = match.first_match("P > { 'að' }")
        if pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        text = "'heillaður að' á sennilega að vera 'heillaður af'"
        detail = "Í samhenginu 'heillaður af e-u' er notuð " "forsetningin 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_preposition_valinn_að(self, match: SimpleTree) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        vp = match.first_match("VP > { 'velja' }")
        if vp is None:
            vp = match.first_match("NP > { 'valinn' }")
        pp = match.first_match('P > { "að" }')
        if vp is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        start, end = pp_að.span
        if " að " in vp.tidy_text:
            text = "'{0}' á sennilega að vera '{1}'".format(vp.tidy_text, vp.tidy_text.replace(" að ", " af "))
        else:
            text = "'{0} að' á sennilega að vera '{0} af'".format(vp.tidy_text)
        detail = "Orðasambandið 'að vera valin/n af e-m' tekur yfirleitt með sér " "forsetninguna 'af', ekki 'að'."
        suggest = pp_að.substituted_text(pp_að, "af")
        if suggest == pp_að.tidy_text or not suggest:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def wrong_að_use(self, match: SimpleTree, context: ContextDict) -> None:
        """Handle a match of a suspect preposition pattern"""
        # Find the offending noun
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'að' }")
        if np is None or pp is None:
            return
        pp_að = pp.first_match('"að"')
        if pp_að is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_að.span
        suggest = match.substituted_text(pp_að, "af")
        text = "Hér á líklega að vera forsetningin 'af' í stað 'að'."
        detail = f"Í samhenginu '{match.tidy_text}' er rétt að nota " f"forsetninguna 'af' í stað 'að'."
        if suggest == match.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AÐ",
                text=text,
                detail=detail,
                original=pp_að.tidy_text,
                suggest=suggest,
            )
        )

    def check_pp_with_place(self, match: SimpleTree) -> None:
        """Check whether the correct preposition is being used with a place name"""
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
        if suggest == match.tidy_text:
            # No need to annotate, no changes were made
            return
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
        """Wrong noun used with a verb, for instance
        'bjóða e-m birginn' instead of 'byrginn'"""
        # TODO: This code is provisional, intended as a placeholder for similar cases
        start, end = match.span
        text = "Mælt er með að rita 'bjóða e-m byrginn' í stað 'birginn'."
        detail = text
        tidy_text = match.tidy_text
        suggest = tidy_text.replace("birginn", "byrginn", 1)
        if suggest == match.tidy_text:
            # No need to annotate, no changes were made
            return
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
        self,
        match: SimpleTree,
        correct_verb: str,
        context: ContextDict,
    ) -> None:
        """Annotate wrong verbs being used with nouns,
        for instance 'byði hnekki' where the verb should
        be 'bíða' -> 'biði hnekki' instead of 'bjóða'"""
        vp = match.first_match("VP > { %verb }", context)
        if vp is None:
            return
        verb = next(ch for ch in vp.children if ch.tcat == "so").own_lemma_mm
        np = match.first_match("NP >> { %noun }", context)
        if np is None:
            return
        start, end = min(vp.span[0], np.span[0]), max(vp.span[1], np.span[1])
        # noun = next(ch for ch in np.leaves if ch.tcat == "no").own_lemma
        text = "Hér á líklega að vera sögnin '{0}' í stað '{1}'.".format(correct_verb, verb)
        detail = "Í samhenginu '{0}' er rétt að nota sögnina '{1}' í stað '{2}'.".format(
            match.tidy_text, correct_verb, verb
        )
        suggest = ""
        # TODO get better suggest value
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
        """Handle a match of a suspect preposition pattern"""
        # Find the offending nominal phrase
        np = match.first_match(" %noun ", context)
        # Find the attached prepositional phrase
        pp = match.first_match("P > { 'af' }")
        if np is None or pp is None:
            return
        pp_af = pp.first_match('"af"')
        if pp_af is None:
            return
        # Calculate the start and end token indices, spanning both phrases
        start, end = pp_af.span
        text = "Hér á líklega að vera forsetningin 'að' í stað 'af'."
        detail = "Í samhenginu '{0}' er rétt að nota forsetninguna 'að' í stað 'af'.".format(match.tidy_text)
        suggest = pp_af.substituted_text(pp_af, "að")
        if suggest == pp_af.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_WRONG_PREP_AF",
                text=text,
                detail=detail,
                original=pp_af.tidy_text,
                suggest=suggest,
            )
        )

    def vera_að(self, match: SimpleTree) -> None:
        """'vera að' in front of verb is unneccessary"""
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
        if "þt" in so.all_variants:
            # The past tense behaves differently, much less likely to be an error
            return
        suggest = PatternMatcher.get_wordform(realso.text.lower(), realso.lemma, realso.cat, so.all_variants)
        if not suggest:
            return
        text = (
            f"Mælt er með að sleppa '{so.tidy_text} að' og beygja frekar sögnina "
            f"'{realso.lemma}' svo hún verði '{suggest}'."
        )
        detail = (
            "Skýrara er að nota beina ræðu ('Ég skil þetta ekki') fremur en "
            "svokallað dvalarhorf ('Ég er ekki að skilja þetta')."
        )
        # TODO better original value!
        if suggest == "vera að":
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_VeraAð",
                text=text,
                detail=detail,
                original="vera að",
                suggest=suggest,
            )
        )

    def dir_loc(self, match: SimpleTree) -> None:
        adv = match.first_match("( 'inn'|'út'|'upp' )")
        if adv is None:
            return
        pp = match.first_match("PP > { P > { ( 'í'|'á'|'um' ) } NP > { ( no_þgf|no_þf|pfn_þgf|pfn_þf ) } }")
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
        if correction == adv.tidy_text:
            # No need to annotate, no changes were made
            return
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
        correction = p.tidy_text[:-1] + "i" + " " + p.tidy_text[-1]
        text = f"Hér á líklega að vera '{correction}' í stað '{p.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, p.tidy_text
        )
        suggest = match.tidy_text.replace(p.tidy_text, correction)
        if suggest == match.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_DIR_LOC",
                text=text,
                detail=detail,
                original=match.tidy_text,
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
        if correction == advp.tidy_text:
            # No need to annotate, no changes were made
            return
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

    def dir_loc_simple(self, match: SimpleTree) -> None:
        advp = match.first_match("ADVP > { ('inn'|'út'|'niður'|'upp') }")
        if advp is None:
            return
        start, end = match.span
        if "niður" in match.tidy_text:
            correction = "niðri"
        else:
            correction = advp.tidy_text + "i"
        text = f"Hér á líklega að vera '{correction}' í stað '{advp.tidy_text}'"
        detail = "Í samhenginu '{0}' er rétt að nota atviksorðið '{1}' í stað '{2}'.".format(
            match.tidy_text, correction, advp.tidy_text
        )
        if correction == advp.tidy_text:
            # No need to annotate, no changes were made
            return
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
        """Subjunctive mood, present tense, is used instead of indicative
        in conditional ("COND"), purpose ("PURP"), relative ("REL")
        or temporal ("TEMP/w") subclauses"""
        vp = match.first_match("VP > so_vh")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        start, end = so.span
        if "þt" in so.all_variants:
            return
        variants = set(so.all_variants) - {"vh"}
        variants.add("fh")
        so_text = so.text.lower()
        suggest = PatternMatcher.get_wordform(so_text, so.lemma, so.cat, variants)
        if suggest == so_text:
            return
        if not suggest:
            return
        text = f"Hér á líklega að nota framsöguhátt sagnarinnar '{so_text}', " f"þ.e. '{suggest}'."
        detail = ""
        sent_kind = ""
        if kind == "COND":
            sent_kind = "skilyrðissetningum á borð við 'Z' í 'X gerir Y ef Z'"
        elif kind == "PURP":
            sent_kind = "tilgangssetningum á borð við 'Z' í 'X gerir Y til þess að Z'"
        elif kind == "TEMP/w":
            sent_kind = "tíðarsetningum á borð við 'Z' í 'X gerði Y áður en Z'"
        elif kind == "REL":
            detail = "Í tilvísunarsetningum er aðeins framsöguháttur sagna tækur."
        else:
            return
        if not detail:
            detail = f"Í {sent_kind} er yfirleitt notaður framsöguháttur sagna."
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_MOOD_" + kind,
                text=text,
                detail=detail,
                original=so.text,
                suggest=emulate_case(suggest, template=so.text),
            )
        )

    def mood_ind(self, kind: str, match: SimpleTree) -> None:
        """Indicative mood is used instead of subjunctive
        in concessive or purpose subclauses"""
        vp = match.first_match("VP > so_fh")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        start, end = so.span
        variants = set(so.all_variants) - {"fh"}
        variants.add("vh")
        so_text = so.text.lower()
        suggest = PatternMatcher.get_wordform(so_text, so.lemma, so.cat, variants)
        if not suggest or suggest == so_text:
            return
        text = f"Hér er réttara að nota viðtengingarhátt " f"sagnarinnar '{so.lemma}', þ.e. '{suggest}'."
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
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_MOOD_" + kind,
                text=text,
                detail=detail,
                original=so.text,
                suggest=emulate_case(suggest, template=so.text),
            )
        )

    def doubledefinite(self, match: SimpleTree) -> None:
        """A definite noun appears with a definite pronoun,
        e.g. 'þessi maðurinn'"""
        no = match.first_match("no_gr")
        if no is None:
            return
        fn = match.first_match("fn")
        if fn is None:
            return
        comma = match.first_match('@","')
        if comma is not None:
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
        suggest = v[0].bmynd.replace("-", "")
        text = f"Hér ætti annaðhvort að sleppa '{fn.tidy_text}' eða " f"breyta '{no.tidy_text}' í '{suggest}'."
        detail = (
            "Hér er notuð tvöföld ákveðni, þ.e. ábendingarfornafn á undan "
            "nafnorði með greini. Það er ekki í samræmi við viðtekinn málstaðal."
        )
        if suggest == match.tidy_text:
            # No need to annotate, no changes were made
            return
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

    def plursub(self, kind: str, match: SimpleTree) -> None:
        """Subject is singular in meaning grammatically, e.g. '40.000 manns', 'meirihluti'"""
        # Check if verb is singular
        ip = match.enclosing_tag("IP")
        if ip is None:
            return
        vp = ip.first_match("VP > so_et")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        no = match.first_match("no_ft")
        if no is None:
            return
        start, end = so.span
        variants = set(so.all_variants) - {"et"}
        variants.add("ft")
        so_text = so.text.lower()
        suggest = PatternMatcher.get_wordform(so_text, so.lemma, so.cat, variants)
        if not suggest or suggest == so_text:
            return
        text = f"Hér er réttara að nota fleirtölu " f"sagnarinnar '{so.lemma}', þ.e. '{suggest}'."
        if kind == "GEN":
            nogen = match.first_match("NP-POSS > { no_ft_ef }")
            if nogen is None:
                return
            detail = f"Þrátt fyrir að eignarfallsliðurinn '{nogen.lemma}' sé eintölumerkingar er aðalnafnliðurinn '{no.lemma}' frumlagið og stjórnar tölu sagnarinnar '{so.lemma}'."
        elif kind == "QUANT":
            detail = f"Fleirtölunafnorðið '{no.lemma}' hefur eintölumerkingu en er málfræðilega fleirtala og sögnin '{so.lemma}' á því að standa í fleirtölu."
        else:
            return

        generic = frozenset(("P_NT_ÍTölu", "P_NT_FjöldiHluti"))  # TODO update list
        # This is more precise, we want to delete the more generic one
        for ann in self._ann:
            if ann.code in generic and ann.start == start and ann.end == end:
                self._ann.remove(ann)
        suggest = emulate_case(suggest, template=so.text)
        if suggest == so.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_PLURSUB_" + kind,
                text=text,
                detail=detail,
                original=so.tidy_text,
                suggest=suggest,
            )
        )

    def singsub(self, kind: str, match: SimpleTree) -> None:
        """Subject is plural in meaning but singular grammatically, e.g. 'Hluti ferðamanna', 'tvíeykið X og Y"""
        # Check if verb is plural
        ip = match.enclosing_tag("IP")
        if ip is None:
            return
        vp = ip.first_match("VP > so_ft")
        if vp is None:
            return
        so = vp.first_match("so")
        if so is None:
            return
        no = match.first_match("no_et")
        if no is None:
            return
        start, end = so.span
        variants = set(so.all_variants) - {"ft"}
        variants.add("et")
        so_text = so.text.lower()
        suggest = PatternMatcher.get_wordform(so_text, so.lemma, so.cat, variants)
        if not suggest or suggest == so_text:
            return
        text = f"Hér er réttara að nota eintölu " f"sagnarinnar '{so.lemma}', þ.e. '{suggest}'."
        if kind == "GEN":
            nogen = match.first_match("NP-POSS > { no_ft_ef }")
            if nogen is None:
                return
            detail = f"Þrátt fyrir að eignarfallsliðurinn '{nogen.lemma}' sé fleirtölumerkingar er aðalnafnliðurinn '{no.lemma}' frumlagið og stjórnar tölu sagnarinnar '{so.lemma}'."
        elif kind == "QUANT":
            detail = f"Eintölunafnorðið '{no.lemma}' hefur fleirtölumerkingu en er málfræðilega eintala og sögnin '{so.lemma}' á því að standa í eintölu."
        elif kind == "AF":
            noaf = match.first_match("PP >> { no_ft_þgf }")
            if noaf is None:
                return
            detail = (
                f"Tala sagnarinnar '{so.lemma}' stjórnast af tölu '{no.lemma}', ekki '{noaf.lemma}' í forsetningarlið."
            )
        else:
            return

        generic = frozenset(("P_NT_ÍTölu", "P_NT_FjöldiHluti"))
        # This is more precise, we want to delete the more generic one
        for ann in self._ann:
            if ann.code in generic and ann.start == start and ann.end == end:
                self._ann.remove(ann)
        if suggest == so.tidy_text:
            # No need to annotate, no changes were made
            return
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_SINGSUB_" + kind,
                text=text,
                detail=detail,
                original=so.text,
                suggest=emulate_case(suggest, template=so.text),
            )
        )

    def né(self, match: SimpleTree) -> None:
        c = match.first_match("'né'")
        if c is None:
            return
        start, end = c.span[0], c.span[1]
        if match.root.first_match("'hvorki'") is not None or "eitt né neitt" in match.root.text:
            # Check to see if VillaNé (P_NT_Né) has been activated
            # and should be deleted
            for ann in self._ann:
                if ann.code == "P_NT_Né" and ann.start == start and ann.end == end:
                    self._ann.remove(ann)
            return

        for ann in self._ann:
            if ann.code == "P_NT_Né" and ann.start == start and ann.end == end:
                # We have already annotated the error, no need to do it twice
                return
        correction = "eða"
        text = "'né' gæti átt að vera 'eða'"
        detail = (
            "'né' er hluti af margorða samtengingunni 'hvorki né' en getur ekki staðið eitt og sér sem aukatenging."
        )
        self._ann.append(
            Annotation(
                start=start,
                end=end,
                code="P_Né",
                text=text,
                detail=detail,
                original="né",
                suggest=correction,
            )
        )

    def add_pattern(self, p: PatternTuple) -> None:
        """Validates and adds a pattern to the class global pattern list"""
        _, pattern, _, ctx = p
        if "%" in pattern:
            assert ctx is not None, "Missing context for pattern with %macro"
        else:
            assert ctx is None, "Unnecessary context given for pattern with no %macro"
        self.PATTERNS.append(p)

    def create_patterns(self) -> None:
        """Initialize the list of patterns and handling functions"""

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
            self.ctx_af = {
                "verb": lambda tree: (tree.own_lemma_mm in verbs_af and not (set(tree.variants) & {"1", "2"}))
            }
            # Catch sentences such as 'Jón leitaði af kettinum'
            self.add_pattern(
                (
                    verbs_af,  # Trigger lemma for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "af" } } }',
                    self.wrong_preposition_af,
                    self.ctx_af,
                )
            )
            # Catch sentences such as 'Vissulega er hægt að brosa af þessu',
            # 'Friðgeir var leitandi af kettinum í allan dag'
            self.add_pattern(
                (
                    verbs_af,  # Trigger lemma for this pattern
                    '. > { (NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "af" } } }',
                    self.wrong_preposition_af,
                    self.ctx_af,
                )
            )
            # Catch "Það sem Jón spurði ekki af...", "Jón spyr (ekki) af því."
            #    self.add_pattern(
            #        (
            #            "spyrja",  # Trigger lemma for this pattern
            #            "IP > { VP >> { 'spyrja' } ADVP > { 'af' } }",
            #            self.wrong_preposition_af,
            #            self.ctx_af,
            #        )
            #    )
            self.add_pattern(
                (
                    "spyrja",  # Trigger lemma for this pattern
                    "VP > { VP > { 'spyrja' } ADVP > { \"af\" } }",
                    self.wrong_preposition_spyrja_af,
                    None,
                )
            )
            # Catch "Jón spyr af því."
            #    self.add_pattern(
            #        (
            #            "spyrja",  # Trigger lemma for this pattern
            #            "IP > { VP >> { 'spyrja' } PP > { 'af' } }",
            #            self.wrong_preposition_af,
            #            None,
            #        )
            #    )
            # Catch "...vegna þess að dýr leita af öðrum smærri dýrum."
            self.add_pattern(
                (
                    "leita",  # Trigger lemma for this pattern
                    "VP > { PP >> { 'leita' } PP > 'af' }",
                    self.wrong_preposition_af,
                    None,
                )
            )

            # Catch "Þetta er mesta vitleysa sem ég hef orðið vitni af", "Hún varð vitni af því þegar kúturinn sprakk"
            self.add_pattern(
                (
                    "vitni",  # Trigger lemma for this pattern
                    "VP > { VP > { ('verða'|'vera') } NP > { \"vitni\" } ADVP > \"af\" }",
                    self.wrong_preposition_vitni_af,
                    None,
                )
            )
            # Catch "Hún gerði grín af því.", "Þetta er mesta vitleysa sem ég hef gert grín af.", "...og gerir grín af sjálfum sér."
            self.add_pattern(
                (
                    "grín",  # Trigger lemma for this pattern
                    # "IP",
                    "VP > { NP > { 'grín' } ( PP|ADVP ) > { \"af\" } }",
                    self.wrong_preposition_grin_af,
                    None,
                )
            )
            # Catch "Hann leiðir (ekki) líkur af því.", "Hann hefur aldrei leitt líkur af því."
            self.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    "VP > { VP > { 'leiða' } NP > { ('líkur' | 'rök' | 'rak') } PP > { \"af\" } }",
                    self.wrong_preposition_leida_af,
                    None,
                )
            )
            # Catch "Tíminn markar upphaf af því."
            self.add_pattern(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP-OBJ > { 'upphaf' PP > { 'af' } } }",
                    self.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar ekki upphaf af því."
            self.add_pattern(
                (
                    frozenset(("upphafinn", "upphaf")),  # Trigger lemma for this pattern
                    "VP > { VP > { 'marka' } NP > { ('upphafinn'|'upphaf') } PP > { 'af' } }",
                    self.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Það markar upphaf af því."
            self.add_pattern(
                (
                    "upphaf",  # Trigger lemma for this pattern
                    "VP > { VP > { VP > { 'marka' } NP-SUBJ > { 'upphaf' } } PP > { 'af' } }",
                    self.wrong_preposition_marka_af,
                    None,
                )
            )
            #    self.add_pattern(
            #        (
            #            "upphefja",  # Trigger lemma for this pattern
            #            "IP",
            #            self.wrong_preposition_marka_af,
            #            None,
            #        )
            #    )
            # Catch "Það hefur ekki markað upphafið af því."
            self.add_pattern(
                (
                    "upphefja",  # Trigger lemma for this pattern
                    "VP > { NP > { 'markaður' } VP > { 'upphefja' } PP > { 'af' } }",
                    self.wrong_preposition_marka_af,
                    None,
                )
            )
            # Catch "Jón leggur hann (ekki) af velli.", "Jón hefur (ekki) lagt hann af velli."
            self.add_pattern(
                (
                    frozenset(("völlur", "vell", "velli")),  # Trigger lemmas for this pattern
                    'VP > { VP > { \'leggja\' } PP > { P > { "af" } NP > { "velli" } } }',
                    self.wrong_preposition_leggja_af,
                    None,
                )
            )
            # Catch "Jón kann það (ekki) utan af."
            self.add_pattern(
                (
                    "kunna",  # Trigger lemma for this pattern
                    "VP > { VP > { 'kunna' } ADVP > { 'utan' } ADVP > { 'af' } }",
                    self.wrong_preposition_utan_af,
                    None,
                )
            )
            # Catch "Honum varð af ósk sinni."
            self.add_pattern(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "(S-MAIN | IP) > { VP > { 'verða' } PP > { 'af' NP > { 'ósk' } } }",
                    self.wrong_preposition_verða_af,
                    None,
                )
            )
            # Catch "...en varð ekki af ósk sinni."
            self.add_pattern(
                (
                    "ósk",  # Trigger lemma for this pattern
                    "IP > { VP > { VP > { 'verða' } PP > { P > { 'af' } NP > { 'ósk' } } } }",
                    self.wrong_preposition_verða_af,
                    None,
                )
            )
            # Catch "Ég varð (ekki) uppvís af athæfinu.", "Hann hafði (ekki) orðið uppvís af því."
            self.add_pattern(
                (
                    "uppvís",  # Trigger lemma for this pattern
                    "VP > { VP > { 'verða' } NP > { 'uppvís' } PP > { 'af' } }",
                    self.wrong_preposition_uppvis_af,
                    None,
                )
            )

        if verbs_að:
            # Create matching patterns with a context that catches the að/af verbs.
            self.ctx_að = {
                "verb": lambda tree: (tree.own_lemma_mm in verbs_að and not (set(tree.variants) & {"1", "2"}))
            }
            # Catch sentences such as 'Jón heillaðist að kettinum'
            self.add_pattern(
                (
                    verbs_að,  # Trigger lemmas for this pattern
                    'VP > { VP >> { %verb } PP >> { P > { "að" } } }',
                    self.wrong_preposition_að,
                    self.ctx_að,
                )
            )
            # Catch sentences such as 'Vissulega er hægt að heillast að þessu'
            self.add_pattern(
                (
                    verbs_að,  # Trigger lemma for this pattern
                    '(NP-PRD | IP-INF) > { VP > { %verb } } PP >> { P > { "að" } }',
                    self.wrong_preposition_að,
                    self.ctx_að,
                )
            )
            # Catch "Þetta er fallegasta kona sem ég hef orðið heillaður að"
            self.add_pattern(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "VP > { VP > [ .* ('verða' | 'vera') ] NP-PRD > [ .* 'heilla' .* ADVP > { \"að\" } ] }",
                    self.wrong_preposition_heillaður_að,
                    None,
                )
            )
            # Catch "Ég hef lengi verið heillaður að henni."
            self.add_pattern(
                (
                    "heilla",  # Trigger lemma for this pattern
                    "NP > { NP >> { 'heilla' } PP > { 'að' } }",
                    self.wrong_preposition_heillaður_að,
                    None,
                )
            )
            # Catch "Ég er (ekki) hluti að heildinni.", "Við höfum öll verið hluti að heildinni."
            self.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { NP > { 'hluti' } PP > { \"að\" } }",
                    # "VP > { VP > { 'vera' NP-PRD > { 'hluti' } } PP > { 'að' } }",
                    self.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Þeir sögðu að ég hefði verið hluti að heildinni."
            self.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "NP > { 'hluti' PP > { \"að\" } }",
                    self.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Þeir sögðu að ég hefði verið hluti að heildinni."  # Two patterns to catch the same sentence due to variable parsing
            self.add_pattern(
                (
                    "hluti",  # Trigger lemma for this pattern
                    "VP > { CP >> { 'hluti' } PP > { \"að\" } }",
                    self.wrong_preposition_hluti_að,
                    None,
                )
            )
            # Catch "Ég hef (ekki) áhyggjur að honum.", "Ég hef áhyggjur að því að honum líði illa."
            self.add_pattern(
                (
                    "áhyggja",  # Trigger lemma for this pattern
                    "VP > { NP > { 'áhyggja' } PP > { \"að\" } }",
                    # "VP > { VP >> { 'áhyggja' } PP > { 'að' } }",
                    self.wrong_preposition_ahyggja_að,
                    None,
                )
            )
            # Catch "Ég hafði ekki lagt mikið að mörkum."
            self.add_pattern(
                (
                    frozenset(("mörk", "mark")),  # Trigger lemmas for this pattern
                    'VP > { VP >> { \'leggja\' } PP > { "að" "mörkum" } }',
                    self.wrong_preposition_að_mörkum,
                    None,
                )
            )
            # Catch "Jón hefur látið gott að sér leiða."
            # self.add_pattern(
            #    (
            #        "leiða",  # Trigger lemma for this pattern
            #        "VP > { VP > { 'láta' } PP > { P > \"að\" } VP > { 'leiða' } }",
            #        self.wrong_preposition_að_leiða,
            #        None,
            #    )
            # )
            # Catch "Ég lét gott að mér leiða."
            self.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    'VP > [ .* VP > { \'láta\' } NP ("að mér"|"að þér"|"að sér") \'leiða\']',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Ég lét (ekki) gott að mér leiða." (In case of different parse)
            self.add_pattern(
                (
                    "leiður",  # Trigger lemma for this pattern
                    'VP > [ VP > [ .* \'láta\' .* ] NP > [ .* "gott" .* ] PP > [ "að" NP > [ ("mér"|"þér"|"sér"|"okkur") ] "leiða" ] ]',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Hann lét (ekki) gott að sér leiða"
            self.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    'VP > [ VP > [ .* \'láta\' .* ] .* NP > [ .* "gott" .* ] PP > [ "að" NP > [ ("mér"|"þér"|"sér"|"okkur") ] ] VP > { \'leiða\' } ]',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "...lét ég (ekki) gott að mér leiða"
            self.add_pattern(
                (
                    "leiða",  # Trigger lemma for this pattern
                    'VP > [ VP > [ .* \'láta\' .* ] .* IP > [ NP > [ .* "gott" PP > [ "að" NP > [ ("mér"|"þér"|"sér"|"okkur") ] ] ] VP > { \'leiða\' } ] ]',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            self.add_pattern(
                (
                    frozenset(("leiða", "leiður")),  # Trigger lemma for this pattern (probably a wrong parse)
                    'VP > [ .* \'láta\' .* NP-OBJ > [ .* "gott" .* ("að mér leiða" | "að sér leiða" | "að þér leiða") ] ]',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            self.add_pattern(
                (
                    frozenset(("leiða", "leiður")),  # Trigger lemma for this pattern (probably a wrong parse)
                    'VP > { IP-INF > { "að" "láta" } NP-PRD > { "gott" } PP > [ "að" ( "mér" | "þér" | "sér" ) "leiða" ] }',
                    self.wrong_preposition_að_leiða,
                    None,
                )
            )
            # Catch "Hún á/fær/hlýtur (ekki) heiðurinn að þessu.", "Hún hafði (ekki) fengið/hlotið heiðurinn að þessu." ÞA: Including 'eiga' here causes double annotation
            self.add_pattern(
                (
                    "heiður",  # Trigger lemma for this pattern
                    (
                        "( "
                        "VP > [ VP-AUX? .* VP > { ( 'fá'|'hljóta' ) } .* NP-OBJ > { 'heiður' PP > { P > { 'að' } NP } } ] "
                        "| "
                        "VP > [ VP-AUX? .* VP > { ( 'fá'|'hljóta' ) } .* NP-OBJ > { 'heiður' } PP > { P > { 'að' } NP } ] "
                        ") "
                    ),
                    self.wrong_preposition_heiður_að,
                    None,
                )
            )
            # Catch "Hún á (ekki) mikið/fullt/helling/gommu... að börnum."
            self.add_pattern(
                (
                    "eiga",  # Trigger lemma for this pattern
                    (
                        "( "
                        "VP > [ VP-AUX? .* VP > { 'eiga' } .* NP-OBJ PP > { P > { 'að' } NP } ] "
                        "| "
                        "VP > [ VP-AUX? .* VP > { 'eiga' } .* NP-OBJ > { PP > { P > { 'að' } NP } } ] "
                        ") "
                    ),
                    self.wrong_preposition_eiga_að,
                    None,
                )
            )
            # Catch "Hún á (ekki) lítið að börnum."
            self.add_pattern(
                (
                    "eiga",  # Trigger lemma for this pattern
                    "VP > { VP > { 'eiga' } ADVP > { 'lítið' } PP > { P > { 'að' } NP } }",
                    self.wrong_preposition_eiga_að,
                    None,
                )
            )
            # Catch "Það er (ekki) til mikið að þessu."
            self.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } NP > { NP >> { 'til' } PP > { 'að' } } }",
                    self.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Mikið er til að þessu."
            self.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "( S|VP ) > { NP VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    self.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Ekki er mikið til að þessu."
            self.add_pattern(
                (
                    "vera",  # Trigger lemma for this pattern
                    "VP > { VP > { 'vera' } ADVP > { 'til' } PP > { 'að' } }",
                    self.wrong_preposition_vera_til_að,
                    None,
                )
            )
            # Catch "Hún hefur (ekki) gagn að þessu.", "Hún hefur (ekki) haft gagn að þessu."
            self.add_pattern(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "VP > { NP > { 'gagn' } PP > { \"að\" } }",
                    self.wrong_preposition_gagn_að,
                    None,
                )
            )
            # Catch "Hvaða gagn hef ég að þessu?"
            self.add_pattern(
                (
                    "gagn",  # Trigger lemma for this pattern
                    "S > { NP > { 'gagn' } IP > { VP > { VP > { 'hafa' } PP > { 'að' } } } }",
                    self.wrong_preposition_gagn_að,
                    None,
                )
            )
            # Catch "Fréttir bárust (ekki) að slysinu."
            self.add_pattern(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "( IP|VP ) > { NP > { 'frétt' } VP > { PP > { P > { 'að' } } } }",
                    self.wrong_preposition_frettir_að,
                    None,
                )
            )
            # Catch "Það bárust (ekki) fréttir að slysinu."
            self.add_pattern(
                (
                    "frétt",  # Trigger lemma for this pattern
                    "NP > { 'frétt' PP > { P > { 'að' } } }",
                    self.wrong_preposition_frettir_að,
                    None,
                )
            )
            # Catch "Hætta stafar (ekki) að þessu.", "Hætta hefur (ekki) stafað að þessu."
            self.add_pattern(
                (
                    "stafa",  # Trigger lemma for this pattern
                    "VP > { VP >> { 'stafa' } ( PP|ADVP ) > { 'að' } }",
                    self.wrong_preposition_stafa_að,
                    None,
                )
            )
            # Catch "Hún er (ekki) ólétt af sínu þriðja barni.", "Hún hefur (ekki) verið ólétt af sínu þriðja barni."
            self.add_pattern(
                (
                    "óléttur",  # Trigger lemma for this pattern
                    "VP > { NP > { 'óléttur' } PP > { 'af' } }",
                    self.wrong_preposition_ólétt_af,
                    None,
                )
            )
            # Catch "Hún heyrði að lausa starfinu.", "Hún hefur (ekki) heyrt að lausa starfinu."
            self.add_pattern(
                (
                    "heyra",  # Trigger lemma for this pattern
                    "( "
                    "VP > { VP >> { 'heyra' } PP > { 'að' } }"
                    "| "
                    "VP > [ VP > { 'heyra' } .* NP > { PP > { 'að' } } .* ]"
                    ") ",
                    self.wrong_preposition_heyra_að,
                    None,
                )
            )
            # Catch "Ég hef (ekki) gaman að henni.", "Ég hef aldrei haft gaman að henni."
            self.add_pattern(
                (
                    "gaman",  # Trigger lemma for this pattern
                    "( "
                    "VP > { VP > { 'hafa' } NP > { 'gaman' } PP > { 'að' } }"
                    "| "
                    "VP > [ .* VP > { 'hafa' } .* NP > { 'gaman' PP > { 'að' } } .* ]"
                    ")",
                    self.wrong_preposition_hafa_gaman_að,
                    None,
                )
            )
            # Catch "Ég var valinn að henni.", "Ég hafði (ekki) verið valinn að henni."
            self.add_pattern(
                (
                    "velja",  # Trigger lemma for this pattern
                    "VP > { VP > { 'velja' } PP > { 'að' } }",
                    # "NP-PRD > { NP-PRD > { 'velja' } PP > { 'að' } }",
                    self.wrong_preposition_valinn_að,
                    None,
                )
            )
            # Catch "Ég var ekki valinn að henni.", "Þau voru sérstaklega valin að stjórninni."
            self.add_pattern(
                (
                    "valinn",  # Trigger lemma for this pattern
                    "VP > { NP > { 'valinn' } PP > { 'að' } }",
                    self.wrong_preposition_valinn_að,
                    None,
                )
            )

        # Verbs used wrongly with particular nouns
        def wrong_noun(nouns: FrozenSet[str], tree: SimpleTree) -> bool:
            """Context matching function for the %noun macro in combinations
            of verbs and their noun objects"""
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & ALL_CASES).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_01: FrozenSet[str] = frozenset(
            [
                "ósigur_þf",
                "hnekkir_þf",
                "álitshnekkir_þf",
                "afhroð_þf",
                "bani_þf",
                "færi_ef",
                "boð_ef",
                "átekt_ef",
            ]
        )
        # The macro %verb expands to "@'bjóða'" which matches terminals
        # whose corresponding token has a meaning with the 'bjóða' lemma.
        # The macro %noun is resolved by calling the function wrong_noun()
        # with the potentially matching tree node as an argument.
        self.ctx_verb_01 = {"verb": "@'bjóða'", "noun": partial(wrong_noun, NOUNS_01)}
        self.add_pattern(
            (
                "bjóða",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda match: self.wrong_verb_use(
                    match,
                    "bíða",
                    self.ctx_verb_01,
                ),
                self.ctx_verb_01,
            )
        )

        NOUNS_02: FrozenSet[str] = frozenset(["haus_þgf", "þvottur_þgf"])
        self.ctx_verb_02 = {"verb": "@'hegna'", "noun": partial(wrong_noun, NOUNS_02)}
        self.add_pattern(
            (
                "hegna",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP-OBJ >> { %noun } }",
                lambda match: self.wrong_verb_use(
                    match,
                    "hengja",
                    self.ctx_verb_02,
                ),
                self.ctx_verb_02,
            )
        )

        # 'af' incorrectly used with particular nouns
        def wrong_noun_af(nouns: FrozenSet[str], tree: SimpleTree) -> bool:
            """Context matching function for the %noun macro in combination
            with 'af'"""
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            try:
                case = (set(tree.variants) & {"nf", "þf", "þgf", "ef"}).pop()
            except KeyError:
                return False
            return (lemma + "_" + case) in nouns

        NOUNS_AF: FrozenSet[str] = frozenset(("beiðni_þgf", "siður_þgf", "tilefni_þgf", "fyrirmynd_þgf"))
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        self.ctx_noun_af = {"noun": partial(wrong_noun_af, NOUNS_AF)}
        af_lemmas = set(n.split("_")[0] for n in NOUNS_AF)
        self.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "PP > { P > { 'af' } NP > { %noun } }",
                lambda match: self.wrong_af_use(match, self.ctx_noun_af),
                self.ctx_noun_af,
            )
        )

        NOUNS_AF_OBJ: FrozenSet[str] = frozenset(
            [
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
            ]
        )
        # The macro %noun is resolved by calling the function wrong_noun_af()
        # with the potentially matching tree node as an argument.
        self.ctx_noun_af_obj = {"noun": partial(wrong_noun_af, NOUNS_AF_OBJ)}
        af_lemmas = set(n.split("_")[0] for n in NOUNS_AF_OBJ)
        self.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "NP > { %noun PP > { P > { 'af' } } }",
                lambda match: self.wrong_af_use(match, self.ctx_noun_af_obj),
                self.ctx_noun_af_obj,
            )
        )
        self.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "VP > { VP >> { %noun } PP > { P > { 'af' } } }",
                lambda match: self.wrong_af_use(match, self.ctx_noun_af_obj),
                self.ctx_noun_af_obj,
            )
        )
        self.add_pattern(
            (
                af_lemmas,  # Trigger lemmas for this pattern
                "VP > { PP > { NP > %noun } PP > { 'af' } }",
                lambda match: self.wrong_af_use(match, self.ctx_noun_af_obj),
                self.ctx_noun_af_obj,
            )
        )

        def wrong_noun_að(nouns: Set[str], tree: SimpleTree) -> bool:
            """Context matching function for the %noun macro in combination
            with 'að'"""
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
        self.ctx_noun_að = {"noun": partial(wrong_noun_að, NOUNS_AÐ)}
        að_lemmas = set(n.split("_")[0] for n in NOUNS_AÐ)
        self.add_pattern(
            (
                að_lemmas,  # Trigger lemma for this pattern
                "PP > { P > { 'að' } NP > { %noun } }",
                lambda match: self.wrong_að_use(match, self.ctx_noun_að),
                self.ctx_noun_að,
            )
        )

        def maybe_place(tree: SimpleTree) -> bool:
            """Context matching function for the %maybe_place macro.
            Returns True if the associated lemma is an uppercase
            word that might be a place name."""
            lemma = tree.lemma
            return lemma[0].isupper() if lemma else False

        # Check prepositions used with place names
        self.ctx_place_names = {"maybe_place": maybe_place}
        self.add_pattern(
            (
                frozenset(("á", "í")),  # Trigger lemmas for this pattern
                "PP > { P > ('á' | 'í') NP > %maybe_place }",
                lambda match: self.check_pp_with_place(match),
                self.ctx_place_names,
            )
        )
        # Check use of 'bjóða e-m birginn' instead of 'bjóða e-m byrginn'
        # !!! TODO: This is a provisional placeholder for similar cases
        self.add_pattern(
            (
                "birgir",  # Trigger lemma for this pattern
                "VP > [ VP > { 'bjóða' } .* NP-IOBJ .* NP-OBJ > { \"birginn\" } ]",
                self.wrong_noun_with_verb,
                None,
            )
        )
        # Check use of "vera að" instead of a simple verb
        self.add_pattern(
            (
                "vera",  # Trigger lemma for this pattern
                "IP > {VP > [VP > { @'vera' } (ADVP|NP-SUBJ)? IP-INF > {TO > nhm}]}",
                lambda match: self.vera_að(match),
                None,
            )
        )

        # Check mood in subclauses

        # Concessive clause - viðurkenningarsetning
        self.add_pattern(
            (
                frozenset(("þrátt fyrir", "þrátt", "þó", "þótt")),  # Trigger lemmas for this pattern
                "CP-ADV-ACK > { IP >> {VP > so_fh} }",
                lambda match: self.mood_ind("ACK", match),
                None,
            )
        )
        # Relative clause - tilvísunarsetning
        self.add_pattern(
            (
                frozenset(("sem", "er")),  # Trigger lemmas for this pattern
                "CP-REL > { IP >> {VP > so_vh} }",
                lambda match: self.mood_sub("REL", match),
                None,
            )
        )
        # Temporal clause - tíðarsetning
        self.add_pattern(
            (
                frozenset(("áður", "eftir", "þangað", "þegar")),  # Trigger lemmas for this pattern
                "CP-ADV-TEMP > { IP >> {VP > so_vh} }",
                lambda match: self.mood_sub("TEMP/w", match),
                None,
            )
        )
        # Conditional clause - skilyrðissetning
        self.add_pattern(
            (
                frozenset(("ef", "svo")),  # Trigger lemmas for this pattern
                "CP-ADV-COND > { IP >> {VP > so_vh} }",
                lambda match: self.mood_sub("COND", match),
                None,
            )
        )
        # Purpose clause - tilgangssetning
        self.add_pattern(
            (
                frozenset(("til", "svo")),  # Trigger lemmas for this pattern
                "CP-ADV-PURP > { IP >> {VP > so_fh} }",
                lambda match: self.mood_ind("PURP", match),
                None,
            )
        )
        # Article errors; demonstrative pronouns and nouns with an article
        self.add_pattern(
            (
                frozenset(("sá", "þessi")),  # Trigger lemmas for this pattern
                "NP > [.* fn .* no_gr]",
                lambda match: self.doubledefinite(match),
                None,
            )
        )

        # Check errors in dir4loc
        def dir4loc(verbs: FrozenSet[str], tree: SimpleTree) -> bool:
            """Context matching function for the %noun macro in combination
            with 'að'"""
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            return lemma in verbs

        VERBS = frozenset(("safna", "kaupa", "læsa", "geyma"))
        # The macro %verb is resolved by calling the function dir4loc()
        # with the potentially matching tree node as an argument.
        self.ctx_dir_loc = {"verb": partial(dir4loc, VERBS)}
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP > { PP > { ADVP > { 'út' } P > { 'í' } NP > { 'búð' } } } }",
                lambda match: self.dir_loc(match),
                self.ctx_dir_loc,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } ADVP > { 'saman' } PP > { ADVP > { 'inn' } P > { 'í' } NP } }",
                lambda match: self.dir_loc(match),
                self.ctx_dir_loc,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } NP > { PP > { ADVP > { 'inn' } } } }",
                lambda match: self.dir_loc(match),
                self.ctx_dir_loc,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { %verb } ADVP > { 'inn' } }",
                lambda match: self.dir_loc(match),
                self.ctx_dir_loc,
            )
        )

        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* VP > { 'hafa' } .* ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "NP > [ ( no_nf|pfn_nf ) PP > [ ADVP > { 'út' } PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > { IP >> [ .* ADVP > { 'út' } ] PP > [ P > { ( 'í'|'á'|'um' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > [ .* VP > { VP > [ 'vera' ] IP >> { ADVP > [ 'út' ] } } .* PP > [ P > [ 'á' ] NP > { ( no_þgf|pfn_þgf ) } ] .* ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > [ VP > [ 'gera' ] NP > [ .* PP > { ADVP > { 'út' } P > [ 'í' ] NP } ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "útá",  # Trigger lemma for this pattern
                "PP > { P > { 'útá' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "útí",  # Trigger lemma for this pattern
                "PP > { P > { 'útí' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'inn' } PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } .* ]",
                #    "( PP|VP|IP ) > [ .* ADVP > { 'inn' } .* PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } .* ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > { IP >> [ .* ADVP > { 'inn' } ] PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "NP > { IP >> { VP > { 'vera' } ADVP > { 'inn' } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP > { 'verða' } ADVP > { 'inn' } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "geyma",  # Trigger lemma for this pattern
                "VP > { VP > { 'geyma' } ADVP > { 'inn' } PP }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inná",  # Trigger lemma for this pattern
                "PP > { P > { 'inná' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inní",  # Trigger lemma for this pattern
                "VP > { VP > [ .* ] NP > { PP > { P > { 'inní' } NP > { ( no_þgf|pfn_þgf ) } } } }",
                # "PP > { P > { 'inní' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > [ VP > { ( 'verða'|'vera' ) } .* ADVP > { 'inn' } PP > { P > { 'á' } } ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > [ VP > { 'vera' } .* ADVP > { 'inn' } PP > { P > { 'í' } } ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        # Catches "Ég hef upp á honum."
        self.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ VP > { ('standa'|'hafa') } .* ADVP > { 'upp' } PP > { P > { ( 'í'|'á' ) } NP > { ( no_þgf|pfn_þgf ) } } ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        # Catches "Það liggur í augum upp."
        self.add_pattern(
            (
                "auga",  # Trigger lemma for this pattern
                "VP > [ VP > [ 'liggja' ] PP > [ P > { ( 'í'|'á' ) } NP > { 'auga' } ] ADVP > [ 'upp' ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "teningur",  # Trigger lemma for this pattern
                "PP > [ .* ADVP > { 'upp' } PP > { P > { ( 'í'|'á' ) } NP > { 'teningur' } } ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "( IP|NP|VP ) > [ IP >> { ADVP > { 'upp' } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )

        self.add_pattern(
            (
                "upp",  # Trigger lemma for this pattern
                "VP > [ VP >> { VP > { VP > { 'hafa' } ADVP > { 'upp' } } } PP > [ P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) ] .* ]",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "uppá",  # Trigger lemma for this pattern
                "VP > { VP > { 'taka' } NP > { PP > { P > { 'uppá' } NP > { ( no_þgf|pfn_þgf|no_þf|pfn_þf ) } } } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "uppí",  # Trigger lemma for this pattern
                "PP > { P > { 'uppí' } NP > { ( no_þgf|pfn_þgf ) } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "uppí",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } PP > { P > { 'uppí' } NP > { ( no_þf|pfn_þf ) } } }",
                lambda match: self.dir_loc_comp(match),
                None,
            )
        )
        self.add_pattern(
            (
                "niður",  # Trigger lemma for this pattern
                "( PP|VP|IP ) > [ .* ADVP > { 'niður' } PP > { P > { ( 'í'|'á' ) } NP > ( no_þgf|pfn_þgf ) } ]",
                lambda match: self.dir_loc_simple(match),
                None,
            )
        )
        self.add_pattern(
            (
                "niður",  # Trigger lemma for this pattern
                "VP > [ VP > { 'vera' } .* PP > { ADVP > { 'niður' } P > { 'í' } NP } ]",
                lambda match: self.dir_loc_simple(match),
                None,
            )
        )
        self.add_pattern(
            (
                "verða",  # Trigger lemma for this pattern
                "VP > { VP > { 'verða' } NP > { ( pfn_þgf|abfn_þgf ) } NP > { 'út' 'um' } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "standa",  # Trigger lemma for this pattern
                "IP > { ADVP > { 'upp' } VP > { VP > { 'vera' } NP > { 'standa' } } }",
                lambda match: self.dir_loc_simple(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] NP > { PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } } }",
                lambda match: self.dir_loc_simple(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] NP > [ .* PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } ] }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP PP >> { NP > { PP > { ADVP > { 'út' } PP > { P > { 'um' } NP } } } } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' ] ADVP > [ 'út' ] PP > { P > [ 'um' ] } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' .* ] NP > { 'út' 'um' } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "útum",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } NP > { 'útum' } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "útum",  # Trigger lemma for this pattern
                "VP > { VP > { 'sækja' } PP > { 'um' } NP > { 'útum' } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > [ 'vera' .* ] ADVP > { 'út' } PP > { P > { 'um' } NP } }",
                lambda match: self.dir_loc_ut_um(match),
                None,
            )
        )
        self.add_pattern(
            (
                "út",  # Trigger lemma for this pattern
                "VP > { VP > { 'gera' } NP > [ .* PP > { ADVP > { 'út' } P > { 'í' } } ] }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "inn",  # Trigger lemma for this pattern
                "VP > { VP >> { ADVP > { 'hér' } } PP > { ADVP > { 'inn' } } }",
                lambda match: self.dir_loc(match),
                None,
            )
        )
        self.add_pattern(
            (
                "Skagi",  # Trigger lemma for this pattern
                "VP > { VP > { 'vera' } PP >> { PP > { ADVP > { 'upp' } P > { 'á' } NP > { 'Skagi' } } } }",
                lambda match: self.dir_loc(match),
                None,
            )
        )

        self.add_pattern(
            (
                "né",  # Trigger lemma for this pattern
                " IP >> { 'né' } ",
                lambda match: self.né(match),
                None,
            )
        )

        def subjsing(nouns: FrozenSet[str], tree: SimpleTree) -> bool:
            """Context matching function for the %noun macro"""
            if not tree.is_terminal:
                return False
            if "et" not in tree.all_variants:
                return False
            lemma = tree.own_lemma
            if not lemma:
                # The passed-in tree node is probably not a terminal
                return False
            return lemma in nouns

        NOUNS_NUM = frozenset(("þríeyki", "tvíeyki", "hluti", "hópur"))
        # The macro %noun is resolved by calling the function subjnum()
        # with the potentially matching tree node as an argument.
        self.ctx_subjsing = {"noun": partial(subjsing, NOUNS_NUM)}

        self.add_pattern(
            (
                NOUNS_NUM,  # Trigger lemmas for this pattern
                "NP-SUBJ >> [ %noun .* 'og' ]",
                lambda match: self.singsub("QUANT", match),
                self.ctx_subjsing,
            )
        )
        self.add_pattern(
            (
                NOUNS_NUM,  # Trigger lemmas for this pattern
                "NP-SUBJ >> [ %noun .* NP-POSS >> { no_ft_ef } ]",
                lambda match: self.singsub("GEN", match),
                self.ctx_subjsing,
            )
        )
        self.add_pattern(
            (
                NOUNS_NUM,
                "NP-SUBJ >> [ %noun .* PP >> [ no_ft_ef  ]]",
                lambda match: self.singsub("AF", match),
                self.ctx_subjsing,
            )
        )

    def run(self) -> None:
        """Apply the patterns to the sentence"""
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
            """Returns True if any of the given trigger lemmas
            occur in the sentence"""
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
                    func(match)
