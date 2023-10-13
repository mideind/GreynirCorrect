"""

    Greynir: Natural language processing for Icelandic

    Error finder for parse trees

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


    This module implements the ErrorFinder class. The class is used
    in checker.py to find occurrences of error-marked nonterminals within
    parse trees and annotate the associated error or warning.

    Specifically, ErrorFinder finds nonterminals that are marked with $tag(error)
    in the CFG (Greynir.grammar). These nonterminals give rise to error
    or warning annotations for their respective token spans.

"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, cast
from typing_extensions import Protocol, TypedDict

import re

from reynir import TOK, Sentence, Tok, correct_spaces
from reynir.binparser import BIN_Terminal, BIN_Token
from reynir.fastparser import Node, ParseForestNavigator
from reynir.settings import VerbSubjects
from reynir.simpletree import SimpleTree
from reynir.verbframe import VerbErrors, VerbFrame

from .annotation import Annotation
from .errtokenizer import emulate_case
from .pattern import PatternMatcher


# Typing stuff
class AnnotationDict(TypedDict):
    """The annotation for a sentence"""

    text: str
    detail: str
    start: int
    end: int
    original: Optional[str]
    suggest: Optional[str]


AnnotationTuple2 = Tuple[str, str]
AnnotationTuple4 = Tuple[str, int, int, str]
AnnotationTuple6 = Tuple[str, str, int, int, str, str]
AnnotationReturn = Union[None, str, AnnotationTuple2, AnnotationTuple4, AnnotationTuple6, AnnotationDict]
AnnotationFunc = Callable[[str, str, Node], AnnotationReturn]


class CastFunction(Protocol):
    """Type annotation protocol for a case casting function"""

    def fget(self, tree: SimpleTree) -> str:
        ...


# Case name prefixes
CASE_NAMES = {"nf": "nefni", "þf": "þol", "þgf": "þágu", "ef": "eignar"}

# Replacements for numeric ordinals, in the various genders and cases
ORDINALS = {
    1: {
        "kk": {"nf": "fyrsti", "þf": "fyrsta", "þgf": "fyrsta", "ef": "fyrsta"},
        "kvk": {"nf": "fyrsta", "þf": "fyrstu", "þgf": "fyrstu", "ef": "fyrstu"},
        "hk": {"nf": "fyrsta", "þf": "fyrsta", "þgf": "fyrsta", "ef": "fyrsta"},
    },
    2: {
        "kk": {"nf": "annar", "þf": "annan", "þgf": "öðrum", "ef": "annars"},
        "kvk": {"nf": "önnur", "þf": "aðra", "þgf": "annarri", "ef": "annarrar"},
        "hk": {"nf": "annað", "þf": "annað", "þgf": "öðru", "ef": "annars"},
    },
    3: {
        "kk": {"nf": "þriðji", "þf": "þriðja", "þgf": "þriðja", "ef": "þriðja"},
        "kvk": {"nf": "þriðja", "þf": "þriðju", "þgf": "þriðju", "ef": "þriðju"},
        "hk": {"nf": "þriðja", "þf": "þriðja", "þgf": "þriðja", "ef": "þriðja"},
    },
    4: {
        "kk": {"nf": "fjórði", "þf": "fjórða", "þgf": "fjórða", "ef": "fjórða"},
        "kvk": {"nf": "fjórða", "þf": "fjórðu", "þgf": "fjórðu", "ef": "fjórðu"},
        "hk": {"nf": "fjórða", "þf": "fjórða", "þgf": "fjórða", "ef": "fjórða"},
    },
    5: {
        "kk": {"nf": "fimmti", "þf": "fimmta", "þgf": "fimmta", "ef": "fimmta"},
        "kvk": {"nf": "fimmta", "þf": "fimmtu", "þgf": "fimmtu", "ef": "fimmtu"},
        "hk": {"nf": "fimmta", "þf": "fimmta", "þgf": "fimmta", "ef": "fimmta"},
    },
    6: {
        "kk": {"nf": "sjötti", "þf": "sjötta", "þgf": "sjötta", "ef": "sjötta"},
        "kvk": {"nf": "sjötta", "þf": "sjöttu", "þgf": "sjöttu", "ef": "sjöttu"},
        "hk": {"nf": "sjötta", "þf": "sjötta", "þgf": "sjötta", "ef": "sjötta"},
    },
    7: {
        "kk": {"nf": "sjöundi", "þf": "sjöunda", "þgf": "sjöunda", "ef": "sjöunda"},
        "kvk": {"nf": "sjöunda", "þf": "sjöundu", "þgf": "sjöundu", "ef": "sjöundu"},
        "hk": {"nf": "sjöunda", "þf": "sjöunda", "þgf": "sjöunda", "ef": "sjöunda"},
    },
    8: {
        "kk": {"nf": "áttundi", "þf": "áttunda", "þgf": "áttunda", "ef": "áttunda"},
        "kvk": {"nf": "áttunda", "þf": "áttundu", "þgf": "áttundu", "ef": "áttundu"},
        "hk": {"nf": "áttunda", "þf": "áttunda", "þgf": "áttunda", "ef": "áttunda"},
    },
    9: {
        "kk": {"nf": "níundi", "þf": "níunda", "þgf": "níunda", "ef": "níunda"},
        "kvk": {"nf": "níunda", "þf": "níundu", "þgf": "níundu", "ef": "níundu"},
        "hk": {"nf": "níunda", "þf": "níunda", "þgf": "níunda", "ef": "níunda"},
    },
}


class ErrorDetectionToken(BIN_Token):

    """A subclass of BIN_Token that adds error detection behavior
    to the base class"""

    _VERB_ERROR_SUBJECTS = VerbSubjects.VERBS_ERRORS

    def __init__(self, t: Tok, original_index: int) -> None:
        """original_index is the index of this token in
        the original token list, as submitted to the parser,
        including not-understood tokens such as quotation marks"""
        super().__init__(t, original_index)
        # Store the capitalization state, carried over from CorrectToken instances.
        # The state is one of (None, "sentence_start", "after_ordinal", "in_sentence").
        # Since some token objects may be instances of Tok, not CorrectToken,
        # we tread carefully here.
        self._cap = getattr(t, "_cap", None)

    @property
    def cap_sentence_start(self) -> bool:
        """True if this token appears at sentence start"""
        return self._cap == "sentence_start"

    @property
    def cap_after_ordinal(self) -> bool:
        """True if this token appears after an ordinal at sentence start"""
        return self._cap == "after_ordinal"

    @property
    def cap_in_sentence(self) -> bool:
        """True if this token appears within a sentence"""
        return self._cap == "in_sentence"

    @classmethod
    def verb_is_strictly_impersonal(cls, verb: str, form: str) -> bool:
        """Return True if the given verb should not be allowed to match
        with a normal (non _op) verb terminal"""
        if "OP" in form and not VerbSubjects.is_strictly_impersonal(verb):
            # We have a normal terminal, but an impersonal verb form. However,
            # that verb is not marked with an error correction from nominative
            # case to another case. We thus return True to prevent token-terminal
            # matching, since we don't have this specified as a verb error.
            return True
        # For normal terminals and impersonal verbs, we allow the match to
        # proceed if we have a specified error correction from a nominative
        # subject case to a different subject case.
        # Example: 'Tröllskessan dagaði uppi' where 'daga' is an impersonal verb
        # having a specified correction from nominative to accusative case.
        return False

    @classmethod
    def verb_cannot_be_impersonal(cls, verb: str, form: str) -> bool:
        """Return True if this verb cannot match an so_xxx_op terminal"""
        # We have a relaxed condition here because we want to catch
        # verbs being used impersonally that shouldn't be. So we don't
        # check for "OP" (impersonal) in the form, but we're not so relaxed
        # that we accept "BH" (imperative) or "NH" (infinitive) forms.
        # We also don't accept plural forms, as those errors would be
        # very improbable ("okkur hlökkum til jólanna").
        return any(f in form for f in ("BH", "NH", "FT"))

    # Variants that must be present in the verb form if they
    # are present in the terminal. We cut away the "op"
    # element of the tuple, since we want to allow impersonal
    # verbs to appear as normal verbs.
    _RESTRICTIVE_VARIANTS = ("sagnb", "lhþt", "bh")

    @classmethod
    def verb_subject_matches(cls, verb: str, subj: str) -> bool:
        """Returns True if the given subject type/case is allowed
        for this verb or if it is an erroneous subject
        which we can flag"""
        return subj in cls._VERB_SUBJECTS.get(verb, set()) or subj in cls._VERB_ERROR_SUBJECTS.get(verb, dict())

    @classmethod
    def verb_matches_arguments(cls, key: str) -> bool:
        """Return True if the given arguments are allowed for this verb
        or if these are erroneous arguments which we can flag"""
        # This function overrides BIN_Token.verb_matches_arguments()
        return VerbFrame.matches_arguments(key) or VerbFrame.matches_error_arguments(key)


class ErrorFinder(ParseForestNavigator):

    """Utility class to find nonterminals in parse trees that are
    tagged as errors in the grammar, and terminals matching
    verb forms marked as errors"""

    _CAST_FUNCTIONS: Dict[str, CastFunction] = {
        "nf": cast(CastFunction, SimpleTree.nominative_np),
        "þf": cast(CastFunction, SimpleTree.accusative_np),
        "þgf": cast(CastFunction, SimpleTree.dative_np),
        "ef": cast(CastFunction, SimpleTree.genitive_np),
    }

    _NON_OP_VERB_FORMS: Mapping[str, Tuple[str, str]] = {
        "lýst": (
            "líst",
            "'Lýst' á sennilega að vera 'líst', þ.e. sögnin 'að líta(st)' " "í stað sagnarinnar 'að ljósta'.",
        ),
    }

    def __init__(self, ann: List[Annotation], sent: Sentence) -> None:
        super().__init__(visit_all=True)
        # Annotation list
        self._ann = ann
        # The original sentence object
        self._sent = sent
        # Token list
        self._tokens = sent.tokens
        # Terminal node list
        self._terminal_nodes = sent.terminal_nodes

    def run(self) -> Any:
        """Start navigating the deep tree structure of the sentence"""
        return super().go(self._sent.deep_tree)

    @staticmethod
    def node_span(node: Node) -> Tuple[int, int]:
        """Return the start and end indices of the tokens
        spanned by the given node"""
        first_token, last_token = node.token_span
        return (first_token.index, last_token.index)

    def cast_to_case(self, case: str, tree: SimpleTree) -> str:
        """Return the contents of a noun phrase node
        inflected in the given case"""
        return self._CAST_FUNCTIONS[case].fget(tree)

    def _simple_tree(self, node: Node) -> Optional[SimpleTree]:
        """Return a SimpleTree instance spanning the deep tree
        of which node is the root"""
        if node is None:
            return None
        first, last = self.node_span(node)
        toklist = self._tokens[first : last + 1]
        return SimpleTree.from_deep_tree(node, toklist, first_token_index=first)

    def node_text(self, node: Node, original_case: bool = False) -> str:
        """Return the text within the span of the node"""

        def text(t: Tok) -> str:
            """If the token t is a word token, return a lower case
            version of its text, unless we have a reason to keep
            the original case, i.e. if it is a lemma that is upper case
            in BÍN"""
            if t.kind != TOK.WORD:
                # Not a word token: keep the original text
                return t.txt
            if len(t.txt) > 1 and t.txt.isupper():
                # All uppercase: keep it that way
                return t.txt
            if t.val and any(m.stofn[0].isupper() for m in t.meanings):
                # There is an uppercase lemma for this word in BÍN:
                # keep the original form
                return t.txt
            # No uppercase lemma in BÍN: return a lower case copy
            return t.txt.lower()

        first, last = self.node_span(node)
        text_func: Callable[[Tok], str] = (lambda t: t.txt) if original_case else text
        return correct_spaces(" ".join(text_func(t) for t in self._tokens[first : last + 1] if t.txt))

    # Functions used to explain grammar errors associated with
    # nonterminals with error tags in the grammar

    def AðvörunHeldur(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'heldur' er ofaukið
        orig_txt = self.node_text(node, original_case=True)
        start, end = self.node_span(node)
        return AnnotationDict(
            text="'{0}' er sennilega ofaukið".format(txt),
            detail="Yfirleitt nægir að nota 'en' í þessu samhengi, 'heldur en' telst frekar óformlegt. "
            "Það á þó stundum rétt hjá sér í löngum setningum ef langt er frá upphaflega samanburðarliðnum til seinni "
            "samanburðarliðarins til að auka skýrleika textans.",
            original=orig_txt,
            start=start,
            end=end,
            suggest="",  # Word should be deleted
        )

    def AðvörunSíðan(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'fyrir tveimur mánuðum síðan'
        orig_txt = self.node_text(node, original_case=True)
        start, end = self.node_span(node)
        return AnnotationDict(
            text="'síðan' er sennilega ofaukið",
            detail=("Yfirleitt er óþarfi að nota orðið 'síðan' í samhengi á borð við " "'fyrir tveimur dögum (síðan)'"),
            start=start,
            end=end,
            original=emulate_case("síðan", template=orig_txt),
            suggest="",
        )

    def VillaVístAð(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'víst að' á sennilega að vera 'fyrst að'
        ch0, _ = node.enum_child_nodes()
        start, end = self.node_span(node)
        orig = self.node_text(node)
        if ch0:
            start, end = self.node_span(ch0)
            orig = self.node_text(ch0, original_case=True)
        return AnnotationDict(
            text="'{0}' á sennilega að vera 'fyrst'".format(orig),
            detail=("Rétt er að nota 'fyrst að' fremur en 'víst að' " " til að tengja saman atburð og forsendu."),
            start=start,
            end=end,
            original=orig,
            suggest=emulate_case("fyrst", template=orig),
        )

    def VillaFráÞvíAð(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # '(allt) frá því' á sennilega að vera '(allt) frá því að'
        children = list(node.enum_child_nodes())
        start, end = self.node_span(node)
        orig_txt = self.node_text(node)
        ch = children[-1]
        if ch:
            start, end = self.node_span(ch)
            orig_txt = self.node_text(ch, original_case=True)
        return AnnotationDict(
            text="'{0}' á sennilega að vera '{0} að'".format(txt),
            detail=(
                "Rétt er að nota samtenginguna 'að' í þessu samhengi, " "til dæmis 'allt frá því að Anna hóf námið'."
            ),
            start=start,
            end=end,
            original=orig_txt,
            suggest=f"{orig_txt} að",
        )

    def VillaAnnaðhvort(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Í stað 'annaðhvort' á sennilega að standa 'annað hvort'
        orig_txt = self.node_text(node, original_case=True)
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Í stað '{0}' á sennilega að standa 'annað hvort'".format(txt),
            detail=(
                "Rita á 'annað hvort' þegar um er að ræða fornöfn, til dæmis "
                "'annað hvort systkinanna'. Rita á 'annaðhvort' í samtengingunni "
                "'annaðhvort eða', til dæmis 'Annaðhvort fer ég út eða þú'."
            ),
            start=start,
            end=end,
            original=orig_txt,
            suggest=emulate_case("annað hvort", template=orig_txt),
        )

    def VillaAnnaðHvort(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Í stað 'annað hvort' á sennilega að standa 'annaðhvort'
        orig_txt = self.node_text(node, original_case=True)
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Í stað '{0}' á sennilega að standa 'annaðhvort'".format(txt),
            detail=(
                "Rita á 'annaðhvort' í samtengingu, til dæmis "
                "'Annaðhvort fer ég út eða þú'. "
                "Rita á 'annað hvort' í tveimur orðum þegar um er að ræða fornöfn, "
                "til dæmis 'annað hvort systkinanna'."
            ),
            start=start,
            end=end,
            original=orig_txt,
            suggest=emulate_case("annaðhvort", template=orig_txt),
        )

    def VillaTvípunkturFs(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Tvípunktur milli forsetningar og nafnliðar
        _, ch1, _ = node.enum_child_nodes()
        start, end = self.node_span(node)
        correct = re.sub(r"\s*:", "", txt)
        if ch1:
            start, end = self.node_span(ch1)
            correct = ""
        return AnnotationDict(
            text="Tvípunktur er óþarfi",
            detail=("Óþarft er að hafa tvípunkt milli forsetningar og nafnliðarins " "sem hún stýrir falli á."),
            start=start,
            end=end,
            original=txt,
            suggest=correct,
        )

    def VillaAnnara(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Í stað fornafnsins 'annarra' í eignarfalli fleirtölu er ritað 'annara'
        d = {"annara": ("annar", "annarra"), "nokkura": ("nokkur", "nokkurra")}
        lemma, correct = d[txt.lower()]
        start, end = self.node_span(node)
        return AnnotationDict(
            text=f"Á líklega að vera '{correct}'",
            detail=f"Fornafnið '{lemma}' er ritað '{correct}' í eignarfalli fleirtölu.",
            start=start,
            end=end,
            original=txt,
            suggest=correct,
        )

    def VillaAnnari(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Í stað fornafnsins 'annarri' í þágufalli eintölu er ritað 'annari'
        d = {"annari": ("annar", "annarri"), "nokkuri": ("nokkur", "nokkurri")}
        lemma, correct = d[txt.lower()]
        start, end = self.node_span(node)
        return AnnotationDict(
            text=f"Á líklega að vera '{correct}'",
            detail=f"Fornafnið '{lemma}' er ritað '{correct}' " "í þágufalli, eintölu, kvenkyni.",
            start=start,
            end=end,
            original=txt,
            suggest=correct,
        )

    def VillaAnnarar(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Í stað fornafnsins 'annarrar' í þágufalli eintölu er ritað 'annarar'
        d = {"annarar": ("annar", "annarrar"), "nokkurar": ("nokkur", "nokkurrar")}
        lemma, correct = d[txt.lower()]
        start, end = self.node_span(node)
        return AnnotationDict(
            text=f"Á líklega að vera '{correct}'",
            detail=f"Fornafnið '{lemma}' er ritað '{correct}' " "í eignarfalli, eintölu, kvenkyni.",
            start=start,
            end=end,
            original=txt,
            suggest=correct,
        )

    def singular_error(self, txt: str, variants: str, node: Node, detail: str) -> AnnotationReturn:
        """Annotate a mismatch between singular and plural
        in subject vs. verb"""
        tnode = self._terminal_nodes[node.start]
        # Find the enclosing inflected phrase
        ip = tnode.enclosing_tag("IP")
        vp: Optional[SimpleTree] = None
        if ip is not None:
            try:
                # Found it: try to locate the main verb
                vp = ip.VP.VP
            except AttributeError:
                try:
                    vp = ip.VP
                except AttributeError:
                    pass
        so = None if vp is None else vp.first_match("so_ft")

        if so is not None:
            start, end = so.span
            vars = set(so.all_variants) - {"ft"}
            vars.add("et")
            suggest = PatternMatcher.get_wordform(so.text.lower(), so.lemma, so.cat, vars)
            return AnnotationDict(
                text=("Sögnin '{0}' á sennilega að vera í eintölu, ekki fleirtölu".format(so.tidy_text)),
                detail=detail,
                start=start,
                end=end,
                original=so.tidy_text,
                suggest=suggest,
            )
        return f"Sögn sem á við '{txt}' á sennilega að vera í eintölu, ekki fleirtölu"

    def VillaFjöldiHluti(self, txt: str, variants: str, node: Node) -> AnnotationReturn:
        # Sögn sem á við 'fjöldi Evrópuríkja' á að vera í eintölu
        # 'mikill:lo'_et_kk/fall? FjöldiHluti_et_kk/fall EfLiður_ft/kyn
        tnode = self._terminal_nodes[node.start]
        # Find the closing inflected phrase
        ip = tnode.enclosing_tag("IP")
        if ip is None:
            return "Sögnin á sennilega að vera í eintölu eins og frumlagið"
        verb = ip.first_match("so_ft")
        if verb is None:
            return "Sögnin á sennilega að vera í eintölu eins og frumlagið"
        start, end = verb.span
        children = list(node.enum_child_nodes())
        _, ch1, ch2 = children
        if ch1 is None or ch2 is None:
            return "Sögnin á sennilega að vera í eintölu eins og frumlagið"
        subjtext = self.node_text(ch1) + " " + self.node_text(ch2)
        vars = set(verb.all_variants) - {"ft"}
        vars.add("et")
        suggestion = PatternMatcher.get_wordform(verb.text.lower(), verb.lemma, verb.cat, vars)
        return AnnotationDict(
            text=f"Sögnin '{verb.tidy_text}' á sennilega að vera í eintölu eins og frumlagið '{subjtext}'",
            detail=f"Orðið '{self.node_text(ch1)}' stjórnar tölu sagnarinnar svo hún á að vera í eintölu.",
            start=start,
            end=end,
            original=verb.tidy_text,
            suggest=suggestion,
        )

    def VillaEinnAf(self, txt: str, variants: str, node: Node) -> AnnotationReturn:
        # Sögn sem á við 'einn af drengjunum' á að vera í eintölu
        return self.singular_error(
            txt,
            variants,
            node,
            f"Nafnliðurinn '{txt}' er í eintölu " f"og með honum á því að vera sögn í eintölu.",
        )

    def VillaEinkunn(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # Fornafn í einkunn er ekki í sama falli og nafnorð,
        # t.d. 'þessum mann'
        wrong_pronoun = self.node_text(node, original_case=True)
        correct_case = variants.split("_")[0]
        pronoun_node = next(node.enum_child_nodes())
        assert pronoun_node is not None
        p = self._simple_tree(pronoun_node)
        assert p is not None
        correct_pronoun = self.cast_to_case(correct_case, p)
        start, end = self.node_span(pronoun_node)
        return AnnotationDict(
            text="'{0}' á sennilega að vera '{1}'".format(wrong_pronoun, correct_pronoun),
            detail=(
                "Fornafnið '{0}' á að vera í {1}falli, eins og "
                "nafnliðurinn sem fylgir á eftir".format(wrong_pronoun, CASE_NAMES[correct_case])
            ),
            start=start,
            end=end,
            original=wrong_pronoun,
            suggest=correct_pronoun,
        )

    def AðvörunSemOg(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'sem' er sennilega ofaukið
        ch1, ch2 = self._terminal_nodes[node.start : node.end]
        start, _ = ch1.span
        _, end = ch2.span
        return AnnotationDict(
            text="'{0}' er að öllum líkindum ofaukið".format(ch1.text),
            detail=("Oft fer betur á að rita 'og', 'og einnig', 'og jafnframt' " " eða 'og auk þess' í stað 'sem og'."),
            start=start,
            end=end,
            original=txt,
            suggest="og einnig",
        )

    def AðvörunAð(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'að' er sennilega ofaukið
        start, end = self.node_span(node)
        return AnnotationDict(
            text="'{0}' er að öllum líkindum ofaukið".format(txt),
            detail=("'að' er yfirleitt ofaukið í samtengingum á " "borð við 'áður en', 'síðan', 'enda þótt' o.s.frv."),
            start=start,
            end=end,
            original=txt,
            suggest="",  # Token should be deleted
        )

    def AðvörunKomma(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Komma er líklega óþörf",
            detail=(
                "Kommu er yfirleitt ofaukið milli frumlags og umsagnar, "
                "milli forsendu og meginsetningar "
                "('áður en ég fer [,] má sækja tölvuna') o.s.frv."
            ),
            start=start,
            end=end,
            original=",",
            suggest="",  # Token should be deleted
        )

    def VillaNé(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        start, end = self.node_span(node)
        return AnnotationDict(
            text="'né' gæti átt að vera 'eða'",
            detail="'né' er hluti af margorða samtengingunni 'hvorki né' en getur ekki "
            "staðið ein og sér sem aðaltenging.",
            original="né",
            start=start,
            end=end,
            suggest="eða",
        )

    def VillaÞóAð(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # [jafnvel] þó' á sennilega að vera '[jafnvel] þó að
        suggestion = f"{txt} að"
        start, end = self.node_span(node)
        return AnnotationDict(
            text="'{0}' á sennilega að vera '{1}' (eða 'þótt')".format(txt, suggestion),
            detail=(
                "Réttara er að nota samtenginguna 'að' í samhengi á borð við " "'jafnvel þó að von sé á sólskini'."
            ),
            start=start,
            end=end,
            original=txt,
            suggest=suggestion,
        )

    def VillaÍTölu(self, txt: str, variants: str, node: Node) -> Optional[AnnotationDict]:
        # Sögn á að vera í sömu tölu og frumlag
        children = list(node.enum_child_nodes())
        ch0, ch1 = children
        assert ch0 is not None
        subject = self.node_text(ch0)
        number = "eintölu" if "et" in variants else "fleirtölu"
        # Find the verb
        ch1node = cast(Node, ch1)
        vptree: Optional[SimpleTree] = self._simple_tree(ch1node)
        so: Optional[SimpleTree] = None
        if vptree is not None:
            vp: Optional[SimpleTree] = vptree.first_match("VP >> so")
            if vp is not None:
                so = vp.first_match("so")
        # Annotate the verb phrase
        assert ch1 is not None
        start, end = self.node_span(ch1)
        detail: str = f"Nafnliðurinn '{subject}' er í {number} og með honum á því að vera sögn í {number}."
        origin = subject
        if so is None:
            return AnnotationDict(
                text="Sögnin á sennilega að vera í {1} eins og frumlagið '{0}'".format(subject, number),
                detail=detail,
                start=start,
                end=end,
                original=origin,
                suggest=None,  # No suggest value available
            )
        origin = so.tidy_text
        sostart, soend = so.span
        end = start + soend
        start = start + sostart
        v = set(so.all_variants)
        if "et" in v:
            v -= {"et"}
            v.add("ft")
        elif "ft" in v:
            v -= {"ft"}
            v.add("et")
        suggest = PatternMatcher.get_wordform(so.text.lower(), so.lemma, "so", v)
        suggest = emulate_case(suggest, template=origin)
        if suggest == origin or not suggest:
            # Avoid meaningless annotation
            return None
        return AnnotationDict(
            text="Sögnin á sennilega að vera í {1}, þ.e. '{0}'".format(suggest, number),
            detail=detail,
            start=start,
            end=end,
            original=origin,
            suggest=suggest,
        )

    def VillaFsMeðFallstjórn(self, txt: str, variants: str, node: Node) -> Optional[AnnotationDict]:
        # Forsetningin z á að stýra x-falli en ekki y-falli
        tnode = self._terminal_nodes[node.start]
        pp = tnode.enclosing_tag("PP")
        obj = None
        if pp is not None:
            try:
                obj = pp.NP
            except AttributeError:
                pass
        if obj:
            assert pp is not None
            preposition = pp.P.text
            suggestion = self.cast_to_case(variants, obj)
            correct_np = correct_spaces(suggestion)
            if correct_np == obj.text:
                # Avoid suggesting the original text
                return None
            start, end = obj.span
            return AnnotationDict(
                text="Á sennilega að vera '{0}'".format(correct_np),
                detail=(
                    "Forsetningin '{0}' stýrir {1}falli.".format(
                        preposition.lower(),
                        CASE_NAMES[variants],
                    )
                ),
                start=start,
                end=end,
                original=obj.text,
                suggest=suggestion,
            )
        # In this case, there's no suggested correction
        # and it is likely a wrong annotation causing false positives
        return None
        # return dict(
        #    text="Forsetningin '{0}' stýrir {1}falli.".format(
        #        txt.split()[0].lower(), CASE_NAMES[variants]
        #    ),
        # )

    def AðvörunSvigaInnihaldNl(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        """Explanatory noun phrase in a different case than the noun phrase
        that it explains"""
        np = self._simple_tree(node)
        assert np is not None
        suggest = self.cast_to_case(variants, np)
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Gæti átt að vera '{0}'".format(suggest),
            detail="Fall liðarins stjórnast af öðru utan liðarins",
            start=start,
            end=end,
            original=np.tidy_text,
            suggest=suggest,
        )

    def VillaSíðastLiðinn(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        """'síðast liðinn' written in two words instead of one"""
        correct = txt.replace(" ", "")
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Venjulega er ritað '{0}'".format(correct),
            detail="'Síðastliðinn' er sjálfstætt og gilt lýsingarorð.",
            start=start,
            end=end,
            original=txt,
            suggest=correct,
        )

    def VillaEndingIR(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'læknirinn' á sennilega að vera 'lækninn'
        # In this case, we need the accusative form
        # of the token in self._tokens[node.start]
        tnode = self._terminal_nodes[node.start]
        suggestion = tnode.accusative_np
        correct_np = correct_spaces(suggestion)
        article = " með greini" if "gr" in tnode.all_variants else ""
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Á sennilega að vera '{0}'".format(correct_np),
            detail=(
                "Karlkyns orð sem enda á '-ir' í nefnifalli eintölu, "
                "eins og '{0}', eru rituð "
                "'{1}' í þolfalli{2}.".format(tnode.canonical_np, correct_np, article)
            ),
            start=start,
            end=end,
            original=txt,
            suggest=suggestion,
        )

    def VillaEndingANA(self, txt: str, variants: str, node: Node) -> AnnotationDict:
        # 'þingflokkana' á sennilega að vera 'þingflokkanna'
        # In this case, we need the genitive form
        # of the token in self._tokens[node.start]
        tnode = self._terminal_nodes[node.start]
        suggestion = tnode.genitive_np
        correct_np = correct_spaces(suggestion)
        canonical_np = tnode.canonical_np
        if canonical_np.endswith("ar"):
            # This might be something like 'landsteinar' which is only plural
            detail = (
                "Karlkyns orð sem enda á '-ar' í nefnifalli fleirtölu, "
                "eins og '{0}', eru rituð "
                "'{1}' með tveimur n-um í eignarfalli fleirtölu, "
                "ekki '{2}' með einu n-i."
            ).format(canonical_np, correct_np, txt)
        else:
            detail = (
                "Karlkyns orð sem enda á '-{3}' í nefnifalli eintölu, "
                "eins og '{0}', eru rituð "
                "'{1}' með tveimur n-um í eignarfalli fleirtölu, "
                "ekki '{2}' með einu n-i."
            ).format(canonical_np, correct_np, txt, canonical_np[-2:])
        start, end = self.node_span(node)
        return AnnotationDict(
            text="Á sennilega að vera '{0}'".format(correct_np),
            detail=detail,
            start=start,
            end=end,
            original=txt,
            suggest=suggestion,
        )

    def VillaManns(self, txt: str, variants: str, node: Node) -> Optional[AnnotationDict]:
        # Sögn á að vera í fleirtölu, líkt og frumlagið
        child = list(node.enum_child_nodes())
        ch0node = child[0]
        assert ch0node is not None
        subject = self.node_text((ch0node))
        # Find the verb
        subjtree: Optional[SimpleTree] = self._simple_tree(ch0node)
        if subjtree is None:
            return None
        if self._sent.tree is None:
            return None
        subjwhole = self._sent.tree.first_match('NP-SUBJ >> { "manns" }')
        if subjwhole is None:
            return None
        ip = subjwhole.enclosing_tag("IP")
        if ip is None:
            return None
        so = ip.first_match("so_et")
        if so is None:
            return None
        start, end = so.span
        vars = set(so.all_variants) - {"et"}
        vars.add("ft")
        suggest = PatternMatcher.get_wordform(so.text.lower(), so.lemma, so.cat, vars)
        return AnnotationDict(
            text=f"Sögn á að vera í fleirtölu líkt og frumlagið '{subject}'",
            detail="Sögnin á að vera í fleirtölu, þrátt fyrir 'manns' í frumlagi.",
            start=start,
            end=end,
            original=so.tidy_text,
            suggest=suggest,
        )

    @staticmethod
    def find_verb_subject(tnode: SimpleTree) -> Optional[SimpleTree]:
        """Starting with a verb terminal node, attempt to find
        the verb's subject noun phrase"""
        subj: Optional[SimpleTree] = None
        # TODO does this belong in pattern.py?
        # First, check within the enclosing verb phrase
        # (the subject may be embedded within it, as in
        # ?'Í dag langaði Páli bróður að fara í sund')
        vp = tnode.enclosing_tag("VP")
        p = None if vp is None else vp.enclosing_tag("VP")
        if p is not None:
            try:
                subj = p.NP_SUBJ
            except AttributeError:
                pass
        if subj is None:
            # If not found there, look within the
            # enclosing IP (inflectional phrase) node, if any
            p = tnode.enclosing_tag("IP")
            if p is not None:
                # Found the inflectional phrase:
                # find the NP-SUBJ node, if any
                try:
                    subj = p.NP_SUBJ
                except AttributeError:
                    pass
        return subj

    @staticmethod
    def find_verb_direct_object(tnode: SimpleTree) -> Optional[SimpleTree]:
        """Starting with a verb terminal node, attempt to find the
        verb's direct object noun phrase"""
        obj: Optional[SimpleTree] = None
        vp = tnode.enclosing_tag("VP")
        if vp is None:
            return obj
        obj = vp.first_match("NP-OBJ")
        if obj is None:
            # The object could have been moved to the front
            p = None if vp is None else vp.enclosing_tag("VP")
            if p is not None:
                try:
                    obj = p.NP_OBJ
                except AttributeError:
                    pass
            if obj is None:
                # Not found in VP, look within enclosing
                # IP node, if any
                p = tnode.enclosing_tag("IP")
                if p is not None:
                    # Found the IP, find NP-OBJ if any
                    try:
                        obj = p.NP_OBJ
                    except AttributeError:
                        pass
        return obj

    def _annotate_ordinal(self, node: Node) -> None:
        """Check for errors in ordinal number terminals ("2.") and annotate them"""
        token = cast("ErrorDetectionToken", node.token)
        if token is None or token.t0 != TOK.ORDINAL:
            # The matched token is not a numeric ordinal: no complaint
            return
        num = cast(int, token.t2)
        if not (1 <= num <= 9):
            # We only want to correct 1. - 9.
            return
        if token.t1[0] not in "0123456789":
            # Probably a Roman numeral - we don't mess with those
            return
        if "." in token.t1[:-1]:
            # Looks like more than one period (2.4.1): leave as-is
            return
        terminal = cast(BIN_Terminal, node.terminal)
        assert terminal is not None
        if len(terminal.variants) < 2:
            # We can only annotate if we have the case and the gender
            return
        correct = ORDINALS[num][terminal.gender or ""][terminal.case or ""]
        if token.cap_sentence_start:
            # The token is at the start of a sentence: suggest an uppercase word
            correct = correct.capitalize()
        self._ann.append(
            Annotation(
                text="Betra væri '{0}'".format(correct),
                detail=("Æskilegt er að rita lágar raðtölur " "með bókstöfum fremur en tölustöfum."),
                start=node.start,
                end=node.start,
                code="number4word",
                original=self.node_text(node),
                suggest=correct,
            )
        )

    def _annotate_verb(self, node: Node) -> None:
        """Annotate a verb (so) terminal"""
        # TODO does this belong in pattern.py?
        terminal = cast(BIN_Terminal, node.terminal)
        tnode = self._terminal_nodes[node.start]
        verb = tnode.lemma

        def annotate_wrong_subject_case(subj_case_abbr: str, correct_case_abbr: str) -> None:
            """Create an annotation that describes a verb having a subject
            in the wrong case"""
            wrong_case = CASE_NAMES[subj_case_abbr]
            # Retrieve the correct case
            correct_case = CASE_NAMES[correct_case_abbr]
            # Try to recover the verb's subject
            subj = self.find_verb_subject(tnode)
            code = "P_WRONG_CASE_" + subj_case_abbr + "_" + correct_case_abbr
            personal = "persónuleg" if correct_case_abbr == "nf" else "ópersónuleg"
            if subj is not None:
                # We know what the subject is: annotate it
                start, end = subj.span
                subj_text = subj.tidy_text
                suggestion = self.cast_to_case(correct_case_abbr, subj)
                correct_np = correct_spaces(suggestion)
                correct_np = emulate_case(correct_np, template=subj_text)
                # Skip the annotation if it suggests the same text as the
                # original one; this can happen if the word forms for two
                # cases are identical
                if subj_text != correct_np:
                    self._ann.append(
                        Annotation(
                            start=start,
                            end=end,
                            code=code,
                            text="Á líklega að vera '{0}'".format(correct_np),
                            detail="Sögnin 'að {0}' er {3}. "
                            "Frumlag hennar á að vera "
                            "í {1}falli í stað {2}falls.".format(verb, correct_case, wrong_case, personal),
                            original=subj_text,
                            suggest=correct_np,
                        )
                    )
            else:
                # We don't seem to find the subject, so just annotate the verb.
                # In this case, there's no suggested correction.
                assert node.token is not None
                index = node.token.index
                self._ann.append(
                    Annotation(
                        start=index,
                        end=index,
                        code=code,
                        text="Frumlag sagnarinnar 'að {0}' " "á að vera í {1}falli".format(verb, correct_case),
                        detail="Sögnin 'að {0}' er {3}. "
                        "Frumlag hennar á að vera "
                        "í {1}falli í stað {2}falls.".format(verb, correct_case, wrong_case, personal),
                        original=verb,
                        suggest=None,  # No suggest value available
                    )
                )

        def annotate_wrong_op_verb_form(verb: str, correct: str, detail: str) -> None:
            """Annotate wrong impersonal verb forms, such as 'Mér lýst'"""
            token = node.token
            assert token is not None
            index = token.index
            self._ann.append(
                Annotation(
                    text="Sögnin '{0}' á sennilega að vera '{1}'".format(verb.lower(), correct),
                    detail=detail,
                    start=index,
                    end=index,
                    code="P_WRONG_OP_FORM",
                    original=verb,
                    suggest=emulate_case(correct, template=verb),
                )
            )

        def annotate_wrong_obj_case(obj_case_abbr: str, correct_case_abbr: str) -> None:
            """Create an annotation that describes a verb having a direct object
            in the wrong case"""
            wrong_case = CASE_NAMES[obj_case_abbr]
            # Retrieve the correct case
            correct_case = CASE_NAMES[correct_case_abbr]
            # Try to recover the verb's direct object
            objtree = self.find_verb_direct_object(tnode)
            if objtree is None:
                return
            code = "P_WRONG_CASE_" + obj_case_abbr + "_" + correct_case_abbr
            if objtree is not None:
                # We know what the object is: annotate it
                start, end = objtree.span
                obj_text: str = objtree.tidy_text
                suggestion: str = self.cast_to_case(correct_case_abbr, objtree)
                correct_np = correct_spaces(suggestion)
                correct_np = emulate_case(correct_np, template=obj_text)
                # Skip the annotation if it suggests the same text as the
                # original one; this can happen if the word forms for two
                # cases are identical
                if obj_text != correct_np:
                    self._ann.append(
                        Annotation(
                            text="Á líklega að vera '{0}'".format(correct_np),
                            detail="Andlag sagnarinnar {0} á að vera "
                            "í {1}falli í stað {2}falls.".format(verb, correct_case, wrong_case),
                            start=start,
                            end=end,
                            code=code,
                            original=obj_text,
                            suggest=correct_np,
                        )
                    )
            else:
                # We don't seem to find the object, so just annotate the verb.
                # In this case, there's no suggested correction.
                assert node.token is not None
                index = node.token.index
                self._ann.append(
                    Annotation(
                        text="Andlag sagnarinnar 'að {0}' " "á að vera í {1}falli".format(verb, correct_case),
                        detail="Andlag sagnarinnar {0} á að vera "
                        "í {1}falli í stað {2}falls.".format(verb, correct_case, wrong_case),
                        start=index,
                        end=index,
                        code=code,
                        original=verb,
                        suggest=None,  # No suggest value available
                    )
                )

        def check_obj() -> None:
            obj: str = ""
            if len(terminal._vparts) < 1 or terminal.variant(0) == "0":
                # No objects to check
                pass
            else:
                obj = "" if len(terminal._vparts) < 2 else terminal.variant(1)
            if not obj:
                return
            obj_errors = VerbErrors.OBJ_ERRORS.get(verb, dict())
            if obj in obj_errors:
                annotate_wrong_obj_case(obj, obj_errors[obj])

        def check_subj() -> None:
            if not terminal.is_subj:
                # Check whether we had to match an impersonal verb
                # with this "normal" (non _subj) terminal
                # Check whether the verb is present in the VERBS_ERRORS
                # dictionary, with an 'nf' entry mapping to another case
                errors = VerbSubjects.VERBS_ERRORS.get(verb, dict())
                if "nf" in errors:
                    # We are using an impersonal verb as a normal verb,
                    # i.e. with a subject in nominative case:
                    # annotate an error
                    annotate_wrong_subject_case("nf", errors["nf"])
                return

            # This is a so_subj terminal
            if not (terminal.is_op or terminal.is_sagnb or terminal.is_nh):
                return
            # This is a so_subj_op, so_subj_sagnb or so_subj_nh terminal
            # Check whether the associated verb is allowed
            # with a subject in this case
            # node points to a fastparser.Node instance
            # tnode points to a SimpleTree instance
            subj_case_abbr = terminal.variant(-1)  # so_1_þgf_subj_op_et_þf
            if subj_case_abbr == "none":
                # so_subj_nh_none or similar:
                # hidden subject ('Í raun þyrfti að yfirfara alla ferla')
                return
            assert subj_case_abbr in {"nf", "þf", "þgf", "ef"}, "Unknown case in " + terminal.name

            # Check whether this is a verb form that is forbidden from
            # impersonal use, such as 'lýst' which cannot be used impersonally
            # as a form of 'ljósta' ('Eldingunni laust niður')
            token = node.token
            assert token is not None
            if token.lower in self._NON_OP_VERB_FORMS:
                correct, detail = self._NON_OP_VERB_FORMS[token.lower]
                annotate_wrong_op_verb_form(token.text, correct, detail)
                return

            # Check whether this verb has an entry in the VERBS_ERRORS
            # dictionary, and whether that entry then has an item for
            # the present subject case
            # TODO: This will get more complex when all arguments are in the same frame
            subj_errors = VerbSubjects.VERBS_ERRORS.get(verb, dict())
            if subj_case_abbr in subj_errors:
                # Yes, this appears to be an erroneous subject case
                annotate_wrong_subject_case(subj_case_abbr, subj_errors[subj_case_abbr])

        check_obj()
        check_subj()

    def visit_token(self, level: int, w: Node) -> None:
        """Entering a terminal/token match node"""
        terminal = cast(BIN_Terminal, w.terminal)
        if terminal.category == "so":
            self._annotate_verb(w)
        # TODO: The following actually reduces GreynirCorrect's score on the
        # iceErrorCorpus test set, so we comment it out for the time being.
        # elif terminal.category == "raðnr":
        #    self._annotate_ordinal(w)

    def visit_nonterminal(self, level: int, node: Node) -> Any:
        """Entering a nonterminal node"""
        if node.is_interior or node.nonterminal is None or node.nonterminal.is_optional:
            # Not an interesting node
            return None
        if not node.nonterminal.has_tag("error"):
            return None
        # This node has a nonterminal that is tagged with $tag(error)
        # in the grammar file (Greynir.grammar)
        suggestion = None
        original = None
        ann_text: str = ""
        ann_detail = None
        start, end = self.node_span(node)
        span_text = self.node_text(node)
        # See if we have a custom text function for this
        # error-tagged nonterminal
        name = node.nonterminal.name
        variants = ""
        if "_" in name:
            # Separate the variants
            ix = name.index("_")
            variants = name[ix + 1 :]
            name = name[:ix]
        # Find the text function by dynamic dispatch
        text_func: Optional[AnnotationFunc] = getattr(self, name, None)
        # The error code in this case is P_NT_ + the name of the error-tagged
        # nonterminal, however after cutting 'Villa'/'Aðvörun' from its front
        is_warning = False
        if name.startswith("Aðvörun"):
            # Warning
            code = "P_NT_" + name[7:]
            is_warning = True
        elif name.startswith("Villa"):
            # Error
            code = "P_NT_" + name[5:]
        else:
            code = "P_NT_" + name
        if text_func is not None:
            # Yes: call it with the nonterminal's spanned text as argument
            ann = text_func(span_text, variants, node)
            if isinstance(ann, str):
                ann_text = ann
            elif isinstance(ann, tuple):
                if len(ann) == 2:
                    ann_text, suggestion = cast(AnnotationTuple2, ann)
                elif len(ann) == 4:
                    ann_text, start, end, suggestion = cast(AnnotationTuple4, ann)
                elif len(ann) == 6:
                    ann_text, ann_detail, start, end, original, suggestion = cast(AnnotationTuple6, ann)
            else:
                if not ann:
                    # Empty or no dict: this means that upon closer inspection,
                    # there was no need to annotate
                    return None
                ann_text = ann.get("text") or ""
                ann_detail = ann.get("detail") or ""
                original = ann.get("original") or ""
                suggestion = ann.get("suggest") or ""
                start = ann.get("start", start)
                end = ann.get("end", end)
        else:
            # No: use a default text
            ann_text = "'{0}' er líklega rangt".format(span_text)
            ann_detail = "Regla {0}".format(node.nonterminal.name)
        self._ann.append(
            # P_NT_ + nonterminal name: Probable grammatical error.
            Annotation(
                start=start,
                end=end,
                code=code,
                text=ann_text,
                detail=ann_detail,
                original=original,
                suggest=suggestion,
                is_warning=is_warning,
            )
        )
        return None
