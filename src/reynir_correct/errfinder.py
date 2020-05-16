"""

    Greynir: Natural language processing for Icelandic

    Error finder for parse trees

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


    This module implements the ErrorFinder class. The class is used
    in checker.py to find occurrences of error-marked nonterminals within
    parse trees and annotate the associated error or warning.

    Specifically, ErrorFinder finds nonterminals that are marked with $tag(error)
    in the CFG (Reynir.grammar). These nonterminals give rise to error
    or warning annotations for their respective token spans.

"""

from reynir import correct_spaces, TOK
from reynir.fastparser import ParseForestNavigator
from reynir.settings import VerbSubjects
from reynir.simpletree import SimpleTree

from .annotation import Annotation
from .errtokenizer import emulate_case


# Case name prefixes
CASE_NAMES = {"nf": "nefni", "þf": "þol", "þgf": "þágu", "ef": "eignar"}


class ErrorFinder(ParseForestNavigator):

    """ Utility class to find nonterminals in parse trees that are
        tagged as errors in the grammar, and terminals matching
        verb forms marked as errors """

    _CAST_FUNCTIONS = {
        "nf": SimpleTree.nominative_np,
        "þf": SimpleTree.accusative_np,
        "þgf": SimpleTree.dative_np,
        "ef": SimpleTree.genitive_np
    }

    def __init__(self, ann, sent):
        super().__init__(visit_all=True)
        # Annotation list
        self._ann = ann
        # The original sentence object
        self._sent = sent
        # Token list
        self._tokens = sent.tokens
        # Terminal node list
        self._terminal_nodes = sent.terminal_nodes

    def go(self):
        """ Start navigating the deep tree structure of the sentence """
        return super().go(self._sent.deep_tree)

    @staticmethod
    def _node_span(node):
        """ Return the start and end indices of the tokens
            spanned by the given node """
        first_token, last_token = node.token_span
        return (first_token.index, last_token.index)

    def cast_to_case(self, case, node):
        """ Return the contents of a noun phrase node
            inflected in the given case """
        return self._CAST_FUNCTIONS[case].fget(node)

    def _simple_tree(self, node):
        """ Return a SimpleTree instance spanning the deep tree
            of which node is the root """
        first, last = self._node_span(node)
        toklist = self._tokens[first : last + 1]
        return SimpleTree.from_deep_tree(node, toklist, first_token_index=first)

    def _node_text(self, node, original_case=False):
        """ Return the text within the span of the node """

        def text(t):
            """ If the token t is a word token, return a lower case
                version of its text, unless we have a reason to keep
                the original case, i.e. if it is a lemma that is upper case
                in BÍN """
            if t.kind != TOK.WORD:
                # Not a word token: keep the original text
                return t.txt
            if len(t.txt) > 1 and t.txt.isupper():
                # All uppercase: keep it that way
                return t.txt
            if t.val and any(m.stofn[0].isupper() for m in t.val):
                # There is an uppercase lemma for this word in BÍN:
                # keep the original form
                return t.txt
            # No uppercase lemma in BÍN: return a lower case copy
            return t.txt.lower()

        first, last = self._node_span(node)
        text_func = (lambda t: t.txt) if original_case else text
        return correct_spaces(
            " ".join(text_func(t) for t in self._tokens[first : last + 1] if t.txt)
        )

    # Functions used to explain grammar errors associated with
    # nonterminals with error tags in the grammar

    def AðvörunHeldur(self, txt, variants, node):
        # 'heldur' er ofaukið
        # !!! TODO: Add suggestion here by replacing
        # !!! 'heldur en' with 'en'
        return dict(
            text="'{0}' er sennilega ofaukið".format(txt),
            detail="Yfirleitt nægir að nota 'en' í þessu samhengi."
        )

    def AðvörunSíðan(self, txt, variants, node):
        # 'fyrir tveimur mánuðum síðan'
        return dict(
            text="'síðan' er sennilega ofaukið",
            detail="Yfirleitt er óþarfi að nota orðið 'síðan' í samhengi á borð við "
                "'fyrir tveimur dögum'",
            suggestion=""
        )

    def VillaVístAð(self, txt, variants, node):
        # 'víst að' á sennilega að vera 'fyrst að'
        return dict(
            text="'{0}' á sennilega að vera 'fyrst að'".format(txt),
            detail="Rétt er að nota 'fyrst að' fremur en 'víst að' "
                " til að tengja saman atburð og forsendu.",
            suggestion="fyrst að"
        )

    def VillaFráÞvíAð(self, txt, variants, node):
        # 'allt frá því' á sennilega að vera 'allt frá því að'
        return dict(
            text="'{0}' á sennilega að vera '{0} að'".format(txt),
            detail="Rétt er að nota samtenginuna 'að' í þessu samhengi, "
                "til dæmis 'allt frá því að Anna hóf námið'.",
            suggestion="{0} að".format(txt)
        )

    def VillaAnnaðhvort(self, txt, variants, node):
        # Í stað 'annaðhvort' á sennilega að standa 'annað hvort'
        return dict(
            text="Í stað '{0}' á sennilega að standa 'annað hvort'".format(txt),
            detail="Rita á 'annað hvort' þegar um er að ræða fornöfn, til dæmis "
                "'annað hvort systkinanna'. Rita á 'annaðhvort' í samtengingu, "
                "til dæmis 'Annaðhvort fer ég út eða þú'.",
            suggestion="annað hvort"
        )

    def VillaAnnaðHvort(self, txt, variants, node):
        # Í stað 'annað hvort' á sennilega að standa 'annaðhvort'
        return dict(
            text="Í stað '{0}' á sennilega að standa 'annaðhvort'".format(txt),
            detail="Rita á 'annaðhvort' í samtengingu, til dæmis "
                "'Annaðhvort fer ég út eða þú'. "
                "Rita á 'annað hvort' í tveimur orðum þegar um er að ræða fornöfn, "
                "til dæmis 'annað hvort systkinanna'.",
            suggestion="annaðhvort"
        )

    def singular_error(self, txt, variants, node, detail):
        """ Annotate a mismatch between singular and plural
            in subject vs. verb """
        tnode = self._terminal_nodes[node.start]
        # Find the enclosing inflected phrase
        p = tnode.enclosing_tag("IP")
        verb = None
        if p is not None:
            try:
                # Found it: try to locate the main verb
                verb = p.VP.VP
            except AttributeError:
                try:
                    verb = p.VP
                except AttributeError:
                    pass
        if verb is not None:
            start, end = verb.span
            return dict(
                text="Sögnin '{0}' á sennilega að vera í eintölu, ekki fleirtölu"
                    .format(verb.tidy_text),
                detail=detail,
                start=start,
                end=end,
            )
        return (
            "Sögn sem á við '{0}' á sennilega að vera í eintölu, ekki fleirtölu"
            .format(txt)
        )

    def VillaFjöldiHluti(self, txt, variants, node):
        # Sögn sem á við 'fjöldi Evrópuríkja' á að vera í eintölu
        return self.singular_error(
            txt, variants, node,
            "Nafnliðurinn '{0}' er í eintölu "
            "og með honum á því að vera sögn í eintölu.".format(txt)
        )

    def VillaEinnAf(self, txt, variants, node):
        # Sögn sem á við 'einn af drengjunum' á að vera í eintölu
        return self.singular_error(
            txt, variants, node,
            "Nafnliðurinn '{0}' er í eintölu "
            "og með honum á því að vera sögn í eintölu.".format(txt)
        )

    def VillaEinkunn(self, txt, variants, node):
        # Fornafn í einkunn er ekki í sama falli og nafnorð,
        # t.d. 'þessum mann'
        wrong_pronoun = self._node_text(node, original_case=True)
        correct_case = variants.split("_")[0]
        pronoun_node = next(node.enum_child_nodes())
        p = self._simple_tree(pronoun_node)
        correct_pronoun = self.cast_to_case(correct_case, p)
        return dict(
            text="'{0}' á sennilega að vera '{1}'"
                .format(wrong_pronoun, correct_pronoun),
            detail="Fornafnið '{0}' á að vera í {1}falli, eins og nafnliðurinn sem fylgir á eftir"
                .format(wrong_pronoun, CASE_NAMES[correct_case]),
            suggestion=correct_pronoun,
        )

    def AðvörunSem(self, txt, variants, node):
        # 'sem' er sennilega ofaukið
        return dict(
            text="'{0}' er að öllum líkindum ofaukið".format(txt),
            detail="Oft fer betur á að rita 'og', 'og einnig' eða 'og jafnframt' "
                " í stað 'sem og'.",
            suggestion="",
        )

    def AðvörunAð(self, txt, variants, node):
        # 'að' er sennilega ofaukið
        return dict(
            text="'{0}' er að öllum líkindum ofaukið".format(txt),
            detail="'að' er yfirleitt ofaukið í samtengingum á "
                "borð við 'áður en', 'síðan', 'enda þótt' o.s.frv.",
            suggestion="",
        )

    def AðvörunKomma(self, txt, variants, node):
        return dict(
            text="Komma er líklega óþörf",
            detail="Kommu er yfirleitt ofaukið milli frumlags og umsagnar, "
                "milli forsendu og meginsetningar "
                "('áður en ég fer [,] má sækja tölvuna') o.s.frv.",
            suggestion=""
        )

    def VillaNé(self, txt, variants, node):
        return dict(
            text="'né' gæti átt að vera 'eða'",
            suggestion="eða"
        )

    def VillaÞóAð(self, txt, variants, node):
        # [jafnvel] þó' á sennilega að vera '[jafnvel] þó að
        suggestion = "{0} að".format(txt)
        return dict(
            text="'{0}' á sennilega að vera '{1}' (eða 'þótt')".format(txt, suggestion),
            detail="Réttara er að nota samtenginguna 'að' í samhengi á borð við "
                "'jafnvel þó að von sé á sólskini'.",
            suggestion=suggestion
        )

    def VillaÍTölu(self, txt, variants, node):
        # Sögn á að vera í sömu tölu og frumlag
        children = list(node.enum_child_nodes())
        assert len(children) == 2
        subject = self._node_text(children[0])
        # verb_phrase = self._node_text(children[1])
        number = "eintölu" if "et" in variants else "fleirtölu"
        # Annotate the verb phrase
        start, end = self._node_span(children[1])
        return dict(
            text="Sögn á sennilega að vera í {1} eins og frumlagið '{0}'"
                .format(subject, number),
            start=start,
            end=end
        )

    def VillaFsMeðFallstjórn(self, txt, variants, node):
        # Forsetningin z á að stýra x-falli en ekki y-falli
        tnode = self._terminal_nodes[node.start]
        p = tnode.enclosing_tag("PP")
        subj = None
        if p is not None:
            try:
                subj = p.NP
            except AttributeError:
                pass
        if subj:
            preposition = p.P.text
            suggestion = preposition + " " + self.cast_to_case(variants, subj)
            correct_np = correct_spaces(suggestion)
            return dict(
                text="Á sennilega að vera '{0}'".format(correct_np),
                detail="Forsetningin '{0}' stýrir {1}falli."
                    .format(
                        preposition.lower(),
                        CASE_NAMES[variants],
                    ),
                suggestion=suggestion
            )
        # In this case, there's no suggested correction
        return dict(
            text="Forsetningin '{0}' stýrir {1}falli."
                .format(txt.split()[0].lower(), CASE_NAMES[variants]),
        )

    def SvigaInnihaldNl(self, txt, variants, node):
        """ Explanatory noun phrase in a different case than the noun phrase
            that it explains """
        np = self._simple_tree(node)
        return (
            "Gæti átt að vera '{0}'"
            .format(self.cast_to_case(variants, np))
        )

    def VillaEndingIR(self, txt, variants, node):
        # 'læknirinn' á sennilega að vera 'lækninn'
        # In this case, we need the accusative form
        # of the token in self._tokens[node.start]
        tnode = self._terminal_nodes[node.start]
        suggestion = tnode.accusative_np
        correct_np = correct_spaces(suggestion)
        article = " með greini" if "gr" in tnode.all_variants else ""
        return dict(
            text="Á sennilega að vera '{0}'".format(correct_np),
            detail="Karlkyns orð sem enda á '-ir' í nefnifalli eintölu, "
                "eins og '{0}', eru rituð "
                "'{1}' í þolfalli{2}.".format(tnode.canonical_np, correct_np, article),
            suggestion=suggestion
        )

    def VillaEndingANA(self, txt, variants, node):
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
        return dict(
            text="Á sennilega að vera '{0}'".format(correct_np),
            detail=detail,
            suggestion=suggestion
        )

    @staticmethod
    def find_verb_subject(tnode):
        """ Starting with a verb terminal node, attempt to find
            the verb's subject noun phrase """
        subj = None
        # First, check within the enclosing verb phrase
        # (the subject may be embedded within it, as in
        # ?'Í dag langaði Páli bróður að fara í sund')
        p = tnode.enclosing_tag("VP").enclosing_tag("VP")
        if p is not None:
            try:
                subj = p.NP_SUBJ
            except AttributeError:
                pass
        if subj is None:
            # If not found there, look within the
            # enclosing IP (inflected phrase) node, if any
            p = tnode.enclosing_tag("IP")
            if p is not None:
                # Found the inflected phrase:
                # find the NP-SUBJ node, if any
                try:
                    subj = p.NP_SUBJ
                except AttributeError:
                    pass
        return subj

    def visit_token(self, level, node):
        """ Entering a terminal/token match node """

        terminal = node.terminal
        if terminal.category != "so":
            # Currently we only need to check verb terminals
            return

        tnode = self._terminal_nodes[node.start]
        verb = tnode.lemma

        def annotate_wrong_subject_case(subj_case_abbr, correct_case_abbr):
            """ Create an annotation that describes a verb having a subject
                in the wrong case """
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
                correct_np = emulate_case(correct_np, subj_text)
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
                                "í {1}falli í stað {2}falls."
                                .format(verb, correct_case, wrong_case, personal),
                            suggest=suggestion
                        )
                    )
            else:
                # We don't seem to find the subject, so just annotate the verb.
                # In this case, there's no suggested correction.
                index = node.token.index
                self._ann.append(
                    Annotation(
                        start=index,
                        end=index,
                        code=code,
                        text="Frumlag sagnarinnar 'að {0}' "
                            "á að vera í {1}falli"
                            .format(verb, correct_case),
                        detail="Sögnin 'að {0}' er {3}. "
                            "Frumlag hennar á að vera "
                            "í {1}falli í stað {2}falls."
                            .format(verb, correct_case, wrong_case, personal),
                    )
                )

        if not terminal.is_subj:
            # Check whether we had to match an impersonal verb
            # with this "normal" (non _subj) terminal
            # Check whether the verb is present in the VERBS_ERRORS
            # dictionary, with an 'nf' entry mapping to another case
            errors = VerbSubjects.VERBS_ERRORS.get(verb, set())
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
        assert subj_case_abbr in {"nf", "þf", "þgf", "ef"}, (
            "Unknown case in " + terminal.name
        )
        # Check whether this verb has an entry in the VERBS_ERRORS
        # dictionary, and whether that entry then has an item for
        # the present subject case
        errors = VerbSubjects.VERBS_ERRORS.get(verb, set())
        if subj_case_abbr in errors:
            # Yes, this appears to be an erroneous subject case
            annotate_wrong_subject_case(subj_case_abbr, errors[subj_case_abbr])

    def visit_nonterminal(self, level, node):
        """ Entering a nonterminal node """
        if node.is_interior or node.nonterminal.is_optional:
            # Not an interesting node
            return None
        if not node.nonterminal.has_tag("error"):
            return None
        # This node has a nonterminal that is tagged with $tag(error)
        # in the grammar file (Reynir.grammar)
        suggestion = None
        ann_text = None
        ann_detail = None
        start, end = self._node_span(node)
        span_text = self._node_text(node)
        # See if we have a custom text function for this
        # error-tagged nonterminal
        name = node.nonterminal.name
        variants = ""
        if "_" in name:
            # Separate the variants
            ix = name.index("_")
            variants = name[ix + 1:]
            name = name[:ix]
        # Find the text function by dynamic dispatch
        text_func = getattr(self, name, None)
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
                    ann_text, suggestion = ann
                else:
                    ann_text, start, end, suggestion = ann
            elif isinstance(ann, dict):
                ann_text = ann.get("text")
                ann_detail = ann.get("detail")
                suggestion = ann.get("suggestion")
                start = ann.get("start", start)
                end = ann.get("end", end)
            else:
                assert False, "Text function {0} returns illegal type".format(name)
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
                suggest=suggestion,
                is_warning=is_warning
            )
        )
        return None

