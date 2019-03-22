"""

    Reynir: Natural language processing for Icelandic

    Error-correcting tokenization layer

    Copyright (C) 2019 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.

       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module adds layers to the bintokenizer.py module in ReynirPackage.
    These layers add token-level error corrections and recommendation flags
    to the token stream.

"""

from collections import defaultdict

from reynir import TOK
from reynir.bintokenizer import DefaultPipeline, MatchingStream

from .settings import (
    AllowedMultiples, WrongCompounds, SplitCompounds, UniqueErrors,
    MultiwordErrors, CapitalizationErrors, TabooWords, ErrorForms
)
from .spelling import Corrector


def emulate_case(s, template):
    """ Return the string s but emulating the case of the template
        (lower/upper/capitalized) """
    if template.isupper():
        return s.upper()
    elif template and template[0].isupper():
        return s.capitalize()
    return s


class CorrectToken:

    """ This class sneakily replaces the tokenizer.Tok tuple in the tokenization
        pipeline. When applying a CorrectionPipeline (instead of a DefaultPipeline,
        as defined in binparser.py in ReynirPackage), tokens get translated to
        instances of this class in the correct() phase. This works due to Python's
        duck typing, because a CorrectToken class instance is able to walk and quack
        - i.e. behave - like a tokenizer.Tok tuple. It adds an _err attribute to hold
        information about spelling and grammar errors, and some higher level functions
        to aid in error reporting and correction. """

    # Use __slots__ as a performance enhancement, since we want instances
    # to be as lightweight as possible - and we don't expect this class
    # to be subclassed or custom attributes to be added
    __slots__ = ("kind", "txt", "val", "_err")

    def __init__(self, kind, txt, val):
        self.kind = kind
        self.txt = txt
        self.val = val
        self._err = None

    def __getitem__(self, index):
        """ Support tuple-style indexing, as raw tokens do """
        return (self.kind, self.txt, self.val)[index]

    @classmethod
    def from_token(cls, token):
        """ Wrap a raw token in a CorrectToken """
        return cls(token.kind, token.txt, token.val)

    @classmethod
    def word(cls, txt, val=None):
        """ Create a wrapped word token """
        return cls(TOK.WORD, txt, val)

    def __repr__(self):
        return (
            "<CorrectToken(kind: {0}, txt: '{1}', val: {2})>"
            .format(TOK.descr[self.kind], self.txt, self.val)
        )

    __str__ = __repr__

    def set_error(self, err):
        """ Associate an Error class instance with this token """
        self._err = err

    def copy_error(self, other, coalesce=False):
        """ Copy the error field from another CorrectToken instance """
        if isinstance(other, list):
            # We have a list of CorrectToken instances to copy from:
            # find the first error in the list, if any, and copy it
            for t in other:
                if self.copy_error(t, coalesce=True):
                    break
        elif isinstance(other, CorrectToken):
            self._err = other._err
            if coalesce and other.error_span > 1:
                # The original token had an associated error
                # spanning more than one token; now we're creating
                # a single token out of then span
                # ('fimm hundruð' -> number token), so we reset
                # the span to one token
                self._err.set_span(1)
        return self._err is not None

    @property
    def error(self):
        """ Return the error object associated with this token, if any """
        # Note that self._err may be a bool
        return self._err

    @property
    def error_description(self):
        """ Return the description of an error associated with this token, if any """
        return self._err.description if hasattr(self._err, "description") else ""

    @property
    def error_code(self):
        """ Return the code of an error associated with this token, if any """
        return self._err.code if hasattr(self._err, "code") else ""

    @property
    def error_span(self):
        """ Return the number of tokens affected by this error """
        return self._err.span if hasattr(self._err, "span") else 1


class Error:

    """ Base class for spelling and grammar errors, warnings and recommendations.
        An Error has a code and can provide a description of itself. """

    def __init__(self, code):
        self._code = code

    @property
    def code(self):
        return self._code
    
    @property
    def description(self):
        """ Should be overridden """
        raise NotImplementedError


class CompoundError(Error):

    """ A CompoundError is an error where words are duplicated, split or not
        split correctly. """

    # C001: Duplicated word removed. Should be corrected.
    # C002: Wrongly compounded words split up. Should be corrected.
    # C003: Wrongly split compounds united. Should be corrected.

    def __init__(self, code, txt, span=1):
        # Compound error codes start with "C"
        super().__init__("C" + code)
        self._txt = txt
        self._span = span

    @property
    def description(self):
        return self._txt

    @property
    def span(self):
        return self._span

    def set_span(self, span):
        """ Reset the span to the given number """
        self._span = span


class UnknownWordError(Error):

    """ An UnknownWordError is an error where the given word form does not
        exist in BÍN or additional vocabularies, and cannot be explained as
        a compound word. """

    # U001: Unknown word. Nothing more is known. Cannot be corrected, only pointed out.

    def __init__(self, code, txt):
        # Unknown word error codes start with "U"
        super().__init__("U" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class CapitalizationError(Error):

    """ A CapitalizationError is an error where a word is capitalized
        incorrectly, i.e. should be lower case but occurs in upper case
        except at the beginning of a sentence, or should be upper case
        but occurs in lower case. """

    # Z001: Word should begin with lowercase letter
    # Z002: Word should begin with uppercase letter

    def __init__(self, code, txt):
        # Capitalization error codes start with "Z"
        super().__init__("Z" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class TabooWarning(Error):

    """ A TabooWarning marks a word that is vulgar or not appropriate
        in formal text. """

    # T001: Taboo word usage warning, with suggested replacement

    def __init__(self, code, txt):
        # Taboo word warnings start with "T"
        super().__init__("T" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class SpellingError(Error):

    """ A SpellingError is an erroneous word that was replaced
        by a much more likely word that exists in the dictionary. """

    # S001: Common errors picked up by unique_errors. Should be corrected
    # S002: Errors handled by spelling.py. Corrections should possibly only be suggested.
    # S003: Erroneously formed word forms picked up by ErrorForms. Should be corrected. TODO split up by nature.
    # S004: Rare word, a more common one has been substituted.

    def __init__(self, code, txt):
        # Spelling error codes start with "S"
        super().__init__("S" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class PhraseError(Error):

    """ A PhraseError is a wrong multiword phrase, where a word is out
        of place in its context. """

    # P_xxx: Phrase error codes

    def __init__(self, code, txt, span):
        # Phrase error codes start with "P", and are followed by
        # a string indicating the type of error, i.e. YI for y/i, etc.
        super().__init__("P_" + code)
        self._txt = txt
        self._span = span

    @property
    def description(self):
        return self._txt

    @property
    def span(self):
        return self._span

    def set_span(self, span):
        """ Reset the span to the given number """
        self._span = span


def parse_errors(token_stream, db):

    """ This tokenization phase is done before BÍN annotation
        and before static phrases are identified. It finds duplicated words,
        and words that have been incorrectly split or should be split. """

    def get():
        """ Get the next token in the underlying stream and wrap it
            in a CorrectToken instance """
        return CorrectToken.from_token(next(token_stream))

    def is_split_compound(token, next_token):
        """ Check whether the combination of the given token and the next
            token forms a split compound. Note that the latter part of
            a split compound is specified as a stem (lemma), so we need
            to check whether the next_token word form has a corresponding
            lemma. Also, a single first part may have more than one (i.e.,
            a set of) subsequent latter part stems.
        """
        txt = token.txt
        next_txt = next_token.txt
        if txt is None or next_txt is None or next_txt.istitle():
            # If the latter part is in title case, we don't see it
            # as a part of split compound
            return False
        if next_txt.isupper() and not txt.isupper():
            # Don't allow a combination of an all-upper-case
            # latter part with anything but an all-upper-case former part
            return False
        if txt.isupper() and not next_txt.isupper():
            # ...and vice versa
            return False
        next_stems = SplitCompounds.DICT.get(txt.lower())
        if not next_stems:
            return False
        _, meanings = db.lookup_word(next_txt.lower(), at_sentence_start=False)
        if not meanings:
            return False
        # If any meaning of the following word has a stem (lemma)
        # that fits the second part of the split compound, we
        # have a match
        return any(m.stofn.replace("-", "") in next_stems for m in meanings)

    token = None
    try:
        # Maintain a one-token lookahead
        token = get()
        while True:
            next_token = get()
            # Make the lookahead checks we're interested in
            # Word duplication (note that word case must also match)
            if (
                token.txt
                and next_token.txt
                and token.txt == next_token.txt
                and token.txt.lower() not in AllowedMultiples.SET
                and token.kind == TOK.WORD
            ):
                # Step to next token
                next_token = CorrectToken.word(token.txt)
                next_token.set_error(
                    CompoundError(
                        "001",
                        "Endurtekið orð ('{0}') var fellt burt".format(token.txt)
                    )
                )
                token = next_token
                continue

            # Splitting wrongly compounded words
            if token.txt and token.txt.lower() in WrongCompounds.DICT:
                correct_phrase = list(WrongCompounds.DICT[token.txt.lower()])
                # Make the split phrase emulate the case of
                # the original token
                if token.txt.isupper():
                    # All upper case
                    for ix, p in enumerate(correct_phrase):
                        correct_phrase[ix] = p.upper()
                else:
                    # First word might be capitalized
                    correct_phrase[0] = emulate_case(correct_phrase[0], token.txt)
                for ix, phrase_part in enumerate(correct_phrase):
                    new_token = CorrectToken.word(phrase_part)
                    if ix == 0:
                        new_token.set_error(
                            CompoundError(
                                "002",
                                "Orðinu '{0}' var skipt upp".format(token.txt),
                                span=len(correct_phrase)
                            )
                        )
                    yield new_token
                token = next_token
                continue

            # Unite wrongly split compounds
            if is_split_compound(token, next_token):
                first_txt = token.txt
                token = CorrectToken.word(token.txt + next_token.txt)
                token.set_error(
                    CompoundError(
                        "003",
                        "Orðin '{0} {1}' voru sameinuð í eitt"
                        .format(first_txt, next_token.txt)
                    )
                )
                continue

            # Yield the current token and advance to the lookahead
            yield token
            token = next_token

    except StopIteration:
        # Final token (previous lookahead)
        if token:
            yield token


class MultiwordErrorStream(MatchingStream):

    """ Class that filters a token stream looking for multi-word
        matches with the MultiwordErrors phrase dictionary,
        and inserting replacement phrases when matches are found """

    def __init__(self, db, token_ctor):
        super().__init__(MultiwordErrors.DICT)
        self._token_ctor = token_ctor
        self._db = db

    def length(self, ix):
        return len(MultiwordErrors.get_replacement(ix))

    def match(self, tq, ix):
        """ This is a complete match of an error phrase;
            yield the replacement phrase """
        replacement = MultiwordErrors.get_replacement(ix)
        db = self._db
        token_ctor = self._token_ctor
        for i, replacement_word in enumerate(replacement):
            # !!! TODO: at_sentence_start
            w, m = db.lookup_word(
                replacement_word, False, False
            )
            if i == 0:
                # Fix capitalization of the first word
                # !!! TODO: handle all-uppercase
                if tq[0].txt.istitle():
                    w = w.title()
            ct = token_ctor.Word(w, m)
            if i == 0:
                ct.set_error(
                    PhraseError(
                        MultiwordErrors.get_code(ix),
                        "Orðasambandið '{0}' var leiðrétt í '{1}'"
                        .format(
                            " ".join(t.txt for t in tq),
                            " ".join(replacement)
                        ),
                        span = len(replacement)
                    )
                )
            else:
                # Set the error field of multiword phrase
                # continuation tokens to True, thus avoiding
                # further meddling with them
                ct.set_error(True)
            yield ct


def handle_multiword_errors(token_stream, db, token_ctor):

    """ Parse a stream of tokens looking for multiword phrases
        containing errors.
        The algorithm implements N-token lookahead where N is the
        length of the longest phrase.
    """

    mwes = MultiwordErrorStream(db, token_ctor)
    yield from mwes.process(token_stream)


# Compound word stuff

# Illegal prefixes that will be split off from the rest of the word
NOT_FORMERS = frozenset(("allra", "alhliða", "fjölnota", "margnota", "ótal"))

# Illegal prefixes that will be substituted
WRONG_FORMERS = {
    "athugana": "athugunar",
    "ferminga": "fermingar",
    "feykna": "feikna",
    "fyrna": "firna",
    "fjarskiptar": "fjarskipta",
    "fjárfestinga": "fjárfestingar",
    "forvarna": "forvarnar",
    "heyrna": "heyrnar",
    "kvartana": "kvörtunar",
    "kvenn": "kven",
    "loftlags": "loftslags",
    "pantana": "pöntunar",
    "ráðninga": "ráðningar",
    "skráninga": "skráningar",
    "ábendinga": "ábendingar",
}


def fix_compound_words(token_stream, db, token_ctor, auto_uppercase):
    """ Fix incorrectly compounded words """

    at_sentence_start = False

    for token in token_stream:

        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue

        if token.kind == TOK.PUNCTUATION or token.kind == TOK.ORDINAL:
            yield token
            # Don't modify at_sentence_start in this case
            continue

        if token.kind != TOK.WORD or not token.val or "-" not in token.val[0].stofn:
            # Not a compound word
            yield token
            at_sentence_start = False
            continue

        # Compound word
        cw = token.val[0].stofn.split("-")

        # Special case for the prefix "ótal" which the compounder
        # splits into ó-tal
        if len(cw) >= 3 and cw[0] == "ó" and cw[1] == "tal":
            cw = ["ótal"] + cw[2:]

        if cw[0] in NOT_FORMERS:
            # Prefix is invalid as such; should be split
            # into two words
            prefix = emulate_case(cw[0], token.txt)
            w, m = db.lookup_word(
                prefix, at_sentence_start, auto_uppercase
            )
            t1 = token_ctor.Word(w, m, token=token)
            t1.set_error(
                CompoundError(
                    "002",
                    "Orðinu '{0}' var skipt upp".format(token.txt),
                    span=2
                )
            )
            yield t1
            at_sentence_start = False
            suffix = token.txt[len(cw[0]):]
            w, m = db.lookup_word(
                suffix, at_sentence_start, auto_uppercase
            )
            token = token_ctor.Word(w, m, token=token)

        elif cw[0] in WRONG_FORMERS:
            # Splice a correct front onto the word
            # ('feyknaglaður' -> 'feiknaglaður')
            correct_former = WRONG_FORMERS[cw[0]]
            corrected = correct_former + token.txt[len(cw[0]):]
            corrected = emulate_case(corrected, token.txt)
            w, m = db.lookup_word(
                corrected, at_sentence_start, auto_uppercase
            )
            t1 = token_ctor.Word(w, m, token=token)
            t1.set_error(
                CompoundError(
                    "004",
                    "Samsetta orðinu '{0}' var breytt í '{1}'"
                    .format(token.txt, corrected)
                )
            )
            token = t1

        yield token
        at_sentence_start = False


def lookup_unknown_words(corrector, token_ctor, token_stream, auto_uppercase):
    """ Try to identify unknown words in the token stream, for instance
        as spelling errors (character juxtaposition, deletion, insertion...) """

    at_sentence_start = False

    # Dict of { (at sentence start, single letter token) : allowed correction }
    single_letter_corrections = {
        (False, "a"): "á",
        (False, "i"): "í",
        (True, "A"): "Á",
        (True, "I"): "Í"
    }

    def is_immune(token):
        """ Return True if the token should definitely not be
            corrected """
        if token.val and len(token.val) == 1 and token.val[0].beyging == "-":
            # This is probably an abbreviation, having a single meaning
            # and no declension information
            return True
        return False

    def correct_word(code, token, corrected, corrected_display):
        """ Return a token for a corrected version of token_txt,
            marked with a SpellingError if corrected_display is
            a string containing the corrected word to be displayed """
        w, m = corrector.db.lookup_word(
            corrected, at_sentence_start, auto_uppercase
        )
        ct = token_ctor.Word(w, m, token=token if corrected_display else None)
        if corrected_display:
            if "." in corrected_display:
                text = (
                    "Skammstöfunin '{0}' var leiðrétt í '{1}'"
                    .format(token.txt, corrected_display)
                )
            else:
                text = (
                    "Orðið '{0}' var leiðrétt í '{1}'"
                    .format(token.txt, corrected_display)
                )
            ct.set_error(SpellingError("{0:03}".format(code), text))
        else:
            # In a multi-word sequence, mark the replacement
            # tokens with a boolean value so that further
            # replacements will not be made
            ct.set_error(True)
        return ct

    for token in token_stream:

        if token.kind == TOK.S_BEGIN:
            yield token
            # A new sentence is starting
            at_sentence_start = True
            continue

        if token.kind == TOK.PUNCTUATION or token.kind == TOK.ORDINAL:
            yield token
            # Don't modify at_sentence_start in this case
            continue

        if token.kind != TOK.WORD or token.error or " " in token.txt:
            # We don't process multi-word composites, or tokens that
            # already have an associated error and eventual correction
            yield token
            # We're now within a sentence
            at_sentence_start = False
            continue

        # The token is a word

        # Check unique errors - some of those may have
        # BÍN annotations via the compounder
        # Examples: 'kvenær' -> 'hvenær', 'starfssemi' -> 'starfsemi'
        # !!! TODO: Handle upper/lowercase
        if token.txt in UniqueErrors.DICT:
            # Note: corrected is a tuple
            corrected = UniqueErrors.DICT[token.txt]
            assert isinstance(corrected, tuple)
            corrected_display = " ".join(corrected)
            for ix, corrected_word in enumerate(corrected):
                if ix == 0:
                    yield correct_word(1, token, corrected_word, corrected_display)
                else:
                    # In a multi-word sequence, we only mark the first
                    # token with a SpellingError
                    yield correct_word(1, token, corrected_word, None)
                at_sentence_start = False
            continue

        # Check wrong word forms, i.e. those that do not exist in BÍN
        # !!! TODO: Some error forms are present in BÍN but in a different
        # !!! TODO: case (for instance, 'á' as a nominative of 'ær').
        # !!! TODO: We are not handling those here.
        # !!! TODO: Handle upper/lowercase
        if not token.val and ErrorForms.contains(token.txt):
            corrected = ErrorForms.get_correct_form(token.txt)
            yield correct_word(2, token, corrected, corrected)
            at_sentence_start = False
            continue

        # Check rare (or nonexistent) words and see if we have a potential correction
        if not token.error and not is_immune(token) and corrector.is_rare(token.txt):
            # Yes, this is a rare word that needs further attention
            corrected = corrector.correct(
                token.txt,
                at_sentence_start=at_sentence_start
            )
            if corrected != token.txt:
                if token.txt[0].lower() == "ó" and corrected == token.txt[1:]:
                    # The correction simply removed "ó" from the start of the
                    # word: probably not a good idea
                    pass
                elif (
                    len(token.txt) == 1
                    and corrected != single_letter_corrections.get(
                        (at_sentence_start, token.txt)
                    )
                ):
                    # Only allow single-letter corrections of a->á and i->í
                    pass
                else:
                    # We have a better candidate: yield it
                    yield correct_word(4, token, corrected, corrected)
                    at_sentence_start = False
                    continue

        # Check for completely unknown and uncorrectable words
        if not token.val:
            # No annotation and not able to correct:
            # mark the token as an unknown word
            token.set_error(
                UnknownWordError(
                    "001", "Óþekkt orð: '{0}'".format(token.txt)
                )
            )

        if token.error is True:
            # Erase the boolean error continuation marker:
            # we no longer need it
            token.set_error(None)

        yield token
        at_sentence_start = False


def fix_capitalization(token_stream, db, token_ctor, auto_uppercase):
    """ Annotate tokens with errors if they are capitalized incorrectly """

    stems = CapitalizationErrors.SET_REV

    def is_wrong(token):
        """ Return True if the word is wrongly capitalized """
        word = token.txt
        if " " in word:
            # Multi-word token: can't be listed in [capitalization_errors]
            return False
        lower = True
        if word.islower():
            # íslendingur -> Íslendingur
            # finni -> Finni
            rev_word = word.title()
        elif word.istitle():
            # Danskur -> danskur
            rev_word = word.lower()
            lower = False
        else:
            # All upper case or other strange capitalization:
            # don't bother
            return False
        meanings = db.meanings(rev_word) or []
        # If we don't find any of the stems of the "corrected"
        # meanings in the corrected error set (SET_REV),
        # the word was correctly capitalized
        if all(m.stofn not in stems for m in meanings):
            return False
        # Potentially wrong, but check for a corner
        # case: the original word may exist in its
        # original case in a non-noun/adjective category,
        # such as "finni" and "finna" as a verb
        if lower and any(m.ordfl not in {"kk", "kvk", "hk"} for m in token.val):
            # Not definitely wrong
            return False
        # Definitely wrong
        return True

    at_sentence_start = False
    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue
        # !!! TODO: Consider whether to overwrite previous error,
        # !!! if token.error is not None
        if token.kind == TOK.WORD and is_wrong(token):
            if token.txt.istitle():
                if not at_sentence_start:
                    original_txt = token.txt
                    w, m = db.lookup_word(
                        token.txt.lower(), at_sentence_start, auto_uppercase
                    )
                    token = token_ctor.Word(w, m, token=token)
                    token.set_error(
                        CapitalizationError(
                            "001",
                            "Orð á að byrja á lágstaf: '{0}'".format(original_txt)
                        )
                    )
            elif token.txt.islower():
                original_txt = token.txt
                w, m = db.lookup_word(
                    token.txt.title(), at_sentence_start, auto_uppercase
                )
                token = token_ctor.Word(w, m, token=token)
                token.set_error(
                    CapitalizationError(
                        "002",
                        "Orð á að byrja á hástaf: '{0}'".format(original_txt)
                    )
                )
        yield token
        if token.kind != TOK.PUNCTUATION and token.kind != TOK.ORDINAL:
            # !!! TODO: This may need to be made more intelligent
            at_sentence_start = False


def check_taboo_words(token_stream):
    """ Annotate taboo words with warnings """

    for token in token_stream:

        # Check taboo words
        if token.kind == TOK.WORD and token.val:
            # !!! TODO: This could be made more efficient if all
            # !!! TODO: taboo word forms could be generated ahead of time
            # !!! TODO: and checked via a set lookup
            for m in token.val:
                stofn = m.stofn.replace("-", "")
                if stofn in TabooWords.DICT:
                    # Taboo word
                    suggested_word = TabooWords.DICT[stofn].split("_")[0]
                    token.set_error(
                        TabooWarning(
                            "001",
                            "Óheppilegt eða óviðurkvæmilegt orð, skárra væri t.d. '{0}'"
                            .format(suggested_word)
                        )
                    )
                    break

        yield token


class Correct_TOK(TOK):

    """ A derived class to override token construction methods
        as required to generate CorrectToken instances instead of
        tokenizer.TOK instances """

    @staticmethod
    def Word(w, m=None, token=None):
        ct = CorrectToken.word(w, m)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct

    @staticmethod
    def Number(w, n, cases=None, genders=None, token=None):
        ct = CorrectToken(TOK.NUMBER, w, (n, cases, genders))
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token, coalesce=True)
        return ct

    @staticmethod
    def Person(w, m=None, token=None):
        ct = CorrectToken(TOK.PERSON, w, m)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct

    @staticmethod
    def Entity(w, token=None):
        ct = CorrectToken(TOK.ENTITY, w, None)
        if token is not None:
            # This token is being constructed in reference to a previously
            # generated token, or a list of tokens, which might have had
            # an associated error: make sure that it is preserved
            ct.copy_error(token)
        return ct


class CorrectionPipeline(DefaultPipeline):

    """ Override the default tokenization pipeline defined in binparser.py
        in ReynirPackage, adding a correction phase """

    def __init__(self, text, auto_uppercase=False):
        super().__init__(text, auto_uppercase)
        self._corrector = None

    # Use the Correct_TOK class to construct tokens, instead of
    # TOK (tokenizer.py) or _Bin_TOK (bintokenizer.py)
    _token_ctor = Correct_TOK

    def correct_tokens(self, stream):
        """ Add a correction pass just before BÍN annotation """
        return parse_errors(stream, self._db)

    def check_spelling(self, stream):
        """ Attempt to resolve unknown words """
        # Create a Corrector on the first invocation
        if self._corrector is None:
            self._corrector = Corrector(self._db)

        # Fix compound words
        stream = fix_compound_words(
            stream, self._db, self._token_ctor, self._auto_uppercase
        )
        # Fix multiword error phrases
        stream = handle_multiword_errors(
            stream, self._db, self._token_ctor
        )
        # Fix single-word errors
        stream = lookup_unknown_words(
            self._corrector, self._token_ctor, stream, self._auto_uppercase
        )
        # Fix the capitalization
        stream = fix_capitalization(
            stream, self._db, self._token_ctor, self._auto_uppercase
        )

        # Check taboo words
        stream = check_taboo_words(stream)

        return stream


def tokenize(text, auto_uppercase=False):
    """ Tokenize text using the correction pipeline, overriding a part
        of the default tokenization pipeline """
    pipeline = CorrectionPipeline(text, auto_uppercase)
    return pipeline.tokenize()

