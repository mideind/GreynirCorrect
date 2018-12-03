"""

    Reynir: Natural language processing for Icelandic

    Error-correcting tokenization layer

    Copyright (C) 2018 Miðeind ehf.

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

from reynir import TOK
from reynir.bintokenizer import DefaultPipeline

from .settings import (
    AllowedMultiples, WrongCompounds, SplitCompounds, UniqueErrors,
    MultiwordErrors, CapitalizationErrors, TabooWords, ErrorForms
)
from .spelling import Corrector


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

    def copy_error(self, other):
        """ Copy the error field from another CorrectToken instance """
        if isinstance(other, list):
            # We have a list of CorrectToken instances to copy from:
            # find the first error in the list, if any, and copy it
            for t in other:
                if self.copy_error(t):
                    break
        elif isinstance(other, CorrectToken):
            self._err = other._err
        return self._err is not None

    @property
    def error(self):
        """ Return the error object associated with this token, if any """
        return self._err

    @property
    def error_description(self):
        """ Return the description of an error associated with this token, if any """
        return "" if self._err is None else self._err.description

    @property
    def error_code(self):
        """ Return the code of an error associated with this token, if any """
        return "" if self._err is None else self._err.code


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

    def __init__(self, code, txt):
        # Compound error codes start with "C"
        super().__init__("C" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


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
        # Taboo word warnings start with T
        super().__init__("T" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


class SpellingError(Error):

    """ An SpellingError is an erroneous word that could be replaced
        by a much more likely word that exists in the dictionary. """

    # S001: Common errors picked up by unique_errors. Should be corrected
    # S002: Errors handled by spelling.py. Corrections should possibly only be suggested.
    # S003: Erroneously formed word forms picked up by ErrorForms. Should be corrected. TODO split up by nature.
    # S004: Unknown word, no correction or suggestions available.

    def __init__(self, code, txt):
        # Spelling error codes start with "S"
        super().__init__("S" + code)
        self._txt = txt

    @property
    def description(self):
        return self._txt


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
                        "001", "Endurtekið orð ('{0}') var fellt burt"
                        .format(token.txt)
                    )
                )
                token = next_token
                continue

            # Splitting wrongly compounded words
            if token.txt and token.txt.lower() in WrongCompounds.DICT:
                for phrase_part in WrongCompounds.DICT[token.txt.lower()]:
                    new_token = CorrectToken.word(phrase_part)
                    new_token.set_error(
                        CompoundError(
                            "002", "Orðinu '{0}' var skipt upp"
                            .format(token.txt)
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
                        "003", "Orðin '{0} {1}' voru sameinuð í eitt"
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


def lookup_unknown_words(corrector, token_ctor, token_stream, auto_uppercase):
    """ Try to identify unknown words in the token stream, for instance
        as spelling errors (character juxtaposition, deletion, insertion...) """

    at_sentence_start = False

    def correct_word(code, token, corrected, corrected_display):
        """ Return a token for a corrected version of token_txt,
            marked with a SpellingError if corrected_display is
            a string containing the corrected word to be displayed """
        w, m = corrector.db.lookup_word(
            corrected, at_sentence_start, auto_uppercase
        )
        ct = token_ctor.Word(w, m, token=token if corrected_display else None)
        if corrected_display:
            ct.set_error(
                SpellingError(
                    "{0:03}".format(code),
                    "Orðið '{0}' var leiðrétt í '{1}'"
                        .format(token.txt, corrected_display)
                )
            )
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

        # Check taboo words
        if token.val:
            # The token has an annotation, so the word is in BÍN
            # !!! TODO: This could be made more efficient if all
            # !!! TODO: taboo word forms could be generated ahead of time
            # !!! TODO: and checked via a set lookup
            for m in token.val:
                if m.stofn in TabooWords.DICT:
                    # Taboo word
                    suggested_word = TabooWords.DICT[m.stofn].split("_")[0]
                    token.set_error(
                        TabooWarning(
                            "001",
                            "Óviðurkvæmilegt orð, skárra væri t.d. '{0}'".format(suggested_word)
                        )
                    )
                    break
            if token.error:
                # Found taboo word
                yield token
                at_sentence_start = False
                continue

        # Check rare (or nonexistent) words and see if we have a potential correction
        if corrector.is_rare(token.txt):
            # Yes, this is a rare one (>=95th percentile in a descending frequency distribution)
            corrected = corrector.correct(token.txt)
            if corrected != token.txt:
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

        yield token
        at_sentence_start = False


def fix_capitalization(token_stream, db, token_ctor, auto_uppercase):
    """ Annotate tokens with errors if they are capitalized incorrectly """

    stems = CapitalizationErrors.SET_REV

    def is_wrong(token):
        """ Return True if the word is wrongly capitalized """
        word = token.txt
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


class _Correct_TOK(TOK):

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
            ct.copy_error(token)
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


class CorrectionPipeline(DefaultPipeline):

    """ Override the default tokenization pipeline defined in binparser.py
        in ReynirPackage, adding a correction phase """

    def __init__(self, text, auto_uppercase=False):
        super().__init__(text, auto_uppercase)
        self._corrector = None

    # Use the _Correct_TOK class to construct tokens, instead of
    # TOK (tokenizer.py) or _Bin_TOK (bintokenizer.py)
    _token_ctor = _Correct_TOK

    def correct(self, stream):
        """ Add a correction pass just before BÍN annotation """
        return parse_errors(stream, self._db)

    def lookup_unknown_words(self, stream):
        """ Attempt to resolve unknown words """
        # Create a Corrector on the first invocation
        if self._corrector is None:
            self._corrector = Corrector(self._db)
        stream = lookup_unknown_words(
            self._corrector, self._token_ctor, stream, self._auto_uppercase
        )
        # Finally, fix the capitalization
        return fix_capitalization(
            stream, self._db, self._token_ctor, self._auto_uppercase
        )


def tokenize(text, auto_uppercase=False):
    """ Tokenize text using the correction pipeline, overriding a part
        of the default tokenization pipeline """
    pipeline = CorrectionPipeline(text, auto_uppercase)
    return pipeline.tokenize()

