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
    # to be subclassed or used in multiple inheritance scenarios
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
            # Word duplication
            if (
                token.txt
                and next_token.txt
                and token.txt.lower() == next_token.txt.lower()
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
    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue
        if token.kind == TOK.WORD and not token.val:  # Hasn't been annotated
            # Check unique errors first
            errkind = 0
            if token.txt in UniqueErrors.DICT:
                # Note: corrected is a tuple
                corrected = UniqueErrors.DICT[token.txt]
                assert isinstance(corrected, tuple)
                errkind = 1
            # Check wrong word forms, TODO split the list up by nature of error
            elif ErrorForms.contains(token.txt):
                corrected = [ErrorForms.get_correct_form(token.txt)]
                errkind = 3
            # Check edit distance errors
            else:
                corrected = corrector.correct(token.txt)
                if corrected != token.txt:
                    errkind = 2
                    corrected = [corrected]
            if errkind != 0:
                # It seems that we are able to correct the word:
                # look it up in the BÍN database
                for ix, corrected_word in enumerate(corrected):
                    w, m = corrector.db.lookup_word(
                        corrected_word, at_sentence_start, auto_uppercase
                    )
                    # Yield a word tuple with meanings
                    ct = token_ctor.Word(w, m, token=token if ix == 0 else None)
                    if ix == 0:
                        # Only mark the first generated token
                        # of a multi-word sequence
                        ct.set_error(
                            SpellingError(
                                "{0:03}".format(errkind),
                                "Orðið '{0}' var leiðrétt í '{1}'"
                                .format(token.txt, " ".join(corrected))
                            )
                        )
                    yield ct
                    at_sentence_start = False
                continue
            # Not able to correct: mark the token as an unknown word
            token.set_error(
                UnknownWordError(
                    "001", "Óþekkt orð: '{0}'".format(token.txt)
                )
            )
        yield token
        if token.kind != TOK.PUNCTUATION and token.kind != TOK.ORDINAL:
            # !!! TODO: This may need to be made more intelligent
            at_sentence_start = False


def fix_capitalization(token_stream, db):
    """ Annotate tokens with errors if they are capitalized incorrectly """

    stems = CapitalizationErrors.SET_REV

    def is_wrong(word):
        """ Return True if the word is wrongly capitalized """
        if word.islower():
            # íslendingur -> Íslendingur
            rev_word = word.title()
        elif word.istitle():
            # Danskur -> danskur
            rev_word = word.lower()
        else:
            # All upper case or other strange capitalization:
            # don't bother
            return False
        meanings = db.meanings(rev_word) or []
        # If we find any of the stems of the "corrected"
        # meanings in the corrected error set (SET_REV),
        # the word was wrongly capitalized
        return any(m.stofn in stems for m in meanings)

    at_sentence_start = False
    for token in token_stream:
        if token.kind == TOK.S_BEGIN:
            yield token
            at_sentence_start = True
            continue
        if token.kind == TOK.WORD and is_wrong(token.txt):
            if token.txt.istitle():
                if not at_sentence_start:
                    original_txt = token.txt
                    token = CorrectToken.word(token.txt.lower())
                    token.set_error(
                        CapitalizationError(
                            "001", "Orð á að byrja á lágstaf: '{0}'".format(original_txt)
                        )
                    )
            elif token.txt.islower():
                original_txt = token.txt
                token = CorrectToken.word(token.txt.title())
                token.set_error(
                    CapitalizationError(
                        "002", "Orð á að byrja á hástaf: '{0}'".format(original_txt)
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
        return fix_capitalization(stream, self._db)


def tokenize(text, auto_uppercase=False):
    """ Tokenize text using the correction pipeline, overriding a part
        of the default tokenization pipeline """
    pipeline = CorrectionPipeline(text, auto_uppercase)
    return pipeline.tokenize()
