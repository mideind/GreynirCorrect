"""

    Greynir: Natural language processing for Icelandic

    Settings module

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


    This module reads and interprets the GreynirCorrect.conf
    configuration file. The file can include other files using the $include
    directive, making it easier to arrange configuration sections into logical
    and manageable pieces.

    Sections are identified like so: [ section_name ]

    Comments start with # signs.

    Sections are interpreted by section handlers.

"""

from typing import Dict, Set, List, Tuple
import os
import threading

from collections import defaultdict

from reynir.basics import ConfigError, LineReader
from reynir.bindb import GreynirBin
from reynir.bintokenizer import StateDict


ErrorFormTuple = Tuple[str, str, int, str, str]

# A set of all strings that should be interpreted as True
TRUE = frozenset(("true", "True", "1", "yes", "Yes"))


class AllowedMultiples:

    # Set of word forms allowed to appear more than once in a row
    SET: Set[str] = set()

    @staticmethod
    def add(word: str) -> None:
        AllowedMultiples.SET.add(word)


class WrongCompounds:

    # Dictionary structure: dict { wrong_compound : "right phrase" }
    DICT: Dict[str, Tuple[str, ...]] = {}

    @staticmethod
    def add(word: str, parts: Tuple[str, ...]) -> None:
        if word in WrongCompounds.DICT:
            raise ConfigError(
                "Multiple definition of '{0}' in wrong_compounds section".format(word)
            )
        assert isinstance(parts, tuple)
        WrongCompounds.DICT[word] = parts


class SplitCompounds:

    # Dict of the form { first_part : set(second_part_stem) }
    DICT: Dict[str, Set[str]] = defaultdict(set)

    @staticmethod
    def add(first_part: str, second_part_stem: str) -> None:
        if (
            first_part in SplitCompounds.DICT
            and second_part_stem in SplitCompounds.DICT[first_part]
        ):
            raise ConfigError(
                "Multiple definition of '{0}' in split_compounds section".format(
                    first_part + " " + second_part_stem
                )
            )
        SplitCompounds.DICT[first_part].add(second_part_stem)


class UniqueErrors:

    # Dictionary structure: dict { wrong_word : (tuple of right words) }
    DICT: Dict[str, Tuple[str, ...]] = dict()

    @staticmethod
    def add(word: str, corr: Tuple[str, ...]) -> None:
        if word in UniqueErrors.DICT:
            raise ConfigError(
                "Multiple definition of '{0}' in unique_errors section".format(word)
            )
        UniqueErrors.DICT[word] = corr


class MultiwordErrors:

    # Dictionary structure: dict { phrase tuple: error specification }
    # List of tuples of multiword error phrases and their word category lists
    LIST: List[Tuple[Tuple[str, ...], str, List[str]]] = []
    # Parsing dictionary keyed by first word of phrase
    DICT: StateDict = defaultdict(list)
    # Error dictionary, { phrase : (error_code, right_phrase, right_parts_of_speech) }
    ERROR_DICT: Dict[Tuple[str, ...], str] = dict()

    @staticmethod
    def add(words: Tuple[str, ...], error: str) -> None:
        if words in MultiwordErrors.ERROR_DICT:
            raise ConfigError(
                "Multiple definition of '{0}' in multiword_errors section".format(
                    " ".join(words)
                )
            )
        MultiwordErrors.ERROR_DICT[words] = error

        # Add to phrase list
        ix = len(MultiwordErrors.LIST)

        a = error.split(",")
        if len(a) != 2:
            raise ConfigError("Expected two comma-separated parameters within $error()")
        code = a[0].strip()
        replacement = a[1].strip().split()

        # Append the phrase and the error specification in tuple form
        MultiwordErrors.LIST.append((words, code, replacement))

        # Dictionary structure: dict { firstword: [ (restword_list, phrase_index) ] }
        MultiwordErrors.DICT[words[0]].append((list(words[1:]), ix))

    @staticmethod
    def get_phrase(ix: int) -> Tuple[str, ...]:
        """ Return the original phrase with index ix """
        return MultiwordErrors.LIST[ix][0]

    @staticmethod
    def get_phrase_length(ix: int) -> int:
        """ Return the count of words in the original phrase with index ix """
        return len(MultiwordErrors.LIST[ix][0])

    @staticmethod
    def get_code(ix: int) -> str:
        """ Return the error code with index ix """
        return MultiwordErrors.LIST[ix][1]

    @staticmethod
    def get_replacement(ix: int) -> List[str]:
        """ Return the replacement phrase with index ix """
        return MultiwordErrors.LIST[ix][2]


class TabooWords:

    # Dictionary structure: dict { taboo_word : (suggested_replacement, explanation) }
    DICT: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def add(word: str, replacement: str, explanation: str) -> None:
        if word in TabooWords.DICT:
            raise ConfigError(
                "Multiple definition of '{0}' in taboo_words section".format(word)
            )
        db = GreynirBin.get_db()
        a = word.split("_")
        _, m = db.lookup_g(a[0])
        if not m or (len(a) >= 2 and all(mm.ordfl != a[1] for mm in m)):
            raise ConfigError(
                "The taboo word '{0}' is not found in BÍN".format(word)
            )
        TabooWords.DICT[word] = (replacement, explanation)


class Suggestions:

    # Dictionary structure: dict { bad_word : [ suggested_replacements ] }
    DICT: Dict[str, List[str]] = {}

    @staticmethod
    def add(word: str, replacements: List[str]) -> None:
        if word in Suggestions.DICT:
            raise ConfigError(
                "Multiple definition of '{0}' in suggestions section".format(word)
            )
        Suggestions.DICT[word] = replacements


class CapitalizationErrors:

    # Set of wrongly capitalized words
    SET: Set[str] = set()
    # Reverse capitalization (íslendingur -> Íslendingur, Danskur -> danskur)
    SET_REV: Set[str] = set()

    @staticmethod
    def emulate_case(s: str, template: str) -> str:
        """ Return the string s but emulating the case of the template
            (lower/upper/capitalized) """
        if template.isupper():
            return s.upper()
        if template and template[0].isupper():
            return s.capitalize()
        return s

    @staticmethod
    def reverse_capitalization(word: str, *, split_on_hyphen: bool = False) -> str:
        """ Return a word with its capitalization reversed (lower <-> upper case) """
        if split_on_hyphen and "-" in word:
            # 'norður-kórea' -> 'Norður-Kórea'
            return "-".join(
                CapitalizationErrors.reverse_capitalization(part)
                for part in word.split("-")
            )
        if word.islower():
            # Lowercase word
            word_rev = word.capitalize()
        elif word.isupper() and len(word) > 1:
            # Multi-letter uppercase acronym
            word_rev = word.capitalize()
        elif word[0].isupper() and word[1:].islower():
            # Uppercase word
            word_rev = word.lower()
        else:
            raise ConfigError("'{0}' cannot have mixed capitalization".format(word))
        return word_rev

    @staticmethod
    def add(word: str) -> None:
        """ Add the given (wrongly capitalized) word stem to the stem set """
        # We support compound words such as 'félags- og barnamálaráðherra' here
        split_on_hyphen = False
        if " " in word:
            prefix, suffix = word.rsplit(" ", maxsplit=1)
            prefix += " "
        else:
            prefix, suffix = "", word
            # Split_on_hyphen is True for e.g. 'norður-kórea' and 'nýja-sjáland'
            split_on_hyphen = "-" in word
        db = GreynirBin().get_db()
        # The suffix may not be in BÍN except as a compound, and in that
        # case we want its hyphenated lemma
        suffix_rev = CapitalizationErrors.reverse_capitalization(
            suffix, split_on_hyphen=split_on_hyphen
        )
        _, m = db.lookup_g(suffix_rev)
        # Only consider lemmas
        m = [mm for mm in m if mm.stofn == mm.ordmynd]
        if not m:
            raise ConfigError(
                "No BÍN meaning for '{0}' (from error word '{1}') in capitalization_errors section".format(
                    suffix_rev, word
                )
            )
        if not prefix:
            # This might be something like 'barnamálaráðherra' which comes out
            # with a lemma of 'barnamála-ráðherra'
            word = CapitalizationErrors.emulate_case(m[0].stofn, template=word)
        else:
            # This might be something like 'félags- og barnamálaráðherra' which comes out
            # with a lemma of 'félags- og barnamála-ráðherra'
            word = prefix + m[0].stofn
        if word in CapitalizationErrors.SET:
            raise ConfigError(
                "Multiple definition of '{0}' in capitalization_errors section".format(
                    word
                )
            )
        # Construct the reverse casing of the word
        word_rev = CapitalizationErrors.reverse_capitalization(
            word, split_on_hyphen=split_on_hyphen
        )
        # Add the word and its reverse case to the set of errors
        CapitalizationErrors.SET.add(word)
        CapitalizationErrors.SET_REV.add(word_rev)


class OwForms:

    # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
    DICT: Dict[str, Tuple[str, str, int, str, str]] = dict()

    @staticmethod
    def contains(word: str) -> bool:
        """ Check whether the word form is in the error forms dictionary,
            either in its original casing or in a lower case form """
        d = OwForms.DICT
        if word.islower():
            return word in d
        return word in d or word.lower() in d

    @staticmethod
    def add(wrong_form: str, meaning: Tuple[str, str, int, str, str]) -> None:
        OwForms.DICT[wrong_form] = meaning

    @staticmethod
    def get_lemma(wrong_form: str) -> str:
        return OwForms.DICT[wrong_form][0]

    @staticmethod
    def get_correct_form(wrong_form: str) -> str:
        """ Return a corrected form of the given word, attempting
            to emulate the lower/upper/title case of the word """
        # First, try the original casing of the wrong form
        c = OwForms.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = OwForms.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    @staticmethod
    def get_id(wrong_form: str) -> int:
        return OwForms.DICT[wrong_form][2]

    @staticmethod
    def get_category(wrong_form: str) -> str:
        return OwForms.DICT[wrong_form][3]

    @staticmethod
    def get_tag(wrong_form: str) -> str:
        return OwForms.DICT[wrong_form][4]


class CIDErrorForms:

    # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
    DICT: Dict[str, ErrorFormTuple] = dict()

    @staticmethod
    def contains(word: str) -> bool:
        """ Check whether the word form is in the error forms dictionary,
            either in its original casing or in a lower case form """
        d = CIDErrorForms.DICT
        return word in d or word.lower() in d

    @staticmethod
    def add(wrong_form: str, meaning: ErrorFormTuple) -> None:
        CIDErrorForms.DICT[wrong_form] = meaning

    @staticmethod
    def get_lemma(wrong_form: str) -> str:
        return CIDErrorForms.DICT[wrong_form][0]

    @staticmethod
    def get_correct_form(wrong_form: str) -> str:
        """ Return a corrected form of the given word, attempting
            to emulate the lower/upper/title case of the word """
        # First, try the original casing of the wrong form
        c = CIDErrorForms.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = CIDErrorForms.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    @staticmethod
    def get_id(wrong_form: str) -> int:
        return CIDErrorForms.DICT[wrong_form][2]

    @staticmethod
    def get_category(wrong_form: str) -> str:
        return CIDErrorForms.DICT[wrong_form][3]

    @staticmethod
    def get_tag(wrong_form: str) -> str:
        return CIDErrorForms.DICT[wrong_form][4]


class CDErrorForms:

    # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
    DICT: Dict[str, ErrorFormTuple] = dict()

    @staticmethod
    def contains(word: str) -> bool:
        """ Check whether the word form is in the error forms dictionary,
            either in its original casing or in a lower case form """
        d = CDErrorForms.DICT
        if word.islower():
            return word in d
        return word in d or word.lower() in d

    @staticmethod
    def add(wrong_form: str, meaning: ErrorFormTuple) -> None:
        CDErrorForms.DICT[wrong_form] = meaning

    @staticmethod
    def get_lemma(wrong_form: str) -> str:
        return CDErrorForms.DICT[wrong_form][0]

    @staticmethod
    def get_correct_form(wrong_form: str) -> str:
        """ Return a corrected form of the given word, attempting
            to emulate the lower/upper/title case of the word """
        # First, try the original casing of the wrong form
        c = CDErrorForms.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = CDErrorForms.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    @staticmethod
    def get_id(wrong_form: str) -> int:
        return CDErrorForms.DICT[wrong_form][2]

    @staticmethod
    def get_category(wrong_form: str) -> str:
        return CDErrorForms.DICT[wrong_form][3]

    @staticmethod
    def get_tag(wrong_form: str) -> str:
        return CDErrorForms.DICT[wrong_form][4]


class Morphemes:

    # dict { morpheme : [ preferred PoS ] }
    BOUND_DICT: Dict[str, List[str]] = {}
    # dict { morpheme : [ excluded PoS ] }
    FREE_DICT: Dict[str, List[str]] = {}

    @staticmethod
    def add(morph: str, boundlist: List[str], freelist: List[str]) -> None:
        if not boundlist:
            raise ConfigError("A definition of allowed PoS is necessary with morphemes")
        Morphemes.BOUND_DICT[morph] = boundlist
        # The freelist may be empty
        Morphemes.FREE_DICT[morph] = freelist


class Settings:

    """ Global settings """

    _lock = threading.Lock()
    loaded = False
    DEBUG = os.environ.get("DEBUG", "").strip() in TRUE

    # Configuration settings from the GreynirCorrect.conf file

    @staticmethod
    def _handle_settings(s: str) -> None:
        """ Handle config parameters in the settings section """
        a: List[str] = s.lower().split("=", maxsplit=1)
        par = a[0].strip().lower()
        val = a[1].strip()
        try:
            if par == "debug":
                Settings.DEBUG = val in TRUE
            else:
                raise ConfigError("Unknown configuration parameter '{0}'".format(par))
        except ValueError:
            raise ConfigError("Invalid parameter value: {0} = {1}".format(par, val))

    @staticmethod
    def _handle_allowed_multiples(s: str) -> None:
        """ Handle config parameters in the allowed_multiples section """
        assert s
        if len(s.split()) != 1:
            raise ConfigError(
                "Only one word per line allowed in allowed_multiples section"
            )
        if s in AllowedMultiples.SET:
            raise ConfigError(
                "'{0}' is repeated in allowed_multiples section".format(s)
            )
        AllowedMultiples.add(s)

    @staticmethod
    def _handle_wrong_compounds(s: str) -> None:
        """ Handle config parameters in the wrong_compounds section """
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between compound word and its parts")
        word = a[0].strip().strip('"')
        parts = a[1].strip().strip('"').split()
        if not word:
            raise ConfigError(
                "Expected word before the comma in wrong_compounds section"
            )
        if len(parts) < 2:
            raise ConfigError("Missing word part(s) in wrong_compounds section")
        if len(word.split()) != 1:
            raise ConfigError(
                "Multiple words not allowed before comma in wrong_compounds section"
            )
        WrongCompounds.add(word, tuple(parts))

    @staticmethod
    def _handle_split_compounds(s: str) -> None:
        """ Handle config parameters in the split_compounds section """
        parts = s.split()
        if len(parts) != 2:
            raise ConfigError("Missing word part(s) in split_compounds section")
        SplitCompounds.add(parts[0], parts[1])

    @staticmethod
    def _handle_unique_errors(s: str) -> None:
        """ Handle config parameters in the unique_errors section """
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between error word and its correction")
        word = a[0].strip()
        if len(word) < 3:
            raise ConfigError(
                "Expected nonempty word before comma in unique_errors section"
            )
        if word[0] != '"' or word[-1] != '"':
            raise ConfigError("Expected word in double quotes in unique_errors section")
        word = word[1:-1]
        corr = a[1].strip()
        if len(corr) < 3:
            raise ConfigError(
                "Expected nonempty word after comma in unique_errors section"
            )
        if corr[0] != '"' or corr[-1] != '"':
            raise ConfigError(
                "Expected word in double quotes after comma in unique_errors section"
            )
        corr = corr[1:-1]
        corr_t = tuple(corr.split())
        if not word:
            raise ConfigError("Expected word before the comma in unique_errors section")
        if len(word.split()) != 1:
            raise ConfigError(
                "Multiple words not allowed before the comma in unique_errors section"
            )
        UniqueErrors.add(word, corr_t)

    @staticmethod
    def _handle_capitalization_errors(s: str) -> None:
        """ Handle config parameters in the capitalization_errors section """
        CapitalizationErrors.add(s)

    @staticmethod
    def _handle_taboo_words(s: str) -> None:
        """ Handle config parameters in the taboo_words section """
        # Start by parsing explanation string off the end (right hand side), if present
        lquote = s.find("\"")
        rquote = s.rfind("\"")
        if (lquote >= 0) != (rquote >= 0):
            raise ConfigError("Explanation string for taboo word should be enclosed in double quotes")
        if lquote >= 0:
            # Obtain explanation from within quotes
            explanation = s[lquote + 1:rquote].strip()
            s = s[:lquote].rstrip()
        else:
            # No explanation
            explanation = ""
        if not s:
            raise ConfigError("Expected taboo word and a suggested replacement")
        a = s.lower().split()
        if len(a) > 2:
            raise ConfigError("Expected taboo word and a suggested replacement")
        taboo = a[0].strip()
        if len(a) == 2:
            replacement = a[1].strip()
        else:
            replacement = taboo
        # Check all replacement words, which are separated by slashes '/'
        if any(r.count("_") != 1 for r in replacement.split("/")):
            raise ConfigError(
                "Suggested replacement(s) should include a word category (_xx)"
            )
        TabooWords.add(taboo, replacement, explanation)

    @staticmethod
    def _handle_suggestions(s: str) -> None:
        """ Handle config parameters in the suggestions section """
        a = s.lower().split()
        if len(a) < 2:
            raise ConfigError(
                "Expected bad word and at least one suggested replacement"
            )
        if any(w.count("_") != 1 for w in a[1:]):
            raise ConfigError(
                "Suggested replacements should include word category (_xx)"
            )
        Suggestions.add(a[0].strip(), [w.strip() for w in a[1:]])

    @staticmethod
    def _handle_multiword_errors(s: str) -> None:
        """ Handle config parameters in the multiword_errors section """
        a = s.lower().split("$error", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected phrase followed by $error(...)")
        phrase = tuple(a[0].strip().split())
        if len(phrase) < 2:
            raise ConfigError("Multiword phrase must contain at least two words")
        error = a[1].strip()
        if len(error) < 3:
            raise ConfigError("Incomplete error specification for multiword phrase")
        if error[0] != "(" or error[-1] != ")":
            raise ConfigError("Error specification should be enclosed in parentheses")
        MultiwordErrors.add(phrase, error[1:-1])

    @staticmethod
    def _handle_ow_forms(s: str) -> None:
        """ Handle config parameters in the ow_forms section """
        split = s.strip().split(";")
        if len(split) != 6:
            raise ConfigError("Expected wrong form;lemma;correct form;id;category;tag")
        wrong_form = split[0].strip()
        correct_form = split[2].strip()
        if wrong_form == correct_form:
            return
            # !!! TODO: Should do this:
            # raise ConfigError(
            #     "Wrong form identical to correct form for '{0}'".format(wrong_form)
            # )
        meaning: ErrorFormTuple = (
            split[1].strip(),  # Lemma (stofn)
            correct_form,  # Correct form (ordmynd)
            int(split[3]),  # Id (utg)
            split[4].strip(),  # Category (ordfl)
            split[5].strip(),  # Tag (beyging)
        )
        OwForms.add(wrong_form, meaning)

    @staticmethod
    def _handle_error_forms(s: str) -> None:
        """ Handle config parameters in the error_forms section """
        split = s.strip().split(";")
        if len(split) != 7:
            raise ConfigError(
                "Expected wrong form;lemma;correct form;id;category;tag;errortype"
            )
        wrong_form = split[0].strip()
        correct_form = split[2].strip()
        if wrong_form == correct_form:
            print(s)
            raise ConfigError(
                "Wrong form identical to correct form for '{0}'".format(wrong_form)
            )
        meaning: ErrorFormTuple = (
            split[1].strip(),  # Lemma (stofn)
            correct_form,  # Correct form (ordmynd)
            int(split[3]),  # Id (utg)
            split[4].strip(),  # Category (ordfl)
            split[5].strip(),  # Tag (beyging)
        )
        etype = split[6].strip()
        if etype == "cid":
            CIDErrorForms.add(wrong_form, meaning)  # context-independent errors
        elif etype == "cd":
            CDErrorForms.add(wrong_form, meaning)  # context-dependent errors
        else:
            raise ConfigError("Wrong error type given, expected 'cid' or 'cd'")

    @staticmethod
    def _handle_morphemes(s: str) -> None:
        """ Process the contents of the [morphemes] section """
        freelist: List[str] = []
        boundlist: List[str] = []
        spl = s.split()
        if len(spl) < 2:
            raise ConfigError(
                "Expected at least a prefix and an attachment specification"
            )
        m = spl[0]
        for pos in spl[1:]:
            if pos:
                if pos.startswith("+"):
                    boundlist.append(pos[1:])
                elif pos.startswith("-"):
                    freelist.append(pos[1:])
                else:
                    raise ConfigError(
                        "Attachment specification should start with '+' or '-'"
                    )
        Morphemes.add(m, boundlist, freelist)

    @staticmethod
    def read(fname: str) -> None:
        """ Read configuration file """

        with Settings._lock:

            if Settings.loaded or UniqueErrors.DICT or AllowedMultiples.SET:
                return

            CONFIG_HANDLERS = {
                "settings": Settings._handle_settings,
                "allowed_multiples": Settings._handle_allowed_multiples,
                "wrong_compounds": Settings._handle_wrong_compounds,
                "split_compounds": Settings._handle_split_compounds,
                "unique_errors": Settings._handle_unique_errors,
                "capitalization_errors": Settings._handle_capitalization_errors,
                "taboo_words": Settings._handle_taboo_words,
                "suggestions": Settings._handle_suggestions,
                "multiword_errors": Settings._handle_multiword_errors,
                "morphemes": Settings._handle_morphemes,
                "ow_forms": Settings._handle_ow_forms,
                "error_forms": Settings._handle_error_forms,
            }
            handler = None  # Current section handler

            rdr = None
            try:
                rdr = LineReader(fname, package_name=__name__)
                for s in rdr.lines():
                    # Ignore comments
                    ix = s.find("#")
                    if ix >= 0:
                        s = s[0:ix]
                    s = s.strip()
                    if not s:
                        # Blank line: ignore
                        continue
                    if s[0] == "[" and s[-1] == "]":
                        # New section
                        section = s[1:-1].strip().lower()
                        if section in CONFIG_HANDLERS:
                            handler = CONFIG_HANDLERS[section]
                            continue
                        raise ConfigError("Unknown section name '{0}'".format(section))
                    if handler is None:
                        raise ConfigError("No handler for config line '{0}'".format(s))
                    # Call the correct handler depending on the section
                    try:
                        handler(s)
                    except ConfigError as e:
                        # Add file name and line number information to the exception
                        # if it's not already there
                        e.set_pos(rdr.fname(), rdr.line())
                        raise e

            except ConfigError as e:
                # Add file name and line number information to the exception
                # if it's not already there
                if rdr:
                    e.set_pos(rdr.fname(), rdr.line())
                raise e

            Settings.loaded = True
