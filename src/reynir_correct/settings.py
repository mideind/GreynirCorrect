"""

    Greynir: Natural language processing for Icelandic

    Settings module

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


    This module reads and interprets the GreynirCorrect.conf
    configuration file. The file can include other files using the $include
    directive, making it easier to arrange configuration sections into logical
    and manageable pieces.

    Sections are identified like so: [ section_name ]

    Comments start with # signs.

    Sections are interpreted by section handlers.

"""

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import os
import threading
from collections import defaultdict

from reynir.basics import ConfigError, LineReader
from reynir.bindb import GreynirBin
from reynir.bintokenizer import StateDict

ErrorFormTuple = Tuple[str, str, int, str, str]
# (lemma, id, cat, correct_word_form, tag, eink, malsnid, stafs, aslatt, beyg)
RitmyndirTuple = Tuple[str, int, str, str, str, int, str, str, str, str]
# (ritregla/spelling rule, category, other detail ) }
DetailsTuple = Tuple[str, str, str]
# A set of all strings that should be interpreted as True
TRUE = frozenset(("true", "True", "1", "yes", "Yes"))
# Einkunn value from Ritmyndir mapped to error code
R_EINKUNN: Mapping[int, str] = {
    0: "R000",
    1: "R001",  # Not an error
    2: "R002",
    3: "R003",
    4: "R004",
    5: "R005",
}


class AllowedMultiples:
    def __init__(self) -> None:
        self.SET: Set[str] = set()

    def add(self, word: str) -> None:
        self.SET.add(word)


class WrongCompounds:
    def __init__(self) -> None:
        # Dictionary structure: dict { wrong_compound : "right phrase" }
        self.DICT: Dict[str, Tuple[str, ...]] = {}

    def add(self, word: str, parts: Tuple[str, ...]) -> None:
        if word in self.DICT:
            raise ConfigError("Multiple definition of '{0}' in wrong_compounds section".format(word))
        assert isinstance(parts, tuple)
        self.DICT[word] = parts


class SplitCompounds:
    # Dict of the form { first_part : set(second_part_stem) }
    def __init__(self) -> None:
        self.DICT: Dict[str, Set[str]] = defaultdict(set)

    def add(self, first_part: str, second_part_stem: str) -> None:
        if first_part in self.DICT and second_part_stem in self.DICT[first_part]:
            raise ConfigError(
                "Multiple definition of '{0}' in split_compounds section".format(first_part + " " + second_part_stem)
            )
        self.DICT[first_part].add(second_part_stem)


class UniqueErrors:
    # Dictionary structure: dict { wrong_word : (tuple of right words) }
    def __init__(self) -> None:
        self.DICT: Dict[str, Tuple[str, ...]] = dict()

    def add(self, word: str, corr: Tuple[str, ...]) -> None:
        if word in self.DICT:
            raise ConfigError("Multiple definition of '{0}' in unique_errors section".format(word))
        self.DICT[word] = corr


class MultiwordErrors:
    def __init__(self) -> None:
        # Dictionary structure: dict { phrase tuple: error specification }
        # List of tuples of multiword error phrases and their word category lists
        self.LIST: List[Tuple[Tuple[str, ...], str, List[str]]] = []
        # Parsing dictionary keyed by first word of phrase
        self.DICT: StateDict = defaultdict(list)
        # Error dictionary, { phrase : (error_code, right_phrase, right_parts_of_speech) }
        self.ERROR_DICT: Dict[Tuple[str, ...], str] = dict()

    def add(self, words: Tuple[str, ...], error: str) -> None:
        if words in self.ERROR_DICT:
            raise ConfigError("Multiple definition of '{0}' in multiword_errors section".format(" ".join(words)))
        self.ERROR_DICT[words] = error

        # Add to phrase list
        ix = len(self.LIST)

        a = error.split(",")
        if len(a) != 2:
            raise ConfigError("Expected two comma-separated parameters within $error()")
        code = a[0].strip()
        replacement = a[1].strip().split()

        # Append the phrase and the error specification in tuple form
        self.LIST.append((words, code, replacement))

        # Dictionary structure: dict { firstword: [ (restword_list, phrase_index) ] }
        self.DICT[words[0]].append((list(words[1:]), ix))

    def get_phrase(self, ix: int) -> Tuple[str, ...]:
        """Return the original phrase with index ix"""
        return self.LIST[ix][0]

    def get_phrase_length(self, ix: int) -> int:
        """Return the count of words in the original phrase with index ix"""
        return len(self.LIST[ix][0])

    def get_code(self, ix: int) -> str:
        """Return the error code with index ix"""
        return self.LIST[ix][1]

    def get_replacement(self, ix: int) -> List[str]:
        """Return the replacement phrase with index ix"""
        return self.LIST[ix][2]


class TabooWords:
    def __init__(self) -> None:
        # Dictionary structure: dict { taboo_word : (suggested_replacement, explanation) }
        self.DICT: Dict[str, Tuple[str, str]] = {}

    def add(self, word: str, replacement: str, explanation: str) -> None:
        if word in self.DICT:
            raise ConfigError("Multiple definition of '{0}' in taboo_words section".format(word))
        db = GreynirBin.get_db()
        a = word.split("_")
        _, m = db.lookup_g(a[0])
        if not m or (len(a) >= 2 and all(mm.ordfl != a[1] for mm in m)):
            raise ConfigError("The taboo word '{0}' is not found in BÍN".format(word))
        self.DICT[word] = (replacement, explanation)


class ToneOfVoiceWords:
    def __init__(self) -> None:
        # Dictionary structure: dict { tone_of_voice : (suggested_replacement, explanation) }
        self.DICT: Dict[str, Tuple[str, str]] = {}

    def add(self, word: str, replacement: str, explanation: str) -> None:
        if word in self.DICT:
            raise ConfigError("Multiple definition of '{0}' in tone_of_voice section".format(word))
        db = GreynirBin.get_db()
        a = word.split("_")
        _, m = db.lookup_g(a[0])
        if not m or (len(a) >= 2 and all(mm.ordfl != a[1] for mm in m)):
            raise ConfigError("The word '{0}' is not found in BÍN".format(word))
        self.DICT[word] = (replacement, explanation)


class ToneOfVoicePatterns:
    def __init__(self) -> None:
        # A path to a python module containing third party tone of voice patterns
        self.PATH: str = ""

    def add(self, fpath: str) -> None:
        self.PATH = fpath


class Suggestions:
    def __init__(self) -> None:
        # Dictionary structure: dict { bad_word : [ suggested_replacements ] }
        self.DICT: Dict[str, List[str]] = {}

    def add(self, word: str, replacements: List[str]) -> None:
        if word in self.DICT:
            raise ConfigError("Multiple definition of '{0}' in suggestions section".format(word))
        self.DICT[word] = replacements


class CapitalizationErrors:
    def __init__(self) -> None:
        # Set of wrongly capitalized words
        self.SET: Set[str] = set()
        # Reverse capitalization (íslendingur -> Íslendingur, Danskur -> danskur)
        self.SET_REV: Set[str] = set()

    def emulate_case(self, s: str, template: str) -> str:
        """Return the string s but emulating the case of the template
        (lower/upper/capitalized)"""
        if template.isupper():
            return s.upper()
        if template and template[0].isupper():
            return s.capitalize()
        return s

    def reverse_capitalization(self, word: str, *, split_on_hyphen: bool = False) -> str:
        """Return a word with its capitalization reversed (lower <-> upper case)"""
        if split_on_hyphen and "-" in word:
            # 'norður-kórea' -> 'Norður-Kórea'
            return "-".join(self.reverse_capitalization(part) for part in word.split("-"))
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

    def add(self, word: str) -> None:
        """Add the given (wrongly capitalized) word stem to the stem set"""
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
        suffix_rev = self.reverse_capitalization(suffix, split_on_hyphen=split_on_hyphen)
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
            word = self.emulate_case(m[0].stofn, template=word)
        else:
            # This might be something like 'félags- og barnamálaráðherra' which comes out
            # with a lemma of 'félags- og barnamála-ráðherra'
            word = prefix + m[0].stofn
        if word in self.SET:
            raise ConfigError("Multiple definition of '{0}' in capitalization_errors section".format(word))
        # Construct the reverse casing of the word
        word_rev = self.reverse_capitalization(word, split_on_hyphen=split_on_hyphen)
        # Add the word and its reverse case to the set of errors
        self.SET.add(word)
        self.SET_REV.add(word_rev)


class OwForms:
    def __init__(self) -> None:
        # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
        self.DICT: Dict[str, Tuple[str, str, int, str, str]] = dict()

    def contains(self, word: str) -> bool:
        """Check whether the word form is in the error forms dictionary,
        either in its original casing or in a lower case form"""
        d = self.DICT
        if word.islower():
            return word in d
        return word in d or word.lower() in d

    def add(self, wrong_form: str, meaning: Tuple[str, str, int, str, str]) -> None:
        self.DICT[wrong_form] = meaning

    def get_lemma(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][0]

    def get_correct_form(self, wrong_form: str) -> str:
        """Return a corrected form of the given word, attempting
        to emulate the lower/upper/title case of the word"""
        # First, try the original casing of the wrong form
        c = self.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = self.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    def get_id(self, wrong_form: str) -> int:
        return self.DICT[wrong_form][2]

    def get_category(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][3]

    def get_tag(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][4]


class CIDErrorForms:
    def __init__(self) -> None:
        # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
        self.DICT: Dict[str, ErrorFormTuple] = dict()

    def contains(self, word: str) -> bool:
        """Check whether the word form is in the error forms dictionary,
        either in its original casing or in a lower case form"""
        d = self.DICT
        return word in d or word.lower() in d

    def add(self, wrong_form: str, meaning: ErrorFormTuple) -> None:
        self.DICT[wrong_form] = meaning

    def get_lemma(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][0]

    def get_correct_form(self, wrong_form: str) -> str:
        """Return a corrected form of the given word, attempting
        to emulate the lower/upper/title case of the word"""
        # First, try the original casing of the wrong form
        c = self.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = self.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    def get_id(self, wrong_form: str) -> int:
        return self.DICT[wrong_form][2]

    def get_category(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][3]

    def get_tag(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][4]


class CDErrorForms:
    def __init__(self) -> None:
        # dict { wrong_word_form : (lemma, correct_word_form, id, cat, tag) }
        self.DICT: Dict[str, ErrorFormTuple] = dict()

    def contains(self, word: str) -> bool:
        """Check whether the word form is in the error forms dictionary,
        either in its original casing or in a lower case form"""
        d = self.DICT
        if word.islower():
            return word in d
        return word in d or word.lower() in d

    def add(self, wrong_form: str, meaning: ErrorFormTuple) -> None:
        self.DICT[wrong_form] = meaning

    def get_lemma(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][0]

    def get_correct_form(self, wrong_form: str) -> str:
        """Return a corrected form of the given word, attempting
        to emulate the lower/upper/title case of the word"""
        # First, try the original casing of the wrong form
        c = self.DICT.get(wrong_form)
        if c is not None:
            # Found it: we're done
            return c[1]
        # Lookup a lower case version
        c = self.DICT.get(wrong_form.lower())
        if c is None:
            # Not found: can't correct
            return wrong_form
        form = c[1]
        if wrong_form.istitle():
            return form.title()
        if wrong_form.isupper():
            return form.upper()
        return form

    def get_id(self, wrong_form: str) -> int:
        return self.DICT[wrong_form][2]

    def get_category(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][3]

    def get_tag(self, wrong_form: str) -> str:
        return self.DICT[wrong_form][4]


class Morphemes:
    def __init__(self) -> None:
        # dict { morpheme : [ preferred PoS ] }
        self.BOUND_DICT: Dict[str, List[str]] = {}
        # dict { morpheme : [ excluded PoS ] }
        self.FREE_DICT: Dict[str, List[str]] = {}

    def add(self, morph: str, boundlist: List[str], freelist: List[str]) -> None:
        if not boundlist:
            raise ConfigError("A definition of allowed PoS is necessary with morphemes")
        self.BOUND_DICT[morph] = boundlist
        # The freelist may be empty
        self.FREE_DICT[morph] = freelist


class Ritmyndir:
    # dict { wrong_word_form : (lemma, id, cat, correct_word_form, tag, eink, malsnid, stafs, aslatt, beyg) }
    # þurrð;10963;kvk;þurðar;þurrðar;EFET;0;URE;;;;1745-1745;KLIM
    # þurrka;425063;so;þurkaði;þurrkaði;;4;VILLA;R4RR;;;;SKOLAVERK
    def __init__(self) -> None:
        self.DICT: Dict[str, RitmyndirTuple] = dict()

    def contains(self, word: str) -> bool:
        """Check whether the word form is in the Ritmyndir dictionary"""
        return word in self.DICT or word.lower() in self.DICT

    def get_lemma(self, wrong_form: str) -> Optional[str]:
        return self.get_entry(wrong_form, 0)

    def get_id(self, wrong_form: str) -> Optional[int]:
        return self.get_entry(wrong_form, 1)

    def get_cat(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 2)

    def get_correct_form(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 3)

    def get_tag(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 4)

    def get_eink(self, wrong_form: str) -> int:
        eink = self.get_entry(wrong_form, 3)
        if not eink:
            eink = 1
        return eink

    def get_malsnid(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 6).split(",")[0]

    def get_stafs(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 7).split(",")[0]

    def get_aslatt(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 8).split(",")[0]

    def get_beyg(self, wrong_form: str) -> str:
        return self.get_entry(wrong_form, 9).split(",")[0]

    def get_code(self, wrong_form: str) -> str:
        code = self.get_stafs(wrong_form)
        if not code:
            code = self.get_aslatt(wrong_form)
        if not code:
            code = self.get_beyg(wrong_form)
        if not code:
            code = self.get_malsnid(wrong_form)
        if not code:
            code = R_EINKUNN[self.get_eink(wrong_form)]
        if not code:
            code = "R001"
        return code

    def get_entry(self, wrong_form: str, index: int) -> Any:
        entry = self.DICT.get(wrong_form, self.DICT.get(wrong_form.lower()))
        return entry[index] if entry else None

    def add(self, wrong_form: str, details: RitmyndirTuple) -> None:
        # TODO Same ritmynd can occur multiple times in the data from different references, how to handle?
        # TODO Also check if the same ritmynd has many different corrections in the data,
        # so we don't just overwrite former values. The DICT entries can be a list of corrections, using defaultdict()
        self.DICT[wrong_form] = details


class RitmyndirDetails:
    def __init__(self) -> None:
        # "Ritmyndir error code" : ("ritregla/spelling rule", "category", "other detail" ) }
        self.DICT: Dict[str, DetailsTuple] = dict()

    def add(self, code: str, details: DetailsTuple) -> None:
        self.DICT[code] = details


class IecNonwords:
    def __init__(self) -> None:
        # Dictionary structure: dict { wrong_word : (tuple of right words) }
        self.DICT: Dict[str, Tuple[str, ...]] = dict()

    def add(self, word: str, corr: Tuple[str, ...]) -> None:
        if word in self.DICT:
            # Happens in the data, just skip it
            # raise ConfigError(
            #    "Multiple definition of '{0}' in IEC nonwords data".format(word)
            # )
            return
        self.DICT[word] = corr


class Icesquer:
    def __init__(self) -> None:
        # Dictionary structure: dict { wrong_word : (tuple of right words) }
        self.DICT: Dict[str, Tuple[str, ...]] = dict()

    def add(self, word: str, corr: Tuple[str, ...]) -> None:
        if word in self.DICT:
            raise ConfigError(f"Multiple definition of '{word}' in IEC nonwords data")
        self.DICT[word] = corr


class WrongFormers:
    def __init__(self) -> None:
        # Dictionary structure: dict { wrong_word : right_word }
        self.DICT: Dict[str, str] = dict()

    def add(self, word: str, corr: str) -> None:
        if word in self.DICT:
            raise ConfigError(f"Multiple definition of '{word}' in WrongFormers")
        self.DICT[word] = corr


class WrongFormersCID:
    def __init__(self) -> None:
        # Dictionary structure: dict { wrong_word : right_word }
        self.DICT: Dict[str, str] = dict()

    def add(self, word: str, corr: str) -> None:
        if word in self.DICT:
            raise ConfigError(f"Multiple definition of '{word}' in WrongFormersCID")
        self.DICT[word] = corr


class Settings:

    """Global settings"""

    def __init__(self):
        self.allowed_multiples = AllowedMultiples()
        self.wrong_compounds = WrongCompounds()
        self.split_compounds = SplitCompounds()
        self.unique_errors = UniqueErrors()
        self.multiword_errors = MultiwordErrors()
        self.taboo_words = TabooWords()
        self.suggestions = Suggestions()
        self.capitalization_errors = CapitalizationErrors()
        self.ow_forms = OwForms()
        self.cid_error_forms = CIDErrorForms()
        self.cd_error_forms = CDErrorForms()
        self.morphemes = Morphemes()
        self.ritmyndir = Ritmyndir()
        self.ritmyndir_details = RitmyndirDetails()
        self.iec_nonwords = IecNonwords()
        self.icesquer = Icesquer()
        self.tone_of_voice_words = ToneOfVoiceWords()
        self.tone_of_voice_patterns = ToneOfVoicePatterns()
        self.wrong_formers = WrongFormers()
        self.wrong_formers_cid = WrongFormersCID()

    _lock = threading.Lock()
    loaded = False
    DEBUG = os.environ.get("DEBUG", "").strip() in TRUE

    # Configuration settings from the GreynirCorrect.conf file

    def _handle_settings(self, s: str) -> None:
        """Handle config parameters in the settings section"""
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

    def _handle_allowed_multiples(self, s: str) -> None:
        """Handle config parameters in the allowed_multiples section"""
        assert s
        if len(s.split()) != 1:
            raise ConfigError("Only one word per line allowed in allowed_multiples section")
        if s in self.allowed_multiples.SET:
            raise ConfigError("'{0}' is repeated in allowed_multiples section".format(s))
        self.allowed_multiples.add(s)

    def _handle_wrong_compounds(self, s: str) -> None:
        """Handle config parameters in the wrong_compounds section"""
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between compound word and its parts")
        word = a[0].strip().strip('"')
        parts = a[1].strip().strip('"').split()
        if not word:
            raise ConfigError("Expected word before the comma in wrong_compounds section")
        if len(parts) < 2:
            raise ConfigError("Missing word part(s) in wrong_compounds section")
        if len(word.split()) != 1:
            raise ConfigError("Multiple words not allowed before comma in wrong_compounds section")
        self.wrong_compounds.add(word, tuple(parts))

    def _handle_split_compounds(self, s: str) -> None:
        """Handle config parameters in the split_compounds section"""
        parts = s.split()
        if len(parts) != 2:
            raise ConfigError("Missing word part(s) in split_compounds section")
        self.split_compounds.add(parts[0], parts[1])

    def _handle_unique_errors(self, s: str) -> None:
        """Handle config parameters in the unique_errors section"""
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between error word and its correction")
        word = a[0].strip()
        if len(word) < 3:
            raise ConfigError("Expected nonempty word before comma in unique_errors section")
        if word[0] != '"' or word[-1] != '"':
            raise ConfigError("Expected word in double quotes in unique_errors section")
        word = word[1:-1]
        corr = a[1].strip()
        if len(corr) < 3:
            raise ConfigError("Expected nonempty word after comma in unique_errors section")
        if corr[0] != '"' or corr[-1] != '"':
            raise ConfigError("Expected word in double quotes after comma in unique_errors section")
        corr = corr[1:-1]
        corr_t = tuple(corr.split())
        if not word:
            raise ConfigError("Expected word before the comma in unique_errors section")
        if len(word.split()) != 1:
            raise ConfigError("Multiple words not allowed before the comma in unique_errors section")
        self.unique_errors.add(word, corr_t)

    def _handle_capitalization_errors(self, s: str) -> None:
        """Handle config parameters in the capitalization_errors section"""
        self.capitalization_errors.add(s)

    def _handle_taboo_words(self, s: str) -> None:
        """Handle config parameters in the taboo_words section"""
        # Start by parsing explanation string off the end (right hand side), if present
        lquote = s.find('"')
        rquote = s.rfind('"')
        if (lquote >= 0) != (rquote >= 0):
            raise ConfigError("Explanation string for taboo word should be enclosed in double quotes")
        if lquote >= 0:
            # Obtain explanation from within quotes
            explanation = s[lquote + 1 : rquote].strip()
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
            raise ConfigError("Suggested replacement(s) should include a word category (_xx)")
        self.taboo_words.add(taboo, replacement, explanation)

    def _handle_tone_of_voice_words(self, s: str) -> None:
        """Handle config parameters in the tone_of_voice section."""
        # Start by parsing explanation string off the end (right hand side), if present
        lquote = s.find('"')
        rquote = s.rfind('"')
        if (lquote >= 0) != (rquote >= 0):
            raise ConfigError("Explanation string for a word should be enclosed in double quotes")
        if lquote >= 0:
            # Obtain explanation from within quotes
            explanation = s[lquote + 1 : rquote].strip()
            s = s[:lquote].rstrip()
        else:
            # No explanation
            explanation = ""
        if not s:
            raise ConfigError("Expected a word to flag and a suggested replacement")
        a = s.split()  # not lower() here
        if len(a) > 2:
            raise ConfigError("Expected a word to flag and a suggested replacement")
        flagged_word = a[0].strip()
        if len(a) == 2:
            replacement = a[1].strip()
        else:
            replacement = flagged_word
        # Check all replacement words, which are separated by slashes '/'
        if any(r.count("_") != 1 for r in replacement.split("/")):
            raise ConfigError("Suggested replacement(s) should include a word category (_xx)")
        self.tone_of_voice_words.add(flagged_word, replacement, explanation)

    def _handle_tone_of_voice_patterns(self, s: str) -> None:
        """Handle module path for external patterns."""
        # the string includes quotes, let's remove them
        p = s.split("=")[1].strip().strip('"')
        # Should only be a path to a python file
        if not os.path.exists(p):
            raise ConfigError("Not a valid path. Expected a path to a Python file")
        self.tone_of_voice_patterns.add(p)

    def _handle_suggestions(self, s: str) -> None:
        """Handle config parameters in the suggestions section"""
        a = s.lower().split()
        if len(a) < 2:
            raise ConfigError("Expected flagged word and at least one suggested replacement")
        if any(w.count("_") != 1 for w in a[1:]):
            raise ConfigError("Suggested replacements should include word category (_xx)")
        self.suggestions.add(a[0].strip(), [w.strip() for w in a[1:]])

    def _handle_multiword_errors(self, s: str) -> None:
        """Handle config parameters in the multiword_errors section"""
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
        self.multiword_errors.add(phrase, error[1:-1])

    def _handle_ow_forms(self, s: str) -> None:
        """Handle config parameters in the ow_forms section"""
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
        self.ow_forms.add(wrong_form, meaning)

    def _handle_error_forms(self, s: str) -> None:
        """Handle config parameters in the error_forms section"""
        split = s.strip().split(";")
        if len(split) != 7:
            raise ConfigError("Expected wrong form;lemma;correct form;id;category;tag;errortype")
        wrong_form = split[0].strip()
        correct_form = split[2].strip()
        if wrong_form == correct_form:
            raise ConfigError("Wrong form identical to correct form for '{0}'".format(wrong_form))
        meaning: ErrorFormTuple = (
            split[1].strip(),  # Lemma (stofn)
            correct_form,  # Correct form (ordmynd)
            int(split[3]),  # Id (utg)
            split[4].strip(),  # Category (ordfl)
            split[5].strip(),  # Tag (beyging)
        )
        etype = split[6].strip()
        if etype == "cid":
            self.cid_error_forms.add(wrong_form, meaning)  # context-independent errors
        elif etype == "cd":
            self.cd_error_forms.add(wrong_form, meaning)  # context-dependent errors
        else:
            raise ConfigError("Wrong error type given, expected 'cid' or 'cd'")

    def _handle_morphemes(self, s: str) -> None:
        """Process the contents of the [morphemes] section"""
        freelist: List[str] = []
        boundlist: List[str] = []
        spl = s.split()
        if len(spl) < 2:
            raise ConfigError("Expected at least a prefix and an attachment specification")
        m = spl[0]
        for pos in spl[1:]:
            if pos:
                if pos.startswith("+"):
                    boundlist.append(pos[1:])
                elif pos.startswith("-"):
                    freelist.append(pos[1:])
                else:
                    raise ConfigError("Attachment specification should start with '+' or '-'")
        self.morphemes.add(m, boundlist, freelist)

    def _handle_ritmyndir(self, s: str) -> None:
        """Handle data from Ritmyndir in Stórasnið in BÍN/DIM"""
        split = s.strip().split(";")
        if len(split) != 13:
            raise ConfigError(
                "Expected lemma, id, cat, wrong_word_form, correct_word_form, tag, eink, malsnid, stafs, aslatt, beyg, age, ref"
            )
        ref = split[12].strip()
        if "SAGA" in ref or ref in {"MILTON", "HALLGP-4", "KLIM", "ONP"}:
            # Skipping errors from very old references, don't represent errors in Modern Icelandic
            return
        wrong_form = split[3].strip()
        correct_form = split[4].strip()
        if wrong_form == correct_form:
            return
        if wrong_form.lower() == correct_form.lower():
            # TODO Skipping capitalization errors for now
            return
        # (lemma, id, cat, correct_word_form, tag, eink, malsnid, stafs, aslatt, beyg)
        meaning: RitmyndirTuple = (
            split[0].strip(),  # Lemma
            int(split[1].strip()),  # id
            split[2].strip(),  # cat
            correct_form,  # correct_word_form
            split[5].strip(),  # tag
            int(split[6].strip()),  # eink
            split[7].strip(),  # malsnid
            split[8].strip(),  # stafs
            split[9].strip(),  # aslatt
            split[10].strip(),  # beyg
        )
        self.ritmyndir.add(wrong_form, meaning)

    def _handle_ritmyndir_details(self, s: str) -> None:
        """Handle data on Ritmyndir categories, including references to the Icelandic Standards"""
        # "Ritmyndir error code" : ("ritregla/spelling rule", "category", "other detail" )
        split = s.split(":", maxsplit=1)
        code = split[0].strip().strip('"')
        dsplit = split[1].split('", "')
        rule = dsplit[0].strip().strip('"')
        cat = dsplit[1].strip().strip('"')
        det = dsplit[2].strip().strip('"')
        self.ritmyndir_details.DICT[code] = (rule, cat, det)

    def _handle_iec_nonwords(self, s: str) -> None:
        """Handle config parameters in the Icelandic Error Corpus Nonwords"""
        a = s.lower().split("\t")
        if len(a) != 2:
            # Happens in the data, just skip it
            # raise ConfigError("Expected tab between error word and its correction")
            return
        word = a[0].strip()
        if len(word) < 1:
            raise ConfigError("Expected nonempty word before comma in unique_errors section")
        corr = a[1].strip()
        if len(corr) < 1:
            raise ConfigError("Expected nonempty word after comma in unique_errors section")
        corr_t = tuple(corr.split())
        if not word:
            raise ConfigError("Expected word before the comma in unique_errors section")
        if len(word.split()) != 1:
            # Happens in the data, just skip it
            return
            # raise ConfigError(
            #    "Multiple words not allowed before the comma in unique_errors section"
            # )
        self.iec_nonwords.add(word, corr_t)

    def _handle_icesquer(self, s: str) -> None:
        """Handle config parameters in the Icelandic Error Corpus Nonwords"""
        a = s.lower().split("\t")
        if len(a) != 2:
            # Happens in the data, just skip it
            # raise ConfigError("Expected tab between error word and its correction")
            return
        word = a[0].strip()
        if len(word) < 1:
            raise ConfigError("Expected nonempty word before comma in unique_errors section")
        corr = a[1].split(";")[0].strip()  # TODO Only the first value for now
        if len(corr) < 1:
            raise ConfigError("Expected nonempty word after comma in unique_errors section")
        corr_t = tuple(corr.split())
        if not word:
            raise ConfigError("Expected word before the comma in unique_errors section")
        if len(word.split()) != 1:
            # Happens in the data, just skip it
            return
            # raise ConfigError(
            #    "Multiple words not allowed before the comma in unique_errors section"
            # )
        self.icesquer.add(word, corr_t)

    def _handle_wrong_formers(self, s: str) -> None:
        """Handle config parameters in the wrong_formers section"""
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between the word and its correction")
        word = a[0].strip().strip('"')
        correction = a[1].strip().strip('"')
        if not word:
            raise ConfigError("Expected word before the comma in wrong_formers section")
        if not correction:
            raise ConfigError("Expected word after the comma in wrong_formers section")
        if len(word.split()) != 1 or len(correction.split()) != 1:
            raise ConfigError("Expected one word on each side in wrong_formers")
        self.wrong_formers.add(word, correction)

    def _handle_wrong_formers_cid(self, s: str) -> None:
        """Handle config parameters in the wrong_formers section"""
        a = s.lower().split(",", maxsplit=1)
        if len(a) != 2:
            raise ConfigError("Expected comma between the word and its correction")
        word = a[0].strip().strip('"')
        correction = a[1].strip().strip('"')
        if not word:
            raise ConfigError("Expected word before the comma in wrong_formers section")
        if not correction:
            raise ConfigError("Expected word after the comma in wrong_formers section")
        if len(word.split()) != 1 or len(correction.split()) != 1:
            raise ConfigError("Expected one word on each side in wrong_formers")
        self.wrong_formers_cid.add(word, correction)

    def read(self, fname: str, external: bool = False) -> None:
        """Read configuration file"""

        with Settings._lock:
            CONFIG_HANDLERS = {
                "settings": Settings._handle_settings,
                "allowed_multiples": Settings._handle_allowed_multiples,
                "wrong_compounds": Settings._handle_wrong_compounds,
                "split_compounds": Settings._handle_split_compounds,
                "unique_errors": Settings._handle_unique_errors,
                "capitalization_errors": Settings._handle_capitalization_errors,
                "taboo_words": Settings._handle_taboo_words,
                "tone_of_voice_words": Settings._handle_tone_of_voice_words,
                "tone_of_voice_patterns": Settings._handle_tone_of_voice_patterns,
                "suggestions": Settings._handle_suggestions,
                "multiword_errors": Settings._handle_multiword_errors,
                "morphemes": Settings._handle_morphemes,
                "ow_forms": Settings._handle_ow_forms,
                "error_forms": Settings._handle_error_forms,
                "auto_ow": Settings._handle_ow_forms,
                "auto_error": Settings._handle_error_forms,
                "iec_nonwords": Settings._handle_iec_nonwords,
                "icesquer": Settings._handle_icesquer,
                "ritmyndir": Settings._handle_ritmyndir,
                "ritmyndir_details": Settings._handle_ritmyndir_details,
                "wrong_formers": Settings._handle_wrong_formers,
                "wrong_formers_ci": Settings._handle_wrong_formers_cid,
            }
            handler = None  # Current section handler

            rdr = None

            try:
                # If an external path is given, use it to read the file
                package_name = None if external else __name__
                rdr = LineReader(fname, package_name=package_name)
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
                        handler(self, s)
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
