"""

    Greynir: Natural language processing for Icelandic

    Spelling correction module

    Copyright © 2025 Miðeind ehf.

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


    This module uses word frequency information extracted from the
    Greynir (greynir.is) database as a basis for guessing the correct
    spelling of words not found in BÍN and not recognized by the
    compound word algorithm.

"""

from typing import Callable, DefaultDict, Iterable, List, Optional, Set, Tuple

import math
import re
import time
from collections import defaultdict
from functools import lru_cache

from reynir import TOK, correct_spaces, tokenize
from reynir.bindb import GreynirBin, ResultTuple
from reynir.bintokenizer import StringIterable

from .ngrams import MAX_ORDER, Ngrams, load_ngrams
from .settings import Settings


EDIT_0_FACTOR = math.log(1.0 / 1.0)
EDIT_REPLACE_FACTOR = math.log(1.0 / 1.25)
EDIT_S_FACTOR = math.log(1.0 / 8.0)
# Edit distance 1 is 48 times more unlikely than 0
EDIT_1_FACTOR = math.log(1.0 / 48.0)
# Edit distance 2 is considerably times more unlikely than 1
EDIT_2_FACTOR = math.log(1.0 / 2048.0)

# Parameter to use for lambda in 'stupid backoff'
LOG_LAMBDA = math.log(0.4)


@lru_cache(maxsize=2048)
def _splits(word: str) -> Tuple[Tuple[str, str], ...]:
    """Return a list of all possible (first, rest) pairs that comprise word."""
    return tuple((word[:i], word[i:]) for i in range(len(word) + 1))


def levenshtein_distance(s1: str, s2: str) -> int:
    """Return the Levenshtein distance between two strings,
    using the Wagner-Fischer iterative algorithm.

    This function is based on code from https://github.com/toastdriven/pylev:

    Copyright (c) 2012, Daniel Lindsley
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the pylev nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL pylev BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    if s1 == s2:
        return 0

    len_1 = len(s1)
    len_2 = len(s2)

    if len_1 == 0:
        return len_2
    if len_2 == 0:
        return len_1

    if len_1 > len_2:
        s2, s1 = s1, s2
        len_2, len_1 = len_1, len_2

    d0: List[int] = [i for i in range(len_2 + 1)]
    d1: List[int] = [j for j in range(len_2 + 1)]

    cost: int
    x_cost: int
    y_cost: int

    for i in range(len_1):
        d1[0] = i + 1
        for j in range(len_2):
            cost = d0[j]

            if s1[i] != s2[j]:
                # Substitution
                cost += 1

                # Insertion
                x_cost = d1[j] + 1
                if x_cost < cost:
                    cost = x_cost

                # Deletion
                y_cost = d0[j + 1] + 1
                if y_cost < cost:
                    cost = y_cost

            d1[j + 1] = cost

        d0, d1 = d1, d0

    return d0[-1]


class Corrector:

    """A spelling corrector class using a word frequency dictionary"""

    # The characters used to form variants of words by insertion
    _ALPHABET = "aábcdðeéfghiíjklmnoópqrstuúvwxyýzþæö"

    # Translate wrongly accented or encoded characters before correcting
    _TRANSLATE = {
        "à": "á",
        "đ": "ð",
        "è": "é",
        "ì": "í",
        "ò": "ó",
        "ô": "ó",  # Possibly ö
        "ù": "ú",
        "ø": "ö",
    }
    _TRANSLATE_REGEX = "(" + "|".join(_TRANSLATE.keys()) + ")"

    _SUBSTITUTE_LIST = [
        # keyboard distance
        # Note: single character substitutions are already carried
        # out in the edit distance algorithm, so they do not need
        # to be repeated here.
        # ("a", ["q", "w", "s", "z"]),
        # ("s", ["w", "e", "d", "x", "z", "a"]),
        # ("d", ["e", "r", "f", "c", "x", "s"]),
        # ("f", ["r", "t", "g", "v", "c", "d"]),
        # ("g", ["t", "y", "h", "b", "v", "f"]),
        # ("h", ["y", "u", "j", "n", "b", "g"]),
        # ("j", ["u", "i", "k", "m", "n", "h"]),
        # ("k", ["i", "o", "l", "m", "j"]),
        # ("l", ["o", "p", "æ", "k"]),
        # ("æ", ["p", "ð", "þ", "l"]),
        # ("q", ["w", "a"]),
        # ("w", ["e", "s", "a", "q"]),
        # ("e", ["r", "d", "s", "w"]),
        # ("r", ["t", "f", "d", "e"]),
        # ("t", ["y", "g", "f", "r"]),
        # ("y", ["u", "h", "g", "t"]),
        # ("u", ["i", "j", "h", "y"]),
        # ("i", ["o", "k", "j", "u"]),
        # ("o", ["p", "l", "k", "i"]),
        # ("p", ["ö", "ð", "æ", "l", "o"]),
        # ("ð", ["ö", "-", "æ", "p"]),
        # ("z", ["a", "s", "x"]),
        # ("x", ["z", "s", "d", "c"]),
        # ("c", ["x", "d", "f", "v"]),
        # ("v", ["c", "f", "g", "b"]),
        # ("b", ["v", "g", "h", "n"]),
        # ("n", ["b", "h", "j", "m"]),
        # ("m", ["n", "j", "k"]),
        # ("þ", ["æ"]),
        # n/nk
        ("áng", ["ang"]),
        ("eing", ["eng"]),
        ("eyng", ["eng"]),
        ("úng", ["ung"]),
        ("íng", ["yng", "ing"]),
        ("ýng", ["yng", "ing"]),
        ("aung", ["öng"]),
        ("ánk", ["ank"]),
        ("eink", ["enk"]),
        ("eynk", ["enk"]),
        ("únk", ["unk"]),
        ("ínk", ["ynk", "ink"]),
        ("ýnk", ["ynk", "ink"]),
        ("aunk", ["önk"]),
        # sníkjuhljóð
        ("dl", ["ll", "rl"]),
        ("dn", ["nn", "rn"]),
        ("rdl", ["rl"]),
        ("rdn", ["rn"]),
        ("sdl", ["sl"]),
        ("sdn", ["sn"]),
        # /j/ø
        ("ýa", ["ýja"]),
        ("ýu", ["ýu"]),
        ("æu", ["æju"]),
        ("ji", ["i", "gi"]),
        ("j", ["gj"]),
        ("ægi", ["agi"]),
        ("eigi", ["egi"]),
        ("eygi", ["egi"]),
        ("ígi", ["igi"]),
        ("ýgi", ["igi"]),
        ("oji", ["ogi"]),
        ("uji", ["ugi"]),
        ("yji", ["ygi"]),
        ("augi", ["ögi"]),
        # ,/ø, f/ø í innstöðu
        ("á", ["ág", "áf"]),
        ("í", ["íg"]),
        ("æ", ["æg", "ei"]),  # áræðanlegur->áreiðanlegur
        ("ú", ["úg", "úf"]),
        ("ó", ["óg", "óf"]),
        # einfaldir/tvöfaldir samhljóðar
        ("g", ["gg"]),
        ("gg", ["g"]),
        ("k", ["kk"]),
        ("kk", ["k"]),
        ("l", ["ll"]),
        ("ll", ["l"]),
        ("m", ["mm"]),
        ("mm", ["m"]),
        ("n", ["nn"]),
        ("nn", ["n"]),
        ("p", ["pp"]),
        ("pp", ["p"]),
        ("r", ["rr"]),
        ("rr", ["r"]),
        ("s", ["ss"]),
        ("ss", ["s"]),
        ("t", ["tt"]),
        ("tt", ["t"]),
        ("gð", ["ggð"]),
        ("ggð", ["gð"]),
        ("gt", ["ggt"]),
        ("ggt", ["gt"]),
        ("gl", ["ggl"]),
        ("ggl", ["gl"]),
        ("gn", ["ggn"]),
        ("ggn", ["gn"]),
        ("kn", ["kkn"]),
        ("kkn", ["kn"]),
        ("kl", ["kkl"]),
        ("kkl", ["kl"]),
        ("kt", ["kkt"]),
        ("kkt", ["kt"]),
        ("pl", ["ppl"]),
        ("ppl", ["pl"]),
        ("pn", ["ppn"]),
        ("ppn", ["pn"]),
        ("pt", ["ppt"]),
        ("ppt", ["pt"]),
        ("tl", ["ttl"]),
        ("ttl", ["tl"]),
        ("tn", ["ttn"]),
        ("ttn", ["tn"]),
        # sérhljóðar
        # ("a", ["á"]),
        # ("e", ["é"]),
        ("ei", ["ey"]),
        ("ey", ["ei"]),
        # ("i", ["í", "y"]),
        # ("o", ["ó", "ö"]),
        # ("u", ["ú"]),
        # ("y", ["i", "ý"]),
        ("je", ["é"]),
        ("æ", ["aí"]),  # Tæland → Taíland
        # zeta og tengdir samhljóðaklasar
        ("z", ["ds", "ðs", "ts"]),  # "s"
        ("zt", ["st"]),
        ("zl", ["sl"]),
        ("nzk", ["nsk"]),
        ("tzt", ["st"]),
        ("ttzt", ["st"]),
        # einföldun, samhljóðaklasar
        ("md", ["fnd"]),
        ("mt", ["fnd"]),
        ("bl", ["fl"]),
        ("bbl", ["fl"]),
        ("bn", ["fn"]),
        ("bbn", ["fn"]),
        ("lgd", ["gld"]),
        ("gld", ["lgd"]),
        ("lgt", ["glt"]),
        ("glt", ["lgt"]),
        ("ngd", ["gnd"]),
        ("gnd", ["ngd"]),
        ("ngt", ["gnt"]),
        ("gnt", ["ngt"]),
        ("lfd", ["fld"]),
        ("fld", ["lfd"]),
        ("lft", ["flt"]),
        ("flt", ["lft"]),
        ("sn", ["stn"]),
        ("rn", ["rfn"]),
        ("rð", ["rgð"]),
        ("rgð", ["rð"]),
        ("ft", ["pt", "ppt"]),
        ("pt", ["ft"]),
        ("ppt", ["ft"]),
        ("nd", ["rnd"]),
        ("st", ["rst"]),
        ("ksk", ["sk"]),
        # annað
        ("kv", ["hv"]),
        ("hv", ["kv"]),
        ("gs", ["x"]),
        ("ks", ["x"]),
        ("x", ["gs", "ks"]),
        # ("v", ["f"]),
        # ("b", ["p"]),
        # ("g", ["k"]),
        # ("d", ["t"]),
        # erlend lyklaborð
        ("ae", ["æ"]),
        # ("t", ["þ"]),
        ("th", ["þ"]),
        # ("d", ["ð"]),
        # ljóslestur
        # ("c", ["æ", "é"]),
        # beygingarendingar
        ("ananna", ["anna", "ana"]),
        ("ana", ["anna"]),
        ("anna", ["ana"]),
    ]

    _SUBSTITUTE: DefaultDict[str, Set[str]] = defaultdict(set)

    for _key, _subs in _SUBSTITUTE_LIST:
        _SUBSTITUTE[_key].update(_subs)

    # Sort the substitution keys in descending order by length
    _SUBSTITUTE_KEYS = sorted(_SUBSTITUTE.keys(), key=lambda x: len(x), reverse=True)
    # Create a regex to extract word fragments ending with substitution keys
    _SUBSTITUTE_REGEX = re.compile("(.*?(" + "|".join(_SUBSTITUTE_KEYS) + "))")

    # The probability thresholds used by the corrector are expressed
    # as occurrence counts in the trigram model, so that they mean the
    # same thing regardless of the size of the corpus behind the model.
    # They are converted to log probabilities in __init__(), using the
    # total unigram count of the loaded model. The counts below reproduce
    # the log probability thresholds that were used with the 2019 model
    # (about 0.8 billion tokens): -16.5 corresponded to about 55
    # occurrences and -12.0 to about 4,900 occurrences.

    # Minimum number of occurrences (in context, after edit-distance and
    # backoff penalties) of a candidate other than the original word
    # in order for it to be returned, when the original word is known
    _MIN_CANDIDATE_COUNT = 55
    # If a unigram occurs at least this often,
    # just assume it's OK without further checking
    _UNIGRAM_ACCEPT_COUNT = 4900
    # Words that occur this often or less are considered rare
    _RARE_MAX_COUNT = 55
    # Minimum number of occurrences, per billion tokens in the model, for a
    # word that is not in BÍN to be considered a "known" word. The 2019
    # model kept no words with fewer than 3 occurrences, so any word in it
    # was known; the 2026 model is larger and keeps words that occur twice.
    _KNOWN_WORD_MIN_PER_BILLION = 4.0

    # The log probability thresholds and the known-word frequency are
    # set per instance in __init__(), derived from the counts above
    _MIN_LOG_PROBABILITY: float
    _UNIGRAM_ACCEPT_THRESHOLD: float
    _RARE_THRESHOLD: float
    _RARE_THRESHOLD_UPPERCASE: float
    _KNOWN_WORD_MIN_FREQUENCY: int

    # Singleton Ngrams dictionary
    _NGRAMS: Optional[Ngrams] = None

    def __init__(self, db: GreynirBin, dictionary: Optional[Ngrams] = None) -> None:
        # Word database
        self._db = db
        # N-gram frequency dictionary
        if dictionary is not None:
            self.ngrams = dictionary
        else:
            if self._NGRAMS is None:
                self.__class__._NGRAMS = load_ngrams()
            assert self._NGRAMS is not None
            self.ngrams = self._NGRAMS
        # Function for log probability of word
        self.logprob = self.ngrams.logprob
        # Function for (adjusted) frequency of word
        self.freq = self.ngrams.adj_freq
        # Calibrate the thresholds to the size of the loaded model.
        # The unigram log probability is log(count + 1) - log(total),
        # and the adjusted frequency is count + 1, so their difference
        # is the log of the total unigram count of the model.
        self._log_total = math.log(self.freq("og")) - self.logprob("og")
        self._MIN_LOG_PROBABILITY = math.log(self._MIN_CANDIDATE_COUNT) - self._log_total
        self._UNIGRAM_ACCEPT_THRESHOLD = math.log(self._UNIGRAM_ACCEPT_COUNT) - self._log_total
        self._RARE_THRESHOLD = math.log(self._RARE_MAX_COUNT) - self._log_total
        # For uppercase words, the rarity threshold is even lower,
        # or half the lowercase one
        self._RARE_THRESHOLD_UPPERCASE = self._RARE_THRESHOLD + math.log(0.5)
        # The frequency is compared against the adjusted frequency (count + 1)
        billions = math.exp(self._log_total) / 1.0e9
        self._KNOWN_WORD_MIN_FREQUENCY = max(3, round(self._KNOWN_WORD_MIN_PER_BILLION * billions))

    def is_known(self, word: str) -> bool:
        """Return True if the word occurs often enough in the trigram
        model to be considered a known word, regardless of BÍN"""
        return self.freq(word) >= self._KNOWN_WORD_MIN_FREQUENCY

    @staticmethod
    def _use_title_case(original_word: str, at_sentence_start: bool) -> bool:
        """Return True if probability queries for this word should also
        try the title case form. This applies to a word that was originally
        in title case, such as 'Ísland', and also to a lower case word at
        a sentence start, since it is then probably in the wrong case and
        should be subject to correction as such."""
        return original_word.istitle() or at_sentence_start

    def _backoff_logprob(self, w: str, context: Tuple[str, ...], use_title: bool) -> float:
        """Return the log probability of the word w in the given context,
        using 'stupid backoff' to shorter contexts when the n-gram
        does not occur in the model"""

        # !!! TODO: We may need a more sophisticated probability function
        # !!! TODO: here, such as Kneser-Ney or Katz

        def logprob_title(*args: str) -> float:
            """Return the log probability of an n-gram as a maximum of
            the log probability of the lower case n-gram and the title case
            n-gram, respectively"""
            ctx, w = args[:-1], args[-1]
            return max(self.logprob(*ctx, w), self.logprob(*ctx, w.title()))

        def freq_title(*args: str) -> int:
            """Return the frequency of an n-gram as a maximum of
            the frequency of the lower case n-gram and the title case
            n-gram, respectively"""
            ctx, w = args[:-1], args[-1]
            return max(self.freq(*ctx, w), self.freq(*ctx, w.title()))

        if use_title:
            logprob = logprob_title
            freq = freq_title
        else:
            # Shortcut to the simple and common query functions
            logprob = self.logprob
            freq = self.freq

        ctx = context
        lamb = 0.0
        while True:
            if not ctx:
                # No context: simply return the logprob of the unigram,
                # multiplied with the current lambda (backoff) factor
                return logprob(w) + lamb
            # !!! TODO: Optimize the following
            cw = ctx + (w,)
            fq = freq(*cw)
            if fq > 1:
                # We have a meaningful frequency here:
                # return the logprob multiplied with the current lambda
                if Settings.DEBUG:
                    print(
                        "stupid_backoff() returning logprob of '{0}' "
                        "which is {1:.3} + {2:.3} = {3:.3}".format(cw, logprob(*cw), lamb, logprob(*cw) + lamb)
                    )
                return logprob(*cw) + lamb
            # Insignificant frequency: back off to a simpler context
            # and use the 'stupid backoff' to reduce the probability
            ctx = ctx[1:]
            # Multiply the prob by 0.4, i.e. add log(0.4) to the logprob
            lamb += LOG_LAMBDA

    def _reject_best(self, best: Tuple[str, float], original_word: str, word: str) -> bool:
        """Return True if the best candidate is not good enough to be
        offered as a replacement for the original word"""
        if self.is_known(word) or self.is_known(original_word):
            # The original word is known to the model, so it may well
            # be correct: only replace it with a candidate that is
            # likely enough in its own right
            return best[1] < self._MIN_LOG_PROBABILITY
        # The original word is unknown to the model: the candidates are
        # all known words, so the best one is accepted even if it is
        # not likely in its own right (such as a rare compound)
        return False

    @property
    def db(self) -> GreynirBin:
        """Return the associated word database"""
        return self._db

    def lookup_word(self, word: str, *, at_sentence_start: bool = False, auto_uppercase: bool = False) -> ResultTuple:
        """Look up the given word in the associated word database"""
        return self._db.lookup_g(word, at_sentence_start, auto_uppercase)

    def subs(self, word: str) -> Iterable[str]:
        """Return all combinations of potential substitutions into the word."""
        # The following yields a list of tuples, for instance
        # [('gl', 'gl'), ('er', 'r'), ('aug', 'g')] for the word "gleraugu"
        fragments: List[Tuple[str, str]] = re.findall(self._SUBSTITUTE_REGEX, word)
        end = 0
        # num_combs is the total number of potential combinations
        num_combs = 1
        # combs is a list of possibilities for each combination slot
        combs: List[List[str]] = []
        # Enumerate through the combination slots
        for frag, sub in fragments:
            end += len(frag)
            if len(frag) > len(sub):
                # The fragment has a constant (fixed) part in front of
                # the combination slot
                combs.append([frag[0 : -len(sub)]])
            # Collect all combinations for this slot
            subs = [sub] + list(self._SUBSTITUTE[sub])
            combs.append(subs)
            # Keep tab of the total number of combinations so far
            num_combs *= len(subs)
        # The word may end with a constant (fixed) suffix
        suffix = word[end:]
        if suffix:
            combs.append([suffix])
        # Prepare the result list, from which we will create result strings
        result = [c[0] for c in combs]
        # Prepare the combinations that we'll be selecting from at each slot
        z = [(c, len(c)) for c in combs]
        # Generate all the combinations, numbered from zero
        for counter in range(num_combs):
            numerator = counter
            for i, (c, d) in enumerate(z):
                # d is the divisor, i.e. the number of combinations for this slot
                if d > 1:
                    numerator, ix = divmod(numerator, d)
                    # Assign the selected combination to the result
                    result[i] = c[ix]
            # assert numerator == 0
            # print(result)
            yield "".join(result)

    def _correct(
        self,
        original_word: str,
        word: str,
        context: Tuple[str, ...],
        at_sentence_start: bool,
    ) -> str:
        """Return best candidate or the original word if none are found"""
        candidates = self.gen_candidates(original_word, word, context, at_sentence_start)
        if not candidates:
            # No good candidates
            return word
        # Find the candidate with the highest probability
        m = max(candidates, key=lambda t: t[1])
        if self._reject_best(m, original_word, word):
            # Best candidate is very unlikely: return the original word
            # print(f"Best candidate {m[0]} is highly unlikely, returning original {word}")
            return word
        # Return the most likely word
        return m[0]

    def gen_candidates(
        self,
        original_word: str,
        word: str,
        context: Tuple[str, ...],
        at_sentence_start: bool,
    ) -> List[Tuple[str, float]]:
        """Find the best candidates for spelling correction for this word.
        Credits for parts of this elegant code are due to Peter Norvig,
        cf. http://nbviewer.jupyter.org/url/norvig.com/ipython/
        How%20to%20Do%20Things%20with%20Words.ipynb"""

        # Note: word is assumed to be in lowercase, while
        # original_word has the original case from the source text

        alphabet = self._ALPHABET

        def in_dictionary(w: str) -> bool:
            """Consider a word to be in-dictionary if it occurs in
            BÍN (potentially also in title case) or
            frequently enough in the trigrams database"""
            if w in self._db or self.is_known(w):
                return True
            wt = w.title()
            return False if wt == w else (wt in self._db or self.is_known(wt))

        def known(words: Iterable[str]) -> Iterable[str]:
            """Return a generator of words that are actually in the dictionary."""
            # A word is known if its lower case form is in the dictionary or
            # if its title form is in the dictionary (for example 'Ísland')
            return (w for w in words if in_dictionary(w))

        def edits0(word: str) -> Set[str]:
            """Return all strings that are zero edits away from word (i.e., just word itself)."""
            return {word}

        def edits1(pairs: Iterable[Tuple[str, str]]) -> Set[str]:
            """Return all strings that are one edit away from this word."""
            # Deletes
            result = {a + b[1:] for (a, b) in pairs if b}
            # Transposes
            result |= {a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) >= 2}
            # Replaces
            result |= {a + c + b[1:] for (a, b) in pairs for c in alphabet if b}
            # Inserts
            result |= {a + c + b for (a, b) in pairs for c in alphabet}
            return result

        # def edits2(pairs: Iterable[Tuple[str, str]]) -> Set[str]:
        #     """ Return all strings that are two edits away from this word. """
        #
        #     def sub_edits1(word: str) -> Set[str]:
        #         pairs = _splits(word)
        #         return edits1(pairs)
        #
        #     return {e2 for e1 in edits1(pairs) for e2 in sub_edits1(e1)}

        def _gen_candidates(original_word: str, word: str) -> Iterable[Tuple[str, float]]:
            """Generate candidates in order of generally decreasing likelihood"""

            use_title = self._use_title_case(original_word, at_sentence_start)

            def P(w: str) -> float:
                return self._backoff_logprob(w, context, use_title)

            e0 = edits0(word)  # | edits0(original_word)
            for c in known(e0):
                yield (c, P(c) + EDIT_0_FACTOR)
            for c in known(self.subs(word)):
                yield (c, P(c) + EDIT_S_FACTOR)
            pairs = _splits(word)
            e1 = edits1(pairs) - e0
            for c in known(e1):
                yield (c, P(c) + EDIT_1_FACTOR)
            # The following edit distance=2 stuff is currently too expensive
            # in terms of processor time and memory
            # e2 = edits2(pairs) - e1 - e0
            # for c in known(e2):
            #     yield (c, P(c) + EDIT_2_FACTOR)

        # First, if the word itself is common enough as a unigram,
        # we don't bother checking it further and just assume it's fine
        log_prob = self.logprob(word)
        if Settings.DEBUG:
            print(
                "Ctx {0}, word '{1}' has logprob {2:.3f}, threshold is {3:.3f}".format(
                    context, original_word, log_prob, self._UNIGRAM_ACCEPT_THRESHOLD
                )
            )
        if log_prob > self._UNIGRAM_ACCEPT_THRESHOLD:
            # print(f"The original word {word} is above the threshold, returning it")
            return []
        # Otherwise, generate replacement candidates
        return list(_gen_candidates(original_word, word))

    def _best_list(
        self,
        original_word: str,
        word: str,
        context: Tuple[str, ...],
        at_sentence_start: bool,
    ) -> List[Tuple[str, float]]:
        """Remove unlikely candidates from list"""
        candidates = self.gen_candidates(original_word, word, context, at_sentence_start)
        if not candidates:
            # No candidates beside the word itself: return an empty list
            # print(f"Candidate {word} is only candidate, returned list is empty")
            return []
        # Return the highest probability candidate
        if Settings.DEBUG:
            for i, (c, log_prob) in enumerate(sorted(candidates, key=lambda t: t[1], reverse=True)[0:5]):
                print("Candidate {0} for {1} is {2} with log_prob {3:.3f}".format(i + 1, word, c, log_prob))
        m = max(candidates, key=lambda t: t[1])
        if self._reject_best(m, original_word, word):
            # Best candidate is very unlikely: return an empty list of suggestions
            # print(f"Best candidate {m[0]} is highly unlikely, returning an empty list")
            return []
        candsort = sorted(candidates, key=lambda t: t[1], reverse=True)[0:5]
        return candsort

    @staticmethod
    def _case_of(text: str) -> Callable[[str], str]:
        """Return the case-function appropriate for text: upper, lower, title, or just str."""
        if text.isupper():
            return str.upper
        if text[0].isupper():
            # We don't use .istitle() and .title() because
            # they consider apostrophes to be word separators
            return lambda s: s[0].upper() + s[1:]
        return str  # noqa

    def _cast(self, word: str) -> str:
        """Cast the word to lowercase and correct accents"""
        return re.sub(
            self._TRANSLATE_REGEX,
            lambda match: self._TRANSLATE[match.group()],
            word.lower(),
        )

    def is_rare(self, word: str, *, sentence_is_uppercase: bool = False) -> bool:
        """Return True if the word is so rare as to be suspicious"""
        wl = word.lower()
        if wl != word:
            # The word is at least partially in uppercase in the text
            if self.logprob(word) >= self._RARE_THRESHOLD_UPPERCASE:
                # The upper case version is not rare
                return False
            if word in self.db:
                # The upper case version is in BÍN: don't consider it rare
                return False
            if (not sentence_is_uppercase) and word.isupper():
                # All-uppercase words in an otherwise not uppercase
                # sentence are probably acronyms, which we don't consider rare
                return False
        # Return True if the lower case version is rare
        return self.logprob(wl) < self._RARE_THRESHOLD

    def correct(self, word: str, *, context: Tuple[str, ...] = (), at_sentence_start: bool = False) -> str:
        """Correct a single word, keeping its case (lower/upper/title) intact.
        The optional context parameter contains a tuple of preceding
        words, used to enable a more accurate probability prediction."""
        return self._case_of(word)(self._correct(word, self._cast(word), context, at_sentence_start))

    def suggest_list(
        self, word: str, *, context: Tuple[str, ...] = (), at_sentence_start: bool = False
    ) -> List[Tuple[str, float]]:
        """Return a list of suggestions for a single word, keeping its case
        (lower/upper/title) intact. The optional context parameter contains
        a tuple of preceding words, used to enable a more accurate probability
        prediction."""
        return list(
            (self._case_of(word)(cased_cand[0]), cased_cand[1])
            for cased_cand in self._best_list(word, self._cast(word), context, at_sentence_start)
        )

    def __getitem__(self, word: str) -> str:
        """For the fun of it, support corrector["myword"] syntax"""
        return self.correct(word)

    def __contains__(self, word: str) -> bool:
        """Support "word" in corrector"""
        return self._db.__contains__(word)

    # pylint: disable=used-before-assignment
    def correct_text(self, text: StringIterable, *, only_rare: bool = False) -> str:
        """Attempt to correct all words within a text, returning the corrected text.
        If only_rare is True, correction is only attempted on rare words."""
        result: List[str] = []
        look_back = -MAX_ORDER + 1
        for token in tokenize(text):
            if token.kind == TOK.WORD:
                if only_rare and not self.is_rare(token.txt):
                    # The word is not rare, so we don't attempt correction
                    result.append(token.txt)
                else:
                    # Correct the word and return the result
                    result.append(self.correct(token.txt, context=tuple(result[look_back:])))
            elif token.txt:
                result.append(token.txt)
            elif token.kind in {TOK.S_BEGIN, TOK.S_END}:
                result.append("")
        return correct_spaces(" ".join(result))


def test() -> None:
    with GreynirBin.get_db() as db:
        c = Corrector(db)

        txts = [
            """
        FF er flokkur með rasisku ívafi og tilhneygjingu til að einkavinavæða alla fjölmiðla
        Íslands og færa þar með elítunni að geta ein haft áhrif á skoðanamyndandi áhri í
        fjölmiðlaheiminum, er ekki viðbúið að svona flokkur gamgi til samstarf við íhaldið
        eftir kosningar en ekki þessa vondu félagshyggjuflokka
            """,
            """
        fæ alveg hræðileg drauma vegna fyrri áfalla og það hjálpar mér að ná góðum svef og þar með
        betri andlegri lýðan og líka til að auka matarlist. Tek samt skýrt fram að ég hef bæði
        missnotað kannabis og ekki. Hef engan áhuga á að vera undir áhrifum kannabis alla dag.
        Mikil munur á að nota og missnota !
            """,
            """
        Bæði , lyf gegn áfengissyki (leiða) , mér hefur ekki leiðst mikið seinustu 30 ár. Gegn
        Taugaveiklun, konan hamrar á mér alla daga , skærur hennar eru langar og strangar. En ef ég fæ
        eina pípu og gríp gitarinn má hún tuða í mér klukkutímum saman.Ég er bæði rólegur og læri hratt
        á gítarinn, eftir 10 ára hjónaband er ég bara ótrúlega heill og stefni hátt. Ég og gitarinn erum
        orðnir samvaxnir. Auðvitað stefnum við á skilnað og þá mun ég sakna skalaæfinganna.
            """,
            """
        biddu nu hæg - var Kvennalistinn eins malefnis hreyfing. Hvað attu við - ef þu telur malefnið
        hafa verið eitt hvert var það? Kannski leikskola fyrir öll börn? Sömu laun fyrir sömu störf?
        Að borgarskipulag tæki mið af þörfum beggja kynja? Að kynjagleraugu væru notuð við gerð
        fjarlaga? Að þjoðfelagið opnaði augun fyrir kynferðsofbeldinu og sifjaspellum? (hvorutveggja
        sagt aðeins viðgangast i utlöndum). Þetta eru aðeins örfa dæmi um malefni sem brunnu a okkur
        og við börðumst fyrir. Ekki ertu i alvöru að tala framlag okkur niður. Tæplega
        telurðu það EITT malefni þo að i grunninn hafi baratta okkar sem stoðum að Kvennaframboðinu
        og -listanum gengið ut a að ,,betri,, helmingur þjoðarinnar öðlast - ekki bara i orði heldur
        einnig a borði - sömu rettindi og raðandi helmingurinn
            """,
            """
        Salvör ekki standa i að reyna að klora yfir mistök þin. Reynsluheimur kvenna visar að sjalsögðu
        til þess að helmingur mannkynsins - -konur - er olikur hinum helmingnum bæði sökum lffræðilegs munar og
        þess að þær eru gerðar að konum (sb de Beauvoir) þe fra frumbernsku er drengjum hrosað fyrir annað en
        stulkum og væntingar foreldra eru aðrar til dætra en sona og auk þess er ætlast til að dætur læri af mæðrum en synir af
        feðrum. Það er þetta sem gerir konur - helming mannkynsins - frabrugðna körlum sem hafa fra örofi alda verið
        ,,raðandi,, kynið. Það var gegn þvi orettlæti að reynsluheimur kvenna speglaðist ekki i politiskum akvörðunum sem við
        sem stofnaði Kvennafranboðið og - listann börðumst gegn - a öllum vigstöðvum. Að skilgreina barattu okkar
        Kvennalistans - fyrir rettindum halfrar þjoðarinnar til að skapa ,,rettlatara samfelag,, - sem eins mals flokk er
        fjarstæða.
            """,
        ]

        def linebreak(txt: str, margin: int = 80, left_margin: int = 0) -> str:
            """Return a nicely column-formatted string representation of the given text,
            where each line is not longer than the given margin (if possible).
            A left margin can be optionally added, as a sequence of spaces.
            The lines are joined by newlines ('\n') but there is no trailing
            newline."""
            result: List[str] = []
            line: List[str] = []
            len_line = 0
            for wrd in txt.split():
                if len_line + 1 + len(wrd) > margin:
                    result.append(" ".join(line))
                    line = []
                    len_line = 0
                line.append(wrd)
                len_line += 1 + len(wrd)
            if line:
                result.append(" ".join(line))
            return "\n".join(" " * left_margin + line for line in result)

        t0 = time.time()

        for t in txts:
            print("\nOriginal:\n")
            print(linebreak(t, left_margin=8))
            print("\nCorrected:\n")
            print(linebreak(c.correct_text(t), left_margin=8))

        t1 = time.time()
        print("\nTotal time: {0:.2f} seconds".format(t1 - t0))

        # Test cases to check:
        # sjalsögðu -> sjálfsögðu
        # olikur -> blikur
        # sb -> b (should probably be sbr.)
        # þe -> þá (should probably be þ.e.)
        # sona -> svona
        # orettlæti -> orettlæti
        # politiskum -> politiskum
        # a -> að (should probably be á)
        # rettlatara -> rettlatara
        # hæg -> hægt
        # biddu -> biddu
        # þjoðfelagið -> þjoðfelagið
        # ertu i alvoru -> ertu alvöru
        # þo -> þá (should probably be þó)
        # FF -> FÉ
        # rasisku -> fasisku
        # íhaldið -> haldið
        # áfalla -> falla
        # mikil munur -> mikill munur


if __name__ == "__main__":
    test()
