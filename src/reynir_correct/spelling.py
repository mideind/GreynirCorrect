"""

    Reynir: Natural language processing for Icelandic

    Spelling correction module

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


    This module uses word frequency information extracted from the
    Reynir (greynir.is) database as a basis for guessing the correct
    spelling of words not found in BÍN and not recognized by the
    compound word algorithm.

"""

import os
import math
import re
import time

from collections import defaultdict
from functools import lru_cache

from reynir import tokenize, correct_spaces, TOK
from reynir.bindb import BIN_Db

from icegrams import Ngrams, MAX_ORDER


_PATH = os.path.dirname(__file__) or "."

EDIT_0_FACTOR = math.log(1.0 / 1.0)
EDIT_REPLACE_FACTOR = math.log(1.0 / 1.25)
EDIT_S_FACTOR = math.log(1.0 / 8.0)
# Edit distance 1 is 48 times more unlikely than 0
EDIT_1_FACTOR = math.log(1.0 / 48.0)
# Edit distance 2 is considerably times more unlikely than 1
EDIT_2_FACTOR = math.log(1.0 / 2048.0)


@lru_cache(maxsize=2048)
def _splits(word):
    """ Return a list of all possible (first, rest) pairs that comprise word. """
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


class Corrector:

    """ A spelling corrector class using a word frequency dictionary """

    # The characters used to form variants of words by insertion
    _ALPHABET = "aábcdðeéfghiíjklmnoópqrstuúvwxyýzþæö"

    # Translate wrongly accented characters before correcting
    _TRANSLATE = {
        "à": "á",
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
        ("æ", ["æg", "ei"]), # áræðanlegur->áreiðanlegur
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
    ]

    _SUBSTITUTE = defaultdict(set)

    for _key, _subs in _SUBSTITUTE_LIST:
        _SUBSTITUTE[_key].update(_subs)

    # Sort the substitution keys in descending order by length
    _SUBSTITUTE_KEYS = sorted(_SUBSTITUTE.keys(), key=lambda x: len(x), reverse=True)
    # Create a regex to extract word fragments ending with substitution keys
    _SUBSTITUTE_REGEX = re.compile("(.*?(" + "|".join(_SUBSTITUTE_KEYS) + "))")

    # Minimum probability of a candidate other than the original
    # word in order for it to be returned
    _MIN_LOG_PROBABILITY = -16.5
    # If a unigram is independently above this threshold,
    # just assume it's OK without further checking
    _UNIGRAM_ACCEPT_THRESHOLD = -12.0
    _RARE_THRESHOLD = -16.5  # Approx frequency 8 in the trigrams database
    # For uppercase words, the rarity threshold is even lower,
    # or half the lowercase one
    _RARE_THRESHOLD_UPPERCASE = _RARE_THRESHOLD + math.log(0.5)
    # Minimum frequency in trigrams database to be considered a "known" word
    _KNOWN_WORD_MIN_FREQUENCY = 3

    def __init__(self, db, dictionary=None):
        # Word database
        self._db = db
        # N-gram frequency dictionary
        self.ngrams = dictionary or Ngrams()
        # Function for log probability of word
        self.logprob = self.ngrams.logprob
        # Function for (adjusted) frequency of word
        self.freq = self.ngrams.adj_freq

    @property
    def db(self):
        """ Return the associated word database """
        return self._db

    def subs(self, word):
        """ Return all combinations of potential substitutions into the word. """
        # The following yields a list of tuples, for instance
        # [('gl', 'gl'), ('er', 'r'), ('aug', 'g')] for the word "gleraugu"
        fragments = re.findall(self._SUBSTITUTE_REGEX, word)
        end = 0
        # num_combs is the total number of potential combinations
        num_combs = 1
        # combs is a list of possibilities for each combination slot
        combs = []
        # Enumerate through the combination slots
        for frag, sub in fragments:
            end += len(frag)
            if len(frag) > len(sub):
                # The fragment has a constant (fixed) part in front of
                # the combination slot
                combs.append([frag[0:-len(sub)]])
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

    def _correct(self, original_word, word, context, at_sentence_start):
        """ Find the best spelling correction for this word.
            Credits for parts of this elegant code are due to Peter Norvig,
            cf. http://nbviewer.jupyter.org/url/norvig.com/ipython/
            How%20to%20Do%20Things%20with%20Words.ipynb """

        # Note: word is assumed to be in lowercase, while
        # original_word has the original case from the source text

        alphabet = self._ALPHABET

        def in_dictionary(w):
            """ Consider a word to be in-dictionary if it occurs in
                BÍN (potentially also in title case) or
                frequently enough in the trigrams database """
            if w in self._db or self.freq(w) >= self._KNOWN_WORD_MIN_FREQUENCY:
                return True
            wt = w.title()
            return False if wt == w else (
                wt in self._db or self.freq(wt) >= self._KNOWN_WORD_MIN_FREQUENCY
            )

        def known(words):
            """ Return a generator of words that are actually in the dictionary. """
            # A word is known if its lower case form is in the dictionary or
            # if its title form is in the dictionary (for example 'Ísland')
            return (w for w in words if in_dictionary(w))

        def edits0(word):
            """ Return all strings that are zero edits away from word (i.e., just word itself). """
            return {word}

        def edits1(pairs):
            """ Return all strings that are one edit away from this word. """
            # Deletes
            result = {a + b[1:]                 for (a, b) in pairs if b}
            # Transposes
            result |= {a + b[1] + b[0] + b[2:]  for (a, b) in pairs if len(b) > 1}
            # Replaces
            result |= {a + c + b[1:]            for (a, b) in pairs for c in alphabet if b}
            # Inserts
            result |= {a + c + b                for (a, b) in pairs for c in alphabet}
            return result

        def edits2(pairs):
            """ Return all strings that are two edits away from this word. """

            def sub_edits1(word):
                pairs = _splits(word)
                return edits1(pairs)

            return {e2 for e1 in edits1(pairs) for e2 in sub_edits1(e1)}

        def gen_candidates(original_word, word):
            """ Generate candidates in order of generally decreasing likelihood """

            def logprob_title(*args):
                """ Return the log probability of an n-gram as a maximum of
                    the log probability of the lower case n-gram and the title case
                    n-gram, respectively """
                # The followin is written to be compatible with Python 3.4
                lp1 = self.logprob(*args)
                args = args[:-1] + (args[-1].title(),)
                lp2 = self.logprob(*args)
                return max(lp1, lp2)
                # On Python >= 3.5, the following works:
                # ctx, w = args[:-1], args[-1]
                # return max(self.logprob(*ctx, w), self.logprob(*ctx, w.title()))

            def freq_title(*args):
                """ Return the frequency of an n-gram as a maximum of
                    the frequency of the lower case n-gram and the title case
                    n-gram, respectively """
                # The followin is written to be compatible with Python 3.4
                lp1 = self.freq(*args)
                args = args[:-1] + (args[-1].title(),)
                lp2 = self.freq(*args)
                return max(lp1, lp2)
                # On Python >= 3.5, the following works:
                # ctx, w = args[:-1], args[-1]
                # return max(self.freq(*ctx, w), self.freq(*ctx, w.title()))

            if original_word.istitle() or at_sentence_start:
                # If we are dealing with a word that was originally in title
                # case, such as 'Ísland', use the title case query functions
                # that try both the title case trigrams and the lower case trigrams.
                # The same applies even if the original word is lower case,
                # if it is at a sentence start, because it is then probably
                # in the wrong case and should be subject to correction as such.
                logprob = logprob_title
                freq = freq_title
            else:
                # Otherwise, just shortcut to the simple and common query functions
                logprob = self.logprob
                freq = self.freq

            def stupid_backoff(w):
                # !!! TODO: We may need a more sophisticated probability function
                # !!! TODO: here, such as Kneser-Ney or Katz
                ctx = context
                lamb = 0.0
                while True:
                    if not ctx:
                        # No context: simply return the logprob of the unigram,
                        # multiplied with the current lambda (backoff) factor
                        return logprob(w) + lamb
                    # !!! TODO: Optimize the following
                    fq = freq(*(tuple(ctx) + (w,)))
                    if fq > 1:
                        # We have a meaningful frequency here:
                        # return the logprob multiplied with the current lambda
                        return logprob(*(tuple(ctx) + (w,))) + lamb
                    # Zero count: back off to a simpler context
                    # and use the 'stupid backoff' to reduce the probability
                    ctx = ctx[1:]
                    # Multiply the prob by 0.4, i.e. add log(0.4) to the logprob
                    lamb += math.log(0.4)

            P = stupid_backoff
            e0 = edits0(word) | edits0(original_word)
            for c in known(e0):
                yield (c, P(c) + EDIT_0_FACTOR)
            for c in known(self.subs(word)):
                yield (c, P(c) + EDIT_S_FACTOR)
            pairs = _splits(word)
            e1 = edits1(pairs) - e0
            for c in known(e1):
                yield (c, P(c) + EDIT_1_FACTOR)
            # The following edit distance=2 stuff is hugely expensive
            # in terms of processor time and memory
            # e2 = edits2(pairs) - e1 - e0
            # for c in known(e2):
            #     yield (c, P(c) + EDIT_2_FACTOR)

        # First, if the word itself is common enough as a unigram,
        # we don't bother checking it further and just assume it's fine
        if self.logprob(word) > self._UNIGRAM_ACCEPT_THRESHOLD:
            # print(f"The original word {word} is above the threshold, returning it")
            return word
        # Otherwise, generate replacement candidates
        candidates = []
        for c, log_prob in gen_candidates(original_word, word):
            # Otherwise, add to candidate list
            candidates.append((c, log_prob))
        if not candidates:
            # No candidates beside the word itself: return it
            # print(f"Candidate {word} is only candidate, returning it")
            return word
        # Return the highest probability candidate
        # for i, (c, log_prob) in enumerate(sorted(candidates, key=lambda t:t[1], reverse=True)[0:5]):
        # print(f"Candidate {i+1} for {word} is {c} with log_prob {log_prob:.3f}")
        m = max(candidates, key=lambda t: t[1])
        if (
            m[1] < self._MIN_LOG_PROBABILITY
            and (word in self.ngrams or original_word in self.ngrams)
        ):
            # Best candidate is very unlikely: return the original word
            # print(f"Best candidate {m[0]} is highly unlikely, returning original {word}")
            return word
        # Return the most likely word
        return m[0]

    @staticmethod
    def _case_of(text):
        """ Return the case-function appropriate for text: upper, lower, title, or just str. """
        if text.isupper():
            return str.upper
        if text[0].isupper():
            # We don't use .istitle() and .title() because
            # they consider apostrophes to be word separators
            return lambda s: s[0].upper() + s[1:]
        return str

    def _cast(self, word):
        """ Cast the word to lowercase and correct accents """
        return re.sub(
            self._TRANSLATE_REGEX,
            lambda match: self._TRANSLATE[match.group()],
            word.lower(),
        )

    def is_rare(self, word, *, sentence_is_uppercase=False):
        """ Return True if the word is so rare as to be suspicious """
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

    def correct(self, word, *, context=(), at_sentence_start=False):
        """ Correct a single word, keeping its case (lower/upper/title) intact.
            The optional context parameter contains a tuple of preceding
            words, used to enable a more accurate probability prediction. """
        return self._case_of(word)(
            self._correct(word, self._cast(word), context, at_sentence_start)
        )

    def __getitem__(self, word):
        """ For the fun of it, support corrector["myword"] syntax """
        return self.correct(word)

    def __contains__(self, word):
        """ Support "word" in corrector """
        return self._db.__contains__(word)

    def correct_text(self, text, *, only_rare=False):
        """ Attempt to correct all words within a text, returning the corrected text.
            If only_rare is True, correction is only attempted on rare words. """
        result = []
        look_back = -MAX_ORDER + 1
        for token in tokenize(text):
            if token.kind == TOK.WORD:
                if only_rare and not self.is_rare(token.txt):
                    # The word is not rare, so we don't attempt correction
                    result.append(token.txt)
                else:
                    # Correct the word and return the result
                    result.append(
                        self.correct(token.txt, context=tuple(result[look_back:]))
                    )
            elif token.txt:
                result.append(token.txt)
            elif token.kind in {TOK.S_BEGIN, TOK.S_END}:
                result.append("")
        return correct_spaces(" ".join(result))


def test():

    with BIN_Db.get_db() as db:
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

        def linebreak(txt, margin=80, left_margin=0):
            """ Return a nicely column-formatted string representation of the given text,
                where each line is not longer than the given margin (if possible).
                A left margin can be optionally added, as a sequence of spaces.
                The lines are joined by newlines ('\n') but there is no trailing
                newline. """
            result = []
            line = []
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
            print(t)
            print("\nCorrected:\n")
            print(linebreak(c.correct_text(t), left_margin=8))

        t1 = time.time()
        print("\nTotal time: {0:.2f} seconds".format(t1 - t0))


if __name__ == "__main__":

    test()
