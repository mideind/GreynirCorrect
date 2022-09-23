#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Evaluation of spelling and grammar correction

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


    This program uses an Icelandic spelling & grammar error corpus
    (https://github.com/antonkarl/iceErrorCorpus) to evaluate the
    performance of the GreynirCorrect package.

    The program reads a development set of hand-annotated texts in
    TEI XML format and automatically annotates errors using GreynirCorrect.
    The machine-generated annotations are then compared with the hand-annotated
    gold reference.

    This program uses Python's multiprocessing.Pool() to perform
    the evaluation using all available CPU cores, simultaneously.

    A normal way to configure this program is to clone the iceErrorCorpus
    repository (from the above path) into a separate directory, and
    then place a symlink to it to the /eval directory. For example:

    $ cd github
    $ git clone https://github.com/antonkarl/iceErrorCorpus
    $ cd GreynirCorrect/eval
    $ ln -s ../../iceErrorCorpus/ .
    $ python eval.py

    An alternate method is to specify a glob path to the error corpus as an
    argument to eval.py:

    $ python eval.py ~/github/iceErrorCorpus/data/**/*.xml

    To measure GreynirCorrect's performance on the test set
    (by default located in ./iceErrorCorpus/testCorpus/):

    $ python eval.py -m

    To measure GreynirCorrect's performance on the test set
    excluding malformed sentences:

    $ python eval.py -m -x

    To run GreynirCorrect on the entire development corpus
    (by default located in ./iceErrorCorpus/data):

    $ python eval.py

    To run GreynirCorrect on 10 files in the development corpus:

    $ python eval.py -n 10

    To run GreynirCorrect on a randomly chosen subset of 10 files
    in the development corpus:

    $ python eval.py -n 10 -r

    To get an analysis report of token comparisons:

    $ python eval.py -a

"""

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Set,
    Union,
    Tuple,
    Iterable,
    cast,
    Any,
    DefaultDict,
    Counter,
)

import os
from collections import defaultdict
from datetime import datetime
import glob
import random
import argparse
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    # For some reason, types seem to be missing from the multiprocessing module
    # but not from multiprocessing.dummy
    import multiprocessing.dummy as multiprocessing
else:
    import multiprocessing

from reynir import _Sentence
from tokenizer import detokenize, Tok, TOK

from reynir_correct.annotation import Annotation
from reynir_correct.checker import AnnotatedSentence, check as gc_check


# Disable Pylint warnings arising from Pylint not understanding the typing module
# pylint: disable=no-member
# pylint: disable=unsubscriptable-object

# The type of a single error descriptor, extracted from a TEI XML file
ErrorDict = Dict[str, Union[str, int, bool]]

# The type of the dict that holds statistical information about sentences
# within a particular content category
SentenceStatsDict = DefaultDict[str, Union[float, int]]

# The type of the dict that holds statistical information about
# content categories
CategoryStatsDict = DefaultDict[str, SentenceStatsDict]

# This tuple should agree with the parameters of the add_sentence() function
StatsTuple = Tuple[
    str, int, bool, bool, int, int, int, int, int, int, int, int, int, int, int, int
]

# Counter of tp, tn, right_corr, wrong_corr, right_span, wrong_span
TypeFreqs = Counter[str]

# Stats for each error type for each content category
# tp, fn, right_corr, wrong_corr, right_span, wrong_span
ErrTypeStatsDict = DefaultDict[str, TypeFreqs]

CatResultDict = Dict[str, Union[int, float, str]]

# Create a lock to ensure that only one process outputs at a time
OUTPUT_LOCK = multiprocessing.Lock()

# Content categories in iceErrorCorpus, embedded within the file paths
GENRES = (
    "essays",
    "onlineNews",
    "wikipedia",
)

# Error codes in iceErrorCorpus that are considered out of scope
# for GreynirCorrect, at this stage at least
OUT_OF_SCOPE = {
    "act4mid",
    "act4pass",
    "adj4noun",
    "adjective-inflection",
    "agreement-pro",  # samræmi fornafns við undanfara  grammar ...vöðvahólf sem sé um dælinguna. Hann dælir blóðinu > Það dælir blóðinu
    "aux",  # meðferð vera og verða, hjálparsagna   wording mun verða eftirminnilegt > mun vera eftirminnilegt
    "bad-contraction",
    "bracket4square",  # svigi fyrir hornklofa  punctuation (Portúgal) > [Portúgal]
    "caps4low",
    "case-verb",
    "case-prep",
    "case-adj",
    "case-collocation",
    # "collocation-idiom",  # fast orðasamband með ógagnsæja merkingu collocation hélt hvorki vindi né vatni > hélt hvorki vatni né vindi
    # "collocation",  # fast orðasamband  collocation fram á þennan dag > fram til þessa dags
    "comma4conjunction",  # komma fyrir samtengingu punctuation ...fara með vald Guðs, öll löggjöf byggir... > ...fara með vald Guðs og öll löggjöf byggir...
    "comma4dash",  # komma fyrir bandstrik  punctuation , > -
    "comma4ex",  # komma fyrir upphrópun    punctuation Viti menn, almúginn... > Viti menn! Almúginn...
    "comma4period",  # komma fyrir punkt    punctuation ...kynnast nýju fólki, er á þrítugsaldri > ...kynnast nýju fólki. Hann er á þrítugsaldri
    "comma4qm",  # komma fyrir spurningarmerki  punctuation Höfum við réttinn, eins og að... > Höfum við réttinn? Eins og að...
    "conjunction",
    "conjunction4comma",  # samtenging fyrir kommu  punctuation ...geta orðið þröngvandi og erfitt getur verið... > ...geta orðið þröngvandi, erfitt getur verið...
    "conjunction4period",  # samtenging fyrir punkt punctuation ...tónlist ár hvert og tónlistarstefnurnar eru orðnar... > ...tónlist ár hvert. Tónlistarstefnurnar eru orðnar...
    "context",  # rangt orð í samhengi  other
    "dash4semicolon",  # bandstrik fyrir semíkommu  punctuation núna - þetta > núna; þetta
    "def4ind",  # ákveðið fyrir óákveðið    grammar skákinni > skák
    "dem-pro",  # hinn í stað fyrir sá; sá ekki til eða ofnotað grammar hinn > sá
    "dem4noun",  # ábendingarfornafn í stað nafnorðs    grammar hinn > maðurinn
    "dem4pers",  # ábendingarfornafn í stað persónufornafns grammar þessi > hún
    "extra-comma",  # auka komma    punctuation stríð, við náttúruna > stríð við náttúruna
    "extra-dem-pro",
    "extra-number",  # tölustöfum ofaukið   other   139,0 > 139
    "extra-period",  # auka punktur punctuation á morgun. Og ... > á morgun og...
    "extra-punctuation",  # auka greinarmerki   punctuation ... að > að
    "extra-space",  # bili ofaukið  spacing 4 . > 4.
    "extra-sub",
    "extra-symbol",  # tákn ofaukið other   Dalvík + gaf... > Dalvík gaf...
    "extra-word",  # orði ofaukið   insertion   augun á mótherja > augu mótherja
    "extra-words",  # orðum ofaukið insertion   ...ég fer að hugsa... > ...ég hugsa...
    "foreign-error",  # villa í útlendu orði    foreign Supurbowl > Super Bowl
    "foreign-name",  # villa í erlendu nafni    foreign Warwixk > Warwick
    "fw4ice",  # erlent orð þýtt yfir á íslensku    style   Elba > Saxelfur
    "gendered",  # kynjað mál, menn fyrir fólk  exclusion   menn hugsa oft > fólk hugsar oft
    "genitive",
    "geta",
    "have",
    "ice4fw",  # íslenskt orð notað í stað erlends      Demókrata öldungarþings herferðarnefndina > Democratic Senatorial Campaign Committee
    "ind4def",  # óákveðið fyrir ákveðið    grammar gítartakta > gítartaktana
    "ind4sub",  # framsöguháttur fyrir vh.  grammar Þrátt fyrir að konfúsíanismi er upprunninn > Þrátt fyrir að konfúsíanismi sé upprunninn
    "indef-pro",  # óákveðið fornafn    grammar enginn > ekki neinn
    "interr-pro",
    "it4nonit",  # skáletrað fyrir óskáletrað       Studdi Isma'il > Studdi Isma'il
    "loan-syntax",  # lánuð setningagerð    style   ég vaknaði upp > ég vaknaði
    "low4caps",
    "marked4unmarked",
    "mid4act",
    "mid4pass",
    "missing-commas",  # kommur vantar utan um innskot  punctuation Hún er jafn verðmæt ef ekki verðmætari en háskólapróf > Hún er verðmæt, ef ekki verðmætari, en háskólapróf
    "missing-conjunction",  # samtengingu vantar    punctuation í Noregi suður að Gíbraltarsundi > í Noregi og suður að Gíbraltarsundi
    "missing-dem-pro",
    "missing-ex",  # vantar upphrópunarmerki    punctuation Viti menn ég komst af > Viti menn! Ég komst af
    "missing-fin-verb",
    "missing-obj",
    "missing-quot",  # gæsalöpp vantar  punctuation „I'm winning > „I'm winning“
    "missing-quots",  # gæsalappir vantar   punctuation I'm winning > „I'm winning“
    "missing-semicolon",  # vantar semíkommu    punctuation Haukar Björgvin Páll > Haukar; Björgvin Páll
    "missing-square",  # vantar hornklofi   punctuation þeir > [þeir]
    "missing-sub",
    "missing-symbol",  # tákn vantar    punctuation 0 > 0%
    "missing-word",  # orð vantar   omission    í Donalda > í þorpinu Donalda
    "missing-words",  # fleiri en eitt orð vantar   omission    því betri laun > því betri laun hlýtur maður
    "nominal-inflection",
    "nonit4it",  # óskáletrað fyrir skáletrað       orðið qibt > orðið qibt
    "noun4adj",
    "noun4dem",  # nafnorð í stað ábendingarfornafns    grammar stærsta klukkan > sú stærsta
    "noun4pro",  # nafnorð í stað fornafns  grammar menntun má nálgast > hana má nálgast
    "number4word",
    "numeral-inflection",
    "pass4act",
    "pass4mid",
    "passive",
    "past4pres",  # sögn í þátíð í stað nútíðar grammar þegar hún leigði spólur > þegar hún leigir spólur
    "perfect4tense",
    "period4comma",  # punktur fyrir kommu  punctuation meira en áður. Hella meira í sig > meira en áður, hella meira í sig
    "period4conjunction",  # punktur fyrir samtengingu  punctuation ...maður vill gera. Vissulega > ...maður vill gera en vissulega
    "period4ex",  # punktur fyrir upphrópun punctuation Viti menn. > Viti menn!
    "pers4dem",  # persónufornafn í staðinn fyrir ábendingarf.  grammar það > þetta
    "pres4past",  # sögn í nútíð í stað þátíðar grammar Þeir fara út > Þeir fóru út
    "pro4noun",  # fornafn í stað nafnorðs  grammar þau voru spurð > parið var spurt
    "pro4reflexive",  # persónufornafn í stað afturbeygðs fn.   grammar Fólk heldur að það geri það hamingjusamt > Fólk heldur að það geri sig hamingjusamt
    "pro-inflection",
    "punctuation",  # greinarmerki  punctuation hún mætti og hann var ekki tilbúinn > hún mætti en hann var ekki tilbúinn
    "qm4ex",  # spurningarmerki fyrir upphrópun punctuation Algjört hrak sjálf? > Algjört hrak sjálf!
    "reflexive4noun",  # afturbeygt fornafn í stað nafnorðs grammar félagið hélt aðalfund þess > félagið hélt aðalfund sinn
    "reflexive4pro",  # afturbeygt fornafn í stað persónufornafns   grammar gegnum líkama sinn > gegnum líkama hans
    "simple4cont",  # nútíð í stað vera að + nafnh. grammar ók > var að aka
    "square4bracket",  # hornklofi fyrir sviga  punctuation [börnin] > (börnin)
    "sub4ind",
    "style",  # stíll   style   urðu ekkert frægir > urðu ekki frægir
    "syntax-other",
    "tense4perfect",
    "unicelandic",  # óíslenskuleg málnotkun    style   ...fer eftir persónunni... > ...fer eftir manneskjunni...
    "upper4lower-proper",  # stór stafur í sérnafni þar sem hann á ekki að vera capitalization  Mál og Menning > Mál og menning
    "upper4lower-noninitial",
    "v3-subordinate",
    "wording",  # orðalag   wording ...gerðum allt í raun... > ...gerðum í raun allt...
    "word4number",
    "wrong-prep",
    "xxx",  # unclassified  unclassified
    "zzz",  # to revisit    unannotated
}

# Default glob path of the development corpus TEI XML files to be processed
# Using a symlink (ln -s /my/location/of/iceErrorCorpus .) can be a good idea
_DEV_PATH = "iceErrorCorpus/data/**/*.xml"

# Default glob path of the test corpus TEI XML files to be processed
_TEST_PATH = "iceErrorCorpus/testCorpus/**/*.xml"

NAMES = {
    "tp": "True positives",
    "tn": "True negatives",
    "fp": "False positives",
    "fn": "False negatives",
    "true_positives": "True positives",
    "true_negatives": "True negatives",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
    "right_corr": "Right correction",
    "wrong_corr": "Wrong correction",
    "ctp": "True positives - error correction",
    "ctn": "True negatives - error correction",
    "cfp": "False positives - error correction",
    "cfn": "False negatives - error correction",
    "right_span": "Right span",
    "wrong_span": "Wrong span",
}

# Three levels: Supercategories, subcategories and error codes
# supercategory: {subcategory : [error code]}
SUPERCATEGORIES: DefaultDict[str, DefaultDict[str, List[str]]] = defaultdict(
    lambda: defaultdict(list)
)

GCtoIEC = {
    "A001": ["abbreviation-period"],
    "A002": ["abbreviation-period"],
    "Z001": ["upper4lower-common", "upper4lower-proper", "upper4lower-noninitial"],
    "Z002": ["lower4upper-initial", "lower4upper-proper", "lower4upper-acro"],
    "Z003": ["upper4lower-common"],
    "Z004": ["upper4lower-common"],
    "Z005": ["upper4lower-common"],
    "Z005/w": ["upper4lower-common"],
    "Z006": ["lower4upper-acro"],
    "E001": ["No responding iEC category"],
    "E002": ["No responding iEC category"],
    "E003": ["No responding iEC category"],
    "E004": ["fw"],
    "C001": ["repeat-word"],
    "C002": ["merged-words"],
    "C003": ["split-compound", "split-word", "split-words"],
    "C004": ["repeat-word"],
    "C004/w": ["repeat-word"],
    "C005": ["split-compound", "split-word", "split-words"],
    "C005/w": ["split-compound", "split-word", "split-words"],
    "Y001/w": ["style", "wording", "context"],
    "C006": ["compound-nonword"],
    "P_NT_Að/w": ["extra-conjunction"],
    "P_NT_AnnaðHvort": ["conjunction"],
    "P_NT_Annaðhvort": ["conjunction"],
    "P_NT_Annara": ["pro-inflection"],
    "P_NT_Annarar": ["pro-inflection"],
    "P_NT_Annari": ["pro-inflection"],
    "P_NT_Einkunn": ["agreement-concord"],
    "P_NT_EinnAf": ["agreement"],
    "P_NT_EndingANA": ["n4nn"],
    "P_NT_EndingIR": ["nominal-inflection"],
    "P_NT_FjöldiHluti": ["agreement"],
    "P_NT_FráÞvíAð": ["missing-conjunction"],
    "P_NT_FsMeðFallstjórn": ["case-prep"],
    "P_NT_Heldur/w": ["conjunction"],
    "P_NT_ÍTölu": ["plural4singular", "singular4plural"],
    "P_NT_Komma/w": ["extra-comma"],
    "P_NT_Né": ["conjunction"],
    "P_NT_Sem/w": ["extra-conjunction"],
    "P_NT_SemOg/w": ["extra-conjunction"],
    "P_NT_Síðan/w": ["extra-word"],
    "P_NT_SíðastLiðinn": ["split-compound"],
    "P_NT_SvigaInnihaldNl": ["case-verb", "case-prep", "case-adj"],
    "P_NT_TvípunkturFs": ["extra-colon"],
    "P_NT_VantarKommu": ["missing-comma"],
    "P_NT_VístAð": ["conjunction"],
    "P_VeraAð": ["cont4simple"],
    "P_NT_ÞóAð": ["conjunction"],
    "P_redundant_word": ["extra-word"],
    "P_wrong_person": ["verb-inflection"],
    "P_wrong_phrase": ["wording"],
    "P_wrong_word": ["wording"],
    "P_wrong_case": ["case-noun"],
    "P_wrong_gender": ["agreement-concord"],
    "P_wrong_number": ["agreement-concord"],
    "P_wrong_form": ["agreement-concord"],
    "P_transposition": ["swapped-letters"],
    "P_WRONG_CASE_nf_þf": ["case-verb"],
    "P_WRONG_CASE_nf_þgf": ["case-verb"],
    "P_WRONG_CASE_nf_ef": ["case-verb"],
    "P_WRONG_CASE_þf_nf": ["case-verb"],
    "P_WRONG_CASE_þf_þgf": ["case-verb"],
    "P_WRONG_CASE_þf_ef": ["case-verb"],
    "P_WRONG_CASE_þgf_nf": ["case-verb"],
    "P_WRONG_CASE_þgf_þf": ["case-verb"],
    "P_WRONG_CASE_þgf_ef": ["case-verb"],
    "P_WRONG_CASE_ef_nf": ["case-verb"],
    "P_WRONG_CASE_ef_þf": ["case-verb"],
    "P_WRONG_CASE_ef_þgf": ["case-verb"],
    "P_WRONG_NOUN_WITH_VERB": ["collocation"],
    "P_WRONG_OP_FORM": ["verb-inflection"],
    "P_WRONG_PLACE_PP": ["wrong-prep"],
    "P_aðaf": ["að4af"],
    "P_afað": ["af4að"],
    "P_kvhv": ["kv4hv"],
    "P_hvkv": ["hv4kv"],
    "P_nn": ["n4nn"],
    "P_n": ["nn4n"],
    "P_yi": ["y4i"],
    "P_iy": ["i4y"],
    "P_yyii": ["ý4í"],
    "P_iiyy": ["í4ý"],
    "P_WRONG_PREP_AÐ": ["að4af"],
    "P_WRONG_PREP_AF": ["af4að"],
    "P_WRONG_VERB_USE": ["collocation"],
    "P_DIR_LOC": ["dir4loc"],
    "P_MOOD_ACK": ["ind4sub-conj"],
    "P_MOOD_REL": ["ind4sub-conj"],
    "P_MOOD_TEMP": ["sub4ind-conj"],
    "P_MOOD_TEMP/w": ["sub4ind-conj"],
    "P_MOOD_COND": ["sub4ind-conj"],
    "P_MOOD_PURP": ["sub4ind-conj"],
    "P_Né": ["extra-conjunction"],
    "P_DOUBLE_DEFINITE": ["extra-dem-pro"],
    "X_number4word": ["number4word"],
    "N001": ["wrong-quot"],
    "N002": ["extra-punctuation"],
    "N002/w": ["extra-punctuation"],
    "N003": ["extra-punctuation"],
    "S001": ["nonword"],
    "S002": ["nonword"],
    "S003": ["nonword"],
    "S004": ["nonword"],
    "S005": ["nonword"],  # No better information available, most likely this
    "S006": ["nonword"],
    "T001": ["taboo-word"],
    "T001/w": ["taboo-word"],
    "U001": ["fw"],
    "U001/w": ["fw"],
    "W001/w": ["nonword"],
    "1ORD42": ["nonword", "merged-words"],
    "2ORD41": ["nonword", "split-word", "split-words"],
    "ANDA4ENDA": ["nonword", "nominal-inflection"],
    "ARI4NI": ["nonword", "adjective-inflection"],
    "ASLAUKABRODD": ["nonword", "extra-accent", "wrong-accent"],
    "ASLAUKASTAF": ["nonword", "extra-letter", "extra-letters"],
    "ASLBRODDVANTAR": ["nonword", "missing-accent", "wrong-accent"],
    "ASLSTAFVANTAR": ["nonword", "missing-letter", "missing-letters"],
    "ASLVITLSTAF": ["nonword", "letter-rep"],
    "ASLVIXL": ["nonword", "swapped-letters"],
    "ASLVIXLBRODD": ["nonword", "wrong-accent"],
    "AUKAG": ["nonword", "extra-letter"],
    "AUKAJ": ["nonword", "adjective-inflection"],
    "AUKARÞFFT": ["nonword", "nominal-inflection", "extra-letter"],
    "B4P": ["nonword", "letter-rep"],
    "BAND-OF": ["nonword", "extra-hyphen"],
    "BAND-VANT": ["nonword", "missing-hyphen"],
    "BEYGSJALD": [
        "nonword",
        "nominal-inflection",
        "verb-inflection",
        "adjective-inflection",
        "numeral-inflection",
    ],
    "BEYGVILLA": [
        "nonword",
        "nominal-inflection",
        "verb-inflection",
        "adjective-inflection",
        "numeral-inflection",
    ],
    "CE4CJE": ["nonword", "nominal-inflection"],
    "CJE4CE": ["nonword", "extra-letter"],
    "CÉ4CE": ["nonword", "extra-accent"],
    "DN4NN": ["nonword", "letter-rep"],
    "E4EI": ["nonword", "missing-letter"],
    "EBEYG": ["nonword", "nominal-inflection"],
    "EFINGU": ["nonword", "nominal-inflection"],
    "EI4E": ["nonword", "extra-letter"],
    "EI4EY": ["nonword", "i4y"],
    "EKKIORD": ["nonword"],
    "ETVILLA": ["nonword", "nominal-inflection"],
    "EY4EI": ["nonword", "y4i"],
    "F4FF": ["nonword", "missing-letter"],
    "F4V": ["nonword", "letter-rep"],
    "FS4PS": ["nonword", "letter-rep"],
    "FT4PT": ["nonword", "letter-rep"],
    "FTVILLA": ["nonword", "nominal-inflection"],
    "G4J": ["nonword", "letter-rep"],
    "G4K": ["nonword", "letter-rep"],
    "GGÐ4GÐ": ["nonword", "extra-letter"],
    "GL4GGL": ["nonword", "missing-letter"],
    "GMVILLA": ["nonword", "verb-inflection"],
    "GN4GGN": ["nonword", "missing-letter"],
    "GN4NG": ["nonword", "swapped-letters"],
    "GST4GGST": ["nonword", "missing-letter"],
    "GV4GGV": ["nonword", "missing-letter"],
    "GVANTAR": ["nonword", "missing-letter"],
    "GÐ4GGÐ": ["nonword", "missing-letter"],
    "HA4LAG": ["nonword", "upper4lower-common"],
    "HK4KVK": ["nonword", "nominal-inflection"],
    "HV4KV": ["nonword", "hv4kv"],
    "I4Y": ["nonword", "i4y"],
    "I4Í": ["nonword", "i4í"],
    "J4G": ["nonword", "letter-rep"],
    "J4GJ": ["nonword", "extra-letter"],
    "JE4É": ["nonword", "extra-letter"],
    "JVANTAR": ["nonword", "missing-letter"],
    "JÉ4JE": ["nonword", "extra-accent"],
    "KKN4KN": ["nonword", "extra-letter"],
    "KKT4KT": ["nonword", "extra-letter"],
    "KN4KKN": ["nonword", "missing-letter"],
    "KT4GT": ["nonword", "letter-rep"],
    "KT4KKT": ["nonword", "missing-letter"],
    "KV4HV": ["nonword", "kv4hv"],
    "KV4HV-FNSP": ["nonword", "kv4hv"],
    "KVK4HK": ["nonword", "nominal-inflection"],
    "KVK4KK": ["nonword", "nominal-inflection"],
    "LAG4HA": ["nonword", "lower4upper-proper"],
    "LG4GL": ["nonword", "swapped-letters"],
    "LLJ4LJ": ["nonword", "extra-letter"],
    "LLST4LST": ["nonword", "extra-letter"],
    "LLT4LT": ["nonword", "extra-letter"],
    "LS4LLS": ["nonword", "missing-letter"],
    "LST4LLST": ["nonword", "missing-letter"],
    "LT4LLT": ["nonword", "missing-letter"],
    "M4FN": ["nonword", "pronun-writing"],
    "M4MM": ["nonword", "missing-letter"],
    "MIÐSTIGV": ["nonword", "adjective-inflection"],
    "MM4M": ["nonword", "extra-letter"],
    "N4NN-END": ["nonword", "nominal-inflection"],
    "N4NN-ORD": ["nonword", "nominal-inflection"],
    "N4NN-SAM": ["nonword", "nominal-inflection"],
    "NG4GN": ["nonword", "swapped-letters"],
    "NGNK": ["nonword", "swapped-letters"],
    "NN4N-END": ["nonword", "nominal-inflection", "adjective-inflection"],
    "NN4N-ORD": ["nonword", "nominal-inflection"],
    "NN4N-SAM": ["nonword", "nominal-inflection"],
    "O4Ó": ["nonword", "missing-accent"],
    "O4Ó-NGNK": ["nonword", "missing-accent"],
    "OF-U": ["nonword", "letter-rep"],
    "P4B": ["nonword", "letter-rep"],
    "P4Þ": ["nonword", "letter-rep"],
    "PL4FL": ["nonword", "letter-rep"],
    "PPL4PL": ["nonword", "extra-letter"],
    "PPN4PN": ["nonword", "extra-letter"],
    "PS4PPS": ["nonword", "missing-letter"],
    "PT4PPT": ["nonword", "missing-letter"],
    "R4RR": ["nonword", "missing-letter"],
    "RFTGR": ["nonword", "missing-letter"],
    "RN4RFN": ["nonword", "pronun-writing"],
    "RN4RÐN": ["nonword", "pronun-writing"],
    "RR4R": ["nonword", "extra-letter"],
    "RS4RFS": ["nonword", "pronun-writing"],
    "SAMS-V": ["nonword", "compound-collocation"],
    "SK4STK": ["nonword", "pronun-writing"],
    "SKSTV": ["nonword", "missing-hyphen"],
    "SL4RSL": ["nonword", "pronun-writing"],
    "SN-TALA-GR": ["nonword", "compound-collocation"],
    # "SO-ÞGF4ÞF": [""],
    "SPMYNDV": ["nonword", "bad-contraction"],
    "SST4ST": ["nonword", "pronun-writing"],
    "ST4RST": ["nonword", "pronun-writing"],
    "ST4SKT": ["nonword", "pronun-writing"],
    "S4AR-EF": ["nonword", "nominal-inflection"],
    "S-EFGR": ["nonword", "nominal-inflection"],
    "STAFAGERD": ["nonword"],  # No corresponding category in iEC?
    "STAFS-ERL": ["nonword", "fw4ice"],
    "STAFSVVIXL": ["nonword", "swapped-letters"],
    "STK4SK": ["nonword", "pronun-writing"],
    "STN4SN": ["nonword", "pronun-writing"],
    "T4TT": ["nonword", "missing-letter"],
    "TOKV": ["nonword", "fw4ice"],  # No corresponding category in iEC?
    "TTN4TN": ["nonword", "pronun-writing", "extra-letter"],
    "U4Y": ["nonword", "u4y"],
    "U4Ú": ["nonword", "missing-accent"],
    "V4F": ["nonword", "letter-rep"],
    "VANTAR-J-FT": ["nonword", "nominal-inflection"],
    "Y4I": ["nonword", "y4i"],
    "Z4S": ["nonword", "letter-rep"],
    "É4JE": ["nonword", "pronun-writing"],
    "Í4I": ["nonword", "í4i"],
    "Í4Ý": ["nonword", "í4ý"],
    "Ý4Y": ["nonword", "ý4y"],
    "Ý4Í": ["nonword", "ý4í"],
    # Malsnid errors
    "URE": ["style", "wording", "context"],
    "VILLA": ["style", "wording", "context"],
    # Einkunn errors
    "R000": ["style"],
    "R002": ["style"],
    "R003": ["style"],
    "R004": ["style"],
    "R005": ["style"],
}
# Value given to float metrics when there is none available
# to avoid magic numbers
NO_RESULTS = -1.0

GCSKIPCODES = frozenset(["E001"])

# Define the command line arguments

parser = argparse.ArgumentParser(
    description=(
        "This program evaluates the spelling and grammar checking performance "
        "of GreynirCorrect on iceErrorCorpus"
    )
)

parser.add_argument(
    "path",
    nargs="?",
    type=str,
    help=f"glob path of XML files to process (default: {_DEV_PATH})",
)

parser.add_argument(
    "-n",
    "--number",
    type=int,
    default=0,
    help="number of files to process (default=all)",
)

parser.add_argument(
    "-c",
    "--cores",
    type=int,
    help=f"number of CPU cores to use (default=all, i.e. {os.cpu_count() or 1})",
)

parser.add_argument(
    "-m",
    "--measure",
    action="store_true",
    help="run measurements on test corpus and output results only",
)

parser.add_argument(
    "-r",
    "--randomize",
    action="store_true",
    help="process a random subset of files",
)

parser.add_argument(
    "-q",
    "--quiet",
    default=None,
    action="store_true",
    help="output results only, not individual sentences",
)

parser.add_argument(
    "-v",
    "--verbose",
    default=None,
    action="store_true",
    help="output individual sentences as well as results, even for the test corpus",
)

parser.add_argument(
    "-x",
    "--exclude",
    action="store_true",
    help="Exclude sentences marked for exclusion",
)

parser.add_argument(
    "-s",
    "--single",
    type=str,
    default="",
    help="Get results for a single error category",
)

parser.add_argument(
    "-a",
    "--analysis",
    action="store_true",
    help="Create an analysis report for token results",
)

parser.add_argument(
    "-f", "--catfile", type=str, default="iceErrorCorpus/errorCodes.tsv"
)

# This boolean global is set to True for quiet output,
# which is the default when processing the test corpus
QUIET = False

# This boolean global is set to True if only a single
# error category should be analyzed
SINGLE = False

# This boolean global is set to True if sentences marked
# with an exclusion flag should be excluded from processing
EXCLUDE = False

# This boolean global is set to True for token-level analysis
ANALYSIS = False


def element_text(element: ET.Element) -> str:
    """Return the text of the given element,
    including all its subelements, if any"""
    return "".join(element.itertext())


class Stats:

    """A container for key statistics on processed files and sentences"""

    def __init__(self) -> None:
        """Initialize empty defaults for the stats collection"""
        self._starttime = datetime.utcnow()
        self._files: Dict[str, int] = defaultdict(int)
        # We employ a trick to make the defaultdicts picklable between processes:
        # instead of the usual lambda: defaultdict(int), use defaultdict(int).copy
        self._sentences: CategoryStatsDict = CategoryStatsDict(
            SentenceStatsDict(int).copy
        )
        self._errtypes: ErrTypeStatsDict = ErrTypeStatsDict(Counter)
        self._true_positives: DefaultDict[str, int] = defaultdict(int)
        self._false_negatives: DefaultDict[str, int] = defaultdict(int)
        self._tp: DefaultDict[str, int] = defaultdict(int)
        self._tn: DefaultDict[str, int] = defaultdict(int)
        self._fp: DefaultDict[str, int] = defaultdict(int)
        self._fn: DefaultDict[str, int] = defaultdict(int)
        self._right_corr: DefaultDict[str, int] = defaultdict(int)
        self._wrong_corr: DefaultDict[str, int] = defaultdict(int)
        self._ctp: DefaultDict[str, int] = defaultdict(int)
        self._ctn: DefaultDict[str, int] = defaultdict(int)
        self._cfp: DefaultDict[str, int] = defaultdict(int)
        self._cfn: DefaultDict[str, int] = defaultdict(int)
        self._right_span: DefaultDict[str, int] = defaultdict(int)
        self._wrong_span: DefaultDict[str, int] = defaultdict(int)
        # reference error code : freq - for hypotheses with the unparsable error code
        self._tp_unparsables: DefaultDict[str, int] = defaultdict(int)

    def add_file(self, category: str) -> None:
        """Add a processed file in a given content category"""
        self._files[category] += 1

    def add_result(
        self,
        *,
        stats: List[StatsTuple],
        true_positives: Dict[str, int],
        false_negatives: Dict[str, int],
        ups: Dict[str, int],
        errtypefreqs: ErrTypeStatsDict,
    ) -> None:
        """Add the result of a process() call to the statistics collection"""
        for sent_result in stats:
            self.add_sentence(*sent_result)
        for k, v in true_positives.items():
            self._true_positives[k] += v
        for k, v in false_negatives.items():
            self._false_negatives[k] += v
        for k, v in ups.items():
            self._tp_unparsables[k] += v

        for okey, d in errtypefreqs.items():  # okey = xtype; d = DefaultDict[str, int]
            for ikey, vv in d.items():  # ikey = tp, fn, ...
                self._errtypes[okey][ikey] += vv  # v = freq for each metric

    def add_sentence(
        self,
        category: str,
        num_tokens: int,
        ice_error: bool,
        gc_error: bool,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        right_corr: int,
        wrong_corr: int,
        ctp: int,
        ctn: int,
        cfp: int,
        cfn: int,
        right_span: int,
        wrong_span: int,
    ) -> None:
        """Add a processed sentence in a given content category"""
        d = self._sentences[category]
        d["count"] += 1
        d["num_tokens"] += num_tokens
        d["ice_errors"] += 1 if ice_error else 0
        d["gc_errors"] += 1 if gc_error else 0
        # True negative: neither iceErrorCorpus nor GC report an error
        true_negative = not ice_error and not gc_error
        d["true_negatives"] += 1 if true_negative else 0
        # True positive: both iceErrorCorpus and GC report an error
        true_positive = ice_error and gc_error
        d["true_positives"] += 1 if true_positive else 0
        # False negative: iceErrorCorpus reports an error where GC doesn't
        false_negative = ice_error and not gc_error
        d["false_negatives"] += 1 if false_negative else 0
        # False positive: GC reports an error where iceErrorCorpus doesn't
        false_positive = gc_error and not ice_error
        d["false_positives"] += 1 if false_positive else 0

        # Stats for error detection for sentence
        d["tp"] += tp
        d["tn"] += tn
        d["fp"] += fp
        d["fn"] += fn
        # Stats for error correction ratio
        d["right_corr"] += right_corr
        d["wrong_corr"] += wrong_corr
        # Stats for error correction for sentence
        d["ctp"] += ctp
        d["ctn"] += ctn
        d["cfp"] += cfp
        d["cfn"] += cfn
        # Stats for error span
        d["right_span"] += right_span
        d["wrong_span"] += wrong_span

        # Causes of unparsable sentences

    def output(self, cores: int) -> None:
        """Write the statistics to stdout"""

        # Accumulate standard output in a buffer, for writing in one fell
        # swoop at the end (after acquiring the output lock)
        if SINGLE:
            bprint(f"")

        num_sentences: int = sum(
            cast(int, d["count"]) for d in self._sentences.values()
        )

        def output_duration() -> None:  # type: ignore
            """Calculate the duration of the processing"""
            dur = int((datetime.utcnow() - self._starttime).total_seconds())
            h = dur // 3600
            m = (dur % 3600) // 60
            s = dur % 60
            # Output a summary banner
            bprint(f"\n" + "=" * 7)
            bprint(f"Summary")
            bprint(f"=" * 7 + "\n")
            # Total number of files processed, and timing stats
            bprint(f"Processing started at {str(self._starttime)[0:19]}")
            bprint(f"Total processing time {h}h {m:02}m {s:02}s, using {cores} cores")
            bprint(f"\nFiles processed:            {sum(self._files.values()):6}")
            for c in GENRES:
                bprint(f"   {c:<13}:           {self._files[c]:6}")
            # Total number of tokens processed
            num_tokens = sum(d["num_tokens"] for d in self._sentences.values())
            bprint(f"\nTokens processed:           {num_tokens:6}")
            for c in GENRES:
                bprint(f"   {c:<13}:           {self._sentences[c]['num_tokens']:6}")
            # Total number of sentences processed
            bprint(f"\nSentences processed:        {num_sentences:6}")
            for c in GENRES:
                bprint(f"   {c:<13}:           {self._sentences[c]['count']:6}")

        def perc(n: int, whole: int) -> str:
            """Return a percentage of total sentences, formatted as 3.2f"""
            if whole == 0:
                return "N/A"
            return f"{100.0*n/whole:3.2f}"

        def write_basic_value(
            val: int, bv: str, whole: int, errwhole: Optional[int] = None
        ) -> None:
            """Write basic values for sentences and their freqs to stdout"""
            if errwhole:
                bprint(
                    f"\n{NAMES[bv]+':':<20}        {val:6} {perc(val, whole):>6}% / {perc(val, errwhole):>6}%"
                )
            else:
                bprint(f"\n{NAMES[bv]+':':<20}        {val:6} {perc(val, whole):>6}%")
            for c in GENRES:
                bprint(f"   {c:<13}:           {self._sentences[c][bv]:6}")

        def calc_PRF(
            tp: int,
            tn: int,
            fp: int,
            fn: int,
            tps: str,
            tns: str,
            fps: str,
            fns: str,
            recs: str,
            precs: str,
        ) -> None:
            """Calculate precision, recall and F0.5-score"""
            # Recall
            if tp + fn == 0:
                result = "N/A"
                recall = 0.0
            else:
                recall = tp / (tp + fn)
                result = f"{recall:1.4f}"
            bprint(f"\nRecall:                     {result}")
            for c in GENRES:
                d = self._sentences[c]
                denominator = d[tps] + d[fns]
                if denominator == 0:
                    bprint(f"   {c:<13}:              N/A")
                else:
                    rc = d[recs] = d[tps] / denominator
                    bprint(f"   {c:<13}:           {rc:1.4f}")
            # Precision
            if tp + fp == 0:
                result = "N/A"
                precision = 0.0
            else:
                precision = tp / (tp + fp)
                result = f"{precision:1.4f}"
            bprint(f"\nPrecision:                  {result}")
            for c in GENRES:
                d = self._sentences[c]
                denominator = d[tps] + d[fps]
                if denominator == 0:
                    bprint(f"   {c:<13}:              N/A")
                else:
                    p = d[precs] = d[tps] / denominator
                    bprint(f"   {c:<13}:           {p:1.4f}")
            # F0.5 score
            if precision + recall > 0.0:
                f05 = 1.25 * (precision * recall) / (0.25 * precision + recall)
                result = f"{f05:1.4f}"
            else:
                f05 = 0.0
                result = "N/A"

            bprint(f"\nF0.5 score:                   {result}")
            for c in GENRES:
                d = self._sentences[c]
                if recs not in d or precs not in d:
                    bprint(f"   {c:<13}:              N/A")
                    continue
                rc = d[recs]
                p = d[precs]
                if p + rc > 0.0:
                    f05 = 1.25 * (p * rc) / (0.25 * p + rc)
                    bprint(f"   {c:<13}:           {f05:1.4f}")
                else:
                    bprint(f"   {c:<13}:           N/A")

        def calc_recall(
            right: int, wrong: int, rights: str, wrongs: str, recs: str
        ) -> None:
            """Calculate precision for binary classification"""
            # Recall
            if right + wrong == 0:
                result = "N/A"
                recall = 0.0
            else:
                recall = right / (right + wrong)
                result = f"{recall:1.4f}"
            bprint(f"\nRecall:                     {result}")
            for c in GENRES:
                d = self._sentences[c]
                denominator = d[rights] + d[wrongs]
                if denominator == 0:
                    bprint(f"   {c:<13}:              N/A")
                else:
                    rc = d[recs] = d[rights] / denominator
                    bprint(f"   {c:<13}:           {rc:1.4f}")

        def calc_error_category_metrics(cat: str) -> CatResultDict:
            """Calculates precision, recall and f0.5-score for a single error code
            N = Number of errors in category z in reference corpus,
            Nall =  number of tokens
            TP = Errors correctly classified as category z
            FP = Errors (or non-errors) incorrectly classified as category z
            FN = Errors in category z in reference but not hypothesis
            Recall = TPz/(TPz+FPz)
            Precision = TPz/(TPz+FNz)
            F0.5-score = 1.25*(P*R)/(0.25*P+R)
            """
            catdict: CatResultDict = {k: v for k, v in self._errtypes[cat].items()}
            tp = cast(int, catdict.get("tp", 0))
            fn = cast(int, catdict.get("fn", 0))
            fp = cast(int, catdict.get("fp", 0))
            recall: float = NO_RESULTS
            precision: float = NO_RESULTS
            ctp = cast(int, catdict.get("ctp", 0))
            cfn = cast(int, catdict.get("cfn", 0))
            cfp = cast(int, catdict.get("cfp", 0))
            crecall: float = NO_RESULTS
            cprecision: float = NO_RESULTS
            catdict["freq"] = tp + fn
            if tp + fn + fp == 0:  # No values in category
                catdict["recall"] = NO_RESULTS
                catdict["precision"] = NO_RESULTS
                catdict["f05score"] = NO_RESULTS
                catdict["crecall"] = NO_RESULTS
                catdict["cprecision"] = NO_RESULTS
                catdict["cf05score"] = NO_RESULTS
            else:
                # Error detection metrics
                # Recall
                if tp + fn != 0:
                    recall = catdict["recall"] = tp / (tp + fn)
                # Precision
                if tp + fp != 0:
                    precision = catdict["precision"] = tp / (tp + fp)
                # F0.5 score
                if recall + precision > 0.0:
                    catdict["f05score"] = (
                        1.25 * (precision * recall) / (0.25 * precision + recall)
                    )
                else:
                    catdict["f05score"] = NO_RESULTS
                # Error correction metrics
                # Recall
                if ctp + cfn != 0:
                    crecall = catdict["crecall"] = ctp / (ctp + cfn)
                # Precision
                if ctp + cfp != 0:
                    cprecision = catdict["cprecision"] = ctp / (ctp + cfp)
                # F0.5 score
                if crecall + cprecision > 0.0:
                    catdict["cf05score"] = (
                        1.25 * (cprecision * crecall) / (0.25 * cprecision + crecall)
                    )
                else:
                    catdict["cf05score"] = NO_RESULTS

                # Correction recall (not used)
                right_corr = cast(int, catdict.get("right_corr", 0))
                if right_corr > 0:
                    catdict["corr_rec"] = right_corr / (
                        right_corr + cast(int, catdict.get("wrong_corr", 0))
                    )
                else:
                    catdict["corr_rec"] = -1.0
                # Span recall
                right_span = cast(int, catdict.get("right_span", 0))
                if right_span > 0:
                    catdict["span_rec"] = right_span / (
                        right_span + cast(int, catdict.get("wrong_span", 0))
                    )
                else:
                    catdict["span_rec"] = NO_RESULTS
            return catdict

        def output_sentence_scores() -> None:  # type: ignore
            """Calculate and write sentence scores to stdout"""

            # Total number of true negatives found
            bprint(f"\nResults for error detection for whole sentences")
            true_positives: int = sum(
                cast(int, d["true_positives"]) for d in self._sentences.values()
            )
            true_negatives: int = sum(
                cast(int, d["true_negatives"]) for d in self._sentences.values()
            )
            false_positives: int = sum(
                cast(int, d["false_positives"]) for d in self._sentences.values()
            )
            false_negatives: int = sum(
                cast(int, d["false_negatives"]) for d in self._sentences.values()
            )

            write_basic_value(true_positives, "true_positives", num_sentences)
            write_basic_value(true_negatives, "true_negatives", num_sentences)
            write_basic_value(false_positives, "false_positives", num_sentences)
            write_basic_value(false_negatives, "false_negatives", num_sentences)

            # Percentage of true vs. false
            true_results = true_positives + true_negatives
            false_results = false_positives + false_negatives
            if num_sentences == 0:
                result = "N/A"
            else:
                result = (
                    perc(true_results, num_sentences)
                    + "%/"
                    + perc(false_results, num_sentences)
                    + "%"
                )
            bprint(f"\nTrue/false split: {result:>16}")
            for c in GENRES:
                d = self._sentences[c]
                num_sents = d["count"]
                true_results = cast(int, d["true_positives"] + d["true_negatives"])
                false_results = cast(int, d["false_positives"] + d["false_negatives"])
                if num_sents == 0:
                    result = "N/A"
                else:
                    result = f"{100.0*true_results/num_sents:3.2f}%/{100.0*false_results/num_sents:3.2f}%"
                bprint(f"   {c:<13}: {result:>16}")

            # Precision, recall, F0.5-score
            calc_PRF(
                true_positives,
                true_negatives,
                false_positives,
                false_negatives,
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "sentrecall",
                "sentprecision",
            )

            # Most common false negative error types
            # total = sum(self._false_negatives.values())
            # if total > 0:
            #    bprint(f"\nMost common false negative error types")
            #    bprint(f"--------------------------------------\n")
            #    for index, (xtype, cnt) in enumerate(
            #        heapq.nlargest(
            #            20, self._false_negatives.items(), key=lambda x: x[1]
            #        )
            #    ):
            #        bprint(f"{index+1:3}. {xtype} ({cnt}, {100.0*cnt/total:3.2f}%)")

            # Most common error types in unparsable sentences
            # tot = sum(self._tp_unparsables.values())
            # if tot > 0:
            #    bprint(f"\nMost common error types for unparsable sentences")
            #    bprint(f"------------------------------------------------\n")
            #    for index, (xtype, cnt) in enumerate(
            #        heapq.nlargest(20, self._tp_unparsables.items(), key=lambda x: x[1])
            #    ):
            #        bprint(f"{index+1:3}. {xtype} ({cnt}, {100.0*cnt/tot:3.2f}%)")

        def output_token_scores() -> None:  # type: ignore
            """Calculate and write token scores to stdout"""

            bprint(f"\n\nResults for error detection within sentences")

            num_tokens = sum(
                cast(int, d["num_tokens"]) for d in self._sentences.values()
            )
            bprint(f"\nTokens processed:           {num_tokens:6}")
            for c in GENRES:
                bprint(f"   {c:<13}:           {self._sentences[c]['num_tokens']:6}")

            tp = sum(cast(int, d["tp"]) for d in self._sentences.values())
            tn = sum(cast(int, d["tn"]) for d in self._sentences.values())
            fp = sum(cast(int, d["fp"]) for d in self._sentences.values())
            fn = sum(cast(int, d["fn"]) for d in self._sentences.values())

            all_ice_errs = tp + fn
            write_basic_value(tp, "tp", num_tokens, all_ice_errs)
            write_basic_value(tn, "tn", num_tokens)
            write_basic_value(fp, "fp", num_tokens, all_ice_errs)
            write_basic_value(fn, "fn", num_tokens, all_ice_errs)

            calc_PRF(
                tp,
                tn,
                fp,
                fn,
                "tp",
                "tn",
                "fp",
                "fn",
                "detectrecall",
                "detectprecision",
            )

            # Stiff: Of all errors in error corpora, how many get the right correction?
            # Loose: Of all errors the tool correctly finds, how many get the right correction?
            # Can only calculate recall.
            bprint(f"\nResults for error correction")
            right_corr = sum(
                cast(int, d["right_corr"]) for d in self._sentences.values()
            )
            wrong_corr = sum(
                cast(int, d["wrong_corr"]) for d in self._sentences.values()
            )
            write_basic_value(right_corr, "right_corr", num_tokens, tp)
            write_basic_value(wrong_corr, "wrong_corr", num_tokens, tp)

            calc_recall(
                right_corr, wrong_corr, "right_corr", "wrong_corr", "correctrecall"
            )

            # Stiff: Of all errors in error corpora, how many get the right span?
            # Loose: Of all errors the tool correctly finds, how many get the right span?
            # Can only calculate recall.

            bprint(f"\nResults for error span")
            right_span = sum(
                cast(int, d["right_span"]) for d in self._sentences.values()
            )
            wrong_span = sum(
                cast(int, d["wrong_span"]) for d in self._sentences.values()
            )
            write_basic_value(right_span, "right_span", num_tokens, tp)
            write_basic_value(wrong_span, "wrong_span", num_tokens, tp)
            calc_recall(
                right_span, wrong_span, "right_span", "wrong_span", "spanrecall"
            )

        def output_error_cat_scores() -> None:
            """Calculate and write scores for each error category to stdout"""
            bprint(f"\n\nResults for each error category in order by frequency")
            freqdict: Dict[str, int] = dict()
            microf05: float = 0.0
            nfreqs: int = 0
            resultdict: Dict[str, CatResultDict] = dict()

            # Iterate over category counts
            for cat in self._errtypes.keys():
                # Get recall, precision and F0.5; recall for correction and span
                catdict = resultdict[cat] = calc_error_category_metrics(cat)

                # Collect micro scores, both overall and for in-scope categories
                freq = cast(int, catdict["freq"])
                assert isinstance(freq, int)
                f05score = cast(float, catdict["f05score"])
                assert isinstance(f05score, float)
                if cat not in OUT_OF_SCOPE:
                    microf05 += f05score * freq
                    nfreqs += freq

                # Create freqdict for sorting error categories by frequency
                freqdict[cat] = freq

            # Print results for each category by frequency
            for k in sorted(freqdict, key=freqdict.__getitem__, reverse=True):
                rk = resultdict[k]
                bprint("{} (in_scope={})".format(k, k not in OUT_OF_SCOPE))
                bprint(
                    "\tTP, FP, FN: {}, {}, {}".format(
                        rk.get("tp", 0),
                        rk.get("fp", 0),
                        rk.get("fn", 0),
                    )
                )
                bprint(
                    "\tRe, Pr, F0.5: {:3.2f}, {:3.2f}, {:3.2f}".format(
                        cast(float, rk.get("recall", 0.0)) * 100.0,
                        cast(float, rk.get("precision", 0.0)) * 100.0,
                        cast(float, rk.get("f05score", 0.0)) * 100.0,
                    )
                )
                if (
                    rk.get("corr_rec", "N/A") == "N/A"
                    or rk.get("span_rec", "N/A") == "N/A"
                ):
                    bprint("\tCorr, span:    N/A,    N/A")
                else:
                    bprint(
                        "\tCorr, span: {:3.2f}, {:3.2f}".format(
                            cast(float, rk.get("corr_rec", 0.0)) * 100.0,
                            cast(float, rk.get("span_rec", 0.0)) * 100.0,
                        )
                    )

            # Micro F0.5-score
            # Results for in-scope categories and all categories
            if nfreqs != 0:
                bprint(
                    "F0.5-score: {:3.2f}".format(
                        microf05 / nfreqs * 100.0,
                    )
                )
            else:
                bprint(f"F0.5-score: N/A")

        def output_supercategory_scores():
            """Error detection results for each supercategory in iEC given
            in SUPERCATEGORIES, each subcategory, and error code"""
            bprint("Supercategory: frequency, F-score")
            bprint("\tSubcategory: frequency, F-score")
            bprint(
                "\t\tError code: frequency, (recall, precision, F-score), (tp, fn, fp)| correct recall"
            )
            totalfreq = 0
            totalf = 0.0
            for supercat in SUPERCATEGORIES:
                # supercategory: {subcategory : error code}
                # entry = supercategory, catlist = {subcategory : error code}
                superblob = ""
                superfreq = 0
                superf = 0.0

                for subcat in SUPERCATEGORIES[supercat]:
                    subblob = ""
                    subfreq = 0
                    subf = 0.0
                    for code in SUPERCATEGORIES[supercat][subcat]:
                        if code not in OUT_OF_SCOPE:
                            et = calc_error_category_metrics(code)
                            if et["f05score"] == "N/A":
                                continue
                            freq = cast(int, et["freq"])
                            fscore = cast(float, et["f05score"])
                            # codework
                            subblob = (
                                subblob
                                + "\t\t{} {}  ({:3.2f}, {:3.2f}, {:3.2f}) ({},{},{})| {}\n".format(
                                    code,
                                    freq,
                                    cast(float, et["recall"]) * 100.0
                                    if "recall" in et
                                    else 0.0,
                                    cast(float, et["precision"]) * 100.0
                                    if "precision" in et
                                    else 0.0,
                                    fscore * 100.0,
                                    cast(int, et["tp"]) if "tp" in et else 0,
                                    cast(int, et["fn"]) if "fn" in et else 0,
                                    cast(int, et["fp"]) if "fp" in et else 0,
                                    cast(float, et["corr_rec"])
                                    if "corr_rec" in et
                                    else 0.0,
                                )
                            )
                            # subwork
                            subfreq += freq
                            subf += fscore * freq * 100.0
                    if subfreq != 0:
                        subblob = (
                            "\t{}   {} {}\n".format(
                                subcat.capitalize(), subfreq, subf / subfreq
                            )
                            + subblob
                        )
                    else:
                        subblob = (
                            "\t{}    0    N/A\n".format(subcat.capitalize()) + subblob
                        )
                    # superwork
                    # freq, f05
                    superblob += subblob
                    superfreq += subfreq
                    superf += subf  # TODO is this correct?
                if superfreq != 0:
                    superblob = (
                        "\n{}   {} {}\n".format(
                            supercat.capitalize(), superfreq, superf / superfreq
                        )
                        + superblob
                    )
                else:
                    superblob = (
                        "\n{}    0    N/A\n".format(supercat.capitalize()) + superblob
                    )
                totalfreq += superfreq
                totalf += superf  # TODO is this correct?
                bprint("".join(superblob))
            bprint("Total frequency: {}".format(totalfreq))
            bprint("Total F-score: {}".format(totalf / totalfreq))

        def output_all_scores():
            """Results for each supercategory in iEC given in SUPERCATEGORIES, each subcategory, and error code, in tsv format."""
            bprint(
                "Category\tfrequency\ttp\tfn\tfp\trecall\tprecision\tF-score\tctp\tcfn\tcfp\tcrecall\tcprecision\tCF-score"
            )
            totalfreq = 0
            totaltp = 0
            totalfn = 0
            totalfp = 0
            totalrecall = 0.0
            totalprecision = 0.0
            totalf = 0.0
            totalctp = 0
            totalcfn = 0
            totalcfp = 0
            totalcrecall = 0.0
            totalcprecision = 0.0
            totalcf = 0.0
            for supercat in SUPERCATEGORIES:
                # supercategory: {subcategory : error code}
                # entry = supercategory, catlist = {subcategory : error code}
                superfreq = 0
                supertp = 0
                superfn = 0
                superfp = 0
                superrecall = 0.0
                superprecision = 0.0
                superf = 0.0
                superctp = 0
                supercfn = 0
                supercfp = 0
                supercrecall = 0.0
                supercprecision = 0.0
                supercf = 0.0
                superblob = ""
                for subcat in SUPERCATEGORIES[supercat]:
                    subfreq = 0
                    subtp = 0
                    subfn = 0
                    subfp = 0
                    subrecall = 0.0
                    subprecision = 0.0
                    subf = 0.0
                    subctp = 0
                    subcfn = 0
                    subcfp = 0
                    subcrecall = 0.0
                    subcprecision = 0.0
                    subcf = 0.0
                    subblob = ""
                    for code in SUPERCATEGORIES[supercat][subcat]:
                        if code not in OUT_OF_SCOPE:
                            et = calc_error_category_metrics(code)
                            freq = cast(int, et["freq"])
                            fscore = cast(float, et["f05score"])
                            cfscore = cast(float, et["cf05score"])
                            # codework
                            subblob = (
                                subblob
                                + "{}\t{}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\n".format(
                                    code,
                                    freq,
                                    cast(int, et["tp"]) if "tp" in et else 0,
                                    cast(int, et["fn"]) if "fn" in et else 0,
                                    cast(int, et["fp"]) if "fp" in et else 0,
                                    cast(float, et["recall"]) * 100.0
                                    if ("recall" in et and float(et["recall"]) > 0.0)
                                    else NO_RESULTS,  # Or "N/A", but that messes with the f-string formatting
                                    cast(float, et["precision"]) * 100.0
                                    if (
                                        "precision" in et
                                        and float(et["precision"]) > 0.0
                                    )
                                    else NO_RESULTS,
                                    fscore * 100.0 if fscore > 0.0 else NO_RESULTS,
                                    cast(int, et["ctp"]) if "ctp" in et else 0,
                                    cast(int, et["cfn"]) if "cfn" in et else 0,
                                    cast(int, et["cfp"]) if "cfp" in et else 0,
                                    cast(float, et["crecall"]) * 100.0
                                    if ("crecall" in et and float(et["crecall"]) > 0.0)
                                    else NO_RESULTS,
                                    cast(float, et["cprecision"]) * 100.0
                                    if (
                                        "cprecision" in et
                                        and float(et["cprecision"]) > 0.0
                                    )
                                    else NO_RESULTS,
                                    cfscore * 100.0 if cfscore > 0.0 else NO_RESULTS,
                                )
                            )
                            # subwork
                            subfreq += freq
                            subtp += cast(int, et["tp"]) if "tp" in et else 0
                            subfn += cast(int, et["fn"]) if "fn" in et else 0
                            subfp += cast(int, et["fp"]) if "fp" in et else 0
                            subrecall += (
                                cast(float, et["recall"]) * freq * 100.0
                                if ("recall" in et and float(et["recall"]) > 0.0)
                                else 0.0
                            )
                            subprecision += (
                                cast(float, et["precision"]) * freq * 100.0
                                if ("precision" in et and float(et["precision"]) > 0.0)
                                else 0.0
                            )
                            subf += fscore * freq * 100.0 if fscore > 0.0 else 0.0
                            subctp += cast(int, et["ctp"]) if "ctp" in et else 0
                            subcfn += cast(int, et["cfn"]) if "cfn" in et else 0
                            subcfp += cast(int, et["cfp"]) if "cfp" in et else 0
                            subcrecall += (
                                cast(float, et["crecall"]) * freq * 100.0
                                if ("crecall" in et and float(et["crecall"]) > 0.0)
                                else 0.0
                            )
                            subcprecision += (
                                cast(float, et["cprecision"]) * freq * 100.0
                                if (
                                    "cprecision" in et and float(et["cprecision"]) > 0.0
                                )
                                else 0.0
                            )
                            subcf += cfscore * freq * 100.0 if cfscore > 0.0 else 0.0

                    if subfreq != 0:
                        subblob = (
                            "\n{}\t{}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\n".format(
                                subcat.capitalize(),
                                subfreq,
                                subtp,
                                subfn,
                                subfp,
                                subrecall / subfreq,
                                subprecision / subfreq,
                                subf / subfreq,
                                subctp,
                                subcfn,
                                subcfp,
                                subcrecall / subfreq,
                                subcprecision / subfreq,
                                subcf / subfreq,
                            )
                            + subblob
                        )
                    else:
                        subblob = (
                            "\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                                subcat.capitalize(),
                                subfreq,
                                subtp,
                                subfn,
                                subfp,
                                NO_RESULTS,
                                NO_RESULTS,
                                NO_RESULTS,
                                subctp,
                                subcfn,
                                subcfp,
                                NO_RESULTS,
                                NO_RESULTS,
                                NO_RESULTS,
                            )
                            + subblob
                        )

                    # superwork
                    superblob += subblob
                    superfreq += subfreq
                    supertp += subtp
                    superfn += subfn
                    superfp += subfp
                    superrecall += subrecall
                    superprecision += subprecision
                    superf += subf
                    superctp += subctp
                    supercfn += subcfn
                    supercfp += subcfp
                    supercrecall += subcrecall
                    supercprecision += subcprecision
                    supercf += subcf
                if superfreq != 0:
                    superblob = (
                        "\n{}\t{}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\n".format(
                            supercat.capitalize(),
                            superfreq,
                            supertp,
                            superfn,
                            superfp,
                            superrecall / superfreq,
                            superprecision / superfreq,
                            superf / superfreq,
                            superctp,
                            supercfn,
                            supercfp,
                            supercrecall / superfreq,
                            supercprecision / superfreq,
                            supercf / superfreq,
                        )
                        + superblob
                    )
                else:
                    superblob = (
                        "\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                            supercat.capitalize(),
                            superfreq,
                            supertp,
                            superfn,
                            superfp,
                            NO_RESULTS,
                            NO_RESULTS,
                            NO_RESULTS,
                            superctp,
                            supercfn,
                            supercfp,
                            NO_RESULTS,
                            NO_RESULTS,
                            NO_RESULTS,
                        )
                        + superblob
                    )
                totalfreq += superfreq
                totaltp += supertp
                totalfn += superfn
                totalfp += superfp
                totalrecall += superrecall
                totalprecision += superprecision
                totalf += superf
                totalctp += superctp
                totalcfn += supercfn
                totalcfp += supercfp
                totalcrecall += supercrecall
                totalcprecision += supercprecision
                totalcf += supercf

                bprint("".join(superblob))
            bprint(
                "\n{}\t{}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\t{}\t{}\t{}\t{:3.2f}\t{:3.2f}\t{:3.2f}\n".format(
                    "Total",
                    totalfreq,
                    totaltp,
                    totalfn,
                    totalfp,
                    totalrecall / totalfreq,
                    totalprecision / totalfreq,
                    totalf / totalfreq,
                    totalctp,
                    totalcfn,
                    totalcfp,
                    totalcrecall / totalfreq,
                    totalcprecision / totalfreq,
                    totalcf / totalfreq,
                )
            )

        # output_duration()
        # output_sentence_scores()
        # output_token_scores()
        # output_error_cat_scores()

        bprint(f"\n\nResults for iEC-categories:")
        # output_supercategory_scores()
        output_all_scores()

        # Print the accumulated output before exiting
        for s in buffer:
            print(s)


def correct_spaces(tokens: List[Tuple[str, str]]) -> str:
    """Returns a string with a reasonably correct concatenation
    of the tokens, where each token is a (tag, text) tuple."""
    return detokenize(
        Tok(TOK.PUNCTUATION if tag == "c" else TOK.WORD, txt, None)
        for tag, txt in tokens
    )


# Accumulate standard output in a buffer, for writing in one fell
# swoop at the end (after acquiring the output lock)
buffer: List[str] = []


def bprint(s: str):
    """Buffered print: accumulate output for printing at the end"""
    buffer.append(s)


def process(fpath_and_category: Tuple[str, str]) -> Dict[str, Any]:

    """Process a single error corpus file in TEI XML format.
    This function is called within a multiprocessing pool
    and therefore usually executes in a child process, separate
    from the parent process. It should thus not modify any
    global state, and arguments and return values should be
    picklable."""

    # Unpack arguments
    fpath, category = fpath_and_category

    # Set up XML namespace stuff
    NS = "http://www.tei-c.org/ns/1.0"
    # Length of namespace prefix to cut from tag names, including { }
    nl = len(NS) + 2
    # Namespace dictionary to be passed to ET functions
    ns = dict(ns=NS)

    # Statistics about processed sentences. These data will
    # be returned back to the parent process.
    stats: List[StatsTuple] = []
    # Counter of iceErrorCorpus error codes (xtypes) encountered
    true_positives: Dict[str, int] = defaultdict(int)
    false_negatives: Dict[str, int] = defaultdict(int)
    # Counter of iceErrorCorpus error codes in unparsable sentences
    ups: Dict[str, int] = defaultdict(int)
    # Stats for each error code (xtypes)
    errtypefreqs: ErrTypeStatsDict = ErrTypeStatsDict(TypeFreqs().copy)

    try:

        if not QUIET:
            # Output a file header
            bprint("-" * 64)
            bprint(f"File: {fpath}")
            bprint("-" * 64)
        # Parse the XML file into a tree
        try:
            tree = ET.parse(fpath)
        except ET.ParseError as e:
            if QUIET:
                bprint(f"000: *** Unable to parse XML file {fpath} ***")
            else:
                bprint(f"000: *** Unable to parse XML file ***")
            raise e
        # Obtain the root of the XML tree
        root = tree.getroot()
        # Iterate through the sentences in the file
        for sent in root.findall("ns:text/ns:body/ns:p/ns:s", ns):
            # Skip sentence if find exclude
            if EXCLUDE:
                exc = sent.attrib.get("exclude", "")
                if exc:
                    continue
            check = False  # When args.single, checking single error code
            # Sentence identifier (index)
            index = sent.attrib.get("n", "")
            tokens: List[Tuple[str, str]] = []
            errors: List[ErrorDict] = []
            # A dictionary of errors by their index (idx field)
            error_indexes: Dict[str, ErrorDict] = {}
            dependencies: List[Tuple[str, ErrorDict]] = []
            analysisblob: List[str] = []
            # Error corpora annotations for sentences marked as unparsable
            # Enumerate through the tokens in the sentence
            for el in sent:
                tag = el.tag[nl:]
                if tag == "revision":
                    # An error annotation starts here, eventually
                    # spanning multiple tokens
                    original = ""
                    corrected = ""
                    # Note the index of the starting token within the span
                    start = len(tokens)
                    # Revision id
                    rev_id = el.attrib["id"]
                    # Look at the original text
                    el_orig = el.find("ns:original", ns)
                    if el_orig is not None:
                        # We have 0 or more original tokens embedded
                        # within the revision tag
                        orig_tokens = [
                            (subel.tag[nl:], element_text(subel)) for subel in el_orig
                        ]
                        tokens.extend(orig_tokens)
                        original = " ".join(t[1] for t in orig_tokens).strip()
                    # Calculate the index of the ending token within the span
                    end = max(start, len(tokens) - 1)
                    # Look at the corrected text
                    el_corr = el.find("ns:corrected", ns)
                    if el_corr is not None:
                        corr_tokens = [element_text(subel) for subel in el_corr]
                        corrected = " ".join(corr_tokens).strip()
                    # Accumulate the annotations (errors)
                    for el_err in el.findall("ns:errors/ns:error", ns):
                        attr = el_err.attrib
                        # Collect relevant information into a dict
                        xtype: str = attr["xtype"].lower()
                        error: ErrorDict = dict(
                            start=start,
                            end=end,
                            rev_id=rev_id,
                            xtype=xtype,
                            in_scope=xtype not in OUT_OF_SCOPE,
                            eid=attr.get("eid", ""),
                            original=original,
                            corrected=corrected,
                        )
                        errors.append(error)
                        # Temporarily index errors by the idx field
                        idx = attr.get("idx")
                        if idx:
                            error_indexes[idx] = error
                        # Accumulate dependencies that need to be "fixed up",
                        # i.e. errors that depend on and refer to other errors
                        # within the sentence
                        if xtype == "dep":
                            dep_id = attr.get("depId")
                            if dep_id:
                                # Note the fact that this error depends on the
                                # error with idx=dep_id
                                dependencies.append((dep_id, error))
                            else:
                                if QUIET:
                                    bprint(f"In file {fpath}:")
                                bprint(
                                    f"\n{index}: *** 'depId' attribute missing for dependency ***"
                                )
                        if SINGLE and xtype == SINGLE:
                            check = True
                else:
                    tokens.append((tag, element_text(el)))
            # Fix up the dependencies, if any
            for dep_id, error in dependencies:
                if dep_id not in error_indexes:
                    if QUIET:
                        bprint(f"In file {fpath}:")
                    bprint(f"\n{index}: *** No error has idx='{dep_id}' ***")
                else:
                    # Copy the in_scope attribute from the original error
                    error["in_scope"] = error_indexes[dep_id]["in_scope"]

            if SINGLE and not check:
                continue

            # Reconstruct the original sentence
            # TODO switch for sentence from original text file
            text = correct_spaces(tokens)
            if not text:
                # Nothing to do: drop this and go to the next sentence
                continue
            # print(text)
            options = {}
            options["annotate_unparsed_sentences"] = True  # True is default
            options["suppress_suggestions"] = False  # False is default
            options["ignore_rules"] = set(
                [
                    "",
                ]
            )
            # Pass it to GreynirCorrect
            pg = [list(p) for p in gc_check(text, **options)]
            s: Optional[_Sentence] = None
            if len(pg) >= 1 and len(pg[0]) >= 1:
                s = pg[0][0]
            if len(pg) > 1 or (len(pg) == 1 and len(pg[0]) > 1):
                # if QUIET:
                #     bprint(f"In file {fpath}:")
                # bprint(
                #     f"\n{index}: *** Input contains more than one sentence *** {text}"
                # )
                pass
            if s is None:
                # if QUIET:
                #     bprint(f"In file {fpath}:")
                # bprint(f"\n{index}: *** No parse for sentence *** {text}")
                pass
            if not QUIET:
                # Output the original sentence
                bprint(f"\n{index}: {text}")
            if not index:
                if QUIET:
                    bprint(f"In file {fpath}:")
                bprint("000: *** Sentence identifier is missing ('n' attribute) ***")

            def sentence_results(
                hyp_annotations: List[Annotation], ref_annotations: List[ErrorDict]
            ) -> Tuple[bool, bool]:
                gc_error = False
                ice_error = False
                unparsable = False
                # Output GreynirCorrect annotations
                for ann in hyp_annotations:
                    if ann.is_error:
                        gc_error = True
                    if ann.code == "E001":
                        unparsable = True
                    if not QUIET:
                        bprint(f">>> {ann}")
                # Output iceErrorCorpus annotations
                xtypes: Dict[str, int] = defaultdict(int)
                for err in ref_annotations:
                    asterisk = "*"
                    xtype = cast(str, err["xtype"])

                    if err["in_scope"]:
                        # This is an in-scope error
                        asterisk = ""
                        ice_error = True
                        # Count the errors of each xtype
                        if xtype != "dep":
                            xtypes[xtype] += 1
                    if unparsable:
                        ups[xtype] += 1
                    if not QUIET:
                        bprint(
                            f"<<< {err['start']:03}-{err['end']:03}: {asterisk}{xtype}"
                        )
                if not QUIET:
                    # Output true/false positive/negative result
                    if ice_error and gc_error:
                        bprint("=++ True positive")
                        for xtype in xtypes:
                            true_positives[xtype] += 1

                    elif not ice_error and not gc_error:
                        bprint("=-- True negative")
                    elif ice_error and not gc_error:
                        bprint("!-- False negative")
                        for xtype in xtypes:
                            false_negatives[xtype] += 1
                    else:
                        assert gc_error and not ice_error
                        bprint("!++ False positive")
                return gc_error, ice_error

            assert s is not None
            assert isinstance(s, AnnotatedSentence)
            gc_error, ice_error = sentence_results(s.annotations, errors)

            def token_results(
                hyp_annotations: Iterable[Annotation],
                ref_annotations: Iterable[ErrorDict],
            ) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
                """Calculate statistics on annotations at the token span level"""
                tp, fp, fn = 0, 0, 0  # tn comes from len(tokens)-(tp+fp+fn) later on
                right_corr, wrong_corr = 0, 0
                ctp, cfp, cfn = (
                    0,
                    0,
                    0,
                )  # ctn comes from len(tokens)-(ctp+cfp+cfn) later on
                right_span, wrong_span = 0, 0
                if not hyp_annotations and not ref_annotations:
                    # No need to go any further
                    return (
                        tp,
                        fp,
                        fn,
                        right_corr,
                        wrong_corr,
                        ctp,
                        cfp,
                        cfn,
                        right_span,
                        wrong_span,
                    )
                y = iter(hyp_annotations)  # GreynirCorrect annotations
                x = iter(ref_annotations)  # iEC annotations
                ytok: Optional[Annotation] = None
                xtok: Optional[ErrorDict] = None

                if ANALYSIS:
                    analysisblob.append("\n{}".format(text))

                    analysisblob.append("\tiEC:")
                    for iec_ann in ref_annotations:
                        analysisblob.append("\t\t{}".format(iec_ann))

                    analysisblob.append("\tGC:")
                    for gc_ann in hyp_annotations:
                        analysisblob.append("\t\t{}".format(gc_ann))

                xspanlast = set([-1])
                try:
                    ytok = next(y)
                    xtok = next(x)
                    while True:
                        ystart, yend = ytok.start, ytok.end
                        xstart, xend = cast(int, xtok["start"]), cast(int, xtok["end"])
                        samespan = False
                        # 1. Error detection
                        # Token span in GreynirCorrect annotation
                        # TODO Usually ystart, yend+1, reset when secondary comparison works
                        yspan = set(range(ystart, yend + 1))
                        # Token span in iEC annotation
                        xspan = set(range(xstart, xend + 1))
                        yorig: Set[str]
                        ysugg: Set[str]
                        if ytok.original:
                            yorig = set(ytok.original.split())
                        else:
                            yorig = set()
                        xorig = set(cast(str, xtok["original"]).split())
                        if ytok.suggest:
                            ysugg = set(ytok.suggest.split())
                        else:
                            ysugg = set()
                        xsugg = set(cast(str, xtok["corrected"]).split())
                        if xspan & yspan:
                            samespan = True

                        # Secondary comparison:
                        # Check if any common tokens
                        # and relatively same span
                        if abs(ystart - xstart) <= 5 or abs(yend - xend) <= 5:
                            if yorig and xorig and yorig.intersection(xorig):
                                samespan = True
                            if ysugg and xsugg and ysugg.intersection(xsugg):
                                samespan = True

                        # iEC error code
                        xtype = cast(str, xtok["xtype"])
                        # By default, use iEC error code
                        # on the GreynirCorrect side as well
                        ytype = xtype
                        if ytok.code in GCtoIEC:
                            # We have a mapping of the GC code
                            if xtype not in GCtoIEC[ytok.code]:
                                # The iEC code is not one that could
                                # correspond to a GC code.
                                # We select the iEC code that most commonly
                                # corresponds to the GC code;
                                # we're going to get an error for
                                # a wrong annotation type anyway, as ytype != xtype.
                                ytype = GCtoIEC[ytok.code][0]
                        else:
                            print("Error tag {} is not supported".format(ytok.code))

                        if ANALYSIS:
                            analysisblob.append(
                                "\tComparing:\n\t          {}\n\t          {} - {} ({})".format(
                                    xtok, ytok, ytok.text, ytype
                                )
                            )
                            # analysisblob.append("\tXspans:   {} | {}".format(xspanlast, xspan))
                        # Multiple tags for same error: Skip rest
                        if xspan == xspanlast:
                            if ANALYSIS:
                                analysisblob.append(
                                    "\t          Same span, skip: {}".format(
                                        cast(str, xtok["xtype"])
                                    )
                                )
                            xtok = None
                            xtok = next(x)
                            continue
                        if ytok.code in GCSKIPCODES:
                            # Skip these errors, shouldn't be compared.
                            if ANALYSIS:
                                analysisblob.append(
                                    "\t          Skip: {}".format(ytok.code)
                                )
                            ytok = None
                            ytok = next(y)
                            continue

                        if samespan:
                            # The annotation spans overlap
                            # or almost overlap and contain the same original value or correction
                            tp += 1
                            errtypefreqs[xtype]["tp"] += 1
                            if ANALYSIS:
                                analysisblob.append("\t          TP: {}".format(xtype))
                            # 2. Span detection
                            if xspan == yspan:
                                right_span += 1
                                errtypefreqs[xtype]["right_span"] += 1
                            else:
                                wrong_span += 1
                                errtypefreqs[xtype]["wrong_span"] += 1
                            # 3. Error correction
                            ycorr = getattr(ytok, "suggest", "")
                            if ycorr == xtok["corrected"]:
                                right_corr += 1
                                errtypefreqs[xtype]["right_corr"] += 1
                                ctp += 1
                                errtypefreqs[xtype]["ctp"] += 1
                            else:
                                wrong_corr += 1
                                errtypefreqs[xtype]["wrong_corr"] += 1
                                cfn += 1
                                errtypefreqs[xtype]["cfn"] += 1
                            xspanlast = xspan
                            xtok, ytok = None, None
                            xtok = next(x)
                            ytok = next(y)
                            continue

                        # The annotation spans do not overlap
                        if yend < xstart:
                            # Extraneous GC annotation before next iEC annotation
                            fp += 1
                            errtypefreqs[ytype]["fp"] += 1
                            cfp += 1
                            errtypefreqs[ytype]["cfp"] += 1
                            if ANALYSIS:
                                analysisblob.append("\t          FP: {}".format(ytype))
                            ytok = None
                            ytok = next(y)
                            continue

                        if ystart > xend:
                            # iEC annotation with no corresponding GC annotation
                            fn += 1
                            errtypefreqs[xtype]["fn"] += 1
                            cfn += 1
                            errtypefreqs[xtype]["cfn"] += 1
                            if ANALYSIS:
                                analysisblob.append("\t          FN: {}".format(xtype))
                            xspanlast = xspan
                            xtok = None
                            xtok = next(x)
                            continue

                        # Should never get here
                        assert False

                except StopIteration:
                    pass

                # At least one of the iterators has been exhausted
                # Process the remainder
                if ANALYSIS and ytok:
                    analysisblob.append("\tDumping rest of GC errors:")
                while ytok is not None:
                    # This is a remaining GC annotation: false positive
                    if ytok.code in GCSKIPCODES:
                        # Skip these errors, shouldn't be a part of the results.
                        if ANALYSIS:
                            analysisblob.append(
                                "\t          Skip: {}".format(ytok.code)
                            )
                        ytok = next(y, None)
                        continue
                    fp += 1
                    ytype = GCtoIEC[ytok.code][0] if ytok.code in GCtoIEC else ytok.code
                    errtypefreqs[ytype]["fp"] += 1
                    cfp += 1
                    errtypefreqs[ytype]["cfp"] += 1
                    if ANALYSIS:
                        analysisblob.append("\t          FP: {}".format(ytype))
                    ytok = next(y, None)

                if ANALYSIS and xtok:
                    analysisblob.append("\tDumping rest of iEC errors:")
                if not xtok:
                    # In case try fails on ytok = next(y)
                    xtok = next(x, None)

                while xtok is not None:
                    # This is a remaining iEC annotation: false negative
                    xstart, xend = cast(int, xtok["start"]), cast(int, xtok["end"])
                    xspan = set(range(xstart, xend + 1))
                    xtype = cast(str, xtok["xtype"])
                    if xspan == xspanlast:
                        # Multiple tags for same error: Skip rest
                        if ANALYSIS:
                            analysisblob.append(
                                "\t          Same span, skip: {}".format(xtype)
                            )
                        xtok = None
                        xtok = next(x, None)
                    else:
                        if ANALYSIS:
                            analysisblob.append("\t          FN: {}".format(xtype))
                        fn += 1
                        errtypefreqs[xtype]["fn"] += 1
                        cfn += 1
                        errtypefreqs[xtype]["cfn"] += 1
                        xspanlast = xspan
                        xtok = next(x, None)

                return (
                    tp,
                    fp,
                    fn,
                    right_corr,
                    wrong_corr,
                    ctp,
                    cfp,
                    cfn,
                    right_span,
                    wrong_span,
                )

            assert isinstance(s, AnnotatedSentence)
            (
                tp,
                fp,
                fn,
                right_corr,
                wrong_corr,
                ctp,
                cfp,
                cfn,
                right_span,
                wrong_span,
            ) = token_results(s.annotations, errors)
            tn = len(tokens) - tp - fp - fn
            ctn = len(tokens) - ctp - cfp - cfn
            # Collect statistics into the stats list, to be returned
            # to the parent process
            if stats is not None:
                stats.append(
                    (
                        category,
                        len(tokens),
                        ice_error,
                        gc_error,
                        tp,
                        tn,
                        fp,
                        fn,
                        right_corr,
                        wrong_corr,
                        ctp,
                        ctn,
                        cfp,
                        cfn,
                        right_span,
                        wrong_span,
                    )
                )
            if ANALYSIS:
                with open("analysis.txt", "a+") as analysis:
                    analysis.write("\n".join(analysisblob))
                analysisblob = []
    except ET.ParseError:
        # Already handled the exception: exit as gracefully as possible
        pass

    finally:
        # Print the accumulated output before exiting
        with OUTPUT_LOCK:
            for txt in buffer:
                print(txt)
            print("", flush=True)

    # This return value will be pickled and sent back to the parent process
    return dict(
        stats=stats,
        true_positives=true_positives,
        false_negatives=false_negatives,
        ups=ups,
        errtypefreqs=errtypefreqs,
    )


def initialize_cats(catfile: str) -> None:
    first = True
    with open(catfile, "r") as cfile:
        for row in cfile:
            split = row.split("\t")
            if first:
                first = False
            else:
                s0, s1, s2 = [s.strip() for s in split[0:3]]
                SUPERCATEGORIES[s0][s1].append(s2)


def main() -> None:
    """Main program"""
    # Parse the command line arguments
    args = parser.parse_args()

    # For a measurement run on the test corpus, the default is
    # quiet operation. We store the flag in a global variable
    # that is accessible to child processes.
    global QUIET
    QUIET = args.measure

    # Overriding flags
    if args.verbose is not None:
        QUIET = False
    # --quiet has precedence over --verbose
    if args.quiet is not None:
        QUIET = True

    global EXCLUDE
    EXCLUDE = args.exclude

    global SINGLE
    SINGLE = args.single

    global ANALYSIS
    ANALYSIS = args.analysis

    # Maximum number of files to process (0=all files)
    max_count = args.number
    # Initialize the statistics collector
    stats = Stats()
    # The glob path of the XML files to process
    path: str = args.path
    # When running measurements only, we use _TEST_PATH as the default,
    # otherwise _DEV_PATH

    initialize_cats(args.catfile)

    if path is None:
        path = _TEST_PATH if args.measure else _DEV_PATH

    def gen_files() -> Iterable[Tuple[str, str]]:
        """Generate tuples with the file paths and categories
        to be processed by the multiprocessing pool"""
        count = 0
        it: Iterable[str]
        if args.randomize and max_count > 0:
            # Randomizing only makes sense if there is a max count as well
            it = glob.glob(path, recursive=True)
            it = random.sample(it, max_count)
        else:
            it = glob.iglob(path, recursive=True)
        for fpath in it:
            # Find out which genre the file belongs to by
            # inference from the file name
            for genre in GENRES:
                if genre in fpath:
                    break
            else:
                assert False, f"File path does not contain a recognized genre: {fpath}"
            # Add the file to the statistics under its genre
            stats.add_file(genre)
            # Yield the file information to the multiprocessing pool
            yield fpath, genre
            count += 1
            # If there is a limit on the number of processed files,
            # and we're done, stop the generator
            if max_count > 0 and count >= max_count:
                break

    # Use a multiprocessing pool to process the articles
    with multiprocessing.Pool(processes=args.cores) as pool:
        # Iterate through the TEI XML files in turn and call the process()
        # function on each file, in a child process within the pool
        for result in pool.imap_unordered(process, gen_files()):
            # Results come back as a dict of arguments that
            # we pass to Stats.add_result()
            stats.add_result(**result)
        # Done: close the pool in an orderly manner
        pool.close()
        pool.join()
    # Finally, acquire the output lock and write the final statistics
    with OUTPUT_LOCK:
        stats.output(cores=args.cores or os.cpu_count() or 1)
        print("", flush=True)


if __name__ == "__main__":
    main()
