#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Evaluation of spelling and grammar correction

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

    To run GreynirCorrect on the entire development corpus
    (by default located in ./iceErrorCorpus/data):

    $ python eval.py

    To run GreynirCorrect on 10 files in the development corpus:

    $ python eval.py -n 10

    To run GreynirCorrect on a randomly chosen subset of 10 files
    in the development corpus:

    $ python eval.py -n 10 -r

"""

from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Iterable,
    cast,
    NamedTuple,
    Any,
    DefaultDict,
)

import os
from collections import defaultdict
from datetime import datetime
import glob
import random
import heapq
import argparse
import xml.etree.ElementTree as ET
import multiprocessing

# import multiprocessing.dummy as multiprocessing

import reynir_correct as gc
from reynir import _Sentence
from tokenizer import detokenize, Tok, TOK


# The type of a single error descriptor, extracted from a TEI XML file
ErrorDict = Dict[str, Union[str, int, bool]]

# The type of the dict that holds statistical information about sentences
# within a particular content category
SentenceStatsDict = Dict[str, Union[float, int]]

# The type of the dict that holds statistical information about
# content categories
CategoryStatsDict = Dict[str, SentenceStatsDict]

# This tuple should agree with the parameters of the add_sentence() function
StatsTuple = Tuple[str, int, bool, bool, int, int, int, int, int, int, int, int]

# Counter of tp, tn, right_corr, wrong_corr, right_span, wrong_span
TypeFreqs = DefaultDict[str, int] 

# Stats for each error type for each content category
# tp, fn, right_corr, wrong_corr, right_span, wrong_span
ErrTypeStatsDict = DefaultDict[str, TypeFreqs]

CatResultDict = Dict[str, Union[int, float, str]]

# Create a lock to ensure that only one process outputs at a time
OUTPUT_LOCK = multiprocessing.Lock()

# Content categories in iceErrorCorpus, embedded within the file paths
CATEGORIES = (
    "essays",
    "onlineNews",
    "wikipedia",
)

# Error codes in iceErrorCorpus that are considered out of scope
# for GreynirCorrect, at this stage at least
OUT_OF_SCOPE = {
    "agreement-pro",  # samræmi fornafns við undanfara  grammar ...vöðvahólf sem sé um dælinguna. Hann dælir blóðinu > Það dælir blóðinu
    "aux",  # meðferð vera og verða, hjálparsagna   wording mun verða eftirminnilegt > mun vera eftirminnilegt
    "bracket4square",  # svigi fyrir hornklofa  punctuation (Portúgal) > [Portúgal]
    # "collocation-idiom",  # fast orðasamband með ógagnsæja merkingu collocation hélt hvorki vindi né vatni > hélt hvorki vatni né vindi
    # "collocation",  # fast orðasamband  collocation fram á þennan dag > fram til þessa dags
    "comma4conjunction",  # komma fyrir samtengingu punctuation ...fara með vald Guðs, öll löggjöf byggir... > ...fara með vald Guðs og öll löggjöf byggir...
    "comma4dash",  # komma fyrir bandstrik  punctuation , > -
    "comma4ex",  # komma fyrir upphrópun    punctuation Viti menn, almúginn... > Viti menn! Almúginn...
    "comma4period",  # komma fyrir punkt    punctuation ...kynnast nýju fólki, er á þrítugsaldri > ...kynnast nýju fólki. Hann er á þrítugsaldri
    "comma4qm",  # komma fyrir spurningarmerki  punctuation Höfum við réttinn, eins og að... > Höfum við réttinn? Eins og að...
    "conjunction4comma",  # samtenging fyrir kommu  punctuation ...geta orðið þröngvandi og erfitt getur verið... > ...geta orðið þröngvandi, erfitt getur verið...
    "conjunction4period",  # samtenging fyrir punkt punctuation ...tónlist ár hvert og tónlistarstefnurnar eru orðnar... > ...tónlist ár hvert. Tónlistarstefnurnar eru orðnar...
    "context",  # rangt orð í samhengi  other
    "dash4semicolon",  # bandstrik fyrir semíkommu  punctuation núna - þetta > núna; þetta
    "def4ind",  # ákveðið fyrir óákveðið    grammar skákinni > skák
    "dem-pro",  # hinn í stað fyrir sá; sá ekki til eða ofnotað grammar hinn > sá
    "dem4noun",  # ábendingarfornafn í stað nafnorðs    grammar hinn > maðurinn
    "dem4pers",  # ábendingarfornafn í stað persónufornafns grammar þessi > hún
    "extra-comma",  # auka komma    punctuation stríð, við náttúruna > stríð við náttúruna
    "extra-number",  # tölustöfum ofaukið   other   139,0 > 139
    "extra-period",  # auka punktur punctuation á morgun. Og ... > á morgun og...
    "extra-punctuation",  # auka greinarmerki   punctuation ... að > að
    "extra-space",  # bili ofaukið  spacing 4 . > 4.
    "extra-symbol",  # tákn ofaukið other   Dalvík + gaf... > Dalvík gaf...
    "extra-word",  # orði ofaukið   insertion   augun á mótherja > augu mótherja
    "extra-words",  # orðum ofaukið insertion   ...ég fer að hugsa... > ...ég hugsa...
    "foreign-error",  # villa í útlendu orði    foreign Supurbowl > Super Bowl
    "fw4ice",  # erlent orð þýtt yfir á íslensku    style   Elba > Saxelfur
    "gendered",  # kynjað mál, menn fyrir fólk  exclusion   menn hugsa oft > fólk hugsar oft
    "ice4fw",  # íslenskt orð notað í stað erlends      Demókrata öldungarþings herferðarnefndina > Democratic Senatorial Campaign Committee
    "ind4def",  # óákveðið fyrir ákveðið    grammar gítartakta > gítartaktana
    "ind4sub",  # framsöguháttur fyrir vh.  grammar Þrátt fyrir að konfúsíanismi er upprunninn > Þrátt fyrir að konfúsíanismi sé upprunninn
    "indef-pro",  # óákveðið fornafn    grammar enginn > ekki neinn
    "it4nonit",  # skáletrað fyrir óskáletrað       Studdi Isma'il > Studdi Isma'il
    "loan-syntax",  # lánuð setningagerð    style   ég vaknaði upp > ég vaknaði
    "missing-commas",  # kommur vantar utan um innskot  punctuation Hún er jafn verðmæt ef ekki verðmætari en háskólapróf > Hún er verðmæt, ef ekki verðmætari, en háskólapróf
    "missing-conjunction",  # samtengingu vantar    punctuation í Noregi suður að Gíbraltarsundi > í Noregi og suður að Gíbraltarsundi
    "missing-ex",  # vantar upphrópunarmerki    punctuation Viti menn ég komst af > Viti menn! Ég komst af
    "missing-quot",  # gæsalöpp vantar  punctuation „I'm winning > „I'm winning“
    "missing-quots",  # gæsalappir vantar   punctuation I'm winning > „I'm winning“
    "missing-semicolon",  # vantar semíkommu    punctuation Haukar Björgvin Páll > Haukar; Björgvin Páll
    "missing-square",  # vantar hornklofi   punctuation þeir > [þeir]
    "missing-symbol",  # tákn vantar    punctuation 0 > 0%
    "missing-word",  # orð vantar   omission    í Donalda > í þorpinu Donalda
    "missing-words",  # fleiri en eitt orð vantar   omission    því betri laun > því betri laun hlýtur maður
    "nonit4it",  # óskáletrað fyrir skáletrað       orðið qibt > orðið qibt
    "noun4dem",  # nafnorð í stað ábendingarfornafns    grammar stærsta klukkan > sú stærsta
    "noun4pro",  # nafnorð í stað fornafns  grammar menntun má nálgast > hana má nálgast
    "past4pres",  # sögn í þátíð í stað nútíðar grammar þegar hún leigði spólur > þegar hún leigir spólur
    "period4comma",  # punktur fyrir kommu  punctuation meira en áður. Hella meira í sig > meira en áður, hella meira í sig
    "period4conjunction",  # punktur fyrir samtengingu  punctuation ...maður vill gera. Vissulega > ...maður vill gera en vissulega
    "period4ex",  # punktur fyrir upphrópun punctuation Viti menn. > Viti menn!
    "pers4dem",  # persónufornafn í staðinn fyrir ábendingarf.  grammar það > þetta
    "pres4past",  # sögn í nútíð í stað þátíðar grammar Þeir fara út > Þeir fóru út
    "pro4noun",  # fornafn í stað nafnorðs  grammar þau voru spurð > parið var spurt
    "pro4reflexive",  # nafnorð í stað afturbeygðs fornafns grammar gefur orku til fólks í kringum það > gefur orku til fólks í kringum sig
    "pro4reflexive",  # persónufornafn í stað afturbeygðs fn.   grammar Fólk heldur að það geri það hamingjusamt > Fólk heldur að það geri sig hamingjusamt
    "punctuation",  # greinarmerki  punctuation hún mætti og hann var ekki tilbúinn > hún mætti en hann var ekki tilbúinn
    "qm4ex",  # spurningarmerki fyrir upphrópun punctuation Algjört hrak sjálf? > Algjört hrak sjálf!
    "reflexive4noun",  # afturbeygt fornafn í stað nafnorðs grammar félagið hélt aðalfund þess > félagið hélt aðalfund sinn
    "reflexive4pro",  # afturbeygt fornafn í stað persónufornafns   grammar gegnum líkama sinn > gegnum líkama hans
    "simple4cont",  # nútíð í stað vera að + nafnh. grammar ók > var að aka
    "square4bracket",  # hornklofi fyrir sviga  punctuation [börnin] > (börnin)
    "style",  # stíll   style   urðu ekkert frægir > urðu ekki frægir
    "sub4ind",  # viðtengingarh. fyrir fh.  grammar Stjórnvöld vildu auka rétt borgara og geri þeim kleift > Stjórnvöld vildu auka rétt borgara og gera þeim kleift
    "unicelandic",  # óíslenskuleg málnotkun    style   ...fer eftir persónunni... > ...fer eftir manneskjunni...
    "upper4lower-proper",  # stór stafur í sérnafni þar sem hann á ekki að vera capitalization  Mál og Menning > Mál og menning
    "wording",  # orðalag   wording ...gerðum allt í raun... > ...gerðum í raun allt...
    "xxx",  # unclassified  unclassified
    "zzz",  # to revisit    unannotated
}

# Default glob path of the development corpus TEI XML files to be processed
_DEV_PATH = "../../mt/iceErrorCorpus/data/**/*.xml"

# Default glob path of the test corpus TEI XML files to be processed
_TEST_PATH = "../../mt/iceErrorCorpus/testCorpus/**/*.xml"

NAMES = {
    "tp" : "True positives",
    "tn" : "True negatives",
    "fp" : "False positives",
    "fn" : "False negatives",
    "true_positives" : "True positives",
    "true_negatives" : "True negatives",
    "false_positives" : "False positives",
    "false_negatives" : "False negatives",
    "right_corr" : "Right correction",
    "wrong_corr" : "Wrong correction",
    "right_span" : "Right span",
    "wrong_span" : "Wrong span"
}

# Supercategories in iceErrorCorpus and corresponding subcategories
SUPERCATEGORIES = {
    "capitalization" : [
        "lower4upper-initial", 
        "lower4upper-proper",
        "lower4upper-acro",
        "upper4lower-common",
        "upper4lower-proper",
        "upper4lower-noninitial",
        "caps4low"
    ],
    "collocation" : [
        "collocation",
        "collocation-idiom",
        "though",
    ],
    "grammar" : [
        "agreement",
        "agreement-concord",
        "agreement-pred",
        "agreement-pro",
        "case-verb",
        "case-prep",
        "case-adj",
        "case-collocation",
        "ind4def",
        "def4ind",
        "ind4sub",
        "sub4ind",
        "verb-inflection",
        "nominal-inflection",
        "numeral-inflection",
        "pro-inflection",
        "plural4singular",
        "singular4plural",
        "conjunction",
        "adjective-inflection",
        "dative-sub",
        "dir4loc",
        "loc4dir",
        "mid4act",
        "act4mid",
        "mid4pass",
        "pass4mid",
        "act4pass",
        "pass4act",
        "passive",
        "new-passive",
        "each",
        "noun4pro",
        "pro4noun",
        "reflexive4pro",
        "pro4reflexive",
        "pres4past",
        "past4pres",
        "tense4perfect",
        "perfect4tense",
        "pers4dem",
        "dem-pro",
        "missing-dem-pro",
        "extra-dem-pro",
        "indef-pro",
        "marked4unmarked",
        "adj4adv",
        "adv4adj",
        "have",
        "cont4simple",
        "missing-inf-part",
        "want",
        "dem4pers",
        "nom4acc-sub",
        "acc4nom-sub",
        "simple4cont",
        "extra-inf-part",
        "gen-escape",
        "genitive",
        "dem4noun",
        "noun4dem",
        "geta",
        "adj4noun",
        "noun4adj",
        "extra-sub",
        "missing-fin-verb",
        "missing-sub",
        "missing-obj",
        "að4af",
        "af4að",
        "wrong-prep",
        "interr-pro",
        "hypercorr",
    ],
    "syntax" : [
        "v3",
        "v3-subordinate",
        "syntax-other",
    ],
    "nonword" : [
        "compound-collocation",
        "compound-nonword",
        "nonword",
    ],
    "omission" : [
        "missing-word",
        "missing-words",
        "missing-prep",
    ],
    "typo" : [
        "swapped-letters",
        "letter-rep",
        "missing-letter",
        "missing-accent",
        "wrong-accent",
        "extra-accent",
        "extra-letter",         
    ],
    "punctuation" : [
        "comma4period",
        "comma4qm",
        "comma4colon",
        "double-punctuation",
        "extra-abbreviation",
        "extra-dash",
        "iteration-colon",
        "missing-colon",
        "missing-comma",
        "missing-commas",
        "missing-period",
        "missing-qm",
        "missing-conjunction",
        "missing-quot",
        "missing-quots",
        "misplaced-quot",
        "wrong-quots",
        "extra-quot",
        "extra-quots",
        "extra-punctuation",
        "extra-comma",
        "extra-period",
        "period4comma",
        "period4colon",
        "period4conjunction",
        "conjunction4period",
        "conjunction4comma",
        "comma4conjunction",
        "period4qm",
        "period-plus-conjunction",
        "comma-plus-conjunction",
        "abbreviation-period",
        "comma4ex",
        "period4ex",
        "semicolon4colon",
        "extra-semicolon",
        "ordinal-period",
        "conjunction-drop",
        "extra-conjunction",
        "semicolon4comma",
        "conjunction4qm",
        "missing-slash",
        "comma4bracket",
        "qm4comma",
        "missing-ex",
        "qm4ex",
        "qm4period",
        "bracket4square",
        "square4bracket",
        "dash4comma",
        "date-period",
        "comma4semicolon",
        "word4dash",
        "dash4word",
        "missing-semicolon",
        "abbreviation",
        "slash4or",
        "dash4period",
        "ex4comma",
        "colon4period",
        "colon4comma",
        "ex4period",
        "extra-colon",
        "bracket4comma",
        "extra-qm",
        "comma4dash",
        "dash4semicolon",
        "wrong-dash",
        "dash4colon",
        "dots4comma",
        "missing-symbol",
        "dots4period",
        "extra-square",
        "bracket4period",
        "word4symbol",
        "nonsup4sup",
        "semicolon4period",
        "period4semicolon",
        "period4dash",
        "missing-square",
        "slash4dash",
        "extra-commas",
        "conjunction4semicolon",
        "missing-bracket",
        "extra-bracket",
        "date-abbreviation",
    ],
    "spacing" : [
        "merged-words",
        "split-compound",
        "split-word",
        "split-words",
        "missing-dash",
        "missing-space",
        "extra-space",
    ],
    "insertion" : [
        "extra-word",
        "extra-words",
        "extra-prep",
        "repeat-word",
        "repeat-word-split",
    ],
    "wording" : [
        "wording",
        "aux",
    ],
    "spelling" : [
        "ngnk",
        "i4y",
        "y4i",
        "í4ý",
        "ý4í",
        "n4nn",
        "nn4n",
        "pronun-writing",
        "kv4hv",
        "hv4kv",
        "name-error",
        "bad-contraction",
    ],
    "foreign" : [
        "fw",
        "foreign-error",
    ],
    "exclusion" : [
        "gendered",
    ],
    "numbers" : [
        "number4word",
        "word4number",
        "extra-number",
        "symbol4number",
        "number4symbol",
        "number-fail",
    ],
    "style" : [
        "style",
        "unicelandic",
        "taboo-word",
        "loan-syntax",
        "u4y",
        "fw4ice",
        "ice4fw",
        "nonit4it",
        "it4nonit",
        "extra-munu",
        "words4abbreviation",
        "abbreviation4words",
    ],
    "other" : [
        "symbol4word",
        "extra-symbol",
        "dep",
        "þar4það",
    ],
    "lexical" : [
        "context",
    ],
    "unnannotated" : [
        "zzz",
        "xxx",
    ]
}

# Supercategories according to SÍM and corresponding subcategories
# Errors go into the first possible category. Error categories that
# can both be independent and dependent of context go under the former.
SIMCATEGORIES = {
    "context-independent" : [ 
        "caps4low",
        "number-fail",
        "lower4upper-proper",
        "lower4upper-acro",
        "though",
        "compound-collocation",
        "compound-nonword",
        "nonword",
        "swapped-letters",
        "letter-rep",
        "missing-letter",
        "missing-accent",
        "wrong-accent",
        "merged-words",
        "split-word",
        "extra-accent",
        "extra-letter",
        "ngnk",
        "i4y",
        "y4i",
        "í4ý",
        "ý4í",
        "n4nn",
        "nn4n",
        "pronun-writing",
        "kv4hv",
        "hv4kv",
        "bad-contraction",
        "fw",
        "foreign-error",
        "abbreviation-period",
        "abbreviation",
    ],
    "context-dependent" : [
        "lower4upper-initial",
        "upper4lower-common",
        "upper4lower-proper",
        "upper4lower-noninitial",
        "collocation",
        "collocation-idiom",
        "missing-word",
        "missing-words",
        "missing-prep",
        "split-compound",
        "split-words",
        "missing-dash",
        "missing-space",
        "extra-space",
        "extra-word",
        "extra-words",
        "extra-prep",
        "repeat-word",
        "repeat-word-split",
        "name-error",
        "gendered",
        "number4word",
        "word4number",
    ],
    "grammar" : [
        "agreement",
        "agreement-concord",
        "agreement-pred",
        "agreement-pro",
        "case-verb",
        "case-prep",
        "case-adj",
        "case-collocation",
        "ind4def",
        "def4ind",
        "ind4sub",
        "sub4ind",
        "verb-inflection",
        "nominal-inflection",
        "numeral-inflection",
        "pro-inflection",
        "plural4singular",
        "singular4plural",
        "conjunction",
        "adjective-inflection",
        "dative-sub",
        "dir4loc",
        "loc4dir",
        "mid4act",
        "act4mid",
        "mid4pass",
        "pass4mid",
        "act4pass",
        "pass4act",
        "passive",
        "new-passive",
        "each",
        "noun4pro",
        "pro4noun",
        "reflexive4pro",
        "pro4reflexive",
        "pres4past",
        "past4pres",
        "tense4perfect",
        "perfect4tense",
        "pers4dem",
        "dem-pro",
        "missing-dem-pro",
        "extra-dem-pro",
        "indef-pro",
        "marked4unmarked",
        "adj4adv",
        "adv4adj",
        "have",
        "cont4simple",
        "missing-inf-part",
        "want",
        "dem4pers",
        "nom4acc-sub",
        "acc4nom-sub",
        "simple4cont",
        "extra-inf-part",
        "gen-escape",
        "genitive",
        "dem4noun",
        "noun4dem",
        "geta",
        "adj4noun",
        "noun4adj",
        "extra-sub",
        "missing-fin-verb",
        "missing-sub",
        "missing-obj",
        "að4af",
        "af4að",
        "wrong-prep",
        "interr-pro",
        "hypercorr",
        "v3",
        "v3-subordinate",
        "syntax-other",
        "aux",
    ],
    "style" : [
        "wording",
        "extra-number",
        "symbol4number",
        "number4symbol",
        "style",
        "unicelandic",
        "taboo-word",
        "loan-syntax",
        "u4y",
        "fw4ice",
        "ice4fw",
        "nonit4it",
        "it4nonit",
        "extra-munu",
        "words4abbreviation",
        "abbreviation4words",
        "symbol4word",
        "extra-symbol",
        "dep",
        "þar4það",
        "context",
    ],
    "punctuation" : [
        "comma4period",
        "comma4qm",
        "comma4colon",
        "double-punctuation",
        "extra-abbreviation",
        "extra-dash",
        "iteration-colon",
        "missing-colon",
        "missing-comma",
        "missing-commas",
        "missing-period",
        "missing-qm",
        "missing-conjunction",
        "missing-quot",
        "missing-quots",
        "misplaced-quot",
        "wrong-quots",
        "extra-quot",
        "extra-quots",
        "extra-punctuation",
        "extra-comma",
        "extra-period",
        "period4comma",
        "period4colon",
        "period4conjunction",
        "conjunction4period",
        "conjunction4comma",
        "comma4conjunction",
        "period4qm",
        "period-plus-conjunction",
        "comma-plus-conjunction",
        "comma4ex",
        "period4ex",
        "semicolon4colon",
        "extra-semicolon",
        "ordinal-period",
        "conjunction-drop",
        "extra-conjunction",
        "semicolon4comma",
        "conjunction4qm",
        "missing-slash",
        "comma4bracket",
        "qm4comma",
        "missing-ex",
        "qm4ex",
        "qm4period",
        "bracket4square",
        "square4bracket",
        "dash4comma",
        "date-period",
        "comma4semicolon",
        "word4dash",
        "dash4word",
        "missing-semicolon",
        "slash4or",
        "dash4period",
        "ex4comma",
        "colon4period",
        "colon4comma",
        "ex4period",
        "extra-colon",
        "bracket4comma",
        "extra-qm",
        "comma4dash",
        "dash4semicolon",
        "wrong-dash",
        "dash4colon",
        "dots4comma",
        "missing-symbol",
        "dots4period",
        "extra-square",
        "bracket4period",
        "word4symbol",
        "nonsup4sup",
        "semicolon4period",
        "period4semicolon",
        "period4dash",
        "missing-square",
        "slash4dash",
        "extra-commas",
        "conjunction4semicolon",
        "missing-bracket",
        "extra-bracket",
        "date-abbreviation",
    ]
}

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
    "-r", "--randomize", action="store_true", help="process a random subset of files",
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

# This boolean global is set to True for quiet output,
# which is the default when processing the test corpus
QUIET = False


def element_text(element: ET.Element) -> str:
    """ Return the text of the given element,
        including all its subelements, if any """
    return "".join(element.itertext())


class Stats:

    """ A container for key statistics on processed files and sentences """

    def __init__(self) -> None:
        """ Initialize empty defaults for the stats collection """
        self._starttime = datetime.utcnow()
        self._files: Dict[str, int] = defaultdict(int)
        # We employ a trick to make the defaultdicts picklable between processes:
        # instead of the usual lambda: defaultdict(int), use defaultdict(int).copy
        self._sentences: CategoryStatsDict = CategoryStatsDict(SentenceStatsDict(int).copy)
        self._errtypes: ErrTypeStatsDict = ErrTypeStatsDict(TypeFreqs(int).copy)
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)
        self._tp: Dict[str, int] = defaultdict(int)
        self._tn: Dict[str, int] = defaultdict(int)
        self._fp: Dict[str, int] = defaultdict(int)
        self._fn: Dict[str, int] = defaultdict(int)
        self._right_corr: Dict[str, int] = defaultdict(int)
        self._wrong_corr: Dict[str, int] = defaultdict(int)
        self._right_span: Dict[str, int] = defaultdict(int)
        self._wrong_span: Dict[str, int] = defaultdict(int)
        self._tp_unparsables: Dict[str, int] = defaultdict(int)  # reference error code : freq - for hypotheses with the unparsable error code

    def add_file(self, category: str) -> None:
        """ Add a processed file in a given content category """
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
        """ Add the result of a process() call to the statistics collection """
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
        self, category: str, num_tokens: int, ice_error: bool, gc_error: bool,
        tp: int, tn: int, fp: int, fn: int, right_corr: int, wrong_corr: int,
        right_span: int, wrong_span: int
    ) -> None:
        """ Add a processed sentence in a given content category """
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
        # Stats for error correction
        d["right_corr"] += right_corr
        d["wrong_corr"] += wrong_corr
        # Stats for error span
        d["right_span"] += right_span
        d["wrong_span"] += wrong_span

        # Causes of unparsable sentences

    def output(self, cores: int) -> None:
        """ Write the statistics to stdout """

        # Accumulate standard output in a buffer, for writing in one fell
        # swoop at the end (after acquiring the output lock)
        num_sentences: int = sum(
            cast(int, d["count"]) for d in self._sentences.values()
        )

        def output_duration() -> None:
            """ Calculate the duration of the processing """
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
            for c in CATEGORIES:
                bprint(f"   {c:<13}:           {self._files[c]:6}")
            # Total number of tokens processed
            num_tokens = sum(d["num_tokens"] for d in self._sentences.values())
            bprint(f"\nTokens processed:           {num_tokens:6}")
            for c in CATEGORIES:
                bprint(f"   {c:<13}:           {self._sentences[c]['num_tokens']:6}")
            # Total number of sentences processed
            bprint(f"\nSentences processed:        {num_sentences:6}")
            for c in CATEGORIES:
                bprint(f"   {c:<13}:           {self._sentences[c]['count']:6}")

        def perc(n: int, whole: int) -> str:
            """ Return a percentage of total sentences, formatted as 3.2f """
            if whole == 0:
                return "N/A"
            return f"{100.0*n/whole:3.2f}"
        
        def write_basic_value(val: int, bv: str, whole: int, errwhole: Optional[int]=None) -> None:
            """ Write basic values for sentences and their freqs to stdout """
            if errwhole:
                bprint(
                    f"\n{NAMES[bv]+':':<20}        {val:6} {perc(val, whole):>6}% / {perc(val, errwhole):>6}%"
                )
            else:
                bprint(f"\n{NAMES[bv]+':':<20}        {val:6} {perc(val, whole):>6}%")
            for c in CATEGORIES:
                bprint(f"   {c:<13}:           {self._sentences[c][bv]:6}")

        def calc_PRF(
            tp: int, tn: int, fp: int, fn: int, tps: str, tns: str, 
            fps: str, fns: str, recs: str, precs: str
        ) -> None:
            """ Calculate precision, recall and F1-score """
            # Recall
            if tp + fn == 0:
                result = "N/A"
                recall = 0.0
            else:
                recall = tp / (tp+fn)
                result = f"{recall:1.4f}"
            bprint(f"\nRecall:                     {result}")
            for c in CATEGORIES:
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
            for c in CATEGORIES:
                d = self._sentences[c]
                denominator = d[tps] + d[fps]
                if denominator == 0:
                    bprint(f"   {c:<13}:              N/A")
                else:
                    p = d[precs] = d[tps] / denominator
                    bprint(f"   {c:<13}:           {p:1.4f}")
            # F1 score
            if precision + recall > 0.0:
                f1 = 2 * precision * recall / (precision + recall)
                result = f"{f1:1.4f}"
            else:
                f1 = 0.0
                result = "N/A"
            bprint(f"\nF1 score:                   {result}")
            for c in CATEGORIES:
                d = self._sentences[c]
                if recs not in d or precs not in d:
                    bprint(f"   {c:<13}:              N/A")
                    continue
                rc = d[recs]
                p = d[precs]
                if p + rc > 0.0:
                    f1 = 2 * p * rc / (p + rc)
                    bprint(f"   {c:<13}:           {f1:1.4f}")
                else:
                    bprint(f"   {c:<13}:           N/A")

        def calc_recall(
            right: int, wrong: int, rights: str, wrongs: str, recs: str
        ) -> None:
            """ Calculate precision for binary classification """
            # Recall
            if right + wrong == 0:
                result = "N/A"
                recall = 0.0
            else:
                recall = right / (right+wrong)
                result = f"{recall:1.4f}"
            bprint(f"\nRecall:                     {result}")
            for c in CATEGORIES:
                d = self._sentences[c]
                denominator = d[rights] + d[wrongs]
                if denominator == 0:
                    bprint(f"   {c:<13}:              N/A")
                else:
                    rc = d[recs] = d[rights] / denominator
                    bprint(f"   {c:<13}:           {rc:1.4f}")

        def calc_error_category_metrics(cat: str) -> CatResultDict:
            """ Calculates precision, recall and f1-score for a single error category
                N = Number of errors in category z in reference corpus, 
                Nall =  number of tokens
                TP = Errors correctly classified as category z
                FP = Errors (or non-errors) incorrectly classified as category z
                FN = Errors in category z in reference but not hypothesis
                Recall = TPz/(TPz+FPz)
                Precision = TPz/(TPz+FNz)
            """
            catdict: CatResultDict = { k: v for k, v in self._errtypes[cat].items() }
            tp = cast(int, catdict.get("tp", 0))
            fn = cast(int, catdict.get("fn", 0))
            fp = cast(int, catdict.get("fp", 0))
            recall: float = 0.0
            precision: float = 0.0
            catdict["freq"] = tp + fn
            if tp + fn + fp == 0:  # No values in category
                catdict["recall"] = "N/A"
                catdict["precision"] = "N/A"
                catdict["fscore"] = "N/A"
            else:
                # Recall
                if tp + fn != 0:
                    recall = catdict["recall"] = tp / (tp + fn)
                # Precision
                if tp + fp != 0:
                    precision = catdict["precision"] = tp / (tp + fp)
                if recall + precision > 0.0:
                    catdict["fscore"] = 2 * precision * recall / (precision + recall)
                else:
                    catdict["fscore"] = 0.0
                # Correction recall
                right_corr = cast(int, catdict.get("right_corr", 0))
                if right_corr > 0:
                    catdict["corr_rec"] = right_corr / (
                        right_corr + cast(int, catdict.get("wrong_corr", 0))
                    )
                else:
                    catdict["corr_rec"] = "N/A"
                # Span recall
                right_span = cast(int, catdict.get("right_span", 0))
                if right_span > 0:
                    catdict["span_rec"] = right_span / (
                        right_span + cast(int, catdict.get("wrong_span", 0))
                    )
                else:
                    catdict["span_rec"] = "N/A"
            return catdict

        def output_sentence_scores() -> None:
            """ Calculate and write sentence scores to stdout """
            ### SENTENCE SCORES
            # Fær sent inn TN, TP, FP og FN - er þá búin að summa það upp
            # Get sent inn aukagildi, sem segir t.d. hvort nota á F0,5 eða F1
            # Og hvað er verið að reikna?
            # Get líka sent inn texta sem segir hvað er verið að reikna,
            # s.s. "Results for Error detection" eða "Results for sentence correctness"?

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
            write_basic_value(false_positives, "false_positives",num_sentences)
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
            for c in CATEGORIES:
                d = self._sentences[c]
                num_sents = d["count"]
                true_results = cast(int, d["true_positives"] + d["true_negatives"])
                false_results = cast(int, d["false_positives"] + d["false_negatives"])
                if num_sents == 0:
                    result = "N/A"
                else:
                    result = f"{100.0*true_results/num_sents:3.2f}%/{100.0*false_results/num_sents:3.2f}%"
                bprint(f"   {c:<13}: {result:>16}")

            # Precision, recall, F1-score
            calc_PRF(true_positives, true_negatives, false_positives, false_negatives, 
                "true_positives", "true_negatives","false_positives", "false_negatives", 
                "sentrecall", "sentprecision")

            # Most common false negative error types
            total = sum(self._false_negatives.values())
            if total > 0:
                bprint(f"\nMost common false negative error types")
                bprint(f"--------------------------------------\n")
                for index, (xtype, cnt) in enumerate(
                    heapq.nlargest(20, self._false_negatives.items(), key=lambda x: x[1])
                ):
                    bprint(f"{index+1:3}. {xtype} ({cnt}, {100.0*cnt/total:3.2f}%)")

            # Most common error types in unparsable sentences
            tot = sum(self._tp_unparsables.values())
            if tot > 0:
                bprint(f"\nMost common error types for unparsable sentences")
                bprint(f"------------------------------------------------\n")
                for index, (xtype, cnt) in enumerate(
                    heapq.nlargest(20, self._tp_unparsables.items(), key=lambda x: x[1])
                ):
                    bprint(f"{index+1:3}. {xtype} ({cnt}, {100.0*cnt/tot:3.2f}%)")

        def output_token_scores() -> None:
            """ Calculate and write token scores to stdout """

            ### TOKEN ERROR SCORES

            bprint(f"\n\nResults for error detection within sentences")

            num_tokens = sum(
                cast(int, d["num_tokens"]) for d in self._sentences.values()
            )
            bprint(f"\nTokens processed:           {num_tokens:6}")
            for c in CATEGORIES:
                bprint(f"   {c:<13}:           {self._sentences[c]['num_tokens']:6}")

            tp = sum(cast(int, d["tp"]) for d in self._sentences.values())
            tn = sum(cast(int, d["tn"]) for d in self._sentences.values())
            fp = sum(cast(int, d["fp"]) for d in self._sentences.values())
            fn = sum(cast(int, d["fn"]) for d in self._sentences.values())

            all_ice_errs = tp+fn
            write_basic_value(tp, "tp", num_tokens, all_ice_errs)
            write_basic_value(tn, "tn", num_tokens)
            write_basic_value(fp, "fp", num_tokens, all_ice_errs)
            write_basic_value(fn, "fn", num_tokens, all_ice_errs)

            calc_PRF(tp, tn, fp, fn, "tp", "tn", "fp", "fn", "detectrecall", "detectprecision")

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
            
            calc_recall(right_corr, wrong_corr, "right_corr", "wrong_corr", "correctrecall")

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
            calc_recall(right_span, wrong_span, "right_span", "wrong_span", "spanrecall")

        def output_error_cat_scores() -> None:
            """ Calculate and write scores for each error category to stdout """
            bprint(f"\n\nResults for each error category in order by frequency")
            freqdict = defaultdict(float)
            micro : float = 0.0
            nfreqs : int  = 0
            microall : float = 0.0
            nfreqsall : int = 0

            # Iterate over category counts
            for cat in self._errtypes.keys():
                # Get recall, precision and F1; recall for correction and span
                calc_error_category_metrics(cat)

                # Collect  micro scores, both overall and for in-scope categories
                if cat not in OUT_OF_SCOPE:
                    micro += self._errtypes[cat]["fscore"]*self._errtypes[cat]["freq"]
                    nfreqs += self._errtypes[cat]["freq"]
                microall += self._errtypes[cat]["fscore"]*self._errtypes[cat]["freq"]
                nfreqsall += self._errtypes[cat]["freq"]

                # Create freqdict for sorting error categories by frequency
                freqdict[cat] = self._errtypes[cat]["freq"]

            # print results for each category by frequency
            for k in sorted(freqdict, key=freqdict.get, reverse=True):
                bprint("{} (in_scope={})".format(k, k not in OUT_OF_SCOPE))
                bprint("\tTP, FP, FN: {}, {}, {}".format(self._errtypes[k]["tp"], self._errtypes[k]["fp"], self._errtypes[k]["fn"]))
                bprint("\tRe, Pr, F1: {:3.2f}, {:3.2f}, {:3.2f}".format(self._errtypes[k]["recall"]*100.0, self._errtypes[k]["precision"]*100.0, self._errtypes[k]["fscore"]*100.0))
                bprint("\tCorr, span: {:3.2f}, {:3.2f}".format(self._errtypes[k]["corr_rec"]*100.0, self._errtypes[k]["span_rec"]*100.0))
           
            # Micro F1-score
            # Results for in-scope categories and all categories
            if nfreqs != 0:
                bprint("Micro F1-score: {:3.2f}  ({:3.2f})".format(micro/nfreqs*100.0, microall/nfreqsall*100.0))
            else:
                bprint(f"Micro F1-score: N/A")

        def output_supercategory_scores(errorcats: Dict[str, List[str]]) -> None:
            # Results for each SÍM category
            for entry, catlist in errorcats.items():
                micro : float = 0.0
                nfreqs : int  = 0
                microall : float = 0.0
                nfreqsall : int = 0
                correcs : float = 0.0
                correcsall : float = 0.0   
                # TODO taka saman corr_rec og span_rec; skoða hvernig fæ F-skor, svipað og fyrir hitt, þegar er ekki með TN inni
                bprint("\n{}:".format(entry.capitalize()))
                for cat in catlist:
                    et = self._errtypes[cat]
                    if et["fscore"] == "N/A":
                        continue
                    if cat not in OUT_OF_SCOPE:
                        micro += et["fscore"]*et["freq"]
                        if et["corr_rec"] != "N/A":
                            correcs += et["corr_rec"]*et["freq"]
                        nfreqs += et["freq"]
                        bprint("\t{}   {:3.2f}   {:3.2f}".format(cat, et["fscore"]*100, et["freq"]))
                    microall += et["fscore"]*et["freq"]
                    if et["corr_rec]"] != "N/A":
                        correcsall += et["corr_rec"]*et["freq"]
                    nfreqsall += et["freq"]
                if nfreqs != 0:
                    bprint("Micro F1-score: {:3.2f}  ({:3.2f})".format(micro/nfreqs*100.0, microall/nfreqsall*100.0))
                    bprint("Error correction recall: {:3.2f} ({:3.2f})".format(correcs/nfreqs*100.0, correcsall/nfreqsall*100.0))
                else:
                    bprint(f"Micro F1-score: N/A")
                    bprint(f"Error correction recall: N/A")

        output_duration()
        output_sentence_scores()
        output_token_scores()
        output_error_cat_scores()

        bprint(f"\n\nResults for iEC-categories:")
        output_supercategory_scores(SUPERCATEGORIES)
        bprint(f"\n\nResults for SÍM-categories:")
        output_supercategory_scores(SIMCATEGORIES)

        # Print the accumulated output before exiting
        for s in buffer:
            print(s)


def correct_spaces(tokens: List[Tuple[str, str]]) -> str:
    """ Returns a string with a reasonably correct concatenation
        of the tokens, where each token is a (tag, text) tuple. """
    return detokenize(
        Tok(TOK.PUNCTUATION if tag == "c" else TOK.WORD, txt, None)
        for tag, txt in tokens
    )

# Accumulate standard output in a buffer, for writing in one fell
# swoop at the end (after acquiring the output lock)
buffer: List[str] = []

def bprint(s: str):
    """ Buffered print: accumulate output for printing at the end """
    buffer.append(s)

def process(fpath_and_category: Tuple[str, str],) -> Dict[str, Any]:

    """ Process a single error corpus file in TEI XML format.
        This function is called within a multiprocessing pool
        and therefore usually executes in a child process, separate
        from the parent process. It should thus not modify any
        global state, and arguments and return values should be
        picklable. """

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
    # Counter of iceErrorCorpus error types (xtypes) encountered
    true_positives: Dict[str, int] = defaultdict(int)
    false_negatives: Dict[str, int] = defaultdict(int)
    # Counter of iceErrorCorpus error types in unparsable sentences
    ups: Dict[str, int] = defaultdict(int)
    # Stats for each error type (xtypes)
    errtypefreqs: ErrTypeStatsDict = ErrTypeStatsDict(TypeFreqs(int).copy)

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
            # Sentence identifier (index)
            index = sent.attrib.get("n", "")
            tokens: List[Tuple[str, str]] = []
            errors: List[ErrorDict] = []
            # A dictionary of errors by their index (idx field)
            error_indexes: Dict[str, ErrorDict] = {}
            dependencies: List[Tuple[str, ErrorDict]] = []
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
                        # We have 0 or more original tokens embedded within the revision tag
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
            
            # Reconstruct the original sentence
            # TODO switch for sentence from original text file
            text = correct_spaces(tokens)
            if not text:
                # Nothing to do: drop this and go to the next sentence
                continue
            
            # Pass it to GreynirCorrect
            pg = [list(p) for p in gc.check(text)]
            s: Optional[_Sentence] = None
            if len(pg) >= 1 and len(pg[0]) >= 1:
                s = pg[0][0]
            if len(pg) > 1 or (len(pg) == 1 and len(pg[0]) > 1):
                if QUIET:
                    bprint(f"In file {fpath}:")
                bprint(f"\n{index}: *** Input contains more than one sentence *** {text}")
            if s is None:
                if QUIET:
                    bprint(f"In file {fpath}:")
                bprint(f"\n{index}: *** No parse for sentence *** {text}")
                continue
            if not QUIET:
                # Output the original sentence
                bprint(f"\n{index}: {text}")
            if not index:
                if QUIET:
                    bprint(f"In file {fpath}:")
                bprint("000: *** Sentence identifier is missing ('n' attribute) ***")


            def sentence_results(hyp_annotations: List[gc.Annotation], ref_annotations: List[ErrorDict]) -> Tuple[bool, bool]:
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
                        bprint(f"<<< {err['start']:03}-{err['end']:03}: {asterisk}{xtype}")
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

            gc_error, ice_error = sentence_results(s.annotations, errors)

            def token_results(
                hyp_annotations: List[gc.Annotation], ref_annotations: List[ErrorDict]
            ) -> Tuple[int, int, int, int, int, int, int]:
                tp, fp, fn = 0, 0, 0  # tn comes from len(tokens)-(tp+fp+fn) later on
                right_corr, wrong_corr = 0, 0
                right_span, wrong_span = 0, 0

                x = (d for d in hyp_annotations) # GreynirCorrect annotations
                y = (l for l in ref_annotations) # iEC annotations
                
                xtok = None
                ytok = None
                try:
                    xtok = next(x)
                    ytok = next(y)
                    while True:

                        # 1. Error detection
                        xtoks = set(range(xtok.start, xtok.end + 1))
                        ytoks = set(
                            range(cast(int, ytok["start"]), cast(int, ytok["end"]) + 1)
                        )
                        ytype = cast(str, ytok["xtype"])
                        if xtoks & ytoks:
                            tp += 1
                            errtypefreqs[ytype]["tp"] += 1
                            # 2. Span detection
                            if xtoks == ytoks:
                                right_span += 1
                                errtypefreqs[ytype]["right_span"] += 1
                            else:
                                wrong_span += 1
                                errtypefreqs[ytype]["wrong_span"] += 1
                            # 3. Error correction
                            # Get the 'corrected' attribute if available,
                            # otherwise use xtok['suggest']
                            xcorr = getattr(xtok, "corrected", xtok.suggest)
                            if xcorr == ytok["corrected"]:
                                right_corr += 1
                                errtypefreqs[ytype]["right_corr"] += 1
                            else:
                                wrong_corr += 1
                                errtypefreqs[ytype]["wrong_corr"] += 1
                            xtok = next(x)
                            ytok = next(y)
                        else:
                            if xtok.start < ytok["start"]:
                                fp += 1
                                errtypefreqs[ytype]["fp"] += 1
                                xtok = next(x)
                            elif xtok.start > ytok["start"]:
                                ytok = next(y)
                                fn += 1
                                errtypefreqs[ytype]["fn"] += 1
                            else:
                                xtok = next(x)
                                ytok = next(y)
                                fp += 1
                                fn += 1
                                errtypefreqs[ytype]["fn"] += 1
                except StopIteration:
                    pass
                if xtok and not ytok:  # Because of exception to try
                    pass
                    # false positive
                if ytok and not xtok:  # Because of exception to try
                    pass
                    # false negative
                return tp, fp, fn, right_corr, wrong_corr, right_span, wrong_span

            tp, fp, fn, right_corr, wrong_corr, right_span, wrong_span = token_results(s.annotations, errors)
            tn: int = len(tokens) - tp - fp - fn
            # Collect statistics into the stats list, to be returned
            # to the parent process
            if stats is not None:
                stats.append((category, len(tokens), ice_error, gc_error, tp, tn, fp, fn, right_corr, wrong_corr, right_span, wrong_span))

    except ET.ParseError:
        # Already handled the exception: exit as gracefully as possible
        pass

    finally:
        # Print the accumulated output before exiting
        with OUTPUT_LOCK:
            for s in buffer:
                print(s)
            if not QUIET:
                print("", flush=True)

    # This return value will be pickled and sent back to the parent process
    return dict(
        stats=stats,
        true_positives=true_positives,
        false_negatives=false_negatives,
        ups=ups,
        errtypefreqs=errtypefreqs,
    )


def main() -> None:
    """ Main program """
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

    # Maximum number of files to process (0=all files)
    max_count = args.number
    # Initialize the statistics collector
    stats = Stats()
    # The glob path of the XML files to process
    path = args.path
    # When running measurements only, we use _TEST_PATH as the default,
    # otherwise _DEV_PATH
    if path is None:
        path = _TEST_PATH if args.measure else _DEV_PATH

    def gen_files() -> Iterable[Tuple[str, str]]:
        """ Generate tuples with the file paths and categories
            to be processed by the multiprocessing pool """
        count = 0
        it: Iterable[str]
        if args.randomize and max_count > 0:
            # Randomizing only makes sense if there is a max count as well
            it = glob.glob(path, recursive=True)
            it = random.sample(it, max_count)
        else:
            it = glob.iglob(path, recursive=True)
        for fpath in it:
            # Find out which category the file belongs to by
            # inference from the file name
            for category in CATEGORIES:
                if category in fpath:
                    break
            else:
                assert (
                    False
                ), f"File path does not contain a recognized category: {fpath}"
            # Add the file to the statistics under its category
            stats.add_file(category)
            # Yield the file information to the multiprocessing pool
            yield fpath, category
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