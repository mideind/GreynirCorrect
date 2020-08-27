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

"""

from typing import Dict, List, Optional, Union, Tuple, Iterable, cast

import os
from collections import defaultdict
from datetime import datetime
import glob
import argparse
import xml.etree.ElementTree as ET
import multiprocessing

import reynir_correct as gc


# The type of a single error descriptor, extracted from a TEI XML file
ErrorDict = Dict[str, Union[str, int, bool]]

# The type of the dict that holds statistical information about sentences
# within a particular content category
SentenceStatsDict = Dict[str, Union[float, int]]

# The type of the dict that holds statistical information about
# content categories
CategoryStatsDict = Dict[str, SentenceStatsDict]

# Create a lock to ensure that only one process outputs at a time
OUTPUT_LOCK = multiprocessing.Lock()

# Content categories in iceErrorCorpus, embedded within the file paths
CATEGORIES = (
    "essays", "onlineNews", "wikipedia",
)

# Error codes in iceErrorCorpus that are considered out of scope
# for GreynirCorrect, at this stage at least
OUT_OF_SCOPE = {
    "agreement-pro",        # samræmi fornafns við undanfara	grammar	...vöðvahólf sem sé um dælinguna. Hann dælir blóðinu > Það dælir blóðinu
    "ind4def",              # óákveðið fyrir ákveðið	grammar	gítartakta > gítartaktana
    "def4ind",              # ákveðið fyrir óákveðið	grammar	skákinni > skák
    "missing-word",         # orð vantar	omission	í Donalda > í þorpinu Donalda
    "missing-words",        # fleiri en eitt orð vantar	omission	því betri laun > því betri laun hlýtur maður
    "missing-commas",       # kommur vantar utan um innskot	punctuation	Hún er jafn verðmæt ef ekki verðmætari en háskólapróf > Hún er verðmæt, ef ekki verðmætari, en háskólapróf
    "missing-conjunction",  # samtengingu vantar	punctuation	í Noregi suður að Gíbraltarsundi > í Noregi og suður að Gíbraltarsundi
    "punctuation",          # greinarmerki	punctuation	hún mætti og hann var ekki tilbúinn > hún mætti en hann var ekki tilbúinn
    "extra-punctuation",    # auka greinarmerki	punctuation	... að > að
    "extra-comma",          # auka komma	punctuation	stríð, við náttúruna > stríð við náttúruna
    "extra-period",         # auka punktur	punctuation	á morgun. Og ... > á morgun og...
    "period4comma",         # punktur fyrir kommu	punctuation	meira en áður. Hella meira í sig > meira en áður, hella meira í sig
    "period4conjunction",   # punktur fyrir samtengingu	punctuation	...maður vill gera. Vissulega > ...maður vill gera en vissulega
    "conjunction4period",   # samtenging fyrir punkt	punctuation	...tónlist ár hvert og tónlistarstefnurnar eru orðnar... > ...tónlist ár hvert. Tónlistarstefnurnar eru orðnar...
    "conjunction4comma",    # samtenging fyrir kommu	punctuation	...geta orðið þröngvandi og erfitt getur verið... > ...geta orðið þröngvandi, erfitt getur verið...
    "comma4conjunction",    # komma fyrir samtengingu	punctuation	...fara með vald Guðs, öll löggjöf byggir... > ...fara með vald Guðs og öll löggjöf byggir...
    "comma4ex",             # komma fyrir upphrópun	punctuation	Viti menn, almúginn... > Viti menn! Almúginn...
    "period4ex",            # punktur fyrir upphrópun	punctuation	Viti menn. > Viti menn!
    "zzz",                  # to revisit	unannotated	
    "xxx",                  # unclassified	unclassified	
    "extra-word",           # orði ofaukið	insertion	augun á mótherja > augu mótherja
    "extra-words",          # orðum ofaukið	insertion	...ég fer að hugsa... > ...ég hugsa...
    "wording",              # orðalag	wording	...gerðum allt í raun... > ...gerðum í raun allt...
    "aux",                  # meðferð vera og verða, hjálparsagna	wording	mun verða eftirminnilegt > mun vera eftirminnilegt
    "foreign-error",        # villa í útlendu orði	foreign	Supurbowl > Super Bowl
    "gendered",             # kynjað mál, menn fyrir fólk	exclusion	menn hugsa oft > fólk hugsar oft
    "style",                # stíll	style	urðu ekkert frægir > urðu ekki frægir
    "unicelandic",          # óíslenskuleg málnotkun	style	...fer eftir persónunni... > ...fer eftir manneskjunni...
    "missing-space",        # vantar bil	spacing	eðlis-og efnafræði > eðlis- og efnafræði
    "extra-space",          # bili ofaukið	spacing	4 . > 4.
    "loan-syntax",          # lánuð setningagerð	style	ég vaknaði upp > ég vaknaði
    "noun4pro",             # nafnorð í stað fornafns	grammar	menntun má nálgast > hana má nálgast
    "pro4noun",             # fornafn í stað nafnorðs	grammar	þau voru spurð > parið var spurt
    "reflexive4noun",       # afturbeygt fornafn í stað nafnorðs	grammar	félagið hélt aðalfund þess > félagið hélt aðalfund sinn
    "pro4reflexive",        # nafnorð í stað afturbeygðs fornafns	grammar	gefur orku til fólks í kringum það > gefur orku til fólks í kringum sig
    "pres4past",            # sögn í nútíð í stað þátíðar	grammar	Þeir fara út > Þeir fóru út
    "past4pres",            # sögn í þátíð í stað nútíðar	grammar	þegar hún leigði spólur > þegar hún leigir spólur
    "pro4reflexive",        # persónufornafn í stað afturbeygðs fn.	grammar	Fólk heldur að það geri það hamingjusamt > Fólk heldur að það geri sig hamingjusamt
    "reflexive4pro",        # afturbeygt fornafn í stað persónufornafns	grammar	gegnum líkama sinn > gegnum líkama hans
    "missing-ex",           # vantar upphrópunarmerki	punctuation	Viti menn ég komst af > Viti menn! Ég komst af
    "qm4ex",                # spurningarmerki fyrir upphrópun	punctuation	Algjört hrak sjálf? > Algjört hrak sjálf!
    "pers4dem",             # persónufornafn í staðinn fyrir ábendingarf.	grammar	það > þetta
    "dem-pro",              # hinn í stað fyrir sá; sá ekki til eða ofnotað	grammar	hinn > sá
    "indef-pro",            # óákveðið fornafn	grammar	enginn > ekki neinn
    "context",              # rangt orð í samhengi	other	
    "fw4ice",               # erlent orð þýtt yfir á íslensku	style	Elba > Saxelfur
    "bracket4square",       # svigi fyrir hornklofa	punctuation	(Portúgal) > [Portúgal]
    "square4bracket",       # hornklofi fyrir sviga	punctuation	[börnin] > (börnin)
    "missing-semicolon",    # vantar semíkommu	punctuation	Haukar Björgvin Páll > Haukar; Björgvin Páll
    "dem4pers",             # ábendingarfornafn í stað persónufornafns	grammar	þessi > hún
    "simple4cont",          # nútíð í stað vera að + nafnh.	grammar	ók > var að aka
    "nonit4it",             # óskáletrað fyrir skáletrað		orðið qibt > orðið qibt
    "comma4dash",           # komma fyrir bandstrik	punctuation	, > -
    "dash4semicolon",       # bandstrik fyrir semíkommu	punctuation	núna - þetta > núna; þetta
    "it4nonit",             # skáletrað fyrir óskáletrað		Studdi Isma'il > Studdi Isma'il
    "extra-symbol",         # tákn ofaukið	other	Dalvík + gaf... > Dalvík gaf...
    "missing-symbol",       # tákn vantar	punctuation	0 > 0%
    "extra-number",         # tölustöfum ofaukið	other	139,0 > 139
    "missing-square",       # vantar hornklofi	punctuation	þeir > [þeir]
    "dem4noun",             # ábendingarfornafn í stað nafnorðs	grammar	hinn > maðurinn
    "noun4dem",             # nafnorð í stað ábendingarfornafns	grammar	stærsta klukkan > sú stærsta
    "ice4fw",               # íslenskt orð notað í stað erlends		Demókrata öldungarþings herferðarnefndina > Democratic Senatorial Campaign Committee
    "upper4lower-proper",   # stór stafur í sérnafni þar sem hann á ekki að vera	capitalization	Mál og Menning > Mál og menning
    "collocation",          # fast orðasamband	collocation	fram á þennan dag > fram til þessa dags
    "collocation-idiom",    # fast orðasamband með ógagnsæja merkingu	collocation	hélt hvorki vindi né vatni > hélt hvorki vatni né vindi
    "ind4sub",              # framsöguháttur fyrir vh.	grammar	Þrátt fyrir að konfúsíanismi er upprunninn > Þrátt fyrir að konfúsíanismi sé upprunninn
    "sub4ind",              # viðtengingarh. fyrir fh.	grammar	Stjórnvöld vildu auka rétt borgara og geri þeim kleift > Stjórnvöld vildu auka rétt borgara og gera þeim kleift
    "comma4period",         # komma fyrir punkt	punctuation	...kynnast nýju fólki, er á þrítugsaldri > ...kynnast nýju fólki. Hann er á þrítugsaldri
    "comma4qm",             # komma fyrir spurningarmerki	punctuation	Höfum við réttinn, eins og að... > Höfum við réttinn? Eins og að...
    "missing-quot",         # gæsalöpp vantar	punctuation	„I'm winning > „I'm winning“
    "missing-quots",        # gæsalappir vantar	punctuation	I'm winning > „I'm winning“
}

# Default glob path of the development corpus TEI XML files to be processed
_DEV_PATH = 'iceErrorCorpus/data/**/*.xml'

# Default glob path of the test corpus TEI XML files to be processed
_TEST_PATH = 'iceErrorCorpus/testCorpus/**/*.xml'

# Define the command line arguments

parser = argparse.ArgumentParser(
    description=(
        "This program evaluates the spelling and grammar checking performance "
        "of GreynirCorrect on iceErrorCorpus"
    )
)

parser.add_argument(
    'path',
    nargs='?',
    type=str,
    help=f"glob path of XML files to process (default: {_DEV_PATH})",
)

parser.add_argument(
    "-n", "--number",
    type=int,
    default=0,
    help="number of files to process (default=all)",
)

parser.add_argument(
    "-c", "--cores",
    type=int,
    help=f"number of CPU cores to use (default=all, i.e. {os.cpu_count() or 1})",
)

parser.add_argument(
    "-m", "--measure",
    action="store_true",
    help="run measurements on test corpus and output results only",
)

parser.add_argument(
    "-q", "--quiet",
    default=None,
    action="store_true",
    help="output results only, not individual sentences",
)

parser.add_argument(
    "-v", "--verbose",
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
        self._sentences: CategoryStatsDict = defaultdict(lambda: defaultdict(int))

    def add_file(self, category: str) -> None:
        """ Add a processed file in a given content category """
        self._files[category] += 1

    def add_sentence(
        self,
        category: str, num_tokens: int,
        ice_error: bool, gc_error: bool
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

    def output(self, cores: int) -> None:
        """ Write the statistics to stdout """
        # Calculate the duration of the processing
        dur = int((datetime.utcnow() - self._starttime).total_seconds())
        h = dur // 3600
        m = (dur % 3600) // 60
        s = (dur % 60)
        # Output a summary banner
        print("\n" + "=" * 7)
        print("Summary")
        print("=" * 7 + "\n")
        # Total number of files processed, and timing stats
        print(f"Processing started at {str(self._starttime)[0:19]}")
        print(f"Total processing time {h}h {m:02}m {s:02}s, using {cores} cores")
        print(f"\nFiles processed:            {sum(self._files.values()):6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._files[c]:6}")
        # Total number of tokens processed
        num_tokens = sum(d["num_tokens"] for d in self._sentences.values())
        print(f"\nTokens processed:           {num_tokens:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['num_tokens']:6}")
        # Total number of sentences processed
        num_sentences = sum(d["count"] for d in self._sentences.values())
        print(f"\nSentences processed:        {num_sentences:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['count']:6}")

        def perc(n):
            """ Return a percentage of total sentences, formatted as 3.2f """
            if num_sentences == 0:
                return "N/A"
            return f"{100.0*n/num_sentences:3.2f}"

        # Total number of true negatives found
        true_negatives = sum(d["true_negatives"] for d in self._sentences.values())
        print(f"\nTrue negatives:             {true_negatives:6} {perc(true_negatives):>6}%")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['true_negatives']:6}")

        # Total number of true positives found
        true_positives = sum(d["true_positives"] for d in self._sentences.values())
        print(f"\nTrue positives:             {true_positives:6} {perc(true_positives):>6}%")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['true_positives']:6}")

        # Total number of false negatives found
        false_negatives = sum(d["false_negatives"] for d in self._sentences.values())
        print(f"\nFalse negatives:            {false_negatives:6} {perc(false_negatives):>6}%")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['false_negatives']:6}")

        # Total number of false positives found
        false_positives = sum(d["false_positives"] for d in self._sentences.values())
        print(f"\nFalse positives:            {false_positives:6} {perc(false_positives):>6}%")
        for c in CATEGORIES:
            print(f"   {c:<13}:           {self._sentences[c]['false_positives']:6}")

        # Percentage of true vs. false
        true_results = true_positives + true_negatives
        false_results = false_positives + false_negatives
        if num_sentences == 0:
            result = "N/A"
        else:
            result = f"{100.0*true_results/num_sentences:3.2f}%/{100.0*false_results/num_sentences:3.2f}%"
        print(f"\nTrue/false split: {result:>16}")
        for c in CATEGORIES:
            d = self._sentences[c]
            num_sentences = d["count"]
            true_results = d["true_positives"] + d["true_negatives"]
            false_results = d["false_positives"] + d["false_negatives"]
            if num_sentences == 0:
                result = "N/A"
            else:
                result = f"{100.0*true_results/num_sentences:3.2f}%/{100.0*false_results/num_sentences:3.2f}%"
            print(f"   {c:<13}: {result:>16}")

        # Recall
        recall = true_positives / (true_positives + false_negatives)
        print(f"\nRecall:                     {recall:1.4f}")
        for c in CATEGORIES:
            d = self._sentences[c]
            denominator = d["true_positives"] + d["false_negatives"]
            if denominator == 0:
                print(f"   {c:<13}:              N/A")
            else:
                rc = d["recall"] = d["true_positives"] / denominator
                print(f"   {c:<13}:           {rc:1.4f}")

        # Precision
        precision = true_positives / (true_positives + false_positives)
        print(f"\nPrecision:                  {precision:1.4f}")
        for c in CATEGORIES:
            d = self._sentences[c]
            denominator = d["true_positives"] + d["false_positives"]
            if denominator == 0:
                print(f"   {c:<13}:              N/A")
            else:
                p = d["precision"] = d["true_positives"] / denominator
                print(f"   {c:<13}:           {p:1.4f}")

        # F1 score
        f1 = 2 * precision * recall / (precision + recall)
        print(f"\nF1 score:                   {f1:1.4f}")
        for c in CATEGORIES:
            d = self._sentences[c]
            if "recall" not in d or "precision" not in d:
                print(f"   {c:<13}:              N/A")
                continue
            rc = d["recall"]
            p = d["precision"]
            f1 = 2 * p * rc / (p + rc)
            print(f"   {c:<13}:           {f1:1.4f}")


def process(
    fpath_and_category: Tuple[str, str],
) -> List[Tuple]:

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

    # Statistics about processed sentences. This list will
    # be returned back to the parent process.
    stats: List[Tuple] = []

    # Accumulate standard output in a buffer, for writing in one fell
    # swoop at the end (after acquiring the output lock)
    buffer: List[str] = []

    def bprint(s: str):
        """ Buffered print: accumulate output for printing at the end """
        buffer.append(s)

    try:

        if not QUIET:
            # Output a file header
            bprint("-" * 64)
            bprint(f"File: {fpath}")
            bprint("-" * 64)
        # Parse the XML file into a tree
        try:
            tree = ET.parse(fpath)
        except ET.ParseError:
            if QUIET:
                bprint(f"000: *** Unable to parse XML file {fpath} ***")
            else:
                bprint(f"000: *** Unable to parse XML file ***")
            return stats
        # Obtain the root of the XML tree
        root = tree.getroot()
        # Iterate through the sentences in the file
        for sent in root.findall("ns:text/ns:body/ns:p/ns:s", ns):
            # Sentence identifier (index)
            index = sent.attrib.get("n", "")
            tokens: List[str] = []
            errors: List[ErrorDict] = []
            # A dictionary of errors by their index (idx field)
            error_indexes: Dict[str, ErrorDict] = {}
            dependencies: List[Tuple[str, ErrorDict]] = []
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
                        orig_tokens = [element_text(subel) for subel in el_orig]
                        tokens.extend(orig_tokens)
                        original = " ".join(orig_tokens).strip()
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
                        xtype = attr["xtype"].lower()
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
                                bprint(f"\n{index}: *** 'depId' attribute missing for dependency ***")
                else:
                    tokens.append(element_text(el))
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
            # !!! TODO: this actually fixes spacing errors, causing them
            # not to be reported by GreynirCorrect.
            text = gc.correct_spaces(" ".join(tokens))
            if not text:
                # Nothing to do: drop this and go to the next sentence
                continue
            # Pass it to GreynirCorrect
            s = gc.check_single(text)
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
            gc_error = False
            ice_error = False
            # Output GreynirCorrect annotations
            for ann in s.annotations:
                if ann.is_error:
                    gc_error = True
                if not QUIET:
                    bprint(f">>> {ann}")
            # Output iceErrorCorpus annotations
            for err in errors:
                asterisk = "*"
                if err["in_scope"]:
                    # This is an in-scope error
                    asterisk = ""
                    ice_error = True
                if not QUIET:
                    bprint(f"<<< {err['start']:03}-{err['end']:03}: {asterisk}{err['xtype']}")
            if not QUIET:
                # Output true/false positive/negative result
                if ice_error and gc_error:
                    bprint("=++ True positive")
                elif not ice_error and not gc_error:
                    bprint("=-- True negative")
                elif ice_error and not gc_error:
                    bprint("!-- False negative")
                else:
                    assert gc_error and not ice_error
                    bprint("!++ False positive")
            # Collect statistics into the stats list, to be returned
            # to the parent process
            if stats is not None:
                stats.append((category, len(tokens), ice_error, gc_error))

    finally:
        # Print the accumulated output before exiting
        with OUTPUT_LOCK:
            for s in buffer:
                print(s)
            if not QUIET:
                print("", flush=True)

    # This return value will be pickled and sent back to the parent process
    return stats


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
        for fpath in glob.iglob(path, recursive=True):
            # Find out which category the file belongs to by
            # inference from the file name
            for category in CATEGORIES:
                if category in fpath:
                    break
            else:
                assert False, f"File path does not contain a recognized category: {fpath}"
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
            # Results come back as lists of arguments (tuples) that
            # we pass to Stats.add_sentence()
            for sent_result in result:
                stats.add_sentence(*sent_result)
        # Done: close the pool in an orderly manner
        pool.close()
        pool.join()

    # Finally, acquire the output lock and write the final statistics
    with OUTPUT_LOCK:
        stats.output(cores=args.cores or os.cpu_count() or 1)
        print("", flush=True)


if __name__ == "__main__":
    main()
