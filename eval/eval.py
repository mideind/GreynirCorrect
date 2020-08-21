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

    A normal way to configure this program is to clone the iceErrorCorpus
    repository (from the above path) into a separate directory, and
    then place a symlink to it to the /eval directory. For example:

    $ cd github
    $ git clone https://github.com/antonkarl/iceErrorCorpus
    $ cd GreynirCorrect/eval
    $ ln -s ../../iceErrorCorpus/ .
    $ python eval.py

    An alternate method is to specify the glob path to the error corpus as an
    argument to eval.py:

    $ python eval.py ~/github/iceErrorCorpus/data/**/*.xml

"""

from typing import Dict, List, Optional, Union

from collections import defaultdict
from datetime import datetime
import glob
import argparse
import xml.etree.ElementTree as ET

import reynir_correct as gc


# The type of a single error descriptor, extracted from a TEI XML file
ErrorDict = Dict[str, Union[str, int, bool]]

# Content categories, embedded within the file paths
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

# Default glob path of the development set TEI XML files to be processed
_DEV_PATH = 'iceErrorCorpus/data/**/*.xml'

# Default glob path of the test set TEI XML files to be processed
_TEST_PATH = 'iceErrorCorpus/testCorpus/**/*.xml'

# Define the command line arguments

parser = argparse.ArgumentParser(
    description="Evaluates spelling and grammar checking performance"
)

parser.add_argument(
    'path',
    nargs='?',
    type=str,
    default=_DEV_PATH,
    help=f"glob path of XML files to process (default: {_DEV_PATH})",
)

parser.add_argument(
    "-n", "--number",
    type=int,
    default=10,
    help="number of files to process (0=all, default: 10)")


def element_text(element: ET.Element) -> str:
    """ Return the text of the given element, including all its subelements, if any """
    return "".join(element.itertext())


class Stats:

    """ A container for key statistics on processed files and sentences """

    def __init__(self) -> None:
        """ Initialize empty defaults for the stats collection """
        self._starttime = datetime.utcnow()
        self._files: Dict[str, int] = defaultdict(int)
        self._sentences: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

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

    def output(self) -> None:
        """ Write the statistics to stdout """
        # Calculate the duration of the processing
        dur = int((datetime.utcnow() - self._starttime).total_seconds())
        h = dur // 3600
        m = (dur % 3600) // 60
        s = (dur % 60)
        # Output a summary banner
        print("\n\n" + "=" * 7)
        print("Summary")
        print("=" * 7 + "\n")
        # Total number of files processed, and timing stats
        print(f"Processing started at {str(self._starttime)[0:19]}")
        print(f"Total processing time {h}h {m:02}m {s:02}s")
        print(f"Files processed:            {sum(self._files.values()):6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._files[c]:5}")
        # Total number of tokens processed
        num_tokens = sum(d["num_tokens"] for d in self._sentences.values())
        print(f"Tokens processed:           {num_tokens:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['num_tokens']:5}")
        # Total number of sentences processed
        num_sentences = sum(d["count"] for d in self._sentences.values())
        print(f"Sentences processed:        {num_sentences:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['count']:5}")
        # Total number of true negatives found
        true_negatives = sum(d["true_negatives"] for d in self._sentences.values())
        print(f"True negatives:             {true_negatives:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['true_negatives']:5}")
        # Total number of true positives found
        true_positives = sum(d["true_positives"] for d in self._sentences.values())
        print(f"True positives:             {true_positives:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['true_positives']:5}")
        # Total number of false negatives found
        false_negatives = sum(d["false_negatives"] for d in self._sentences.values())
        print(f"False negatives:            {false_negatives:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['false_negatives']:5}")
        # Total number of false positives found
        false_positives = sum(d["false_positives"] for d in self._sentences.values())
        print(f"False positives:            {false_positives:6}")
        for c in CATEGORIES:
            print(f"   {c:<13}:            {self._sentences[c]['false_positives']:5}")
        # Percentage of true vs. false
        true_results = true_positives + true_negatives
        false_results = false_positives + false_negatives
        if num_sentences == 0:
            result = "N/A"
        else:
            result = f"{100.0*true_results/num_sentences:3.2f}%/{100.0*false_results/num_sentences:3.2f}%"
        print(f"True/false split: {result:>16}")
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


def process(category: str, fpath: str, stats: Stats=None) -> None:
    """ Process a single error corpus file in TEI XML format """
    NS = "http://www.tei-c.org/ns/1.0"
    # Length of namespace prefix to cut from tag names, including { }
    nl = len(NS) + 2
    # Namespace dictionary to be passed to ET functions
    ns = dict(ns=NS)
    # Parse the XML file into a tree
    tree = ET.parse(fpath)
    # Obtain the root of the XML tree
    root = tree.getroot()
    # Output a file header
    print("\n" + "-" * 64)
    print(f"File: {fpath}")
    print("-" * 64)
    # Iterate through the sentences in the file
    for sent in root.findall("ns:text/ns:body/ns:p/ns:s", ns):
        # Sentence index
        index = int(sent.attrib.get("n", 0))
        tokens: List[str] = []
        errors: List[ErrorDict] = []
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
                        eid=attr["eid"],
                        original=original,
                        corrected=corrected,
                    )
                    errors.append(error)
            else:
                tokens.append(element_text(el))
        # Reconstruct the original sentence as a shallow-tokenized text string
        text = " ".join(tokens).strip()
        if not text:
            # Nothing to do: drop this and go to the next sentence
            continue
        # Pass it to GreynirCorrect
        try:
            s = gc.check_single(text)
        except StopIteration:
            print(f"\n{index:03}: *** No parse for sentence *** {text}")
            continue
        # Output the original sentence
        print(f"\n{index:03}: {text}")
        if index == 0:
            print("000: *** Sentence index is missing ('n' attribute) ***")
        gc_error = False
        ice_error = False
        # Output GreynirCorrect annotations
        for ann in s.annotations:
            if ann.is_error:
                gc_error = True
            print(f">>> {ann}")
        # Output iceErrorCorpus annotations
        for err in errors:
            asterisk = "*"
            if err["in_scope"]:
                asterisk = ""
                ice_error = True
            print(f"<<< {err['start']:03}-{err['end']:03}: {asterisk}{err['xtype']}")
        # Output true/false positive/negative result
        if ice_error and gc_error:
            print("=++ True positive")
        elif not ice_error and not gc_error:
            print("=-- True negative")
        elif ice_error and not gc_error:
            print("!-- False negative")
        else:
            assert gc_error and not ice_error
            print("!++ False positive")
        # Collect statistics
        if stats is not None:
            stats.add_sentence(category, len(tokens), ice_error, gc_error)


def main() -> None:
    """ Main program """
    # Parse the command line arguments
    args = parser.parse_args()
    # Count the processed files
    count = 0
    max_count = args.number
    # Initialize the statistics collector
    stats = Stats()
    # Process each TEI XML file in turn
    for fpath in glob.iglob(args.path, recursive=True):
        # Find out which category the file belongs to by
        # inference from the file name
        for category in CATEGORIES:
            if category in fpath:
                break
        else:
            assert False, f"File path does not contain a recognized category: {fpath}"
        stats.add_file(category)
        process(category, fpath, stats)
        count += 1
        if max_count > 0 and count >= max_count:
            break
    stats.output()
    print()


if __name__ == "__main__":
    main()
