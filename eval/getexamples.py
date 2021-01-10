#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Collection of error category examples

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


    The program reads a development set of hand-annotated texts in
    TEI XML format and collects all examples for each error category.

    A way to configure this program is to clone the iceErrorCorpus
    repository (from the above path) into a separate directory, and
    then place a symlink to it to the /eval directory. For example:

    $ cd github
    $ git clone https://github.com/antonkarl/iceErrorCorpus
    $ cd GreynirCorrect/eval
    $ ln -s ../../iceErrorCorpus/ .

"""

from collections import defaultdict
import xml.etree.ElementTree as ET
import glob
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
from tokenizer import detokenize, Tok, TOK

from eval import OUT_OF_SCOPE

# Default glob path of the development corpus TEI XML files to be processed
_DEV_PATH = "iceErrorCorpus/data/**/*.xml"

# Default glob path of the test corpus TEI XML files to be processed
_TEST_PATH = "iceErrorCorpus/testCorpus/**/*.xml"

CATS: DefaultDict[str, List[str]] = defaultdict(list)


def element_text(element: ET.Element) -> str:
    """ Return the text of the given element,
        including all its subelements, if any """
    return "".join(element.itertext())


def correct_spaces(tokens: Iterable[Tuple[str, str]]) -> str:
    """ Returns a string with a reasonably correct concatenation
        of the tokens, where each token is a (tag, text) tuple. """
    return detokenize(
        Tok(TOK.PUNCTUATION if tag == "c" else TOK.WORD, txt, None)
        for tag, txt in tokens
    )


def get_examples(fpath: str) -> None:

    """ Process a single error corpus file in TEI XML format """

    # Set up XML namespace stuff
    NS = "http://www.tei-c.org/ns/1.0"
    # Length of namespace prefix to cut from tag names, including { }
    nl = len(NS) + 2
    # Namespace dictionary to be passed to ET functions
    ns = dict(ns=NS)

    # Accumulate standard output in a buffer, for writing in one fell
    # swoop at the end (after acquiring the output lock)
    buffer: List[str] = []

    def bprint(s: str):
        """ Buffered print: accumulate output for printing at the end """
        buffer.append(s)

    try:
        # Parse the XML file into a tree
        try:
            tree = ET.parse(fpath)
        except ET.ParseError as e:
            bprint(f"000: *** Unable to parse XML file ***")
            raise e
        # Obtain the root of the XML tree
        root = tree.getroot()
        # Iterate through the sentences in the file
        for sent in root.findall("ns:text/ns:body/ns:p/ns:s", ns):
            tokens: List[Tuple[str, str]] = []
            errors = []
            # A dictionary of errors by their index (idx field)
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
                        error: DefaultDict[str, Union[int, bool, str]] = defaultdict(
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
                else:
                    tokens.append((tag, element_text(el)))

            # Reconstruct the original sentence
            # TODO switch for sentence from original text file
            text = correct_spaces(tokens)
            if not text:
                # Nothing to do: drop this and go to the next sentence
                continue
            for item in errors:
                xtype = cast(str, item["xtype"])
                CATS[xtype].append(
                    "{}\t{}-{}\t{}\t{}\t{}\n".format(
                        text,
                        item["start"],
                        item["end"],
                        item["in_scope"],
                        item["original"],
                        item["corrected"],
                    )
                )

    except ET.ParseError:
        # Already handled the exception: exit as gracefully as possible
        pass


if __name__ == "__main__":
    it = glob.iglob(_DEV_PATH, recursive=True)
    for fpath in it:
        get_examples(fpath)

    for xtype in CATS:
        with open("examples/" + xtype + ".txt", "w") as myfile:
            for example in CATS[xtype]:
                myfile.write(example)
