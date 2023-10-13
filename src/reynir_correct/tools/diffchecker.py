#!/usr/bin/env python

"""
Easy way to look at input and output together.
To run for the file 'prufa.txt':
$ python diffchecker.py prufa.txt

"""
from typing import Dict, Iterable, Iterator, Set, Union

import argparse
import sys

from reynir_correct.wrappers import check_errors

# File types for UTF-8 encoded text files
ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

# Define the command line arguments
parser = argparse.ArgumentParser(description="Corrects Icelandic text")

parser.add_argument(
    "infile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to correct",
)


def gen(f: Iterator[str]) -> Iterable[str]:
    """Generate the lines of text in the input file"""
    yield from f


def main() -> None:
    options: Dict[str, Union[str, bool, Set[str]]] = {}
    options["format"] = "text"  # text, json, csv, m2
    options["annotations"] = True
    options["all_errors"] = True
    # options["input"] = open("prufa.txt", "r")
    options["one_sent"] = False
    options["generate_suggestion_list"] = True
    options["ignore_comments"] = True  # Only used here
    options["annotate_unparsed_sentences"] = True
    options["suppress_suggestions"] = False
    options["replace_html_escapes"] = True
    options["ignore_wordlist"] = set()
    options["spaced"] = False
    options["print_all"] = True
    options["ignore_rules"] = set()
    options["suggest_not_correct"] = False
    options["extra_config"] = "config/ExtraWords.conf"
    options["extra_tone_of_voice"] = True

    args = parser.parse_args()
    infile = args.infile
    if infile is sys.stdin and sys.stdin.isatty():
        # terminal input is empty, most likely no value was given for infile:
        # Nothing we can do
        print("No input has been given, nothing can be returned")
        sys.exit(1)
    itering = gen(infile)
    for sent in itering:
        sent = sent.strip()
        if not sent:
            continue
        if sent.startswith("#"):
            # Comment string, want to show it with the examples
            if not options.get("ignore_comments"):
                print(sent)
            continue
        options["input"] = sent
        x = check_errors(**options)
        # Here we can compare x to gold by zipping sentences
        # from output and gold together and iterating in a for loop
        print(sent.strip())
        if x:
            print(x)
        print("=================================")
    """
    # Used when output shouldn't be split into sentences
    options["input"] = infile
    x = check_errors(**options)
    if x:
        print(x)
    print("=================================")
    """


if __name__ == "__main__":
    main()
