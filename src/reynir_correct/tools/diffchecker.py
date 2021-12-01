#!/usr/bin/env python

"""
Easy way to look at input and output together.
To run for the file 'prufa.txt':
$ python diffchecker.py prufa.txt

"""
from reynir_correct.wrappers import check_errors

from typing import (
    Iterator,
    Iterable,
    Dict,
    Union,
)
import sys
import argparse

# File types for UTF-8 encoded text files
ReadFile = argparse.FileType("r", encoding="utf-8")
WriteFile = argparse.FileType("w", encoding="utf-8")

# Define the command line arguments
parser = argparse.ArgumentParser(description="Corrects Icelandic text")

parser.add_argument(
    "inputfile",
    nargs="?",
    type=ReadFile,
    default=sys.stdin,
    help="UTF-8 text file to correct",
)


def gen(f: Iterator[str]) -> Iterable[str]:
    """Generate the lines of text in the input file"""
    yield from f


def main():

    options: Dict[str, Union[str, bool]] = {}
    # Hægt að biðja um annað til að fá frekari upplýsingar!
    options["format"] = "text"
    options["annotations"] = True
    options["all_errors"] = True

    # options["infile"] = open("prufa.txt", "r")
    options["one_sent"] = True
    # options["generate_suggestion_list"] = True

    args = parser.parse_args()
    inputfile = args.inputfile
    if inputfile == sys.stdin and sys.stdin.isatty():
        # terminal input is empty, most likely no value was given for infile:
        # Nothing we can do
        # inputfile = open("prufa.txt", "r")
        print("No input has been given, nothing can be returned")
        raise ValueError
    itering = gen(inputfile)
    for sent in itering:
        print("=================================")
        if sent.startswith("#"):
            # Comment string, want to show it with the examples
            print(sent.strip())
            sent = next(itering)
        options["infile"] = sent
        x = check_errors(**options)
        # Hér væri auðvelt að bæta við einhverjum samanburði við gull, skila bara T/F hvort sé eins
        # Þyrfti þá að zippa  saman setningar úr testskjali og gullskjali til að nota í forlykkju
        print(sent.strip())
        print(x)


if __name__ == "__main__":
    main()
