#!/usr/bin/env python
"""

    Greynir: Natural language processing for Icelandic

    Spelling and grammar checking module

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


    This is an executable program wrapper (main module) for the GreynirCorrect
    package. It can be used to invoke the corrector from the command line,
    or via fork() or exec(), with the command 'correct'. The main() function
    of this module is registered as a console_script entry point in setup.py.

"""

from typing import Dict, Union

import argparse
import sys

from .wrappers import check_errors

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
parser.add_argument(
    "outfile",
    nargs="?",
    type=WriteFile,
    default=sys.stdout,
    help="UTF-8 output text file",
)

parser.add_argument(
    "--suppress_suggestions",
    "-ss",
    action="store_true",
    help="Suppress more agressive error suggestions",
)

parser.add_argument("--spaced", "-sp", help="Separate tokens with spaces", action="store_true")

# Determines the output format
parser.add_argument(
    "--format",
    "-f",
    nargs="?",
    type=str,
    default="text",
    help="Determine output format.\ntext: Corrected text only.\ncsv: One token per line in CSV format.\njson: One token per line in JSON format.\nm2: M2 format, GEC standard.",
)

# Determines whether we supply only token-level annotations or also sentence-level annotations
parser.add_argument(
    "--all_errors",
    "-a",
    help="Annotate both grammar and spelling errors",
    action="store_true",
)

# Add --grammar for compatibility; works the same as --all_errors
parser.add_argument(
    "--grammar",
    "-g",
    help="Annotate both grammar and spelling errors",
    action="store_true",
)

# Add --json for compatibility; works the same as --format=json
parser.add_argument(
    "--json",
    "-j",
    help="Output in JSON format",
    action="store_true",
)

# Add --csv for compatibility; works the same as --format=csv
parser.add_argument(
    "--csv",
    "-c",
    help="Output in CSV format",
    action="store_true",
)

# Add --normalize
parser.add_argument(
    "--normalize",
    "-n",
    help="Normalize punctuation",
    action="store_true",
)

parser.add_argument(
    "--sentence_prefilter",
    help="Run a heuristic filter on sentences to determine whether they are probably correct. Probably correct sentences will not go through the full parsing process.",
    action="store_true",
)
parser.add_argument(
    "--flesch",
    help="Calculate Flesch readability score for the input text",
    action="store_true",
)
parser.add_argument(
    "--rare_words",
    help="Show rare words in the input text",
    action="store_true",
)

parser.add_argument(
    "--tov_config",
    nargs=1,
    type=str,
    help="Add additional use-specific rules in a configuration file to check for custom tone-of-voice issues. Uses the same format as the default GreynirCorrect.conf file",
    default=None,
)

parser.add_argument(
    "--suggest_not_correct",
    help="Instead of directly changing the text, some stylistic errors are presented as suggestions only.",
    action="store_true",
    default=False,
)


def from_args(args: argparse.Namespace) -> Dict[str, Union[str, bool]]:
    """Fill options with information from args"""
    format = args.format
    if args.json:
        format = "json"
    elif args.csv:
        format = "csv"
    return {
        "input": args.infile,
        "suppress_suggestions": args.suppress_suggestions,
        "format": format,
        "spaced": args.spaced,
        "normalize": args.normalize,
        "all_errors": args.all_errors or args.grammar,
        "sentence_prefilter": args.sentence_prefilter,
        "tov_config": args.tov_config,
        "suggest_not_correct": args.suggest_not_correct,
        "flesch": args.flesch,
        "rare_words": args.rare_words,
    }


def main() -> None:
    """Main function, called when the 'correct' command is invoked"""

    args = parser.parse_args()
    # Fill options with information from args
    if args.infile is sys.stdin and sys.stdin.isatty():
        # terminal input is empty, most likely no value was given for infile:
        # Nothing we can do
        print("No input has been given, nothing can be returned")
        sys.exit(1)
    options = from_args(args)

    print(check_errors(**options), file=args.outfile)


if __name__ == "__main__":
    main()
