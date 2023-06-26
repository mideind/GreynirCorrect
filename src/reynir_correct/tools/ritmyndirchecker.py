#!/usr/bin/env python

"""
Easy way to compare error categories in Ritmyndir and the metadata,
checking for new ones that need to be added.

$ python ritmyndirchecker.py

"""
from reynir_correct.settings import RitmyndirDetails
from reynir_correct.settings import Ritmyndir

from typing import (
    Set,
)

# Error codes that are not explicitly an error
IGNORE = set(["GAM", "SO-ÞGF4ÞF", "OSB-BMYND", "SJALD", "STAD", "AV"])


def main() -> None:
    ritmyndir = Ritmyndir()
    ritmyndir_details = RitmyndirDetails()
    allcodes: Set[str] = set()
    for entry in ritmyndir.DICT:
        allcodes.add(ritmyndir.get_code(entry))
    detcodes: Set[str] = set()
    for keycode in ritmyndir_details.DICT:
        detcodes.add(keycode)

    # Compare
    newcodes = allcodes - detcodes - IGNORE
    outdated = detcodes - allcodes - IGNORE

    print("=====================")
    print("New codes:")
    for item in newcodes:
        print("\t{}".format(item))
    print("=====================")
    print("Outdated/unused codes:")
    for item in outdated:
        print("\t{}".format(item))


if __name__ == "__main__":
    main()
