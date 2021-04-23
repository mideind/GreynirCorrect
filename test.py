# type: ignore

import sys
import reynir_correct as rc


def display_annotations(sent):
    print("\nSetning:")
    print(sent.text)
    print("\nNiðurstaða tókunar:")
    for ix, tok in enumerate(sent.tokens):
        print("{0:03} {1}".format(ix, tok.txt or ""))
    print("\nSetningatré:")
    print("[Ekkert]" if sent.tree is None else sent.tree.flat)
    print("\nVillur:")
    for ann in sent.annotations:
        print("{0}".format(ann))
    print("")


txt = (
    "Einn af drengjunum fóru í sund af gefnu tilefni.\n"
    "Mig hlakkaði til.\nÉg dreymdi kött.\n"
    "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, langaði að horfa á sjónvarpið.\n"
    "Páli, vini mínum, langaði að horfa á sjónvarpið.\n"
    "Hestinum Skjóna vantaði hamar.\n"
    "\"Önnu kveið # fyrir skóladeginum.\"\n"
)

print("\nUpphaflegur texti: '{0}'".format(txt))
for pg in rc.check(txt, split_paragraphs=True):
    for sent in pg:
        display_annotations(sent)
    print("---")

sys.exit(0)

import time

from reynir_correct.spelling import Corrector
from reynir.bindb import GreynirBin

with GreynirBin.get_db() as db:
    c = Corrector(db)  # type: Corrector


def test(c, word):
    t0 = time.time()
    result = list(c.subs(word))
    valid = [r for r in result if r in c]
    t1 = time.time()
    print("Word: {0}, combinations: {1}, time {2:.3f} secs".format(word, len(result), t1 - t0))
    print(result)
    print(valid)


test(c, "hæstarréttarlögmaður")
test(c, "fangageimslan")
test(c, "ollíugeimir")

