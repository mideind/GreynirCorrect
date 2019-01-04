

import sys
import reynir_correct as rc

txt = "Einn af drengjunum fóru í sund af gefnu tilefni."
print("\nUpphafleg setning: '{0}'".format(txt))
sent = rc.check_single(txt)
print("\nNiðurstaða tókunar:")
for ix, tok in enumerate(sent.tokens):
	print("{0:03} {1}".format(ix, tok.txt or ""))
print("\nVillur:")
for ann in sent.annotations:
	print("{0:03}-{1:03} {2:6} {3}".format(ann["start"], ann["end"], ann["code"], ann["text"]))
print("")

sys.exit(0)


from reynir_correct.spelling import Corrector
from reynir.bindb import BIN_Db
with BIN_Db.get_db() as db: c = Corrector(db)
import time

def test(word):
    t0 = time.time()
    result = list(c.subs(word))
    valid = [r for r in result if r in c]
    t1 = time.time()
    print("Word: {0}, combinations: {1}, time {2:.3f} secs".format(word, len(result), t1 - t0))
    print(result)
    print(valid)


test("hæstarréttarlögmaður")
test("fangageimslan")
test("ollíugeimir")

