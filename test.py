
from reynir_correct.spelling import Corrector
from reynir.bindb import BIN_Db
with BIN_Db.get_db() as db: c = Corrector(db)
import time

def test(word):
    t0 = time.time()
    result = list(c.test_subs(word))
    valid = [r for r in result if r in c]
    t1 = time.time()
    print("Word: {0}, combinations: {1}, time {2:.3f} secs".format(word, len(result), t1 - t0))
    print(result)
    print(valid)


test("hæstarréttarlögmaður")
test("fangageimslan")
test("ollíugeimir")

