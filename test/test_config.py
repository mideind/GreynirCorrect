from typing import Dict, Union
import reynir_correct
import pytest
from reynir_correct.wrappers import test_grammar as wrap_check

@pytest.fixture(scope="module")
def rc_extra():
    """Provide a module-scoped GreynirCorrect instance as a test fixture"""
    settings = reynir_correct.Settings()
    settings.read("../reynir_correct/config/GreynirCorrect.conf")
    settings.read("../reynir_correct/config/ExtraWords.conf")
    r = reynir_correct.GreynirCorrect(settings)
    yield r
    # Do teardown here
    r.__class__.cleanup()


def check(p, rc_extra, options={}):
    """Return a corrected, normalized string form of the input along with the tokens"""
    options["input"] = [p]
    options["one_sent"] = False

    return wrap_check(rc=rc_extra, **options)



def test_tov_words(rc_extra, verbose=False):
    s, g = check(
        "Ég er með bankareikning og góðan þjónustufulltrúa.", rc_extra)
    assert len(s) == 37
    errors = {4, 7}
    for ix in range(len(g)):
        print(ix)
        if ix in errors:
            assert g[ix].error_code == "V001/w"
        else:
            assert not g[ix].error_code

if __name__ == "__main__":
    r = rc_extra()
    test_tov_words(r, verbose=True)