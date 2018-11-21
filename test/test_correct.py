"""

    test_correct.py

    Tests for ReynirCorrect module

    Copyright(C) 2018 by Miðeind ehf.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import reynir_correct as rc
import tokenizer


def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))


def test_correct(verbose=False):
    """ Test the spelling and grammar correction module """

    g = rc.tokenize(
        "Kexið er gott báðumegin, sagði sagði Cthulhu og rak sig uppundir þakið. "
        "Það var aldrey aftaka veður í gær."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 24
    assert g[4].error_code == "C002"
    assert g[6].error_code == "C001"
    assert g[7].error_code == "U001"
    assert g[11].error_code == "C002"
    assert g[19].error_code == "S001"
    assert g[20].error_code == "C003"

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "báðum megin" in s
    assert "upp undir" in s
    assert "aldrei" in s
    assert "aldrey" not in s
    assert "aftakaveður" in s
    assert "sagði sagði" not in s

    g = rc.tokenize(
        "Hann borðaði alltsaman en allsekki það sem ég gaf honum. "
        "Þið hafið hafið mótið að viðstöddum fimmhundruð áhorfendum."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 25
    assert g[3].error_code == "C002"
    assert g[4].error_code == "C002"
    assert g[6].error_code == "C002"
    assert g[21].error_code == "C002"

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "allt saman" in s
    assert "alls ekki" in s
    assert "hafið hafið" in s
    assert "fimm hundruð" in s

    g = rc.tokenize(
        "Ég gaf honum klukkustundar frest áður áður en hann fékk 50 ml af lyfinu. "
        "Langtíma þróun sýnir 25% hækkun hækkun frá 1. janúar 1980."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 24
    assert g[4].error_code == "C003"
    assert g[5].error_code == "C001"
    assert g[19].error_code == "C001"

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "klukkustundarfrest" in s
    assert "áður áður" not in s
    assert "hækkun hækkun" not in s


if __name__ == "__main__":

    test_correct(verbose=True)
