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
    assert g[19].error_code == "S002"
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

    assert len(g) == 23
    assert g[5].error_code == "C001"  # áður áður
    assert g[18].error_code == "C001"  # hækkun hækkun

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "áður áður" not in s
    assert "hækkun hækkun" not in s
    assert "klukkustundar frest" not in s
    assert "klukkustundarfrest" in s
    assert "Langtíma þróun" not in s
    assert "Langtímaþróun" in s

    # Check allowed_multiples

    g = rc.tokenize(
        "Þetta gerði gerði ekkert fyrir mig. Bóndinn á Á á á á fjalli."
    )

    g = list(g)
    if verbose: dump(g)

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "gerði gerði" in s
    assert "á Á á á á" in s

    # Check wrong_compounds

    g = rc.tokenize(
        "Það voru allskonar kökur á borðinu en ég vildi samt vera annarsstaðar."
    )

    g = list(g)
    if verbose: dump(g)

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "allskonar" not in s
    assert "alls konar" in s
    assert "annarsstaðar" not in s
    assert "annars staðar" in s

    # Check split_compounds

    g = rc.tokenize(
        "Ég fór bakdyra megin inn í auka herbergi og sótti uppáhalds bragðtegund af ís. "
        "Langtíma spá gerir ráð fyrir aftaka veðri. SÉR ÍSLENSKAN BELGING MÁ FINNA VÍÐA."
    )

    g = list(g)
    if verbose: dump(g)

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "bakdyra megin" not in s
    assert "bakdyramegin" in s
    assert "auka herbergi" not in s
    assert "aukaherbergi" in s
    assert "uppáhalds bragðtegund" not in s
    assert "uppáhaldsbragðtegund" in s
    assert "Langtíma spá" not in s
    assert "Langtímaspá" in s
    assert "aftaka veðri" not in s
    assert "aftakaveðri" in s
    assert "SÉR ÍSLENSKAN" not in s
    assert "SÉRÍSLENSKAN" in s

    # Check unique_errors

    g = rc.tokenize(
        "Björgvinn tók efitr þvi að han var jafvel ókeipis."
    )

    g = list(g)
    if verbose: dump(g)

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Björgvinn" not in s
    assert "Björgvin" in s
    assert "efitr" not in s
    assert "eftir" in s
    assert "þvi" not in s
    assert "því" in s
    assert "han " not in s
    assert "hann " in s
    assert "jafvel" not in s
    assert "jafnvel" in s
    assert "ókeipis" not in s
    assert "ókeypis" in s

    # Check error_forms

    g = rc.tokenize(
        "Fellibylir og jafvel HVIRFILBYLIR gengu yfir hús bróðurs míns."
    )

    g = list(g)
    if verbose: dump(g)

    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Fellibylir" not in s
    assert "Fellibyljir" in s
    assert "jafvel" not in s
    assert "jafnvel" in s
    assert "HVIRFILBYLIR" not in s
    assert "HVIRFILBYLJIR" in s
    assert "bróðurs" not in s
    assert "bróður" in s


if __name__ == "__main__":

    test_correct(verbose=True)
