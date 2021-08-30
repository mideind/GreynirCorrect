# type: ignore
"""

    test_tokenizer.py

    Tests for GreynirCorrect module

    Copyright (C) 2021 by Miðeind ehf.

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


    This module tests the token-level error detection and correction
    of GreynirCorrect.

"""

from typing import Iterable, cast
import reynir_correct as rc
import tokenizer


def dump(tokens):
    print("\n{1}\n{0} tokens:\n"
        .format(
            len(tokens),
            tokenizer.correct_spaces(" ".join(t.txt for t in tokens if t.txt))
        )
    )
    for token in tokens:
        err = token.error_description
        if err:
            print("{0}".format(token.txt))
            print("   {0}: {1}".format(token.error_code, err))


def gen_to_string(g):
    return tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt))


def roundtrip(s: str) -> str:
    return rc.detokenize(cast(Iterable[tokenizer.Tok], rc.tokenize(s)))


def test_correct(verbose=False):
    """ Test the spelling and grammar correction module """

    g = rc.tokenize(
        "Kexið er gott báðumegin, sagði sagði Cthulhu og rak sig uppundir þakið. "
        "Það var aldrey aftaka veður í gær."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 24
    assert g[4].error_code == "C002"  # báðumegin -> báðum megin
    assert g[6].error_code == "C001"  # sagði sagði
    #assert g[7].error_code == "U001/w"  # Cthulhu
    assert g[11].error_code == "C002"  # uppundir -> upp undir
    assert g[19].error_code == "S001"  # aldrey
    assert g[20].error_code == "C003"  # aftaka veður -> aftakaveður

    s = gen_to_string(g)
    assert "báðum megin" in s
    assert "upp undir" in s
    assert "aldrei" in s
    assert "aldrey" not in s
    assert "aftakaveður" in s
    assert "sagði sagði" not in s

    g = rc.tokenize(
        "Müller sagði að hann hefði ýtrekað þurft að ræsa cyclotroninn."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 13
    #assert g[1].error_code == "U001/w"  # Müller
    assert g[6].error_code == "S004"  # ýtrekað -> ítrekað
    assert g[10].error_code == "U001"  # cyclotroninn

    s = gen_to_string(g)
    assert "Müller" in s
    assert "ítrekað" in s
    assert "ýtrekað" not in s
    assert "cyclotroninn" in s

    g = rc.tokenize(
        "Hann borðaði alltsaman en allsekki það sem ég gaf honum. "
        "Þið hafið hafið mótið að viðstöddum fimmhundruð áhorfendum."
    )

    g = list(g)
    if verbose: dump(g)

    assert len(g) == 25
    assert g[3].error_code == "C002"
    assert g[3].error_span == 2
    assert g[6].error_code == "C002"
    assert g[21].error_code == "C002"

    s = gen_to_string(g)
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

    s = gen_to_string(g)
    assert "áður áður" not in s
    assert "hækkun hækkun" not in s
    assert "klukkustundar frest" not in s
    assert "klukkustundarfrest" in s
    assert "Langtíma þróun" not in s
    assert "Langtímaþróun" in s


def test_allowed_multiples(verbose=False):
    """ Test allowed_multiples """

    g = rc.tokenize(
        "Þetta gerði gerði ekkert fyrir mig. Bóndinn á Á á á á fjalli."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
    assert "gerði gerði" in s
    assert "á Á á á á" in s


def test_wrong_compounds(verbose=False):
    """ Check wrong_compounds """

    g = rc.tokenize(
        "Það voru allskonar kökur á borðinu en ég vildi samt vera annarsstaðar."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
    assert "allskonar" not in s
    assert "alls konar" in s
    assert "annarsstaðar" not in s
    assert "annars staðar" in s


def test_split_compounds(verbose=False):
    """ Check split_compounds """

    g = rc.tokenize(
        "Ég fór bakdyra megin inn í auka herbergi og sótti uppáhalds bragðtegund "
        "af ís. Langtíma spá gerir ráð fyrir aftaka veðri. AFNÁM VERÐTRYGGINGAR "
        "ER GRUNDVALLAR ATRIÐI."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
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
    assert "GRUNDVALLAR ATRIÐI" not in s
    assert "GRUNDVALLARATRIÐI" in s


def test_unique_errors(verbose=False):
    """ Check unique_errors """

    g = rc.tokenize("Hann er einhverskonar asni en það er a.m.k rétt.")
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "a.m.k " not in s
    assert "a. m. k " not in s
    assert "a. m. k. " in s
    assert "einhverskonar" not in s
    assert "einhvers konar" in s
    assert g[3].val[0].stofn == "einhver"
    assert g[4].val[0].stofn == "konar"

    g = rc.tokenize(
        "Björgvinn tók efitr þvi að han var jafvel ókeipis."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
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

    g = rc.tokenize("Mér er sama þótt hann deyji enda er hann einhversslags asni.")
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "deyji" not in s
    assert "deyi" in s
    assert "einhversslags" not in s
    assert "einhvers lags" in s
    assert g[10].val[0].stofn == "einhver"
    assert g[11].val[0].stofn == "lag"


def test_error_forms(verbose=False):
    """ Check error_forms """

    g = rc.tokenize(
        "Fellibylir og jafvel HVIRFILBYLIR gengu yfir hús bróðurs míns."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
    assert "Fellibylir" not in s
    assert "Fellibyljir" in s
    assert "jafvel" not in s
    assert "jafnvel" in s
    assert "HVIRFILBYLIR" not in s
    assert "HVIRFILBYLJIR" in s
    assert "bróðurs" not in s
    assert "bróður" in s


def test_capitalization_errors(verbose=False):
    """ Check capitalization_errors """

    g = rc.tokenize(
        "Umhverfis- og auðlindaráðherra hitti Félags- og barnamálaráðherra í gær "
        "ásamt Fjármála- og Efnahagsráðherra en hann var á fundi með Fjármálaráðherra."
    )
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "Umhverfis- og auðlindaráðherra" in s
    assert "félags- og barnamálaráðherra" in s
    assert "fjármála- og efnahagsráðherra" in s
    assert "fjármálaráðherra" in s

    g = rc.tokenize(
        "Íslenskir menn drápu Danska menn og Gyðinga í evrópu gegn mótmælum "
        "Eistneskra sjálfstæðismanna."
    )

    g = list(g)
    if verbose: dump(g)

    s = gen_to_string(g)
    assert s.startswith("Íslenskir ")
    assert "Danska" not in s
    assert "danska" in s
    # !!! This is presenly commented out in GreynirCorrect.conf
    # assert "Gyðinga" not in s
    # assert "gyðinga" in s
    assert "evrópu" not in s
    assert "Evrópu" in s
    assert "Eistneskra" not in s
    assert "eistneskra" in s
    assert "sjálfstæðismanna" not in s
    assert "Sjálfstæðismanna" in s

    g = rc.tokenize(
        "finnar finna Finna hvar sem þeir leita en finnarnir fóru "
        "og hittu finnana."
    )
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert s.startswith("Finnar ")
    assert "finnarnir" not in s
    assert "Finnarnir" in s
    assert "finnana" not in s
    assert "Finnana" in s
    assert "Finna" in s

    g = rc.tokenize(
        "Gyðingurinn sagði að Lenínisminn tröllriði öllu en Eskimóinn taldi að "
        "það ætti fremur við um Marxismann en Sjítann."
    )
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert s.startswith("Gyðingurinn ")
    assert "Lenínisminn" not in s
    assert "lenínisminn" in s
    assert "Eskimóinn" not in s
    assert "eskimóinn" in s
    assert "Sjítann" not in s
    assert "sjítann" in s
    assert "Marxismann" not in s
    assert "marxismann" in s

    g = rc.tokenize(
        "30. Desember á ég afmæli en ég held upp á það 20. JÚLÍ "
        "af því að mamma á afmæli þriðja Janúar."
    )
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "Desember" not in s
    assert "desember" in s
    assert "JÚLÍ" in s
    assert "júlí" not in s
    assert "Júlí" not in s
    assert "Janúar" not in s
    assert "janúar" in s

    g = rc.tokenize(
        "30. Janúar á mamma afmæli en ég á afmæli í Febrúar. "
        "17. Ágúst kemur Ágúst í heimsókn en þriðja Júlí verður sungið fyrir okkur."
    )
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "30. Janúar" not in s
    assert "30. janúar" in s
    assert "Febrúar" not in s
    assert "febrúar" in s
    # FIXME:
    # assert "17. Ágúst" not in s
    # assert "17. ágúst" in s
    assert "kemur Ágúst" in s
    assert "kemur ágúst" not in s
    assert "þriðja Júlí" not in s
    assert "þriðja júlí" in s


def test_capitalization_of_numbers(verbose=False):
    g = rc.tokenize("Fjögur hundruð manns komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjögur hundruð manns" in s
    g = rc.tokenize("FJÖGUR HUNDRUÐ manns komu saman í dag.")
    s = gen_to_string(g)
    assert "FJÖGUR HUNDRUÐ manns" in s
    g = rc.tokenize("Fjögur Hundruð manns komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjögur hundruð manns" in s
    g = rc.tokenize("Fjögur Þúsund manns komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjögur þúsund manns" in s
    g = rc.tokenize("Fjórar Milljónir manna komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjórar milljónir manna" in s
    g = rc.tokenize("Fjórar MILLJÓNIR manna komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjórar milljónir manna" in s
    g = rc.tokenize("fjögur hundruð manns komu saman í dag.")
    s = gen_to_string(g)
    assert "Fjögur hundruð manns" in s
    g = rc.tokenize("400 þúsund manns komu saman í dag.")
    s = gen_to_string(g)
    assert "400 þúsund manns" in s
    g = rc.tokenize("400 Þúsund manns komu saman í dag.")
    s = gen_to_string(g)
    assert "400 þúsund manns" in s

    g = rc.tokenize("Tilkynnt var að Fjögur hundruð manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjögur hundruð manns" in s
    g = rc.tokenize("Tilkynnt var að FJÖGUR HUNDRUÐ manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "FJÖGUR HUNDRUÐ manns" in s
    g = rc.tokenize("Tilkynnt var að Fjögur Hundruð manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjögur hundruð manns" in s
    g = rc.tokenize("Tilkynnt var að Fjögur Þúsund manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjögur þúsund manns" in s
    g = rc.tokenize("Tilkynnt var að Fjórar Milljónir manna kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjórar milljónir manna" in s
    g = rc.tokenize("Tilkynnt var að Fjórar MILLJÓNIR manna kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjórar milljónir manna" in s
    g = rc.tokenize("Tilkynnt var að fjögur hundruð manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "fjögur hundruð manns" in s
    g = rc.tokenize("Tilkynnt var að 400 þúsund manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "400 þúsund manns" in s
    g = rc.tokenize("Tilkynnt var að 400 Þúsund manns kæmu saman í dag.")
    s = gen_to_string(g)
    assert "400 þúsund manns" in s

    s = roundtrip("Jón greiddi Hundrað Þúsund Dollara fyrir bílinn.")
    assert "hundrað þúsund dollara" in s
    s = roundtrip("Jón greiddi hundrað Þúsund Dollara fyrir bílinn.")
    assert "hundrað þúsund dollara" in s
    s = roundtrip("Jón greiddi hundrað þúsund Dollara fyrir bílinn.")
    assert "hundrað þúsund dollara" in s
    s = roundtrip("Jón greiddi Hundrað þúsund Dollara fyrir bílinn.")
    assert "hundrað þúsund dollara" in s
    s = roundtrip("Jón greiddi Hundrað þúsund dollara fyrir bílinn.")
    assert "hundrað þúsund dollara" in s
    s = roundtrip("Jón greiddi Tíu þúsund dollara fyrir bílinn.")
    assert "tíu þúsund dollara" in s
    s = roundtrip("Jón greiddi Fjögur þúsund dollara fyrir bílinn.")
    assert "fjögur þúsund dollara" in s
    s = roundtrip("Jón greiddi 4 Þúsund dollara fyrir bílinn.")
    assert "4 þúsund dollara" in s
    s = roundtrip("Jón greiddi 4 þúsund Dollara fyrir bílinn.")
    assert "4 þúsund dollara" in s
    s = roundtrip("Jón greiddi 4 Þúsund Dollara fyrir bílinn.")
    assert "4 þúsund dollara" in s

    s = roundtrip("Jón greiddi Hundrað þúsund danskar krónur fyrir bílinn.")
    assert "hundrað þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi hundrað Þúsund danskar krónur fyrir bílinn.")
    assert "hundrað þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi hundrað þúsund Danskar krónur fyrir bílinn.")
    assert "hundrað þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi hundrað þúsund danskar Krónur fyrir bílinn.")
    assert "hundrað þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi Hundrað þúsund Danskar krónur fyrir bílinn.")
    assert "hundrað þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi Tíu þúsund danskar krónur fyrir bílinn.")
    assert "tíu þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi Fjögur þúsund danskar krónur fyrir bílinn.")
    assert "fjögur þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi FJÖGUR ÞÚSUND DANSKAR KRÓNUR fyrir bílinn.")
    assert "FJÖGUR ÞÚSUND DANSKAR KRÓNUR" in s
    s = roundtrip("Jón greiddi 4 Þúsund danskar krónur fyrir bílinn.")
    assert "4 þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi 4 þúsund Danskar krónur fyrir bílinn.")
    assert "4 þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi 4 Þúsund danskar Krónur fyrir bílinn.")
    assert "4 þúsund danskar krónur" in s
    s = roundtrip("Jón greiddi 4 ÞÚSUND DANSKAR KRÓNUR fyrir bílinn.")
    assert "4 ÞÚSUND DANSKAR KRÓNUR" in s

    # !!! TODO
    #s = roundtrip("Gísli notaði 200 Grömm af hveiti í kökuna.")
    #assert "200 grömm" in s
    #s = roundtrip("Gísli notaði Hundrað Grömm af hveiti í kökuna.")
    #assert "hundrað grömm" in s
    #s = roundtrip("Gísli notaði Hundrað grömm af hveiti í kökuna.")
    #assert "hundrað grömm" in s
    #s = roundtrip("Gísli notaði 20 Millilítra af vanilludropum í kökuna.")
    #assert "20 millilítra" in s
    #s = roundtrip("Gísli notaði 20 Kíló af pipar í kökuna.")
    #assert "20 kíló" in s


def test_taboo_words(verbose=False):
    g = rc.tokenize(
        "Jón sagði að hún væri múhameðstrúarmaður en hún svaraði að "
        "hann væri hommatittur og negri með lítinn typpaling."
    )
    g = list(g)
    if verbose: dump(g)
    assert len(g) == 21
    errors = {6, 13, 15, 18}
    for ix, _ in enumerate(g):
        if ix in errors:
            assert g[ix].error_code == "T001/w"  # Taboo word
        else:
            assert not g[ix].error_code


def test_multiword_errors(verbose=False):
    sent = """
        Af gefnu tilefni fékk hann vilja sýnum framgengt við hana í auknu mæli
        og að mestu leiti en hún helti úr skálum reiði sinnar.
    """
    g = rc.tokenize(sent)
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)

    assert "Af gefnu" not in s
    assert "Að gefnu" in s
    assert "vilja sýnum" not in s
    assert "vilja sínum" in s
    assert "í auknu mæli" not in s
    assert "í auknum mæli" in s
    assert "að mestu leiti" not in s
    assert "að mestu leyti" in s
    assert "hún helti" not in s
    assert "hún hellti" in s


def test_complex(verbose=False):
    s = """
        biddu nu hæg - var Kvennalistinn eins malefnis hreyfing. Hvað attu við - ef þu telur malefnið
        hafa verið eitt hvert var það? Kannski leikskola fyrir öll börn? Sömu laun fyrir sömu störf?
        Að borgarskipulag tæki mið af þörfum beggja kynja? Að kynjagleraugu væru notuð við gerð
        fjarlaga? Að þjoðfelagið opnaði augun fyrir kynferðsofbeldinu og sifjaspellum? (hvorutveggja
        sagt aðeins viðgangast i utlöndum). Þetta eru aðeins örfa dæmi um malefni sem brunnu a okkur
        og við börðumst fyrir. Ekki ertu i alvöru að tala framlag okkur niður. Tæplega
        telurðu það EITT malefni þo að i grunninn hafi baratta okkar sem stoðum að Kvennaframboðinu
        og -listanum gengið ut a að ,,betri,, helmingur þjoðarinnar öðlast - ekki bara i orði heldur
        einnig a borði - sömu rettindi og raðandi helmingurinn
    """
    g = rc.tokenize(s)
    g = list(g)
    if verbose: dump(g)
    s = gen_to_string(g)
    assert "malefnis" not in s
    assert "málefnis" in s
    assert "attu" not in s
    assert "áttu" in s
    assert "fjarlaga" not in s
    assert "fjárlaga" in s
    assert "kynferðsofbeldinu" not in s
    assert "kynferðisofbeldinu" in s
    assert "þjoðarinnar" not in s
    assert "þjóðarinnar" in s
    # The following is not corrected; it only gets an annotation
    #assert "raðandi" not in s
    #assert "ráðandi" in s
    assert "örfa" not in s
    assert "örfá" in s


if __name__ == "__main__":

    test_correct(verbose=True)
    test_allowed_multiples(verbose=True)
    test_split_compounds(verbose=True)
    test_wrong_compounds(verbose=True)
    test_unique_errors(verbose=True)
    test_error_forms(verbose=True)
    test_capitalization_errors(verbose=True)
    test_taboo_words(verbose=True)
    test_multiword_errors(verbose=True)
    test_complex(verbose=True)
