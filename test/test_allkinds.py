# type: ignore
"""

    test_allkinds.py

    Tests for GreynirCorrect module

    Copyright (C) 2022 by Miðeind ehf.

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


    This module tests both the token-level error detection and correction
    and the sentence-level annotation functionality of GreynirCorrect.

"""
from typing import Optional, Set

from reynir_correct import GreynirCorrectAPI, detokenize

from .utils import check_sentence, correct_grammar_format

# Global settings object for the tests
api = GreynirCorrectAPI.from_options(one_sent=False)


def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))


def normalize(g):
    """Return a corrected, normalized string form of the token list in g"""
    return detokenize(g, normalize=True)


def check(p: str, ignore_rules: Optional[Set[str]] = None, suppress_suggestions: bool = False):
    """Return a corrected, normalized string form of the input along with the tokens"""
    return correct_grammar_format(
        text=[p], api=api, ignore_rules=ignore_rules, suppress_suggestions=suppress_suggestions
    )


def test_punctuation(verbose=False):
    # Quotation marks
    s, g = check('Hann var kallaður ,,pottormur" og var "hrekkjusvín".')
    assert "„pottormur“" in s
    assert "„hrekkjusvín“" in s

    # Ellipsis
    s, g = check("Ég veit ekki...")
    assert "..." not in s
    # Ath. þetta finnst í s, en í viðmótinu birtist þetta ekki í hríslutrénu.
    assert "…" in s


def test_multiple_spaces(verbose=False):
    s, g = check("Hér         er langt bil.")
    assert "Hér er" in s
    assert "  " not in s


def test_doubling(verbose=False):
    """Test words that are erroneously written twice or more often"""
    # Simple instance
    s, g = check("Ég hélt mér mér fast í sætið.")
    assert "hélt mér fast" in s
    assert "mér mér" not in s
    assert len(g) == 9
    assert g[3].error_code == "C001"  # mér

    # Test many instances in same sentence
    s, g = check("Potturinn kom ekki ekki í ljós ljós fyrr en en í dag dag.")
    assert "kom ekki í" in s
    assert "ekki ekki" not in s
    assert "í ljós fyrr" in s
    assert "ljós ljós" not in s
    assert "fyrr en í" in s
    assert "en en" not in s
    assert "í dag." in s
    assert "dag dag" not in s
    assert len(g) == 12
    errors = {3, 5, 7, 9}
    for ix in range(len(g)):
        if ix in errors:
            assert g[ix].error_code == "C001"  # ekki, ljós, en, dag
        else:
            assert not g[ix].error_code

    # Here the first word is just an uppercase version of the following common noun. This should be corrected.
    s, g = check("Slysið slysið átti sér stað í gærkvöldi.")
    assert g[2].error_code == "C004/w"  # slysið, bara uppástunga, ekki leiðrétt

    # Testing multiple words in a row. This should be corrected.
    s, g = check("Það er stanslaust fjör fjör fjör fjör fjör fjör fjör fjör í sveitinni.")
    assert len(g) == 9
    assert "stanslaust fjör í" in s
    assert "fjör fjör" not in s
    assert g[4].error_code == "C001"

    # 'á' has different meanings here, 'á á' should be marked as a possible error
    # 'en en' should not be accepted
    s, g = check("Ég á á sem heitir Lína langsokkur en en en hún kann ekki að jarma.")
    assert len(g) == 16 or len(g) == 15  # á á is corrected but is marked differently
    # assert "á á" in s  # Now corrected by default, but can be handled differently with a special option
    # assert "Ég á sem" not in s
    assert "langsokkur en hún" in s
    assert "en en" not in s
    assert g[3].error_code == "C004/w"  # á
    assert g[8].error_code == "C001" or g[7].error_code == "C001"  # en


def test_accepted_doubling(verbose=False):
    # Test examples with a comma between. This should be accepted.
    s, g = check("Lífið, sem er flokkar, flokkar potta.")
    assert len(g) == 11
    assert "flokkar, flokkar" in s
    assert not g[4].error_code

    # Another example with a comma between. This should be accepted.
    s, g = check("Lífið er svaka, svaka gaman.")
    assert len(g) == 9
    assert "svaka, svaka" in s
    assert not g[3].error_code

    # Test whether lowercase instance following uppercase instance is corrected
    # First, these are separate words and should be accepted but possible error pointed out.
    # 'í í' should be pointed out but not corrected, as 'í í' can occur,
    # for instance as a particle and then a preposition.
    s, g = check("Finnur finnur gull í í Tálknafirði.")
    assert len(g) == 9 or len(g) == 7
    # assert "Finnur finnur" in s
    # assert not "Finnur gull" in s
    assert g[2].error_code == "C004/w"  # Finnur
    assert g[4].error_code == "C004/w" or g[5].error_code == "C004/w"

    # Same example except now the common noun is capitalized in the beginning,
    # followed by the proper noun. This should be accepted.
    s, g = check("Finnur Finnur gull í Tálknafirði?")
    # assert len(g) == 8        # TODO útfæra að þetta er ekki leiðrétt
    # assert "Finnur Finnur" in s

    # Here are separate words, as Gaukur is a proper noun. This should be accepted.
    s, g = check("Gaukur gaukur slasaðist í slysinu.")
    assert len(g) == 8
    # TODO þetta á líklega frekar heima í checker.py, ath. hvort þetta er bæði
    # sérnafn og samnafn og þannig.
    # assert "Gaukur gaukur" in s
    # assert "Gaukur slasaðist" not in s
    assert g[2].error_code == "C004/w"  # gaukur

    # 'heldur' is allowed to appear more than once in a row. This should be accepted.
    s, g = check("Kvikan heldur heldur mikið í jörðina.")
    assert len(g) == 9
    # assert "Kvikan heldur heldur mikið" in s
    # assert "Kvikan heldur mikið" not in s
    assert g[3].error_code == "C004/w"  # heldur

    # 'gegn' has different meanings here, should be accepted
    s, g = check("Hún var góð og gegn gegn Svíum í úrslitaleiknum.")
    # assert len(g) == 12
    # assert "og gegn gegn Svíum" in s
    # assert "og gegn Svíum" not in s
    assert g[6].error_code == "C004/w"


def test_wrong_compounds(verbose=False):
    s, g = check(
        "Fötin koma í margskonar litum og fara afturábak afþvíað annarstaðar " "eru fjögurhundruð mikilsháttar hestar."
    )
    assert len(g) == 23
    assert "margs konar" in s
    assert "aftur á bak" in s
    assert "af því að" in s
    assert "annars staðar" in s
    assert "fjögur hundruð" in s
    assert "mikils háttar" in s
    errors = {4, 9, 12, 15, 17, 18}
    for ix in range(len(g)):
        if ix in errors:
            assert (
                g[ix].error_code == "C002"
            )  # margs konar, aftur á bak, af því að, annars staðar, fjögur hundruð, mikils háttar
        else:
            assert not g[ix].error_code

    s, g = check("Vagninn fór niðrá torg og svo ofan í níuhundruð samskonar seinnihluta.")
    assert "niður á" in s
    assert "níu hundruð" in s
    assert "sams konar" in s
    assert "seinni hluta" in s
    errors = {3, 10, 11, 13}
    for ix, t in enumerate(g):
        if ix in errors:
            assert t.error_code == "C002"  # niður á, níu hundruð, sams konar, seinni hluta
        else:
            assert not t.error_code


def test_split_compounds(verbose=False):
    s, g = check("Aðal inngangur að auka herbergi er gagn stæður öðrum gangi.")
    g = list(g)
    assert len(g) == 10
    assert "Aðalinngangur" in s
    assert "Aðal inngangur" not in s
    assert "aukaherbergi" in s
    assert "auka herbergi" not in s
    assert "gagnstæður" in s
    assert "gagn stæður" not in s
    errors = {1, 3, 5}
    for ix in range(len(g)):
        if ix in errors:
            assert g[ix].error_code == "C003"  # Aðalinngangur, aukaherbergi, gagnstæður
        else:
            assert not g[ix].error_code

    s, g = check("Myndar drengurinn er hálf undarlegur kvenna megin.")
    assert len(g) == 7
    assert "Myndardrengurinn" in s
    assert "Myndar drengurinn" not in s
    assert "hálfundarlegur" in s
    assert "hálf undarlegur" not in s
    assert "kvennamegin" in s
    assert "kvenna megin" not in s
    errors = {1, 3, 4}
    for ix, t in enumerate(g):
        if ix in errors:
            assert t.error_code == "C003"  # Myndardrengurinn, hálfundarlegur, kvennamegin
        else:
            assert not t.error_code

    s, g = check("Hann var full fljótur á sér, en það var líka full ástæða til að fá full réttindi.")
    assert len(g) == 20
    assert "fullfljótur" in s
    assert "full fljótur" not in s
    assert "full ástæða" in s
    assert "fullástæða" not in s
    assert "full réttindi" in s
    assert "fullréttindi" not in s
    errors = {3}
    for ix, t in enumerate(g):
        if ix in errors:
            assert t.error_code == "C003"  # fullfljótur
        else:
            assert not t.error_code


def test_unique_context_independent_errors(verbose=False):
    # Known, unique, context independent spelling errors - S001
    s, g = check("Fomaður fór til fljúgjandi augnæknis í liltu andyri Svíþjóðar.")
    assert "Formaður" in s
    assert "Fomaður" not in s
    assert "fljúgandi" in s
    assert "fljúgjandi" not in s
    assert "augnlæknis" in s
    assert "augnæknis" not in s
    assert "litlu" in s
    assert "liltu" not in s
    assert "anddyri" in s
    assert "andyri" not in s
    assert g[1].error_code in frozenset(["S004", "ASLSTAFVANTAR"])  # Fomaður
    assert g[4].error_code in frozenset(["S001", "ASLAUKASTAF"])  # fljúgjandi
    assert g[5].error_code in frozenset(["S001", "ASLSTAFVANTAR"])  # augnæknis
    assert g[7].error_code in frozenset(["S001", "ASLVIXL"])  # liltu
    assert g[8].error_code in frozenset(["S001", "EKKIORD"])  # andyri

    s, g = check("Mér tóskt að fá áfarm ókeipis ríkistjórn.")
    assert "tókst" in s
    assert "tóskt" not in s
    assert "áfram" in s
    assert "áfarm" not in s
    assert "ókeypis" in s
    assert "ókeipis" not in s
    assert "ríkisstjórn" in s
    assert "ríkistjórn" not in s
    assert g[2].error_code in frozenset(["S001", "ASLVIXL"])  # tóskt
    assert g[5].error_code in frozenset(["S001", "ASLVIXL"])  # áfarm
    assert g[6].error_code in frozenset(["S001", "EI4EY"])  # ókeipis
    assert g[7].error_code in frozenset(["S001", "EKKIORD"])  # ríkistjórn

    s, g = check("Þar sat Gunan og fyldist með framistöðu liðisins í framlenginunni mikklu.")
    assert "sat Gunna og" in s
    assert "sat Gunan og" not in s
    assert " Gunna " in s
    assert " Gunan " not in s
    assert " fylgdist " in s
    assert " fyldist " not in s
    assert " frammistöðu " in s
    assert " framistöðu " not in s
    assert " liðsins " in s
    assert " liðisins " not in s
    assert " framlengingunni " in s
    assert " framlenginunni " not in s
    assert " miklu." in s
    assert " mikklu." not in s
    assert g[3].error_code in frozenset(["S001", "S005"])  # Gunan
    assert g[5].error_code == "S004"  # fyldist; TODO Á að vera S001?
    assert g[7].error_code in frozenset(["S001", "M4MM"])  # framistöðu
    assert g[8].error_code in frozenset(["S001", "ASLAUKASTAF"])  # liðisins
    assert g[10].error_code in frozenset(["S001", "ASLSTAFVANTAR"])  # framlenginunni
    assert g[11].error_code == "S004"  # mikklu; TODO Á að vera S001?


def test_other_context_independent_spelling_errors(verbose=False):
    # aðrar ósamhengisháðar, einskiptisvillur og bullorð
    # S002 ef þetta er leiðréttanlegt;
    # S003 ef aðeins uppástunga finnst;
    # U001 ef ekkert finnst
    # TODO Fæ ég einhvern tímann W001??
    s, g = check("Ég fyldist með fóboltanum í sjóvvarpinu í gærköldi.")
    # TODO breytir þessu í fylgist, er aðeins líklegra út af tíðni. Bæta við þekktar villur/heil beygingardæmi?
    # assert "Ég fylgdist með" in s
    # assert "Ég fyldist með" not in s
    assert "með fótboltanum í" in s
    assert "með fóboltanum í" not in s
    assert "í sjónvarpinu í" in s
    assert "í sjóvvarpinu í" not in s
    # TODO getur leiðrétt gærköld→gærkvöld, en gærkvöldi virðist ekki vera í safninu.
    assert "í gærkvöldi" in s
    assert "í gærköldi" not in s

    # assert g[2].error_code == "S002"  # TODO endar sem S004; "Checking rare word 'fyldist'"
    # assert g[4].error_code == "S002"  # TODO endar sem S004
    # assert g[6].error_code == "S002"  # TODO endar sem S004
    # assert g[8].error_code == "S002"  # TODO endar sem S004

    s, g = check("Ég fór í ljós tisvar í vigu og mædi regullega í lígamsrætt.")
    # Current output (2020-09-08):
    # 'Ég fór í ljós tvisvar í viku og mæli regullega í lígamsrætt .'
    assert "ljós tvisvar í" in s
    assert "í viku og" in s
    # assert "og mæti" in s
    # assert "reglulega í" in s     # TODO leiðréttist ekki, kemur með uppástungu
    # assert "í líkamsrækt" in s    # TODO leiðréttist ekki, of flókin villa.

    # assert g[5].error_code == "S002"      # TODO endar sem S004
    # assert g[7].error_code == "S002"      # TODO endar sem S004
    # assert g[9].error_code == "S002"      # TODO endar sem S004
    # assert g[10].error_code == "S002"     # TODO endar sem W001/w
    assert g[12].error_code == "U001"


def test_context_dependent_spelling_errors(verbose=False):
    # Context dependent spelling errors - P_xxx

    s, g = check("Alla sýna lífdaga hljóp hún allt kvað fætur toga af ástæðulausu.")
    assert "Alla sína lífdaga" in s
    assert "allt hvað fætur" in s
    assert "toga að ástæðulausu" in s
    # TODO fæ villukóðana á fyrsta orðið í fasta frasanum en ætti að fá á villuorðið sjálft.
    assert g[0].error_code == "P_yyii" or g[2].error_code == "S005"  # sína
    assert g[6].error_code == "P_kvhv" or g[7].error_code == "S005"  # hvað
    assert g[10].error_code == "P_afað"  # að

    # Context dependent spelling errors - P_xxx
    s, g = check("Kvað sem á bjátar lifir en í glæðunum.")
    assert "Hvað sem" in s
    assert "lifir enn í" in s
    # TODO villan kemur í fyrsta orðið í fasta frasanum en ætti að fá á villuorðið sjálft.
    assert g[1].error_code == "P_kvhv"  # Hvað
    assert g[5].error_code == "P_nn"  # enn


def test_homophones(verbose=False):
    # ruglingsmengin
    s, g = check("Hann heyrði lágvært kvísl í myrkrinu.")
    # s = normalize(g)
    # assert "hvísl" in s       # TODO eftir að útfæra
    # assert "kvísl" not in s   # TODO eftir að útfæra
    # assert g[4].error_code == "S006"  # TODO eftir að útfæra villukóða

    s, g = check("Kirtillinn flæktist fyrir fótum hennar í fermingunni.")
    # s = normalize(g)
    # assert "Kyrtillinn" in s          # TODO eftir að útfæra
    # assert "Kirtillinn" not in s      # TODO eftir að útfæra
    # assert g[0].error_code == "S006"  # TODO eftir að útfæra

    s, g = check("Tímanum líkur á því að kvatt er til þess að kvika ekki frá sinni stöðu.")
    # assert "lýkur" in s       # TODO eftir að útfæra
    # assert "líkur" not in s   # TODO eftir að útfæra
    # assert "hvatt" in s       # TODO eftir að útfæra
    # assert "kvatt" not in s   # TODO eftir að útfæra
    # assert "hvika" in s       # TODO eftir að útfæra
    # assert "kvika" not in s
    # assert g[2].error_code == "S006"
    # assert g[6].error_code == "S006"
    # assert g[11].error_code == "S006"

    s, g = check("Við rímum húsið til að leifa eldinum ekki að hvelja fólkið.")
    # assert "rýmum" in s       # TODO eftir að útfæra
    # assert "rímum" not in s
    # assert "leyfa" in s       # TODO eftir að útfæra
    # assert "leifa" not in s
    # assert "kvelja" in s      # TODO eftir að útfæra
    # assert "hvelja" not in s
    # assert g[2].error_code == "S006"
    # assert g[6].error_code == "S006"
    # assert g[10].error_code == "S006"


def test_paradigm_spelling_errors(verbose=False):
    # Unique errors in whole paradigm - S001
    s, g = check("Það var leiðilegt en þæginlegt að koma tímalega á áfangastað um fjögurleitið.")
    # assert "leiðinlegt" in s   # TODO er þetta ekki í þekktu villunum sem á eftir að koma inn?
    assert "þægilegt" in s  # TODO sama
    # assert "tímanlega" in s     # TODO sama
    assert "fjögurleytið" in s  # TODO sama
    assert g[3].error_code == "W001/w"
    assert g[5].error_code == "S001"
    assert g[8].error_code == "W001/w"
    assert g[12].error_code == "S004"

    s, g = check(
        "Barnið var fjagra ára þegar það fór janframt til ýmissra annara landa að "
        "leita að síðastu kúinni en það var til einskins."
    )
    assert "fjögurra" in s
    assert "jafnframt" in s
    assert "ýmissa" in s
    # assert "annarra" in s          # TODO leiðréttist ekki, er í ErrorForms en er samhengisháð villa
    assert "síðustu" in s
    assert "kúnni" in s
    # assert "einskis" in s         # TODO leiðréttist ekki.
    # assert "einskins" not in s

    # errors = {3, 8, 10, 11, 16, 17, 22}
    # for ix in range(len(g)):                 # TODO virkar ekki, eftir að útfæra villukóðann og skipta villunum upp eftir eðli
    #    if ix in errors:                       # TODO þarf þá að uppfæra dæmin.
    #        assert g[ix].error_code == "S007"  # fjagra, janframt, ýmissra, annara, síðastu, kúinni, einskins
    #    else:
    #        assert not g[ix].error_code


def test_rare_word_errors(verbose=False):
    # S004, spelling.py
    s, g = check("Hann finur fyri alls kins verkjum.")
    # assert "Hann finnur" in s         # TODO Fæ uppástungu en ekki nógu sterka leiðréttingu
    # assert "fyrir" in s               # TODO Virkar ekki
    assert "kyns" in s
    # assert g[2].error_code == "S004"  # TODO virkar ekki, fæ S001
    # assert g[3].error_code == "S004"  # TODO virkar ekki, fæ W001 virðist vera.
    assert g[5].error_code == "S004"

    s, g = check("Hann skoðaði arða gluggs en leists ekki vel á neinn.")
    assert g[3].error_code == "W001/w"  # arða
    assert g[4].error_code == "W001/w"  # gluggs
    assert g[6].error_code == "W001/w"  # leists


def test_wrong_abbreviations(verbose=False):
    # S005
    s, g = check("Karlinn datt þ.á.m. í amk. fimm polla.")
    assert "þ. á m." in s
    assert "þ.á.m." not in s
    assert "a.m.k. " in s
    assert "amk. " not in s
    assert g[3].error_code == "A001"
    assert g[5].error_code == "A002"

    s, g = check("Eftir ca 10 mínútur datt hann í pollinn.")
    assert "ca. " in s
    assert "ca " not in s
    assert g[2].error_code == "A001"

    s, g = check("Forsetinn ofl. gengu út um dyrnar.")
    assert "o.fl. " in s  # TODO býr þetta til en setur aukapunkt aftan við!
    assert "ofl. " not in s
    assert g[2].error_code == "A002"

    s, g = check("Ríkissjóður tók m.a 30 ma lán.")
    assert "m.a. " in s
    assert "m.a " not in s
    assert g[3].error_code == "A002"


def test_capitalization(verbose=False):
    # NOTE "Eskimói" has now been added to BÍN, and as an error in Storasnid_ord.csv, but not to Storasnid_ritm.csv.
    s, g = check(
        "Einn Aríi, Búddisti, Eskimói, Gyðingur, sjálfstæðismaður, " "Múslími og Sjíti gengu inn á bar í evrópu."
    )
    assert "aríi" in s
    assert "búddisti" in s
    # assert "eskimói" in s or "inúíti" in s  # both the cap and taboo error are corrected.
    # assert "gyðingur" in s
    assert "Sjálfstæðismaður" in s
    assert "múslími" in s
    assert "sjíti" in s
    assert "Evrópu" in s
    assert g[2].error_code == "Z001"  # aríi
    assert g[4].error_code == "Z001"  # búddisti
    # assert g[6].error_code == "Z001" or g[6].error_code == "T001/w"  # eskimói/inúíti
    # assert g[8].error_code == "Z001"  # gyðingur
    assert g[10].error_code == "Z002"  # Sjálfstæðismaður
    assert g[12].error_code == "Z001"  # múslími
    assert g[14].error_code == "Z001"  # sjíti

    s, g = check("Á íslandi búa íslendingar og í danmörku búa Danskir danir í Nóvember en ekki fríslendingar.")
    assert "Íslandi" in s
    assert "Íslendingar" in s
    assert "Danmörku" in s
    # assert "danskir" in s            # Most likely an adjective at the start of a named entity (Danska ríkisútvarpið, Íslensk erfðagreining)
    assert "Danir" in s
    assert "nóvember" in s
    assert "Nóvember" not in s
    assert "Fríslendingar" in s
    assert g[2].error_code == "Z002"  # Íslandi
    assert g[4].error_code == "Z002"  # Íslendingar
    assert g[7].error_code == "Z002"  # Danmörku
    # assert g[9].error_code == "Z001"  # danskir
    assert g[10].error_code == "Z002"  # Danir
    assert g[12].error_code == "Z003"  # nóvember
    assert g[15].error_code == "Z002"  # Fríslendingar

    s, g = check("Í norður-belfast og norður-Belfast er rigning en ekki í suður-afríku hjá suður-afríkumönnum.")
    # assert "Norður-Belfast" in s
    assert "Suður-Afríku" in s
    assert "Suður-Afríkumönnum" in s
    # assert g[2].error_code == "Z002"  # Norður-Belfast
    # assert g[4].error_code == "Z002"  # Norður-Belfast
    assert g[10].error_code == "Z002"  # Suður-Afríku
    assert g[12].error_code == "Z002"  # Suður-Afríkumönnum

    s, g = check("Þau gengu frá suður-kóreu og þaðan á norður-hliðina og norður- og suðurhlutann.")
    assert "Suður-Kóreu" in s
    assert "norðurhliðina" in s
    assert "norður-" in s
    assert g[4].error_code == "Z002"  # Suður-Kóreu

    s, g = check("Í norður-kóreu er menningin önnur en í suður-kóreu og allt önnur á nýja-sjálandi.")
    assert "Norður-Kóreu" in s
    assert "Suður-Kóreu" in s
    assert "Nýja-Sjálandi" in s
    assert g[2].error_code == "Z002"  # Norður-Kóreu
    assert g[8].error_code == "Z002"  # Suður-Kóreu
    assert g[13].error_code == "Z002"  # Nýja-Sjálandi

    s, g = check("Í karíbahafi eru Karabísk mál töluð.")
    assert "Karíbahafi" in s
    assert "karabísk" in s
    assert g[2].error_code == "Z002"  # Karíbahafi
    assert g[4].error_code == "Z001"  # karabísk

    s, g = check("Á internetinu finna Súnnítar ýmsar Upplýsingar.")
    assert "súnnítar" in s
    # assert "upplýsingar" in s
    assert g[4].error_code == "Z001"  # súnnítar

    s, g = check(
        "Ég er með Alzheimers-sjúkdóm en Félags- og Barnamálaráðherra ekki og heldur ekki Ferðamála-, iðnaðar- og nýsköpunarráðherra."
    )
    assert "alzheimers-sjúkdóm" in s
    assert "félags- og barnamálaráðherra" in s
    assert "ferðamála-, iðnaðar- og nýsköpunarráðherra" in s
    assert g[4].error_code == "Z001"  # alzheimers-sjúkdóm
    assert g[6].error_code == "Z001"  # félags- og barnamálaráðherra
    assert g[11].error_code == "Z001"  # ferðamála-, iðnaðar- og nýsköpunarráðherra

    s, g = check(
        "Þau gengu laugaveg í sumar og fóru á þingvelli, heimsóttu austur-evrópu, vestur-evrópu, Austurevrópskt fólk, mið-austurlönd og litla-hraun."
    )
    assert "Laugaveg" in s
    assert "Þingvelli" in s
    assert "Austur-Evrópu" in s
    assert "Vestur-Evrópu" in s
    assert "austurevrópskt" in s
    assert "Mið-Austurlönd" in s
    assert "Litla-Hraun" in s
    assert g[3].error_code == "Z002"  # Laugaveg
    assert g[9].error_code == "Z002"  # Þingvelli
    assert g[12].error_code == "Z002"  # Austur-Evrópu
    assert g[14].error_code == "Z002"  # Vestur-Evrópu
    assert g[16].error_code == "Z001"  # austurevrópskt
    assert g[19].error_code == "Z002"  # Mið-Austurlönd
    assert g[21].error_code == "Z002"  # Litla-Hraun

    s, g = check("Þjóðin tók þátt í vetrarólympíuleikunum og sumarólympíuleikunum en líbanar ekki.")
    assert "Vetrarólympíuleikunum" in s
    assert "Sumarólympíuleikunum" in s
    assert "Líbanar" in s
    assert g[5].error_code == "Z002"  # Vetrarólympíuleikunum
    assert g[7].error_code == "Z002"  # Sumarólympíuleikunum
    assert g[9].error_code == "Z002"  # Líbanar

    s, g = check(
        "Nýr Loftslagsráðherra, Innviðaráðherra og Umhverfisráðherra er Afróasískur, "
        "talar Dravídamál, fylgir Lútherstrú og er miðflokksmaður."
    )
    assert "loftslagsráðherra" in s
    assert "innviðaráðherra" in s
    assert "umhverfisráðherra" in s
    assert "afróasískur" in s
    assert "dravídamál" in s
    assert "lútherstrú" in s
    assert "Miðflokksmaður" in s
    assert g[2].error_code == "Z001"  # loftslagsráðherra
    assert g[4].error_code == "Z001"  # innviðaráðherra
    assert g[6].error_code == "Z001"  # umhverfisráðherra
    assert g[8].error_code == "Z001"  # afróasískur
    assert g[11].error_code == "Z001"  # dravídamál
    assert g[14].error_code == "Z001"  # lútherstrú
    assert g[17].error_code == "Z002"  # Miðflokksmaður

    s, g = check(
        "Hann er Suðurkákasískur, tínir Unnarfald, býr í neðra-breiðholti og elskar Múmínálfa og óskarsverðlaunin."
    )
    assert "suðurkákasískur" in s
    assert "unnarfald" in s
    assert "Neðra-Breiðholti" in s
    assert "múmínálfa" in s
    assert "Óskarsverðlaunin" in s
    assert g[3].error_code == "Z001"  # suðurkákasískur
    assert g[6].error_code == "Z001"  # unnarfald
    assert g[10].error_code == "Z002"  # Neðra-Breiðholti
    assert g[13].error_code == "Z001"  # múmínálfa
    assert g[15].error_code == "Z002"  # Óskarsverðlaunin

    s, g = check("Í Seinni Heimsstyrjöldinni gerðist meira en í Kalda Stríðinu.")
    assert "heimsstyrjöldinni" in s
    assert "stríðinu" in s
    assert g[3].error_code == "Z001"  # heimsstyrjöldinni
    assert g[9].error_code == "Z001"  # stríðinu

    s, g = check(
        "Ég tala Víetnömsku, Indónesísku, er Afrísk-amerísk, karíbi, Karíbskur, austur-evrópubúi og vestur-evrópubúi"
    )
    assert "víetnömsku" in s
    assert "indónesísku" in s
    assert "afrísk-amerísk" in s
    assert "Karíbi" in s
    assert "karíbskur" in s
    assert "Austur-Evrópubúi" in s
    assert "Vestur-Evrópubúi" in s
    assert g[3].error_code == "Z001"  # heimsstyrjöldinni
    assert g[5].error_code == "Z001"  # indónesísku
    assert g[8].error_code == "Z001"  # afrísk-amerísk
    assert g[10].error_code == "Z002"  # Karíbi
    assert g[12].error_code == "Z001"  # karíbskur
    assert g[14].error_code == "Z002"  # Austur-Evrópubúi
    assert g[16].error_code == "Z002"  # Vestur-Evrópubúi


def test_acronyms(verbose=False):
    s, g = check("Hún skrifar fyrir Dv og Rúv.")
    assert "DV" in s
    assert "RÚV" in s
    assert g[4].error_code == "Z006"  # DV
    assert g[6].error_code == "Z006"  # RÚV

    s, g = check("Guðrún lék hlutverk Ms. Abercrombie í þáttunum")
    assert "Ms." in s
    assert g[4].error_code != "Z006"

    s, g = check("Sigurður lék hlutverk Mr. Smith í leikritinu")
    assert "Mr." in s
    assert g[4].error_code != "Z006"

    s, g = check("Hr. Hnetusmjör hélt tónleika í Kópavogi í kvöld")
    assert "Hr." in s
    assert g[1].error_code != "Z006"

    s, g = check("Hann er Suðurkóreskur og er suður-kóreumaður frá suður-kóreu.")
    assert "suðurkóreskur" in s
    assert "Suður-Kóreumaður" in s
    assert "Suður-Kóreu" in s

    s, g = check("Hann er Norðurkóreskur og er norður-kóreumaður frá norður-kóreu.")
    assert "norðurkóreskur" in s
    assert "Norður-Kóreumaður" in s
    assert "Norður-Kóreu" in s

    s, g = check("Hann er Nýsjálenskur og er nýsjálendingur frá nýja-sjálandi.")
    assert "nýsjálenskur" in s
    assert "Nýsjálendingur" in s
    assert "Nýja-Sjálandi" in s

    s, g = check("Hann er Suðurafrískur og er suður-afríkumaður frá suður-afríku.")
    assert "suðurafrískur" in s
    assert "Suður-Afríkumaður" in s
    assert "Suður-Afríku" in s

    s, g = check("Þau heimsóttu norðurland og hittu norðlendinga í Meistaradeild.")
    assert "Norðurland" in s
    assert "Norðlendinga" in s
    # assert "meistaradeild" in s

    s, g = check("Haldið er upp á Páskadag, Verslunarmannahelgina, Jólin og Bóndadag.")
    assert "páskadag" in s
    assert "verslunarmannahelgina" in s
    assert "jólin" in s
    assert "bóndadag" in s
    assert g[5].error_code == "Z001"  # páskadag
    assert g[7].error_code == "Z001"  # verslunarmannahelgina
    assert g[9].error_code == "Z001"  # jólin
    assert g[11].error_code == "Z001"  # bóndadag

    s, g = check(
        "Talað var við Dómsmálaráðherra, Ríkissaksóknara, Biskupinn og Doktorinn " "á Mánudögum og Þriðjudögum."
    )
    assert "dómsmálaráðherra" in s
    # assert "ríkissaksóknara" in s
    # assert "biskupinn" in s
    assert "doktorinn" in s
    assert "mánudögum" in s
    assert "þriðjudögum" in s
    assert g[4].error_code == "Z001"  # dómsmálaráðherra
    # assert g[6].error_code == "Z001"    #ríkissaksóknara
    # assert g[8].error_code == "Z001"  # biskupinn
    assert g[10].error_code == "Z001"  # doktorinn
    assert g[12].error_code == "Z001"  # mánudögum
    assert g[14].error_code == "Z001"  # þriðjudögum

    s, g = check(
        "Þau læra Íslensku og Landafræði með Allsherjarþinginu og Öryggisráðinu en líka um Indóevrópsk mál og Óðinshana."
    )
    assert "íslensku" in s
    assert "landafræði" in s
    assert "allsherjarþinginu" in s
    assert "öryggisráðinu" in s
    assert "indóevrópsk" in s
    assert g[3].error_code == "Z001"  # íslensku
    assert g[5].error_code == "Z001"  # landafræði
    assert g[7].error_code == "Z001"  # allsherjarþinginu
    assert g[9].error_code == "Z001"  # öryggisráðinu
    assert g[13].error_code == "Z001"  # indóevrópsk
    assert g[16].error_code == "Z001"  # óðinshana

    s, g = check("Í Vín má kaupa Vínartertu og Rínarvín en Póstmódernismi og Maríutásur eru vinsælar.")
    assert "Vín" in s
    assert "vínartertu" in s
    assert "rínarvín" in s
    assert "póstmódernismi" in s
    assert "maríutásur" in s
    assert g[5].error_code == "Z001"  # vínartertu
    assert g[7].error_code == "Z001"  # rínarvín
    assert g[9].error_code == "Z001"  # póstmódernismi
    assert g[11].error_code == "Z001"  # maríutásur


def test_inflectional_errors(verbose=False):
    # beygingarvillur
    # sama og test_error_forms í test_tokenizer.py
    s, g = check("Tréið gekk til rekstar rúmmsins.")
    assert "Tréð" in s
    assert "rekstrar" in s
    assert "rúmsins" in s
    assert g[1].error_code == "S002"  # tréð
    assert g[4].error_code == "S002" or g[4].error_code == "ASLSTAFVANTAR"  # rekstrar
    assert g[5].error_code == "S002"  # rúmsins

    s, g = check("Þér finndist víðfermt í árverkni.")
    assert "fyndist" in s
    assert "víðfeðmt" in s  # TODO greinir sem so. víð-ferma!
    assert "árvekni" in s
    assert g[2].error_code in frozenset(["S001", "I4Y"])  # fyndist
    assert g[3].error_code == "S004"  # víðfeðmt
    assert g[5].error_code in frozenset(["S001", "EKKIORD"])  # árvekni

    s, g = check("Ein kúin kom aldrei til baka vegna eldingunnar.")
    assert "kýrin" in s
    assert "eldingarinnar" in s
    assert g[2].error_code == "BEYGVILLA"  # kýrin
    assert g[7].error_code == "S002"  # eldingarinnar

    s, g = check("Lítum til áttunda áratugsins til upplýsingu.")
    assert "áratugarins" in s
    assert "upplýsingar" in s
    assert g[4].error_code == "S001"  # áratugarins
    # assert g[6].error_code == "P_NT_FsMeðFallstjórn"  # TODO Picked up in sentence-level annotation, not available in the tokens

    s, g = check("Nánar tiltekið árins 1978, fjóru og hálfu ári fyrir byltinguna.")
    assert "ársins" in s
    assert "fjórum" in s
    assert g[2].error_code == "S001"  # ársins
    assert g[5].error_code in frozenset(["S001", "ASLSTAFVANTAR"])  # fjórum

    s, g = check("Frumkvöðullinn aldist upp í litlu sjávarþorpi án föðurs og ýmsra þæginda.")
    assert "ólst upp" in s
    assert "föður" in s
    assert "ýmissa" in s
    assert g[2].error_code in frozenset(["S002", "S001"])  # ólst
    assert g[8].error_code == "BEYGVILLA"  # föður
    assert g[10].error_code == "S002"  # ýmissa

    s, g = check(
        "Friðsælari leið hefði verið að hundruðir leituðu í geiminum að kílómeter " "af féinu vegna ástandins."
    )
    assert "Friðsælli" in s
    assert "hundruð" in s
    assert "geimnum" in s
    # assert "kílómetra" in s # TODO Parsing chooses 'kílómetri'
    # assert "fénu" in s # TODO cut out of sentence for now
    assert "ástandsins" in s
    assert g[1].error_code == "S002"  # friðsælli
    assert g[6].error_code in frozenset(["S001", "BEYGVILLA"])  # hundruð
    assert g[9].error_code in frozenset(["S001", "I40-ÞGF"])  # geimnum
    assert g[11].error_code == "S001"  # kílómetra
    assert g[13].error_code == "S002"  # fénu
    assert g[15].error_code == "S002"  # ástandsins

    s, g = check("Loks gekk hann til Selfosss tuttugusta dag samningins.")
    assert "Selfoss" in s
    assert "tuttugasta" in s
    assert "samningsins" in s
    assert g[5].error_code == "S002"  # Selfoss
    assert g[6].error_code in frozenset(["S001", "BEYGVILLA"])  # tuttugasta
    assert g[8].error_code in frozenset(["S001", "S-EFGR"])  # samningsins


def test_wrong_first_parts(verbose=False):
    # C004: Rangur fyrri hluti í samsetningu    (heyrna-laus, náms-skrá)

    s, g = check("Kvenngormar eru feyknaskemmtilegir en ekki fyrnauppteknir.")
    assert "Kvengormar" in s
    assert "Kvenngormar" not in s
    # assert "feiknaskemmtilegir" in s      # TODO virkar ekki eins og er
    # assert "feyknaskemmtilegir" not in s
    assert "firnauppteknir" in s
    assert "fyrnauppteknir" not in s

    # assert g[1].error_code == "C004"    # Fæ C002; eftir að útfæra villukóða
    # assert g[3].error_code == "C004"    # Virkar ekki
    # assert g[6].error_code == "C004"    # TODO fæ C002

    s, g = check("Ég fékk heyrnatól hjá eyrnarlækninum.")
    assert "heyrnartól" in s
    assert "heyrnatól" not in s
    # assert "eyrnalækninum" in s       # TODO Ætti að virka þegar geri ný orðanet
    # assert "eyrnarlækninum" not in s
    assert g[3].error_code == "C006"
    # assert g[5].error_code == "C006"  # TODO virkar ekki

    s, g = check("Lundúnarloftið er næringaríkt í ár.")
    # assert "Lundúnaloftið" in s
    # assert "Lundúnarloftið" not in s
    assert "næringarríkt" in s
    assert "næringaríkt" not in s
    # assert g[1].error_code == "C006"
    assert g[3].error_code == "C006"

    s, g = check("Öldungardeildarþingmaðurinn keyrði díselbíl á hringveginum.")
    # assert "Öldungadeildarþingmaðurinn" in s      # TODO Ætti að virka þegar ný orðanet
    # assert "Öldungardeildarþingmaðurinn" not in s
    # assert "dísilbíl" in s                        # TODO Ætti að virka þegar ný orðanet
    # assert "díselbíl" not in s
    # assert g[1].error_code == "C006"
    # assert g[3].error_code == "C006"


def test_single_first_parts(verbose=False):
    # C003: Stakir fyrri hlutar í setningu sem alveg viss um      (all kaldur, )
    # C005: Mögulega stakir fyrri hlutar en ekki viss, uppástunga um sameiningu
    s, g = check("Hann var all kaldur þegar hann fannst enda var hann hálf ber.")
    # assert "allkaldur" in s           # Sameina ekki, því 'kaldur' gæti verið no.
    # assert "all kaldur" not in s
    assert "hálfber" in s
    assert "hálf ber" not in s
    assert g[3].error_code in frozenset(["C005/w", "S001"])  # all
    assert g[11].error_code == "C003"  # hálfber

    s, g = check("Hún setti honum afar kosti í for vinnunni.")
    assert "afarkosti" in s
    assert "afar kosti" not in s
    assert "forvinnunni" in s
    assert "for vinnunni" not in s
    assert g[4].error_code == "C003"
    assert g[6].error_code == "C003"

    s, g = check("Hér er afbragðs matur fyrir allsherjar gesti í langtíma gistingu.")
    assert "afbragðsmatur" in s
    assert "allsherjargesti" in s
    assert "langtímagistingu" in s
    assert g[3].error_code == "C003"
    assert g[5].error_code == "C003"
    assert g[7].error_code == "C003"


def test_single_last_parts(verbose=False):
    # M003: Stakir seinni hlutar í setningu     (græn keri, arf beri, barn dómur)

    s, g = check("Hann gekk í barn dóm þegar hann komst að því að hún var líka í hópi græn kera.")
    # assert "barndóm" in s                 # TODO Eftir að útfæra
    # assert "barn dóm" not in s
    # assert "grænkeri" in s                # TODO Eftir að útfæra
    # assert "græn keri" not in s
    # assert g[4].error_code == "C003"      # TODO Eftir að útfæra villukóða
    # assert g[16].error_code == "C003"     # TODO Eftir að útfæra villukóða


def test_wrong_parts(verbose=False):
    # M004: Rangt orð finnst í samsetningu      (trukkalessa, kúardella)
    # Getur verið tabúorð, röng beygingarmynd, ...
    s, g = check("Loftlagsmál eru vandamál skráningastarfsmanna.")
    assert "Loftslagsmál" in s
    assert "Loftlagsmál" not in s
    assert "skráningarstarfsmanna" in s
    assert "skráningastarfsmanna" not in s
    # assert g[1].error_code == "M004"      # TODO Eftir að útfæra villukóða, fæ C002
    # assert g[4].error_code == "M004"      # TODO Eftir að útfæra villukóða, fæ C002

    s, g = check("Trukkalessur þola ekki kúardellu.")
    # assert "kúadellu" in s            # TODO Virkar ekki eins og er.
    # assert "kúardellu" not in s
    # assert g[1].error_code == "T001"    # TODO Eftir að útfæra. Tabúorð ætti að merkja sem slík.
    # assert g[4].error_code == "M004"    # TODO Eftir að útfæra, fæ U001. En beygingarvillur?


def test_non_single_first_parts(verbose=False):
    # C002: Fyrri hluti á að vera stakur        (fjölnotapappír, ótalmargir)
    s, g = check("Það er alhliðavandamál hvað ótalmargir fjölnotahestar eru afarleiðinlegir.")
    assert "alhliða vandamál" in s
    assert "alhliðavandamál" not in s
    # assert "ótal margir" in s         # TODO virkar ekki því "ótalmargur" er í BÍN!
    # assert "ótalmargir" not in s      # TODO Ætla að merkja slíkar færslur sem villur í CID/CD_error_forms
    assert "fjölnota hestar" in s
    assert "fjölnotahestar" not in s
    # assert "afar leiðinlegir" in s    # TODO virkar ekki af því að "afarleiðinlegur" er í BÍN!
    # assert "afarleiðinlegir" not in s # TODO Ætla að merkja slíkar færslur sem villur í CID/CD_error_forms
    assert g[3].error_code == "C002"  # TODO Eftir að útfæra, fæ C002
    # assert g[5].error_code == "C002"  # TODO Eftir að útfæra
    # assert g[6].error_code == "C002"  # TODO Eftir að útfæra, fæ C002
    # assert g[8].error_code == "C002"  # TODO Virkar ekki

    s, g = check("Það er betra að vera ofgóður en ofursvalur.")
    # assert "of góður" in s        # TODO Virkar ekki eins og er
    # assert "ofgóður" not in s
    # assert "ofur svalur" in s       # TODO Reglurnar segja til um að þetta sé réttara en ég er bara ekki sammála!
    # assert "ofursvalur" not in s
    # assert g[6].error_code == "C002"  # TODO
    # assert g[8].error_code == "C002"  # TODO

    s, g = check("Það er allrabest að eiga ótalhesta í margnotapokanum.")
    assert "allra best" in s
    assert "allrabest" not in s
    assert "ótal hesta" in s
    assert "ótalhesta" not in s
    # assert "margnota pokanum" in s  # TODO: Þetta er í BÍN en merkt sem rangur rithtáttur. Hugsanleg eitthvað sem við getum gert?
    # assert "margnotapokanum" not in s
    assert g[3].error_code == "C002"
    # assert g[6].error_code == "C002"  # TODO Ætti að virka...
    # assert g[8].error_code == "C002"

    s, g = check("Það er lágmarkskurteisi að tebollinn sé velsætur.")
    assert "lágmarkskurteisi" in s
    assert "lágmarks kurteisi" not in s
    # assert "vel sætur" in s           # TODO virkar ekki
    # assert "velsætur" not in s            # TODO virkar ekki
    assert not g[3].error_code
    # assert g[7].error_code == "C002"


def test_inquiry_verb_forms(verbose=False):
    # athuga hvort eintöluform séu til staðar?
    # athuga hvort fleirtala sé leiðrétt
    s, g = check("Þegar þið hafið hrært deigið setjiði það í ofninn.")
    # assert "setjið " in s      # TODO eftir að útfæra
    # assert "setjið þið" in s   # TODO eftir að útfæra
    # assert g[6].error_code == "Q001"   # TODO eftir að útfæra villukóða

    s, g = check("Eftir að kakan kemur úr ofninum náiði í kremið.")
    # assert "náið þið" in s    # TODO eftir að útfæra, spelling.py virðist taka á þessu
    # assert "náiði" not in s     # TODO eftir að útfæra
    # assert g[7].error_code == "Q001"    # TODO eftir að útfæra villukóða


def test_taboo_words(verbose=False):
    # Simple test
    s, g = check("Júðarnir og hommatittirnir hoppuðu ásamt halanegrunum.")
    assert g[1].error_code == "T001/w"
    assert g[3].error_code == "T001/w"
    assert g[6].error_code == "T001/w"

    s, g = check("Jón ætlaði að afhomma Pál en Múhameðstrúarmennirnir komu í veg fyrir það.")
    assert g[4].error_code == "T001/w"
    assert g[7].error_code == "T001/w"

    # Should not be allowed in compounds
    s, g = check("Merartussan henti mér af kuntubaki.")
    assert g[1].error_code == "T001" or g[1].error_code == "Y001/w"
    # assert g[5].error_code == "T001" or g[5].error_code == "Y001/w" # TODO Finnst ekki


def test_wrong_whitespace(verbose=False):
    s, g = check("Þetta var gert ti lað vekja hrútinn ein sog til stóð.")
    # assert "til að" in s      # TODO eftir að bæta við
    # assert "ti lað" not in s  # TODO eftir að bæta við
    # assert "eins og" in s
    # assert "ein sog" not in s
    # assert g[4].error_code == "S005"
    # assert g[8].error_code == "S005"


def test_correct_words(verbose=False):
    # Athuga hvort hér greinist nokkuð villa
    s, g = check("Ég fann nokkurs konar skógardverg ofan í skúffunni en David " "Schwimmer vissi allt um mannætuapana.")
    assert s == "Ég fann nokkurs konar skógardverg ofan í skúffunni en David " "Schwimmer vissi allt um mannætuapana."
    assert not any(w.error_code for w in g)

    s, g = check(
        "Ökumaður bílaleigubíls komst í hann krappan á Grandanum í Reykjavík skömmu " "fyrir klukkan 11 í dag."
    )
    assert s == "Ökumaður bílaleigubíls komst í hann krappan á Grandanum í " "Reykjavík skömmu fyrir klukkan 11 í dag."
    assert not any(w.error_code for w in g)

    s, g = check(
        "Þá telur hann kjarasamninga stuðla að stöðugleika sem einnig undirbyggi " "frekari stýrivaxtalækkanir."
    )
    assert s == "Þá telur hann kjarasamninga stuðla að stöðugleika sem einnig " "undirbyggi frekari stýrivaxtalækkanir."
    assert not any(w.error_code for w in g)


# Tests for checker.py


def test_NP_agreement(verbose=False):
    # Beygingarsamræmi
    # fjöldi X gerðu... P_NT_FjöldiHluti
    # einn af X gerðu

    # Sambeyging / rétt orðmynd notuð
    s = "Ég fór frá Pétur Páli um miðnætti."
    # TODO Fæ ekki villu, 'Pétur Páli' er sameinað í nafn áður en
    # fallið er tékkað virðist vera.
    # check_sentence(api, s, [(3, 5, "P_NT_X")])
    s = "Hann hélt utan um dóttir sína."
    check_sentence(api, s, [(3, 4, "P_NT_FsMeðFallstjórn")])
    s = "Barnið var með kaldar fingur en heitar fætur."
    # TODO villurnar greinast ekki, vantar líklega reglur.
    # check_sentence(api, s, [(4, 6, "P_NT_KynInnanNafnliðar"), (6, 8, "P_NT_Fall")])
    s = "Miklar umræður eiga sér stað innan verkalýðsfélagsins Eflingu."
    # TODO  FsMeFallstjórn greinir villu, en nær ekki yfir Eflingu,
    # eitthvað skrýtið á ferðinni!
    # Vil fá reglu sem heitir FallInnanNafnliðar og á að ná yfir 5, 8.
    # check_sentence(api, s, [(5, 7, "P_NT_FsMeðFallstjórn")])
    s = "Fyrirtækið er rekið með fimm prósent halla en verðið er sjö prósent lægra."
    # TODO hvorug villan greinist. Vil reglu sem heitir Prósent...
    # eða eitthvað í þá áttina.
    # Ath. hvort það sé nokkuð regla sem heitir það núna.
    # check_sentence(api, s, [(5, 7, "P_NT_Prósent"), (11, 13, "P_NT_Prósent")])
    s = "Stúlkan kom ásamt fleirum konum í bæinn."
    # TODO villan greinist ekki, eftir að höndla
    # check_sentence(api, s, [(3, 5, "P_NT_Fleirum")])
    s = "Þetta er einhvert mesta óheillaráð sem ég hef heyrt."
    # TODO villan greinist sem S001, viljum við höndla þetta sem beygingarsamræmisvillu
    # frekar? Þetta er ósamhengisháð.
    # check_sentence(api, s, [(2, 2, "P_NT_Einhver")])
    s = "Hún heyrði einhvað frá háaloftinu."
    # TODO villan greinist sem S001, viljum við höndla þetta frekar sem
    # beygingarsamræmisvillu? Þetta er ósamhengisháð.
    # check_sentence(api, s, [(2, 2, "P_NT_Einhver")])


def test_number_agreement(verbose=False):
    # Tala
    s = "Fleiri en einn slasaðist í árekstrinum."
    check_sentence(api, s, [(3, 3, "P_NT_ÍTölu")])
    s = "Hann er einn þeirra sem slasaðist í árekstrinum."
    # check_sentence(api, s, [(5, 6, "P_NT_Þeirra")])        # TODO engin villa finnst.
    s = "Minnihluti starfsmanna samþykktu samninginn."
    check_sentence(api, s, [(2, 2, "P_NT_FjöldiHluti")])
    # TODO villan greinist, en ætti að vera staðsett á sögninni
    # til að hægt sé að leiðrétta hana... Hvernig er þetta leiðrétt? Er bara ábending?
    s = "Helmingur landsmanna horfðu á barnaefnið."
    check_sentence(api, s, [(2, 2, "P_NT_FjöldiHluti")])
    # TODO villan greinist en ætti að vera staðsett á sögninni svo hægt sé
    # að leiðrétta hana. Skoða hvernig/hvort villan er leiðrétt.
    s = "Hér eru tuttugu og ein appelsínur."
    # check_sentence(api, s, [()])


def test_gender_agreement(verbose=False):
    # Kyn
    # s = "Foreldrar hans voru skildir."
    # TODO þetta mætir afgangi en væri gott að koma inn.
    # check_sentence(api, s, [(4, 5, "P_NT_Foreldrar")])
    # s = "Stúlkan varð ekki var við hávaðann."
    # TODO fæ enga villu, eftir að útfæra.
    # check_sentence(api, s, [(3, 4, "P_NT_SagnfyllingKyn")])
    pass


def test_verb_agreement(verbose=False):
    # Sagnir
    s = "Konunni vantar að kaupa rúðusköfu."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_þf")])
    s = "Mér kvíðir fyrir að byrja í skólanum."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Ég dreymi um skjaldbökur sem synda um hafið."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_nf_þf")])
    s = "Feimni drengurinn hélt sig til hlés þar til þolinmæðin þraut."
    # TODO: Engin villa greinist; er þetta í Verbs.conf?
    # check_sentence(api, s, [(3, 3, "P_WRONG_CASE_þf_þgf"), (8, 8, "P_WRONG_CASE_nf_þf")])
    s = "Kúrekinn hafði upp á kúnum á sléttunni."
    # TODO greinist ekki, þetta á algerlega eftir að útfæra betur þegar þetta er
    # komið inn í Verbs.conf. Þetta er líklega ekki réttur villukóði.
    # check_sentence(api, s, [(2, 2, "P_WRONG_PARTICLE_uppi")])
    s = "Maðurinn dáðist af málverkinu."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")])
    s = "Barnið á hættu á að detta í brunninn."
    # TODO erfitt að eiga við, líklega ekki réttur villukóði, bæta við Verbs.conf.
    # check_sentence(api, s, [(1, 1, "P_WRONG_FORM")])
    s = "Hetjan á heiður að björguninni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Ferðafólkið fór erlendis að leita lamba."
    # TODO villan greinist ekki. Komið í Verbs.conf? Líklega ekki réttur villukóði.
    # check_sentence(api, s, [(2, 3, "P_WRONG_PARTICLE_til_útlanda")])
    s = "Túlkurinn gaf í skin að mælandi hefði misskilið túlkinn."
    # TODO villan greinist ekki. Komið í Verbs.conf? Líklega ekki réttur villukóði.
    # check_sentence(api, s, [(2, 4, "P_WRONG_PP_í_skyn")])


def test_hvor_annar(verbose=False):
    # s = "Drengirnir héldu fast utan um hvorn annan."
    # check_sentence(api, s, [(3, 7, "P_NT_HvorAnnar")])      # TODO engin villa greinist; eftir að útfæra villureglu
    # s = "Hringirnir voru í hvorum öðrum."
    # check_sentence(api, s, [(2, 5, "P_NT_HvorAnnar")])      # TODO engin villa greinist; eftir að útfæra villureglu
    pass


def test_phrasing(verbose=False):
    s = "Ég vill ekki gera mál úr þessu."
    # TODO þetta virkar, en skoða lengdina.
    check_sentence(
        api, s, [(0, 1, "P_wrong_person"), (1, 1, "S005")]
    )  # TODO breyta aftur í 0, 1, P_wrong_person ef ætlunin er að merkja aðeins eina villu.
    s = "Konur vilja í auknu mæli koma að sjúkraflutningum."
    # TODO á kannski að greina þetta öðruvísi? Fastur frasi? Skoða líka lengdina.
    check_sentence(api, s, [(2, 4, "P_wrong_gender"), (3, 3, "S005")])
    s = "Ég veit ekki hvort að ég komi í kvöld."
    check_sentence(api, s, [(4, 4, "P_NT_Að/w")])
    s = "Meðan veislunni stendur verður frítt áfengi í boði."
    # check_sentence(api, s, [(0, 3, "P_NT_MeðanStendur")])      # TODO engin villa finnst; hér þarf merkingargreiningu!


def test_munu(verbose=False):
    # s = "Ég mun aldrei gleyma þessu."
    # check_sentence(api, s, [(1, 1, "P_NT_Munu")])
    # s = "Hundurinn mun verða vinur minn að eilífu."
    # check_sentence(api, s, [(1, 1, "P_NT_Munu")])
    pass


def test_vera(verbose=False):
    # vera að + so.nh.
    s = "Ég er ekki að skilja þetta."
    check_sentence(api, s, [(1, 4, "P_VeraAð")])
    s = "Ég er að fara í sund þegar ég fæ símtalið."
    # check_sentence(api, s, [(1, 3, "P_VeraAð")])
    s = "Hún er að skrifa vel."
    # check_sentence(api, s, [(1, 3, "P_VeraAð")]) # Greinist ekki lengur sem villa, undanskil 3.p. pfn. því geta verið ómannleg.
    s = "Kristín er að skrifa vel."
    check_sentence(api, s, [(1, 3, "P_VeraAð")])

    s = "Það gekk mikið á þegar Jack Valentine var ekki að sofa."
    # check_sentence(api, s, [(7, 10, "P_VeraAð")])   # TODO 'Jack Valentine' is an entity, like Ikea. Should it be allowed to fire an error rule?


def test_nhm(verbose=False):
    # s = "Ég ætla fara í búð."
    # check_sentence(api, s, [(2, 3, "P_Að")])        # TODO villan greinist ekki, eftir að útfæra. Ætti að vera í Verbs.conf
    # s = "Hún ætlar að fara lesa um skjaldbökur."
    # check_sentence(api, s, [(3, 4, "P_Að")])        # TODO villan greinist ekki, eftir að útfæra. Ætti að vera í Verbs.conf
    pass


def test_new_passive(verbose=False):
    # s = "Það var gert grein fyrir stöðu mála."
    # check_sentence(api, s, [(2, 2, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu
    # s = "Lagt verður áhersla á að skoða reikningana."
    # check_sentence(api, s, [(0, 0, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu
    # s = "Það verður lagt áherslu á að skoða reikningana."
    # check_sentence(api, s, [(2, 4, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu
    pass


def test_verb_arguments(verbose=False):
    # TODO breyta prófuninni svo falli að mynsturgreininum.
    s, g = check("Vefurinn bíður upp á bestu fréttirnar.")
    g = list(g)
    if verbose:
        dump(g)
    s = normalize(g)
    # assert "Vefurinn býður" in s  # TODO eftir að útfæra
    # assert "Vefurinn bíður" not in s  # TODO eftir að útfæra
    # assert g[2].error_code == "V001"  # TODO eftir að ákveða villukóða

    s = "Kirkjuna bar við himinn þegar við komum þar um morguninn."
    # TODO Verbs.conf ætti að dekka þetta -- útfæra goggunarröð?
    check_sentence(api, s, [(3, 3, "P_NT_FsMeðFallstjórn")])


def test_complex_sentences(verbose=False):
    _ = (
        "Drengurinn dreif sig inn þegar hann heyrði í bjöllunni af því að hann "
        "vildi sjá hvort það væri kominn nýr kennari en sem betur fer var gamli "
        "kennarinn á sínum stað svo að hann settist niður í rólegheitum og tók "
        "upp bækurnar."
    )
    # check_sentence(api, s, [0, 0, "P_COMPLEX"])      # TODO eftir að útfæra
    _ = (
        "Tromman sem var í skápnum sem hafði brotnað í óveðrinu sem var daginn "
        "sem þau keyptu kexið sem var ónýtt þegar þau komu úr búðinni sem þau "
        "keyptu það í hafði skekkst."
    )
    # check_sentence(api, s, [0, 0, "P_COMPLEX"])      # TODO eftir að útfæra


def test_tense_mood(verbose=False):
    s = "Ég kláraði verkefnið þrátt fyrir að ég var syfjaður."
    check_sentence(api, s, [(6, 6, "P_MOOD_ACK")])
    s = "Hann kemur ef hann geti."
    check_sentence(api, s, [(4, 4, "P_MOOD_COND")])
    # s = "Hún kemur ef það sé gott veður."
    # check_sentence(api, s, [(2, 7, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    # s = "Hún segir að veðrið var gott í dag."
    # check_sentence(api, s, [(1, 8, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    # s = "Hann sagði að veðrið er gott í dag."
    # check_sentence(api, s, [(1, 8, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    pass


def test_noun_style(verbose=False):
    # Ekki í forgangi
    # nafnorðastíll
    # s = "Stofnunin framkvæmdi könnun á aðstæðum á vinnustað."
    # check_sentence(api, s, [(1, 3, "P_Nafnorðastíll")])        # TODO greinist ekki, eftir að útfæra -- þetta gæti virkað vel í Verbs.conf!
    pass


def test_missing_word(verbose=False):
    # Ekki í forgangi
    # s = "Það er mjög mikilvægt þið lesið þennan póst."
    # check_sentence(api, s, [(4, 4, "P_NT_Að")])    # TODO engin villa finnst, eftir að útfæra
    # s = "Það mjög mikilvægt að þið lesið þennan póst."
    # check_sentence(api, s, [(1, 1, "P_NT_SögnVantar")])     # TODO engin villa finnst, eftir að útfæra
    pass


def test_foreign_sentences(verbose=False):
    s = (
        "Brooks Koepka lék hringinn á þremur undir pari og er því "
        "líkt og Thomas og Schauffele á tíu höggum undir pari."
    )
    # check_sentence(api, s, [(14, 14, "U001/w")])        # Changed so capitalized unknown words are not marked as errors
    s = (
        "If you asked people to try to picture hunting for truffles, the "
        "expensive subterranean fungi, many would no doubt imagine men "
        "with dogs going through woodlands in France or Italy."
    )
    check_sentence(api, s, [(0, 31, "E004")], is_foreign=True)
    s = "Rock and roll er rollicking great fun."
    check_sentence(api, s, [(0, 6, "E004")], is_foreign=True)


def test_conjunctions(verbose=False):
    s = "Ef að pósturinn kemur ekki á morgun missi ég vitið."
    check_sentence(api, s, [(1, 1, "P_NT_Að/w")])
    s = "Hafsteinn vissi svarið þótt að hann segði það ekki upphátt."
    check_sentence(api, s, [(4, 4, "P_NT_Að/w")])
    s = "Hafsteinn vissi svarið þó hann segði það ekki upphátt."
    check_sentence(api, s, [(3, 3, "P_NT_ÞóAð")])
    s = "Ég kem á hátíðina víst að pabbi þinn kemst ekki."
    check_sentence(api, s, [(4, 4, "P_NT_VístAð")])
    s = "Ég kem á hátíðina fyrst að pabbi þinn kemst ekki."
    check_sentence(api, s, [(5, 5, "P_NT_Að/w")])
    s = "Hatturinn passar á höfuðið nema að það sé eyrnaband undir honum."
    check_sentence(api, s, [(5, 5, "P_NT_Að/w")])
    s = "Hún grét þegar að báturinn sást ekki lengur."
    check_sentence(api, s, [(3, 3, "P_NT_Að/w")])
    s = "Hún hélt andliti á meðan að hann horfði til hennar."
    check_sentence(api, s, [(4, 4, "P_NT_Að/w")])
    s = "Annaðhvort ferðu í buxurnar núna."
    # check_sentence(api, s, [(5, 5, "P_NT_AnnaðhvortEða")])     # TODO engin villa finnst, eftir að útfæra
    s = "Hvorki hatturinn passaði á höfuðið."
    # check_sentence(api, s, [(0, 0, "P_NT_HvorkiNé")])          # TODO engin villa finnst, eftir að útfæra


def test_impersonal_verbs(verbose=False):
    s = "Ég dreymdi að það væri hundur í fiskabúrinu mínu."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_nf_þf")])
    s = "Hestinum dreymdi að það væri hundur í fiskabúrinu."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_þf")])
    s = "Mér klæjar undan áburðinum."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_þf")])
    # s = "Hann sagði að konan hefði misminnt að potturinn væri með loki."
    # check_sentence(api, s, [(3, 3, "P_WRONG_CASE_nf_þf")])
    s = "Bréfberinn spurði hvort Páli vantaði fleiri frímerki."
    check_sentence(api, s, [(3, 3, "P_WRONG_CASE_þgf_þf")])
    s = "Lögfræðingnum sem ég fékk til þess að verja mig í jarðaberjastuldarmálinu " "hlakkaði til að losna við mig."
    # TODO greinist, en spanið gæti verið réttara.
    check_sentence(api, s, [(0, 10, "P_WRONG_CASE_þgf_nf")])
    s = "Tröllskessan dagaði uppi."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_nf_þf")])
    s = "Báturinn rak á land."
    # check_sentence(api, s, [(0, 0, "P_SUBJ_CASE")])        # TODO erfitt að ráða við þar sem 'rak' er líka persónuleg sögn


def test_correct_sentences(verbose=False):
    s = "Ráðist var í úttektina vegna ábendinga sem bárust embættinu " "frá notendum þjónustunnar."
    check_sentence(api, s, [])
    s = "Upp úr krafsinu hafði maðurinn samtals 51 pakka af kjúklingabringum " "og 9,2 kíló að auki."
    check_sentence(api, s, [])
    s = "Á göngudeild gigtar á Landspítalanum sé tilvísunum forgangsraðað " "og er meðalbiðtími innan marka."
    check_sentence(api, s, [])


def test_correction_is_valid(verbose=False):
    # Check that 'Ferðavefir' is not corrected (or suggested) to 'Ferðavefur',
    # which doesn't work grammatically
    pass
    # TODO this is corrected, why does that happen?
    # check_sentence(api, s, [])
    # Check that 'hátekju' is not corrected (or suggested) to 'hátekjum'
    # within a hyphenated composite word
    # TODO this is corrected, why?
    # sent = rc.check_single(s)
    # check_sentence(api, s, [])


def test_corrected_sentences(verbose=False):
    # Setningar sem þáttast ekki upprunalega út af villum.
    # Villurnar eru svo leiðréttar í errtokenizer.py.
    # Hér vil ég athuga hvort setningin er þáttuð aftur.
    # TODO prófa hér.
    # s = "Alla sína lífdaga hljóp hún allt hvað fætur toga að ástæðulausu."
    pass


def test_compounds():
    s = "Ég hitti fjármála-og efnahagsráðherra."
    check_sentence(api, s, [(2, 2, "S005")])


def test_styles():
    result = api.correct("Spanendurnir afdjöfluðu á afarorðunum.")
    a = result.sentences[0]
    assert len(a.annotations) == 3
    assert a.annotations[0].code == "Y001/w"
    assert "úrelt" in a.annotations[0].detail
    assert a.annotations[1].code == "Y001/w"
    assert "sjaldgæft" in a.annotations[1].detail
    assert a.annotations[2].code == "Y001/w"
    assert "úrelt" in a.annotations[2].detail
    result = api.correct("Jón átti höfundarétt og spaghetti fyrir sveitastjórnarkosningarnar.")
    a = result.sentences[0]
    assert len(a.annotations) == 3
    assert a.annotations[0].code == "Y001/w"
    assert "villa" in a.annotations[0].detail
    assert "höfundarétt" in a.annotations[0].detail
    assert "spagettí" in a.annotations[1].text
    assert a.annotations[2].code == "Y001/w"
    assert "villa" in a.annotations[2].detail
    assert "sveitastjórnarkosningarnar" in a.annotations[2].detail
    result = api.correct("Kamesið mitt er ferlega óhreint.")
    a = result.sentences[0]
    assert len(a.annotations) == 1
    assert a.annotations[0].code == "Y001/w"
    assert "úrelt" in a.annotations[0].detail
    assert "Kamesið" in a.annotations[0].detail
    assert "Kamersið" in a.annotations[0].detail
    assert a.annotations[0].suggest == "Kamersið"
    result = api.correct("Páll kyngdi belgverska konfektinu.")
    a = result.sentences[0]
    assert len(a.annotations) == 1
    assert a.annotations[0].code == "Y001/w"
    assert "gamalt" in a.annotations[0].detail
    assert "belgverska" in a.annotations[0].detail
    assert "belgíska" in a.annotations[0].detail
    assert a.annotations[0].suggest == "belgíska"


def test_né():
    # Attn. error codes 'P_Né' and 'P_NT_Né' are used,
    # depending on where the error is handled
    s = "Ég fór út né gekk heim."
    check_sentence(api, s, [(3, 3, "P_NT_Né")])
    s = "Ég hvorki fór út né gekk heim."
    check_sentence(api, s, [])
    s = "Við keyptum brauð né ost."
    check_sentence(api, s, [(3, 3, "P_Né")])
    s = "Við keyptum annaðhvort brauð né ost."
    check_sentence(api, s, [(4, 4, "P_Né")])
    s = "Ég hugsa ekki um hvað Bretland gerir né hvað Ísland gerir."
    check_sentence(api, s, [(7, 7, "P_NT_Né")])
    s = "Hann er sakaður um kynþáttafordóma en eftir því sem ég best veit eru fylgdarlaus ungmenni ekki kynþáttur, þjóð né þjóðarbrot."
    check_sentence(api, s, [(19, 19, "P_Né")])
    s = "Alþjóðaheilbrigðisstofnunin hefur gefið út að suðurafríska afbrigði kórónuveirunnar leiði ekki til verri veikinda né sé það banvænna en önnur."
    check_sentence(api, s, [(13, 13, "P_NT_Né")])

    # Some false positives to watch out for
    s = "Hvorki þarf sýnatöku vegna loka sóttkvíar í dag né í lok smitgátar."
    check_sentence(api, s, [])
    s = "Við erum búnir að vera hér í sólarhring og það er ekki enn búið að bæta eitt né neitt."
    check_sentence(api, s, [])
    s = "Læknarnir sögðu að símanum hafi verið smyglað inn í fangelsið en lögreglan vildi hvorki staðfesta né tjá sig um fréttina."
    check_sentence(api, s, [])


def test_ignore_rules(verbose=False):
    """Test error annotation deletion. In some cases, when a certain error type is skipped, another similar error type is applied."""
    # doubling - C001
    ignore_rules = {"C001"}
    s, g = check("Ég hélt mér mér fast í sætið.", ignore_rules=ignore_rules)
    assert not g[3].error_code

    # wrong compounds - C002
    ignore_rules = {"C002"}
    s, g = check(
        "Fötin koma í margskonar litum og fara afturábak afþvíað annarstaðar " "eru fjögurhundruð mikilsháttar hestar.",
        ignore_rules=ignore_rules,
    )
    for ix in range(len(g)):
        # Setningin þáttast ekki fyrst villurnar finnast ekki
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # split compounds
    ignore_rules = {"C003", "C005/w"}
    s, g = check("Aðal inngangur að auka herbergi er gagn stæður öðrum gangi.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # unique_context_independent_errors
    ignore_rules = {
        "S001",
        "S004",
        "ASLSTAFVANTAR",
        "ASLAUKASTAF",
        "ASLVIXL",
        "EKKIORD",
    }
    s, g = check("Fomaður fór til fljúgjandi augnæknis í liltu andyri Svíþjóðar.", ignore_rules=ignore_rules)

    # Errors in fixed phrases (multiword errors)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}
    ignore_rules = {"S005", "P_yyii", "P_kvhv", "P_afað"}
    s, g = check("Alla sýna lífdaga hljóp hún allt kvað fætur toga af ástæðulausu.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # subject case
    ignore_rules = {"P_WRONG_CASE_nf_þf"}
    s, g = check("Ég dreymdi að það væri hundur í fiskabúrinu mínu.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # Pattern errors
    ignore_rules = {"P_NT_Heldur"}
    s, g = check("Gíraffi er stærri heldur en fíll.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # check_style - Y001/w
    ignore_rules = {"Y001/w"}
    s, g = check("Hún er æxling og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # check_taboo_words - T001/w
    ignore_rules = {"T001/w"}
    s, g = check("Hann er typpalingur og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # late_fix_merges - S005
    ignore_rules = {"S005"}
    s, g = check("Hún á fimm miljónir króna og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}
    s, g = check("Það er 1,8 millarður króna sem labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # late_fix_capitalization
    # Z001
    ignore_rules = {"Z001"}
    s, g = check("Hann var Félags- og barnamálaráðherra og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "S005"}

    # Z002
    ignore_rules = {"Z002"}
    s, g = check("félags- og barnamálaráðherra labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "S005"}

    # Z004
    ignore_rules = {"Z004"}
    s, g = check("500 Milljónir löbbuðu um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # Z005
    ignore_rules = {"Z005"}
    s, g = check("Fimm Hundruð milljónir löbbuðu um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # fix_capitalization
    # Z002
    ignore_rules = {"Z002"}
    s, g = check("Hún heitir hrafnhildur benediktsdóttir og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "U001"}
    s, g = check("Hann heitir ásþór harðarson og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "U001"}

    # Z006
    ignore_rules = {"Z006"}
    s, g = check("Hann var í Así og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "S004"}

    # Z003
    ignore_rules = {"Z003"}
    s, g = check("Hann datt 15. Apríl og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # lookup_unknown_words
    # Ritmyndir errors - EI4EY
    ignore_rules = {"EI4EY"}
    s, g = check("Gíraffi er stærri heldur en fíll.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # S001 - Icelandic error corpus nonwords
    ignore_rules = {"S001"}
    s, g = check("Hann á þriðjun í starfsemi og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # S002 - CIDErrorForms
    ignore_rules = {"S002"}
    s, g = check("Hann saknar aðalspurningunnar og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # U001
    ignore_rules = {"U001"}
    s, g = check("Hún er blurbilosiru og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}

    # C006
    ignore_rules = {"C006"}
    s, g = check("Hún var kvennmaður og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001", "NN4N-ORD"}
    s, g = check("Hann var feyknaglaður og labbaði um herbergið.", ignore_rules=ignore_rules)
    for ix in range(len(g)):
        assert not g[ix].error_code or g[ix].error_code in {"E001"}


def test_suppress_suggestions(verbose=False):
    x, y = check(
        "Það var leiðilegt en þæginlegt að koma tímalega á áfangastað um fjögurleitið.",
        suppress_suggestions=False,
    )
    s, g = check(
        "Það var leiðilegt en þæginlegt að koma tímalega á áfangastað um fjögurleitið.",
        suppress_suggestions=True,
    )
    assert y != g


if __name__ == "__main__":
    pass
