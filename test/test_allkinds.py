"""

    test_allkinds.py

    Tests for ReynirCorrect module

    Copyright(C) 2019 by Miðeind ehf.

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


    This module tests both the token-level error detection and correction
    and the sentence-level annotation functionality of ReynirCorrect.

"""

# Run with 'pytest -v' for verbose mode

import reynir_correct as rc
import tokenizer

# tests for errtokenizer.py

def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))

def test_punctuation(verbose=False):
    # Quotation marks
    g = rc.tokenize("Hann var kallaður ,,pottormur\" og var \"hrekkjusvín\".")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "„pottormur“" in s
    assert "„hrekkjusvín“" in s

    # Ellipsis
    g = rc.tokenize("Ég veit ekki...")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "..." not in s
    assert "…" in s                 # Ath. þetta finnst í s, en í viðmótinu birtist þetta ekki í hríslutrénu.

def test_multiple_spaces(verbose=False):
    pass
    g = rc.tokenize("Hér         er langt bil.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Hér er" in s
    assert "  " not in s

def test_doubling(verbose=False):
    """ Test words that are erroneously written twice or more often"""
    # Simple instance
    g = rc.tokenize("Ég hélt mér mér fast í sætið.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "hélt mér fast" in s
    assert "mér mér" not in s
    assert len(g) == 9
    assert g[3].error_code == "C001"    # mér
    
    # Test many instances in same sentence
    g = rc.tokenize("Potturinn kom ekki ekki í ljós ljós fyrr en en í dag dag.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
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
    for ix, t in enumerate(g):
        if ix in errors:
            assert g[ix].error_code == "C001"  # ekki, ljós, en, dag
        else:
            assert not g[ix].error_code

    # Here the first word is just an uppercase version of the following common noun. This should be corrected.
    g = rc.tokenize("Slysið slysið átti sér stað í gærkvöldi.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert len(g) == 9            # TODO útfæra þetta í checker.py, þarf að hafa upplýsingar um meanings.
    # assert "Slysið átti" in s

    # Testing multiple words in a row. This should be corrected.
    g = rc.tokenize("Það er stanslaust fjör fjör fjör fjör fjör fjör fjör fjör í sveitinni.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 9
    assert "stanslaust fjör í" in s
    assert "fjör fjör" not in s
    assert g[4].error_code == "C001"

    # 'á' has different meanings here, 'á á' should be accepted
    # 'en en' should not be accepted
    g = rc.tokenize("Ég á á sem heitir Lína langsokkur en en en hún kann ekki að jarma.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 16
    assert "á á" in s
    assert "Ég á sem" not in s
    assert "langsokkur en hún" in s
    assert "en en" not in s
    assert g[8].error_code == "C001"

def test_accepted_doubling(verbose=False):
    # Test examples with a comma between. This should be accepted.
    g = rc.tokenize("Lífið, sem er flokkar, flokkar potta.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 11
    assert "flokkar, flokkar" in s
    assert not g[4].error_code

    # Another example with a comma between. This should be accepted.
    g = rc.tokenize("Lífið er svaka, svaka gaman.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 9
    assert "svaka, svaka" in s
    assert not g[3].error_code

    # Test whether lowercase instance following uppercase instance is corrected    
    # First, these are separate words and should be accepted. 
    # 'í í' should be pointed out but not corrected, as 'í í' can occur, 
    # for instance as a particle and then a preposition.
    g = rc.tokenize("Finnur finnur gull í í Tálknafirði.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 9
    assert "Finnur finnur" in s
    assert not "Finnur gull" in s
    #assert g[4].error_code == "W001"    # TODO útfæra

    # Same example except now the common noun is capitalized in the beginning,
    # followed by the proper noun. This should be accepted.
    g = rc.tokenize("Finnur Finnur gull í Tálknafirði?")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert len(g) == 8    # TODO útfæra að þetta er ekki leiðrétt
    # assert "Finnur Finnur" in s

    # Here are separate words, as Gaukur is a proper noun. This should be accepted.
    g = rc.tokenize("Gaukur gaukur slasaðist í slysinu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 8
    assert "Gaukur gaukur" in s         # TODO þetta á líklega frekar heima í checker.py, ath. hvort þetta er bæði sérnafn og samnafn og þannig.
    assert "Gaukur slasaðist" not in s

    # 'heldur' is allowed to appear more than once in a row. This should be accepted.
    g = rc.tokenize("Kvikan heldur heldur mikið í jörðina.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 9
    assert "Kvikan heldur heldur mikið" in s
    assert "Kvikan heldur mikið" not in s

    # 'gegn' has different meanings hera, should be accepted
    g = rc.tokenize("Hún var góð og gegn gegn Svíum í úrslitaleiknum.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 12
    assert "og gegn gegn Svíum" in s
    assert "og gegn Svíum" not in s

def test_wrong_compounds(verbose=False):
    g = rc.tokenize("Fötin koma í margskonar litum og fara afturábak afþvíað annarstaðar eru fjögurhundruð mikilsháttar hestar.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 23
    assert "margs konar" in s
    assert "aftur á bak" in s
    assert "af því að" in s
    assert "annars staðar" in s
    assert "fjögur hundruð" in s
    assert "mikils háttar" in s
    errors = {4, 9, 12, 15, 17, 18}
    for ix, t in enumerate(g):
        if ix in errors:
            assert g[ix].error_code == "C002"  # margs konar, aftur á bak, af því að, annars staðar, fjögur hundruð, mikils háttar
        else:
            assert not g[ix].error_code

    g = rc.tokenize("Rólan fór niðrá torg og svo ofan í níuhundruð samskonar seinnihluta.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "niður á" in s
    assert "níu hundruð" in s
    assert "sams konar" in s
    assert "seinni hluta" in s
    errors = {3, 10, 11, 13}
    for ix, t in enumerate(g):
        if ix in errors:
            assert g[ix].error_code == "C002"  # niður á, níu hundruð, sams konar, seinni hluta
        else:
            assert not g[ix].error_code

def test_split_compounds(verbose=False):
    g = rc.tokenize("Aðal inngangur að auka herbergi er gagn stæður öðrum gangi.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert len(g) == 10
    assert "Aðalinngangur" in s
    assert "Aðal inngangur" not in s
    assert "aukaherbergi" in s
    assert "auka herbergi" not in s
    assert "gagnstæður" in s
    assert "gagn stæður" not in s
    errors = {1, 3, 5}
    for ix, t in enumerate(g):
        if ix in errors:
            assert g[ix].error_code == "C003"  # Aðalinngangur, aukaherbergi, gagnstæður
        else:
            assert not g[ix].error_code


    g = rc.tokenize("Myndar drengurinn er hálf undarlegur kvenna megin.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
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
            assert g[ix].error_code == "C003"  # Myndardrengurinn, hálfundarlegur, kvennamegin
        else:
            assert not g[ix].error_code

def test_unique_context_independent_errors(verbose=False):
    # Known, unique, context independent spelling errors - S001
    g = rc.tokenize("Fomaður fór til fljúgjandi augnæknis í liltu andyri Svíþjóðar.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
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
    #assert g[1].error_code == "S001"   # TODO virkar ekki; endar sem S004
    assert g[4].error_code == "S001"
    assert g[5].error_code == "S001"
    assert g[7].error_code == "S001"
    assert g[8].error_code == "S001"

    g = rc.tokenize("Mér tóskt að fá áfarm ókeipis ríkistjórn.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "tókst" in s
    assert "tóskt" not in s
    assert "áfram" in s
    assert "áfarm" not in s
    assert "ókeypis" in s
    assert "ókeipis" not in s
    # assert "ríkisstjórn" in s     # TODO eftir að útfæra
    # assert "ríkistjórn" in s      # TODO eftir að útfæra
    assert g[2].error_code == "S001"
    assert g[5].error_code == "S001"
    assert g[6].error_code == "S001"
    # assert g[7].error_code == "S001"      # TODO eftir að útfæra

    g = rc.tokenize("Þar sat Gunan og fylgdist með frammistöðu liðisins í framlenginunni miklu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "sat Gunna og" in s              # TODO Virkar ekki eins og er út af hástaf í unique_errors
    # assert "sat Gunan og" not in s          # TODO Virkar ekki eins og er út af hástaf í unique_errors
    assert "frammistöðu liðsins í" in s
    assert "frammistöðu liðisins í" not in s
    assert "í framlengingunni miklu" in s     # TODO Þetta virkar í viðmóti en allt í einu ekki hér.
    assert "í framlenginunni miklu" not in s
    # assert g[3].error_code == "S001"  # TODO virkar ekki út af hástaf í unique_errors, setja inn þegar það er lagað
    assert g[8].error_code == "S001"
    assert g[10].error_code == "S001"

def test_other_context_independent_spelling_errors(verbose=False):
    # aðrar ósamhengisháðar, einskiptisvillur og bullorð 
    # S002 ef þetta er leiðréttanlegt; 
    # S003 ef aðeins uppástunga finnst;
    # U001 ef ekkert finnst
    # TODO Fæ ég einhvern tímann W001??
    g = rc.tokenize("Ég fyldist með fóboltanum í sjóvvarpinu í gærköldi.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    #assert "Ég fylgdist með" in s       # TODO breytir þessu í fylgist, er aðeins líklegra út af tíðni. Bæta við þekktar villur/heil beygingardæmi?
    #assert "Ég fyldist með" not in s    
    assert "með fótboltanum í" in s
    assert "með fóboltanum í" not in s
    assert "í sjónvarpinu í" in s
    assert "í sjóvvarpinu í" not in s
    assert "í gærkvöldi" in s           # TODO getur leiðrétt gærköld→gærkvöld, en gærkvöldi virðist ekki vera í safninu.
    assert "í gærköldi" not in s

    # assert g[2].error_code == "S002"  # TODO endar sem S004; "Checking rare word 'fyldist'"
    # assert g[4].error_code == "S002"  # TODO endar sem S004
    # assert g[6].error_code == "S002"  # TODO endar sem S004
    # assert g[8].error_code == "S002"  # TODO endar sem S004


    g = rc.tokenize("Ég fór í ljós tisvar í vigu og mædi regullega í lígamsrætt.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "ljós tvisvar í" in s
    # assert "í viku og" in s     # TODO leiðréttist í 'eigu'
    assert "og mæti" in s
    # assert "reglulega í" in s     # TODO leiðréttist ekki, kemur með uppástungu
    # assert "í líkamsrækt" in s    # TODO leiðréttist ekki, of flókin villa.

    # assert g[5].error_code == "S002"      # TODO endar sem S004
    # assert g[7].error_code == "S002"      # TODO endar sem S004
    # assert g[9].error_code == "S002"      # TODO endar sem S004
    # assert g[10].error_code == "S002"     # TODO endar sem S003! Endar sem bara uppástunga
    assert g[12].error_code == "U001"

def test_context_dependent_spelling_errors(verbose=False):
    # Context dependent spelling errors - P_xxx

    g = rc.tokenize("Alla sýna lífdaga hljóp hún allt kvað fætur toga af ástæðulausu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Alla sína lífdaga" in s
    assert "allt hvað fætur" in s
    assert "toga að ástæðulausu" in s
    # TODO fæ villukóðana á fyrsta orðið í fasta frasanum en ætti að fá á villuorðið sjálft.
    assert g[1].error_code == "P_yi"    # sína
    assert g[6].error_code == "P_khv"  # hvað
    assert g[10].error_code == "P_aðaf" # að


    # Context dependent spelling errors - P_xxx
    g = rc.tokenize("Kvað sem á bjátar lifir en í glæðunum.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Hvað sem" in s
    assert "lifir enn í" in s
    # TODO villan kemur í fyrsta orðið í fasta frasanum en ætti að fá á villuorðið sjálft.
    assert g[1].error_code == "P_khv"   # Hvað
    assert g[5].error_code == "P_n"   # enn

def test_homophones(verbose=False):
    # ruglingsmengin
    g = rc.tokenize("Hann heyrði lágvært kvísl í myrkrinu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    #assert "hvísl" in s    # TODO eftir að útfæra
    # assert "kvísl" not in s   # TODO eftir að útfæra
    # assert g[4].error_code == "S006"  # TODO eftir að útfæra villukóða

    g = rc.tokenize("Kirtillinn flæktist fyrir fótum hennar í fermingunni.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "Kyrtillinn" in s  # TODO eftir að útfæra
    # assert "Kirtillinn" not in s  # TODO eftir að útfæra
    # assert g[0].error_code == "S006"  # TODO eftir að útfæra

    g = rc.tokenize("Tímanum líkur á því að kvatt er til þess að kvika ekki frá sinni stöðu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "lýkur" in s    # TODO eftir að útfæra
    # assert "líkur" not in s   # TODO eftir að útfæra
    # assert "hvatt" in s    # TODO eftir að útfæra
    # assert "kvatt" not in s   # TODO eftir að útfæra
    # assert "hvika" in s    # TODO eftir að útfæra
    # assert "kvika" not in s
    # assert g[2].error_code == "S006"
    # assert g[6].error_code == "S006"
    # assert g[11].error_code == "S006"

    g = rc.tokenize("Við rímum húsið til að leifa eldinum ekki að hvelja fólkið.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "rýmum" in s    # TODO eftir að útfæra
    # assert "rímum" not in s
    # assert "leyfa" in s    # TODO eftir að útfæra
    # assert "leifa" not in s
    # assert "kvelja" in s   # TODO eftir að útfæra
    # assert "hvelja" not in s
    # assert g[2].error_code == "S006"
    # assert g[6].error_code == "S006"
    # assert g[10].error_code == "S006"

def test_paradigm_spelling_errors(verbose=False):

    # Unique errors in whole paradigm - S001
    g = rc.tokenize("Það var leiðilegt en þæginlegt að koma tímalega á áfangastað um fjögurleitið.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    #assert "leiðinlegt" in s   # TODO er þetta ekki í þekktu villunum sem á eftir að koma inn?
    #assert "þægilegt" in s     # TODO sama
    #assert "tímanlega" in s     # TODO sama
    # assert "fjögurleytið" in s    # TODO sama
    # assert g[3].error_code == S007  # lagfært en með spelling.py, endar sem S002 eða S003.
    # assert g[5].error_code == S007  # lagfært en með spelling.py, endar sem S002 eða S003.
    # assert g[8].error_code == S007  # TODO greinist ekki, eftir að setja inn
    # assert g[12].error_code == S007  # lagfært en með spelling.py, endar sem S002 eða S003.

    g = rc.tokenize("Barnið var fjagra ára þegar það fór janframt til ýmissra annara landa að leita að síðastu kúinni en það var til einskins.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "fjögurra" in s
    assert "jafnframt" in s
    assert "ýmissa" in s
    #assert "annarra" in s          # TODO leiðréttist ekki, er í ErrorForms en er samhengisháð villa
    assert "síðustu" in s
    assert "kúnni" in s
    # assert "einskis" in s         # TODO leiðréttist ekki.
    # assert "einskins" not in s

    errors = {3, 8, 10, 11, 16, 17, 22}
    #for ix, t in enumerate(g):                 # TODO virkar ekki, eftir að útfæra villukóðann og skipta villunum upp eftir eðli 
    #    if ix in errors:                       # TODO þarf þá að uppfæra dæmin.
    #        assert g[ix].error_code == "S007"  # fjagra, janframt, ýmissra, annara, síðastu, kúinni, einskins
    #    else:
    #        assert not g[ix].error_code

def test_rare_word_errors(verbose=False):
    # S004, spelling.py
    g = rc.tokenize("Hann finur fyri alls kins verkjum.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    #assert "Hann finnur" in s      # TODO Fæ uppástungu en ekki nógu sterka leiðréttingu
    # assert "fyrir" in s           # TODO Virkar ekki
    assert "kyns" in s
    # assert g[2].error_code == "S004"  # TODO virkar ekki, fæ S001
    # assert g[3].error_code == "S004"  # TODO virkar ekki, fæ W001 virðist vera.
    assert g[5].error_code == "S004"


    g = rc.tokenize("Hann skoðaði arða gluggs en leists ekki vel á neinn.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "aðra" in s                # TODO Virðist ekki virka!
    # assert "glugga" in s              # TODO Virðist ekki virka!
    # assert "leist" in s               # TODO Virðist ekki virka!
    # assert g[3].error_code == "S004"  # TODO Virðist ekki virka!
    # assert g[4].error_code == "S004"  # TODO Virðist ekki virka! Finn S001
    # assert g[6].error_code == "S004"  # TODO Virðist ekki virka! Finn S001

def test_wrong_abbreviations(verbose=False):
    # TODO vil fá A001

    g = rc.tokenize("Karlinn datt þ.á.m. í amk. fimm polla.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "þ. á m." in s  # TODO ekki gert rétt eins og er, þarf að bæta við þekktar villur. Get búið til sérfall í errtokenizer.py, verið með lítið safn í ReynirCorrect.conf.
    # assert "þ.á.m." not in s
    #assert "a.m.k." in s    # TODO býr þetta til og S001, en setur aukapunkt og býr til nýja setningu eftir þetta!
    assert "amk." not in s
    # assert g[3].error_code == "S001"      # TODO eftir að útfæra
    # assert g[7].error_code == "S001"      # TODO eftir að útfæra

    g = rc.tokenize("Eftir ca 10 mínútur datt hann í pollinn.")
    g = list(g)
    if verbose: dump(g)
    # assert "ca." in s     # TODO eftir að bæta við algengar villur
    # assert "ca " not in s
    # assert g[2].error_code == "S001"      # TODO eftir að útfæra

    g = rc.tokenize("Forsetinn ofl. gengu út um dyrnar.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "o.fl." in s       # TODO býr þetta til en setur aukapunkt aftan við!
    # assert "ofl." not in s
    # assert g[2].error_code == "S001"      # TODO eftir að útfæra

def test_capitalization(verbose=False):
    g = rc.tokenize("Einn Aríi, Búddisti, Eskimói, Gyðingur, sjálfstæðismaður, Múslími og Sjíti gengu inn á bar í evrópu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "aríi" in s        # TODO leiðréttist ekki því að Aríi var til staðar. Búin að laga í BinErrata.conf, svo verður ekki vandamál þegar næsta útgáfa ord.compressed kemur.
    assert "búddisti" in s
    assert "eskimói" in s
    assert "gyðingur" in s
    # assert "Sjálfstæðismaður" in s    # TODO leiðréttist ekki
    # assert "múslími" in s     # Leiðréttist ekki
    # assert "sjíti" in s       # TODO leiðréttist í sjíta. Þarf að láta einræðanlegar villur gera þetta og svo ekkert stoppa það
    # assert "Evrópu" in s
    # assert g[2].error_code == "Z001"  # aríi, TODO leiðréttist ekki eins og er, gerist með nýrri útgáfu ord.compressed.
    assert g[4].error_code == "Z001"  # búddisti
    assert g[6].error_code == "Z001"  # eskimói
    assert g[8].error_code == "Z001"    # gyðingur
    # assert g[10].error_code == "Z002"   # Sjálfstæðismaður, leiðréttist ekki því fannst í BÍN. Búin að uppfæra BinErrata.conf.
    # assert g[12].error_code == "Z002"   # múslími, TODO leiðréttist ekki
    # assert g[14].error_code == "Z002"     # sjíti; TODO leiðréttist ekki 

    g = rc.tokenize("Á íslandi búa íslendingar og í danmörku búa Danskir danir í Nóvember.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Íslandi" in s
    assert "Íslendingar" in s
    assert "Danmörku" in s
    assert "danskir" in s
    assert "Danir" in s
    assert "nóvember" in s
    assert "Nóvember" not in s
    assert g[2].error_code == "Z002"    # Íslandi
    assert g[4].error_code == "Z002"    # Íslendingar
    assert g[7].error_code == "Z002"    # Danmörku
    assert g[9].error_code == "Z001"    # danskir
    assert g[10].error_code == "Z002"   # Danir
    assert g[12].error_code == "Z003"   # nóvember

def test_inflectional_errors(verbose=False):
    # beygingarvillur
    # sama og test_error_forms í test_tokenizer.py
    g = rc.tokenize("Tréið gekk til rekstar rúmmsins.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Tréð" in s
    assert "rekstrar" in s
    assert "rúmsins" in s
    # assert g[1].error_code == "B001"    # tréð, TODO eftir að útfæra
    # assert g[4].error_code == "B001"    # rekstrar, TODO eftir að útfæra
    # assert g[5].error_code == "B001"    # rúmsins, TODO eftir að útfæra

    g = rc.tokenize("Þér finndist víðfermt í árverkni.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "fyndist" in s
    # assert "víðfeðmt" in s     # TODO ekki komið inn
    assert "árvekni" in s
    # assert g[2].error_code == "B001"    # fyndist, TODO eftir að útfæra
    # assert g[3].error_code == "B001"    # víðfeðmt, TODO eftir að útfæra
    # assert g[5].error_code == "B001"    # árvekni, TODO eftir að útfæra

    g = rc.tokenize("Ein kúin kom aldrei til baka vegna eldingunnar.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "kýrin" in s        # TODO eftir að setja inn
    # assert "eldingarinnar" in s    # TODO eftir að setja inn
    # assert g[2].error_code == "B001"    # kýrin, TODO eftir að útfæra
    # assert g[8].error_code == "B001"    # eldingarinnar, TODO eftir að útfæra

    g = rc.tokenize("Lítum til áttunda áratugsins til upplýsingu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "áratugarins" in s
    # assert "upplýsingar" in s      # TODO eftir að setja inn
    # assert g[4].error_code == "B001"    # áratugarins, TODO eftir að útfæra
    # assert g[6].error_code == "B001"    # upplýsingar, TODO eftir að útfæra

    g = rc.tokenize("Nánar tiltekið árins 1978, fjóru og hálfu ári fyrir byltinguna.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "ársins" in s
    assert "fjórum" in s
    # assert g[3].error_code == "B001"    # ársins, TODO eftir að útfæra
    # assert g[5].error_code == "B001"    # fjórum, TODO eftir að útfæra

    g = rc.tokenize("Frumkvöðullinn aldist upp í litlu sjávarþorpi án föðurs og ýmsra þæginda.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "ólst upp" in s
    assert "föður" in s
    assert "ýmissa" in s
    # assert g[2].error_code == "B001"    # ólst, TODO eftir að útfæra
    # assert g[8].error_code == "B001"    # föður, TODO eftir að útfæra
    # assert g[10].error_code == "B001"   # ýmissa, TODO eftir að útfæra

    g = rc.tokenize("Friðsælari leið hefði verið að hundruðir leituðu í geiminum að kílómeter af féinu vegna ástandins.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    #assert "Friðsælli" in s        # TODO eftir að setja inn
    assert "hundruð" in s
    assert "geimnum" in s
    #assert "kílómetra" in s    # TODO eftir að setja inn
    assert "fénu" in s
    assert "ástandsins" in s
    # assert g[1].error_code == "B001"      # friðsælli, TODO eftir að útfæra
    # assert g[6].error_code == "B001"      # hundruð, TODO eftir að útfæra
    # assert g[9].error_code == "B001"      # geimnum, TODO eftir að útfæra
    # assert g[11].error_code == "B001"     # kílómetra, TODO eftir að útfæra
    # assert g[13].error_code == "B001"     # fénu, TODO eftir að útfæra
    # assert g[15].error_code == "B001"     # ástandsins, TODO eftir að útfæra

    g = rc.tokenize("Loks gekk hann til Selfosss tuttugusta dag samningins.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Selfoss" in s
    assert "tuttugasta" in s
    assert "samningsins" in s
    # assert g[5].error_code == "B001"        # Selfoss, TODO eftir að útfæra
    # assert g[6].error_code == "B001"        # tuttugasta, TODO eftir að útfæra
    # assert g[8].error_code == "B001"        # samningsins, TODO eftir að útfæra

def test_wrong_first_parts(verbose=False):
    # M001: Rangur fyrri hluti í samsetningu    (heyrna-laus, náms-skrá)

    g = rc.tokenize("Kvenngormar eru feyknaskemmtilegir en ekki fyrnauppteknir.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Kvengormar" in s
    assert "Kvenngormar" not in s
    # assert "feiknaskemmtilegir" in s      # TODO virkar ekki eins og er
    # assert "feyknaskemmtilegir" not in s
    assert "firnauppteknir" in s
    assert "fyrnauppteknir" not in s

    # assert g[1].error_code == "M001"    # Fæ C002; eftir að útfæra villukóða
    # assert g[3].error_code == "M001"    # Virkar ekki
    # assert g[6].error_code == "M001"    # TODO fæ C002

def test_single_first_parts(verbose=False):
    # M002: Stakir fyrri hlutar í setningu      (all kaldur, )

    g = rc.tokenize("Hann var all kaldur þegar hann fannst enda var hann hálf ber.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "allkaldur" in s           # TODO virkar ekki; gerir all að U001
    # assert "all kaldur" not in s
    # assert "hálfber" in s             # TODO virkar ekki
    # assert "hálf ber" not in s
    # assert g[3].error_code == "M002"  # TODO eftir að útfæra villukóða
    # assert g[10].error_code == "M002" # TODO eftir að útfæra villukóða

    g = rc.tokenize("Hún setti honum afar kosti í for vinnunni.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "afarkosti" in s           # TODO virkar ekki
    # assert "afar kosti" in s          # TODO virkar ekki
    # assert "forvinnunni" in s         # TODO virkar ekki
    # assert "for vinnunni" not in s    # TODO virkar ekki
    # assert g[4].error_code == "M002"  # TODO Eftir að útfæra
    # assert g[6].error_code == "M002"  # TODO Eftir að útfæra

def test_single_last_parts(verbose=False):
    # M003: Stakir seinni hlutar í setningu     (græn keri, arf beri, barn dómur)

    g = rc.tokenize("Hann gekk í barn dóm þegar hann komst að því að hún var líka í hópi græn kera.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "barndóm" in s                 # TODO Eftir að útfæra
    # assert "barn dóm" not in s
    # assert "grænkeri" in s                # TODO Eftir að útfæra
    # assert "græn keri" not in s
    # assert g[4].error_code == "M003"      # TODO Eftir að útfæra villukóða
    # assert g[16].error_code == "M003"     # TODO Eftir að útfæra villukóða

def test_wrong_parts(verbose=False):
    # M004: Rangt orð finnst í samsetningu      (trukkalessa, kúardella)
    # Getur verið tabúorð, röng beygingarmynd, ...
    g = rc.tokenize("Loftlagsmál eru vandamál skráningastarfsmanna.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "Loftslagsmál" in s
    assert "Loftlagsmál" not in s
    assert "skráningarstarfsmanna" in s
    assert "skráningastarfsmanna" not in s
    # assert g[1].error_code == "M004"      # TODO Eftir að útfæra villukóða, fæ C002 
    # assert g[4].error_code == "M004"      # TODO Eftir að útfæra villukóða, fæ C002

    g = rc.tokenize("Trukkalessur þola ekki kúardellu.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "kúadellu" in s            # TODO Virkar ekki eins og er.
    # assert "kúardellu" not in s
    # assert g[1].error_code == "T001"    # TODO Eftir að útfæra. Tabúorð ætti að merkja sem slík.
    # assert g[4].error_code == "M004"    # TODO Eftir að útfæra, fæ U001. En beygingarvillur?

def test_non_single_first_parts(verbose=False):
    # M005: Fyrri hluti á að vera stakur        (fjölnotapappír, ótalmargir)
    g = rc.tokenize("Það er alhliðavandamál hvað ótalmargir fjölnotahestar eru afarleiðinlegir.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert "alhliða vandamál" in s
    assert "alhliðavandamál" not in s
    # assert "ótal margir" in s         # TODO virkar ekki
    # assert "ótalmargir" not in s      # TODO virkar ekki
    assert "fjölnota hestar" in s
    assert "fjölnotahestar" not in s
    # assert "afar leiðinlegir" in s    # TODO virkar ekki
    # assert "afarleiðinlegir" not in s
    # assert g[3].error_code == "M005"  # TODO Eftir að útfæra, fæ C002
    # assert g[5].error_code == "M005"  # TODO Eftir að útfæra
    # assert g[6].error_code == "M005"  # TODO Eftir að útfæra, fæ C002
    # assert g[8].error_code == "M005"  # TODO Eftir að útfæra

    g = rc.tokenize("Það er betra að vera ofgóður en ofursvalur.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "of góður" in s        # TODO Virkar ekki eins og er
    # assert "ofgóður" not in s
    # assert "ofur svalur" in s       # TODO Reglurnar segja til um að þetta sé réttara en ég er bara ekki sammála!
    # assert "ofursvalur" not in s
    # assert g[6].error_code == "M005"  # TODO Eftir að útfæra villukóða
    # assert g[8].error_code == "M005"  # TODO Eftir að útfæra villukóða

def test_inquiry_verb_forms(verbose=False):
    # athuga hvort eintöluform séu til staðar?
    # athuga hvort fleirtala sé leiðrétt
    g = rc.tokenize("Þegar þið hafið hrært deigið setjiði það í ofninn.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "setjið " in s      # TODO eftir að útfæra
    # assert "setjið þið" in s   # TODO eftir að útfæra
    # assert g[6].error_code == "Q001"   # TODO eftir að útfæra villukóða

    g = rc.tokenize("Eftir að kakan kemur úr ofninum náiði í kremið.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "náið þið" in s    # TODO eftir að útfæra, spelling.py virðist taka á þessu
    # assert "náiði" not in s     # TODO eftir að útfæra
    # assert g[7].error_code == "Q001"    # TODO eftir að útfæra villukóða

def test_taboo_words(verbose=False):
    # Simple test
    g = rc.tokenize("Júðarnir og hommatittirnir hoppuðu ásamt halanegrunum.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert g[1].error_code == "T001"    # TODO eftir að útfæra
    # assert g[3].error_code == "T001"    # TODO eftir að útfæra
    # assert g[6].error_code == "T001"    # TODO eftir að útfæra

    # Should not be allowed in compounds
    g = rc.tokenize("Merartussan henti mér af kuntubaki.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert g[1].error_code == "T001"    # TODO eftir að útfæra; gæti verið M004
    # assert g[5].error_code == "T001"    # TODO eftir að útfæra; gæti verið M004

def test_wrong_whitespace(verbose=False):
    g = rc.tokenize("Þetta var gert ti lað vekja hrútinn ein sog til stóð.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "til að" in s      # TODO eftir að bæta við
    # assert "ti lað" not in s  # TODO eftir að bæta við
    # assert "eins og" in s
    # assert "ein sog" not in s
    # assert g[4].error_code == "S005"
    # assert g[8].error_code == "S005"

def test_correct_words(verbose=False):
    # Athuga hvort hér greinist nokkuð villa
    g = rc.tokenize("Ég fann nokkurs konar skógardverg ofan í skúffunni en David Schwimmer vissi allt um mannætuapana.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert s == "Ég fann nokkurs konar skógardverg ofan í skúffunni en David Schwimmer vissi allt um mannætuapana."
    for w in g:
        assert not w.error_code

    g = rc.tokenize("Ökumaður bílaleigubíls komst í hann krappan á Grandanum í Reykjavík skömmu fyrir klukkan 11 í dag.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert s == "Ökumaður bílaleigubíls komst í hann krappan á Grandanum í Reykjavík skömmu fyrir klukkan 11 í dag."
    for w in g:
        assert not w.error_code

    g = rc.tokenize("Þá telur hann kjarasamninga stuðla að stöðugleika sem einnig undirbyggi frekari stýrivaxtalækkanir.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    assert s == "Þá telur hann kjarasamninga stuðla að stöðugleika sem einnig undirbyggi frekari stýrivaxtalækkanir."
    for w in g:
        assert not w.error_code

# Tests for checker.py

def check_sentence(s, annotations):
    """ Check whether a given single sentence gets the
        specified annotations when checked """

    def check_sent(sent):
        assert sent is not None
        if sent.tree is None:
            # If the sentence should not parse, call
            # check_sentence with annotations=None
            assert annotations is None
            return
        assert annotations is not None
        assert sent.tree is not None
        if not annotations:
            # This sentence is not supposed to have any annotations
            assert (not hasattr(sent, "annotations")) or len(sent.annotations) == 0
            return
        assert hasattr(sent, "annotations")
        assert len(sent.annotations) == len(annotations)
        for a, (start, end, code) in zip(sent.annotations, annotations):
            print("{} {} {}".format(a.start, a.end, a.code))    # TODO taka út þegar búin að aflúsa prófanirnar
            assert a.start == start
            assert a.end == end
            assert a.code == code
    print(s)
    # Test check_single()
    check_sent(rc.check_single(s))
    # Test check()
    for pg in rc.check(s):
        for sent in pg:
            check_sent(sent)
    # Test check_with_stats()
    for pg in rc.check_with_stats(s)["paragraphs"]:
        for sent in pg:
            check_sent(sent)

def test_NP_agreement(verbose=False):
    # Beygingarsamræmi
    # fjöldi X gerðu... P_NT_FjöldiHluti
    # einn af X gerðu

    # Sambeyging / rétt orðmynd notuð
    s = "Ég fór frá Pétur Páli um miðnætti."
    #check_sentence(s, [(3, 5, "P_NT_X")])        # TODO Fæ ekki villu, 'Pétur Páli' er sameinað í nafn áður en fallið er tékkað virðist vera.
    s = "Hann hélt utan um dóttir sína."
    check_sentence(s, [(3, 4, "P_NT_FsMeðFallstjórn")])
    s = "Barnið var með kaldar fingur en heitar fætur."
    # check_sentence(s, [(4, 6, "P_NT_KynInnanNafnliðar"), (6, 8, "P_NT_Fall")])     # TODO villurnar greinast ekki, vantar líklega reglur.
    s = "Miklar umræður eiga sér stað innan verkalýðsfélagsins Eflingu."
    check_sentence(s, [(5, 7, "P_NT_FsMeðFallstjórn")])      # TODO  FsMeFallstjórn greinir villu, en nær ekki yfir Eflingu, eitthvað skrýtið á ferðinni! Vil fá reglu sem heitir FallInnanNafnliðar og á að ná yfir 5, 8.
    s = "Fyrirtækið er rekið með fimm prósent halla en verðið er sjö prósent lægra."
    # check_sentence(s, [(5, 7, "P_NT_Prósent"), (11, 13, "P_NT_Prósent")])     # TODO hvorug villan greinist. Vil reglu sem heitir Prósent... eða eitthvað í þá áttina. Ath. hvort það sé nokkuð regla sem heitir það núna.
    s = "Stúlkan kom ásamt fleirum konum í bæinn."
    # check_sentence(s, [(3, 5, "P_NT_Fleirum")])           # TODO villan greinist ekki, eftir að höndla
    s = "Þetta er einhvert mesta óheillaráð sem ég hef heyrt."
    # check_sentence(s, [(2, 2, "P_NT_Einhver")])     # TODO villan greinist sem S001, viljum við höndla þetta sem beygingarsamræmisvillu frekar? Þetta er ósamhengisháð.
    s = "Hún heyrði einhvað frá háaloftinu."
    #check_sentence(s, [(2, 2, "P_NT_Einhver")])        # TODO villan greinist sem S001, viljum við höndla þetta frekar sem beygingarsamræmisvillu? Þetta er ósamhengisháð.

def test_number_agreement(verbose=False):
    # Tala
    s = "Fleiri en einn slasaðist í árekstrinum."
    check_sentence(s, [(3, 5, "P_NT_ÍTölu")])           # TODO virkar en ég skil ekki lengdina, af hverju er 'í' haft með? Hvað er ÍTölu að gera?
    s = "Hann er einn þeirra sem slasaðist í árekstrinum."
    # check_sentence(s, [(5, 6, "P_NT_Þeirra")])        # TODO engin villa finnst.
    s = "Minnihluti starfsmanna samþykktu samninginn."
    check_sentence(s, [(0, 1, "P_NT_FjöldiHluti")])     # TODO villan greinist, en ætti að vera staðsett á sögninni til að hægt sé að leiðrétta hana... Hvernig er þetta leiðrétt? Er bara ábending?
    s = "Helmingur landsmanna horfðu á barnaefnið."
    check_sentence(s, [(0, 1, "P_NT_FjöldiHluti")])     # TODO villan greinist en ætti að vera staðsett á sögninni svo hægt sé að leiðrétta hana. Skoða hvernig/hvort villan er leiðrétt.
    s = "Hér eru tuttugu og ein appelsínur."
    # check_sentence(s, [()])

def test_gender_agreement(verbose=False):
    # Kyn
    s = "Foreldrar hans voru skildir."
    # check_sentence(s, [(4, 5, "P_NT_Foreldrar")])     # TODO þetta mætir afgangi en væri gott að koma inn.
    s = "Stúlkan varð ekki var við hávaðann."
    # check_sentence(s, [(3, 4, "P_NT_SagnfyllingKyn")])    # TODO fæ enga villu, eftir að útfæra.

def test_verb_agreement(verbose=False):
    # Sagnir
    s = "Konunni vantar að kaupa rúðusköfu."
    check_sentence(s, [(0, 0, "P_WRONG_CASE_þgf_þf")])      # TODO virkar vel, en skil ekki lengdina. Af hverju er það ekki 0, 1? Hvernig birtist þetta í viðmótinu?
    s = "Mér kvíðir fyrir að byrja í skólanum."
    check_sentence(s, [(0, 0, "P_WRONG_CASE_þgf_nf")])      # TODO virkar, en athuga lengdina, og hvernig þetta er leiðrétt í viðmóti. Er sögninni líka breytt?
    s = "Ég dreymi um skjaldbökur sem synda um hafið."
    # check_sentence(s, [(0, 0, "P_WRONG_CASE_nf_þf")])       # TODO villan greinist ekki! Ath. líka lengdina.
    s = "Feimni drengurinn hélt sig til hlés þar til þolinmæðin þraut."
    # check_sentence(s, [(3, 3, "P_WRONG_CASE_þf_þgf"), (8, 8, "P_WRONG_CASE_nf_þf")])      # Engin villa greinist; er þetta í Verbs.conf?
    s = "Kúrekinn hafði upp á kúnum á sléttunni."
    # check_sentence(s, [(2, 2, "P_WRONG_PARTICLE_uppi")])    # TODO greinist ekki, þetta á algerlega eftir að útfæra betur þegar þetta er komið inn í Verbs.conf. Þetta er líklega ekki réttur villukóði.
    s = "Maðurinn dáðist af málverkinu."
    # check_sentence(s, [(2, 2, "P_WRONG_PP_að")])      # Engin villa greinist. Eftir að útfæra, þetta er líklega ekki réttur villukóði.
    s = "Barnið á hættu á að detta í brunninn."
    # check_sentence(s, [(1, 1, "P_WRONG_FORM")])       # TODO erfitt að eiga við, líklega ekki réttur villukóði, bæta við Verbs.conf.
    s = "Hetjan á heiður að björguninni."
    # check_sentence(s, [(3, 4, "P_WRONG_PP_af")])      # TODO villan greinist ekki. Komið í Verbs.conf? Líklega ekki réttur villukóði.
    s = "Ferðafólkið fór erlendis að leita lamba."
    # check_sentence(s, [(2, 3, "P_WRONG_PARTICLE_til_útlanda")])       # TODO villan greinist ekki. Komið í Verbs.conf? Líklega ekki réttur villukóði.
    s = "Túlkurinn gaf í skin að mælandi hefði misskilið túlkinn."
    # check_sentence(s, [(2, 4, "P_WRONG_PP_í_skyn")])      # TODO villan greinist ekki. Komið í Verbs.conf? Líklega ekki réttur villukóði.

def test_hvor_annar(verbose=False):
    s = "Drengirnir héldu fast utan um hvorn annan."
    #check_sentence(s, [(3, 7, "P_NT_HvorAnnar")])      # TODO engin villa greinist; eftir að útfæra villureglu
    s = "Hringirnir voru í hvorum öðrum."
    #check_sentence(s, [(2, 5, "P_NT_HvorAnnar")])      # TODO engin villa greinist; eftir að útfæra villureglu

def test_phrasing(verbose=False):
    s = "Ég vill ekki gera mál úr þessu."
    check_sentence(s, [(0, 1, "P_wrong_person")])       # TODO þetta virkar, en skoða lengdina.
    s = "Konur vilja í auknu mæli koma að sjúkraflutningum."
    check_sentence(s, [(2, 4, "P_wrong_gender")])       # TODO á kannski að greina þetta öðruvísi? Fastur frasi? Skoða líka lengdina.
    s = "Ég veit ekki hvort að ég komi í kvöld."
    check_sentence(s, [(4, 4, "P_NT_Að")])              
    s = "Meðan veislunni stendur verður frítt áfengi í boði."
    # check_sentence(s, [(0, 3, "P_NT_MeðanStendur")])      # TODO engin villa finnst

def test_munu(verbose=False):
    s = "Ég mun aldrei gleyma þessu."
    #check_sentence(s, [(1, 1, "P_NT_Munu")])
    s = "Hundurinn mun verða vinur minn að eilífu."
    #check_sentence(s, [(1, 1, "P_NT_Munu")])

def test_vera(verbose=False):
    # vera að + so.nh.
    s = "Ég er ekki að skilja þetta."
    #check_sentence(s, [(1, 6, "P_NT_VeraAð")])     # TODO villan greinist ekki, eftir að útfæra
    s = "Ég var að fara út með ruslið þegar ég fékk símtalið."
    #check_sentence(s, [(1, 11, "P_NT_VeraAð")])    # TODO villan greinist ekki, eftir að útfæra
    s = "Hún er að skrifa vel."
    # check_sentence(s, [(1, 6, "P_NT_VeraAð")])    # TODO villan greinist ekki, eftir að útfæra

def test_nhm(verbose=False):
    s = "Ég ætla fara í búð."
    # check_sentence(s, [(2, 3, "P_Að")])        # TODO villan greinist ekki, eftir að útfæra. Ætti að vera í Verbs.conf
    s = "Hún ætlar að fara lesa um skjaldbökur."
    # check_sentence(s, [(3, 4, "P_Að")])        # TODO villan greinist ekki, eftir að útfæra. Ætti að vera í Verbs.conf

def test_new_passive(verbose=False):
    s = "Það var gert grein fyrir stöðu mála."
    # check_sentence(s, [(2, 2, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu
    s = "Lagt verður áhersla á að skoða reikningana."
    #check_sentence(s, [(0, 0, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu
    s = "Það verður lagt áherslu á að skoða reikningana."   
    # check_sentence(s, [(2, 4, "P_NT_NýjaÞolmynd")])         # TODO villan greinist ekki, eftir að útfæra villureglu

def test_verb_arguments(verbose=False):
    # TODO breyta prófuninni svo falli að mynsturgreininum.
    g = rc.tokenize("Vefurinn bíður upp á bestu fréttirnar.")
    g = list(g)
    if verbose: dump(g)
    s = tokenizer.correct_spaces(" ".join(t.txt for t in g if t.txt is not None))
    # assert "Vefurinn býður" in s  # TODO eftir að útfæra
    # assert "Vefurinn bíður" not in s  # TODO eftir að útfæra
    # assert g[2].error_code == "V001"  # TODO eftir að ákveða villukóða

    s = "Kirkjuna bar við himinn þegar við komum þar um morguninn."
    check_sentence(s, [(2, 9, "P_NT_FsMeðFallstjórn")])    # TODO Verbs.conf ætti að dekka þetta -- útfæra goggunarröð?

def test_complex_sentences(verbose=False):
    s = "Drengurinn dreif sig inn þegar hann heyrði í bjöllunni af því að hann vildi sjá hvort það væri kominn nýr kennari en sem betur fer var gamli kennarinn á sínum stað svo að hann settist niður í rólegheitum og tók upp bækurnar."
    # check_sentence(s, [0, 0, "P_COMPLEX"])      # TODO eftir að útfæra
    s = "Tromman sem var í skápnum sem hafði brotnað í óveðrinu sem var daginn sem þau keyptu kexið sem var ónýtt þegar þau komu úr búðinni sem þau keyptu það í hafði skekkst."
    # check_sentence(s, [0, 0, "P_COMPLEX"])      # TODO eftir að útfæra

def test_tense_mood(verbose=False):
    s = "Hann kemur ef hann geti."
    # check_sentence(s, [(2, 5, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    s = "Hún kemur ef það sé gott veður."
    # check_sentence(s, [(2, 7, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    s = "Hún segir að veðrið var gott í dag."
    # check_sentence(s, [(1, 8, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra
    s = "Hann sagði að veðrið er gott í dag."
    # check_sentence(s, [(1, 8, "P_NT_TíðHáttur")])     # TODO villan finnst ekki, eftir að útfæra

def test_noun_style(verbose=False):
    # Ekki í forgangi
    # nafnorðastíll
    s = "Stofnunin framkvæmdi könnun á aðstæðum á vinnustað."
    # check_sentence(s, [(1, 3, "P_Nafnorðastíll")])        # TODO greinist ekki, eftir að útfæra -- þetta gæti virkað vel í Verbs.conf!

def test_missing_word(verbose=False):
    # Ekki í forgangi
    s = "Það er mjög mikilvægt þið lesið þennan póst."
    # check_sentence(s, [(4, 4, "P_NT_Að")])    # TODO engin villa finnst, eftir að útfæra
    s = "Það mjög mikilvægt að þið lesið þennan póst."
    #check_sentence(s, [(1, 1, "P_NT_SögnVantar")])     # TODO engin villa finnst, eftir að útfæra

def test_foreign_sentences(verbose=False):
    s = "Brooks Koepka lék hringinn á þremur undir pari og er því líkt og Thomas og Schauffele á tíu höggum undir pari. "
    check_sentence(s, [(0, 0, "U001"), (1, 1, "U001"), (15, 15, "U001")])
    s = "If you asked people to try to picture hunting for truffles, the expensive subterranean fungi, many would no doubt imagine men with dogs going through woodlands in France or Italy."
    # check_sentence(s, [(0, 33, "E004")])      # TODO þetta virðist ekki virka; strandar á því að setningin greinist ekki!
    s = "Rock and roll er skemmtilegt."
    check_sentence(s, [(0, 5, "E004")])

def test_conjunctions(verbose=False):
    s = "Ef að pósturinn kemur ekki á morgun missi ég vitið."
    check_sentence(s, [(1, 1, "P_NT_Að")])
    s = "Hafsteinn vissi svarið þótt að hann segði það ekki upphátt."
    check_sentence(s, [(4, 4, "P_NT_Að")])
    s = "Hafsteinn vissi svarið þó hann segði það ekki upphátt."
    check_sentence(s, [(3, 3, "P_NT_ÞóAð")])
    s = "Ég kem á hátíðina víst að pabbi þinn kemst ekki."
    # check_sentence(s, [(4, 5, "P_NT_VístAð")])            # TODO engin villa finnst! Var ekki búið að útfæra þetta?
    s = "Ég kem á hátíðina fyrst að pabbi þinn kemst ekki."
    check_sentence(s, [(5, 5, "P_NT_Að")])
    s = "Hatturinn passar á höfuðið nema að það sé eyrnaband undir honum."
    check_sentence(s, [(5, 5, "P_NT_Að")])
    s = "Hún grét þegar að báturinn sást ekki lengur."
    check_sentence(s, [(3, 3, "P_NT_Að")])
    s = "Hún hélt andliti á meðan að hann horfði til hennar."
    check_sentence(s, [(4, 4, "P_NT_Að")])
    s = "Annaðhvort ferðu í buxurnar núna."
    # check_sentence(s, [(5, 5, "P_NT_AnnaðhvortEða")])     # TODO engin villa finnst, eftir að útfæra
    s = "Hvorki hatturinn passaði á höfuðið."
    # check_sentence(s, [(0, 0, "P_NT_HvorkiNé")])          # TODO engin villa finnst, eftir að útfæra

def test_impersonal_verbs(verbose=False):
    s = "Ég dreymdi að það væri hundur í fiskabúrinu mínu."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE_nf_þf")])            # TODO setningin fær ekki þáttun. Þetta er inni í Verbs.conf, af hverju er þetta ekki höndlað?
    s = "Hestinum dreymdi að það væri hundur í fiskabúrinu."
    check_sentence(s, [(0, 0, "P_WRONG_CASE_þgf_þf")])    
    s  = "Mér klæjar undan áburðinum."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE_þgf_þf")])           # TODO villa greinist ekki... eftir að útfæra?
    s = "Hann sagði að konan hefði misminnt að potturinn væri með loki."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE")])                # TODO villa greinist ekki, eftir að útfæra?
    s = "Bréfberinn spurði hvort hún vantaði fleiri frímerki."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE")])                # TODO villa greinist ekki, eftir að útfæra?
    s = "Lögfræðingnum sem ég fékk til þess að verja mig í jarðaberjastuldarmálinu hlakkaði til að losna við mig."
    check_sentence(s, [(0, 2, "P_WRONG_CASE_þgf_nf")])          # TODO greinist, en skoða lengdina.
    s = "Tröllskessan dagaði uppi."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE")])        # TODO villa greinist ekki; eftir að útfæra? Setja í Verbs.conf?
    s = "Báturinn rak á land."
    # check_sentence(s, [(0, 0, "P_SUBJ_CASE")])        # TODO villa greinist ekki; eftir að útfæra? Setja í Verbs.conf?

def test_correct_sentences(verbose=False):
    s = "Ráðist var í úttektina vegna ábendinga sem bárust embættinu frá notendum þjónustunnar. "
    check_sentence(s, [])
    s = "Upp úr krafsinu hafði maðurinn samtals 51 pakka af kjúklingabringum og 9,2 kíló að auki."
    check_sentence(s, [])
    s = "Á göngudeild gigtar á Landspítalanum sé tilvísunum forgangsraðað og er meðalbiðtími innan marka."
    check_sentence(s, [])

def test_corrected_sentences(verbose=False):
    # Setningar sem þáttast ekki upprunalega út af villum.
    # Villurnar eru svo leiðréttar í errtokenizer.py.
    # Hér vil ég athuga hvort setningin er þáttuð aftur.
    # TODO prófa hér.
    s = "Alla sína lífdaga hljóp hún allt hvað fætur toga að ástæðulausu."


if __name__ == "__main__":
    pass
