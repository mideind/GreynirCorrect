# type: ignore
"""

    test_annotator.py

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


    This module tests the sentence-level annotation functionality
    of GreynirCorrect.

    Potential test sentences:

        Tillaga formanna þingflokkana var rædd í gær, eftir að frumvarpið var sett í kælir.
        Manninum á verkstæðinu vantaði hamar.
        Það var auðséð að henni langaði að fara til sólarlanda.
        Mitt í hamaganginum hlakkaði Jónasi til að fá sér kakó.
        Jón hefur aðra sögu að segja heldur en Páll.
        Ég hætti við að fara víst að Sigga var veik.
        Víst að Sigga var veik hætti ég við að fara.
        Frá því ég sá hana fyrst var ég ástfanginn.
        Annað hvort er þetta della eða þetta virkar vel. Annaðhvort systkinanna mun örugglega greiða mér.
        Fjöldi Evrópuríkja hafa mótmælt áformum Breta. Ég var viss um að fjöldi stuðningsmanna Liverpool myndu fagna.
        Einn af drengjunum voru komnir með flensu, meðan einn af læknunum þurftu að fara heim.
        Allt Viðreisnarfólk, sem og Píratar, tóku þátt í atkvæðagreiðslunni.
        Ég gekk frá skrifborðinu, áður en að ég ók bílnum heim. Ég hef verið frískur síðan að ég tók fúkkalyfið.
        Meðan að tölvan er í viðgerð, get ég lítið unnið.
        Mér var ekki sama um þetta, jafnvel þó hjúkrunarfræðingurinn reyndi að hughreysta mig.
        Þó veðrið væri vont, gátum við þvegið bílinn.
        Hundurinn hans Páls fóru í bað í gær.
        Allir kettirnir í götunni var að elta mýs.
        Samhliða leiksýningin talaði ég við Páll um vandamálið.
        Ég las síðustu bók Guðrúnar (sú sem ég minntist á við þig) og fannst hún býsna góð.

"""


import pytest

import reynir_correct

from .utils import check_sentence


@pytest.fixture(scope="module")
def api() -> reynir_correct.GreynirCorrectAPI:
    """Provide a module-scoped GreynirCorrect instance as a test fixture"""
    r = reynir_correct.GreynirCorrectAPI.from_options()
    yield r


def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))


def test_multiword_phrases(api):
    s = "Einn af drengjunum fór í sund af gefnu tilefni."
    check_sentence(api, s, [(6, 8, "P_afað")])


def test_error_finder(api):
    """Test errors that are found by traversing the detailed
    parse tree in checker.py (ErrorFinder class)"""
    s = "Einn af drengjunum fóru í sund."
    check_sentence(api, s, [(3, 3, "P_NT_EinnAf")])
    s = "Fjöldi þingmanna greiddu atkvæði gegn tillögunni."
    check_sentence(api, s, [(2, 2, "P_NT_FjöldiHluti")])
    s = "Jón borðaði ís þar sem að hann var svangur."
    check_sentence(api, s, [(5, 5, "P_NT_Að/w")])
    s = 'Jón "borðaði" ís þar sem að hann var svangur.'
    check_sentence(api, s, [(1, 1, "N001"), (3, 3, "N001"), (7, 7, "P_NT_Að/w")])
    s = "Jón borðaði ís þó hann væri svangur."
    check_sentence(api, s, [(3, 3, "P_NT_ÞóAð")])
    s = 'Jón "borðaði" ís þó hann væri svangur.'
    check_sentence(api, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 5, "P_NT_ÞóAð")])
    s = "Jón borðaði ís jafnvel þó hann væri svangur."
    check_sentence(api, s, [(3, 4, "P_NT_ÞóAð")])
    s = 'Jón "borðaði" ís jafnvel þó hann væri svangur.'
    check_sentence(api, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 6, "P_NT_ÞóAð")])
    s = "Jón borðaði ís þótt hann væri svangur."
    check_sentence(api, s, [])
    s = 'Jón "borðaði" ís þótt hann væri svangur.'
    check_sentence(api, s, [(1, 1, "N001"), (3, 3, "N001")])
    s = "Ég féll fyrir annað hvort fegurð hennar eða gáfum."
    check_sentence(api, s, [(3, 4, "P_NT_AnnaðHvort")])
    s = "Ég talaði við annaðhvort barnanna."
    check_sentence(api, s, [(3, 3, "P_NT_Annaðhvort")])
    s = "Ég hef verið slappur frá því ég fékk sprautuna."
    check_sentence(api, s, [(5, 5, "P_NT_FráÞvíAð")])
    s = "Ég hef verið slappur allt frá því ég fékk sprautuna."
    check_sentence(api, s, [(6, 6, "P_NT_FráÞvíAð")])
    s = "Friðgeir vildi vera heima víst að Sigga yrði að vera heima."
    # TODO no longer annotated, due to changes in verb frames
    # check_sentence(rc, s, [(4, 4, "P_NT_VístAð")])
    s = "Víst að Sigga var heima ákvað Friðgeir að vera heima."
    check_sentence(api, s, [(0, 0, "P_NT_VístAð")])
    s = "Friðgeir taldi víst að Sigga yrði að vera heima."
    check_sentence(api, s, [])
    s = "Ég er ekki meiri fáviti heldur en þú."
    check_sentence(api, s, [(4, 4, "T001/w"), (5, 5, "P_NT_Heldur/w")])


def test_ordinals(api):
    # NOTE: Commented out as this functionality increases the number of
    # false positives on the iceErrorCorpus test set.
    # s = "4. barnið fæddist í gær, en það er 3. strákur þeirra hjóna."
    # check_sentence(rc, s, [(0, 0, "X_number4word"), (8, 8, "X_number4word")])
    # sent = rc.parse_single(s)
    # assert sent.annotations[0].suggest == "Fjórða"
    # assert sent.annotations[1].suggest == "þriðji"
    s = "5. Ákæran beinist gegn Jóni og Friðberti."
    check_sentence(api, s, [])
    # s = "2. deildin fer vel af stað í vetur."
    # check_sentence(rc, s, [(0, 0, "X_number4word")])
    s = "XVII. kafli: Um landsins gagn og nauðsynjar."
    check_sentence(api, s, [])


def test_pronoun_annara(api):
    s = (
        "Allir í hans bekk, auk nokkurra nemenda úr öðrum bekkjum, "
        "umsjónakennara og fjögurra annara kennara "
        "hafa verið sendir í sjö daga sóttkví."
    )
    check_sentence(api, s, [(12, 12, "S004"), (15, 15, "R4RR")])
    s = " Mér er annara um símann minn en orðspor mitt."
    # TODO 'annara' is changed to 'annarra' due to better token-level data
    # and a difficult syntax pattern
    # check_sentence(rc, s, [])


def test_impersonal_verbs(api):
    s = "Mig hlakkaði til."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þf_nf")])
    s = "Mér hlakkaði til."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Ég dreymdi köttinn."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_nf_þf")])
    s = "Mér dreymdi köttinn."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_þf")])
    # The following should not parse
    s = "Ég dreymdi kettinum."
    check_sentence(api, s, None)
    s = "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, " "langaði að horfa á sjónvarpið."
    check_sentence(api, s, [(0, 11, "P_WRONG_CASE_þgf_þf")])
    s = "Pál, sem hefur verið landsliðsmaður í fótbolta í sjö ár, " "langaði að horfa á sjónvarpið."
    check_sentence(api, s, [])
    s = "Pál kveið fyrir skóladeginum."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þf_nf")])
    s = "Páli kveið fyrir skóladeginum."
    check_sentence(api, s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Unga fólkinu skortir aðhald."
    check_sentence(api, s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    # FIXME:
    # s = "Ég held að músinni hafi kviðið fyrir að hitta köttinn."
    # check_sentence(rc, s, [(3, 3, "P_WRONG_CASE_þgf_nf")])
    s = "Hestinum Grímni vantaði hamar."
    # s = "Hestinum Skjóna vantaði hamar."
    check_sentence(api, s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    s = "Stóra manninum sem vinnur á verkstæðinu vantaði hamar."
    check_sentence(api, s, [(0, 5, "P_WRONG_CASE_þgf_þf")])


def test_foreign_sentences(api):
    check_sentence(
        api,
        "It was the best of times, it was the worst of times.",
        [(0, 13, "E004")],
        is_foreign=True,
    )
    check_sentence(
        api,
        "Praise the Lord.",
        [(0, 1, "E004")],  # Note: the tokenizer amalgams 'Praise the Lord' into one token
        is_foreign=True,
    )
    check_sentence(
        api,
        "Borðaðu Magnyl og Xanax eagerly in Rushmore for the holidays.",
        [(0, 8, "E004")],
        is_foreign=True,
    )  # Note: Example needed to be made longer due to 'in' appearing as an Icelandic error


def test_number(api):
    check_sentence(api, "Vinnuvika sjómanna eru 7 heilir dagar.", [(2, 2, "P_NT_ÍTölu")])
    check_sentence(api, "Hjón borðar matinn sinn.", [(1, 1, "P_NT_ÍTölu")])
    check_sentence(api, "Ég borðum matinn minn.", [(1, 1, "P_NT_ÍTölu")])


def test_correct_sentences(api):
    check_sentence(api, "Pál langaði að horfa á sjónvarpið.", [])
    check_sentence(api, "Mig dreymdi mús sem elti kött.", [])
    check_sentence(
        api,
        "Ég held að músin hafi kviðið fyrir að hitta köttinn.",
        [],
        ignore_warnings=True,
    )
    check_sentence(api, "Músin kveið fyrir að hitta köttinn.", [])
    check_sentence(
        api,
        "Páll hlakkaði til jólanna og að hitta strákinn sem hlakkaði til páskanna.",
        [],
    )
    check_sentence(api, "Ég hlakka til að sjá nýju Aliens-myndina.", [])


def test_corrected_meanings(api) -> None:
    s = """
    Þannig fundust stundum engin bréfaskipti á milli lífsförunauta í annars ríkulegum bréfasöfnum.
    """
    check_sentence(api, s.rstrip(), [])
    s = """
    Þeir hafa líka þennan Beach Boys-hljóm og virkilega fallegar raddanir,"
    sagði Jardine, en platan hans nefnist A Postcard fram California.
    """
    # Note: "A Postcard" is tokenized as one entity token and should not
    # be reported as an error or annotation
    check_sentence(
        api,
        s.rstrip(),
        [(11, 11, "N001")],
    )


def test_lhþt_variant(api) -> None:
    """Check for a regression in the handling of LHÞT variants in BinPackage"""
    s = (
        "Minna þekkt eru sem dæmi rafgas og kvarka-límeindarafgas, "
        "Bose-Einstein þétting og oddskiptaeindaþétting, sérstætt efni, "
        "vökvakristall, ofurstraumefni og ofurþéttefni og einnig "
        "meðseglunar- og járnseglunarhamir segulmagnaðra efna."
    )
    try:
        api.correct(s)
    except ValueError:
        assert False, "Regression in handling of LHÞT variants in BinPackage"


if __name__ == "__main__":
    from reynir_correct import GreynirCorrect

    gc = GreynirCorrect()
    test_multiword_phrases(gc)
    test_impersonal_verbs(gc)
    test_error_finder(gc)
    test_correct_sentences(gc)
    test_foreign_sentences(gc)
    test_number(gc)
    test_corrected_meanings(gc)
