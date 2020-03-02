"""

    test_annotator.py

    Tests for ReynirCorrect module

    Copyright (C) 2020 by Miðeind ehf.
    Original author: Vilhjálmur Þorsteinsson

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


    This module tests the sentence-level annotation functionality
    of ReynirCorrect.

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


@pytest.fixture(scope="module")
def rc():
    """ Provide a module-scoped Greynir instance as a test fixture """
    r = reynir_correct.ReynirCorrect()
    yield r
    # Do teardown here
    r.__class__.cleanup()


def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))


def check_sentence(rc, s, annotations):
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
            assert a.start == start
            assert a.end == end
            assert a.code == code

    # Test check_single()
    check_sent(rc.parse_single(s))
    # Test check()
    for pg in reynir_correct.check(s):
        for sent in pg:
            check_sent(sent)
    # Test check_with_stats()
    for pg in reynir_correct.check_with_stats(s)["paragraphs"]:
        for sent in pg:
            check_sent(sent)


def test_multiword_phrases(rc):
    s = "Einn af drengjunum fóru í sund af gefnu tilefni."
    check_sentence(rc, s, [(3, 3, "P_NT_EinnAf"), (6, 8, "P_aðaf")])


def test_error_finder(rc):
    """ Test errors that are found by traversing the detailed
        parse tree in checker.py (ErrorFinder class) """
    s = "Fjöldi þingmanna greiddu atkvæði gegn tillögunni."
    check_sentence(rc, s, [(2, 2, "P_NT_FjöldiHluti")])
    s = "Jón borðaði ís þar sem að hann var svangur."
    check_sentence(rc, s, [(5, 5, "P_NT_Að/w")])
    s = "Jón \"borðaði\" ís þar sem að hann var svangur."
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (7, 7, "P_NT_Að/w")])
    s = "Jón borðaði ís þó hann væri svangur."
    check_sentence(rc, s, [(3, 3, "P_NT_ÞóAð")])
    s = "Jón \"borðaði\" ís þó hann væri svangur."
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 5, "P_NT_ÞóAð")])
    s = "Jón borðaði ís jafnvel þó hann væri svangur."
    check_sentence(rc, s, [(3, 4, "P_NT_ÞóAð")])
    s = "Jón \"borðaði\" ís jafnvel þó hann væri svangur."
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 6, "P_NT_ÞóAð")])
    s = "Jón borðaði ís þótt hann væri svangur."
    check_sentence(rc, s, [])
    s = "Jón \"borðaði\" ís þótt hann væri svangur."
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001")])
    s = "Ég féll fyrir annað hvort fegurð hennar eða gáfum."
    check_sentence(rc, s, [(3, 4, "P_NT_AnnaðHvort")])
    s = "Ég talaði við annaðhvort barnanna."
    check_sentence(rc, s, [(3, 3, "P_NT_Annaðhvort")])
    s = "Ég hef verið slappur frá því ég fékk sprautuna."
    check_sentence(rc, s, [(4, 5, "P_NT_FráÞvíAð")])
    s = "Ég hef verið slappur allt frá því ég fékk sprautuna."
    check_sentence(rc, s, [(4, 6, "P_NT_FráÞvíAð")])
    s = "Friðgeir vildi vera heima víst að Sigga yrði að vera heima."
    check_sentence(rc, s, [(4, 5, "P_NT_VístAð")])
    s = "Friðgeir taldi víst að Sigga yrði að vera heima."
    check_sentence(rc, s, [])
    s = "Ég er ekki meiri fáviti heldur en þú."
    check_sentence(rc, s, [(5, 5, "P_NT_Heldur/w")])


def test_impersonal_verbs(rc):
    s = "Mig hlakkaði til."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_þf_nf")])
    s = "Mér hlakkaði til."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Ég dreymdi köttinn."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_nf_þf")])
    s = "Mér dreymdi köttinn."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_þgf_þf")])
    # The following should not parse
    s = "Ég dreymdi kettinum."
    check_sentence(rc, s, None)
    s = (
        "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, "
        "langaði að horfa á sjónvarpið."
    )
    check_sentence(rc, s, [(0, 11, "P_WRONG_CASE_þgf_þf")])
    s = (
        "Pál, sem hefur verið landsliðsmaður í fótbolta í sjö ár, "
        "langaði að horfa á sjónvarpið."
    )
    check_sentence(rc, s, [])
    s = "Pál kveið fyrir skóladeginum."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_þf_nf")])
    s = "Páli kveið fyrir skóladeginum."
    check_sentence(rc, s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Unga fólkinu skortir aðhald."
    check_sentence(rc, s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    s = "Ég held að músinni hafi kviðið fyrir að hitta köttinn."
    check_sentence(rc, s, [(3, 3, "P_WRONG_CASE_þgf_nf")])
    s = "Hestinum Grímni vantaði hamar."
    # s = "Hestinum Skjóna vantaði hamar."
    check_sentence(rc, s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    check_sentence(rc, "Ég hlakka til að sjá nýju Aliens-myndina.", [])


def test_foreign_sentences(rc):
    check_sentence(
        rc,
        "It was the best of times, it was the worst of times.",
        [(0, 13, "E004")]
    )
    check_sentence(
        rc,
        "Praise the Lord.",
        [(0, 3, "E004")]
    )
    check_sentence(
        rc,
        "Borðaðu Magnyl og Xanax in Rushmore.",
        [(0, 6, "E004")]
    )


def test_number(rc):
    check_sentence(
        rc,
        "Vinnuvika sjómanna eru 7 heilir dagar.",
        [(2, 5, "P_NT_ÍTölu")]
    )
    check_sentence(
        rc,
        "Hjón borðar matinn sinn.",
        [(1, 3, "P_NT_ÍTölu")]
    )
    check_sentence(
        rc,
        "Ég borðum matinn minn.",
        [(1, 3, "P_NT_ÍTölu")]
    )


def test_correct_sentences(rc):
    check_sentence(rc, "Pál langaði að horfa á sjónvarpið.", [])
    check_sentence(rc, "Mig dreymdi mús sem var að elta kött.", [])
    check_sentence(rc, "Ég held að músin hafi kviðið fyrir að hitta köttinn.", [])


if __name__ == "__main__":

    from reynir_correct import ReynirCorrect

    rc = ReynirCorrect()
    test_multiword_phrases(rc)
    test_impersonal_verbs(rc)
    test_error_finder(rc)
    test_correct_sentences(rc)
    test_foreign_sentences(rc)
    test_number(rc)
