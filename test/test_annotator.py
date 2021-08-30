# type: ignore
"""

    test_annotator.py

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

from typing import List, Optional, Tuple

import pytest

import reynir_correct


@pytest.fixture(scope="module")
def rc():
    """Provide a module-scoped GreynirCorrect instance as a test fixture"""
    r = reynir_correct.GreynirCorrect()
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


def check_sentence(
    rc: reynir_correct.GreynirCorrect,
    s: str,
    annotations: Optional[List[Tuple[int, int, str]]],
    is_foreign: bool = False,
    ignore_warnings: bool = False,
) -> None:
    """Check whether a given single sentence gets the
    specified annotations when checked"""

    def check_sent(sent: reynir_correct.AnnotatedSentence) -> None:
        assert sent is not None
        if sent.tree is None and not is_foreign:
            # If the sentence should not parse, call
            # check_sentence with annotations=None
            assert annotations is None
            return
        assert annotations is not None
        if not is_foreign:
            assert sent.tree is not None
        # Compile a list of error annotations, omitting warnings
        sent_errors = [a for a in sent.annotations if not a.code.endswith("/w")]
        if not annotations:
            # This sentence is not supposed to have any annotations
            if ignore_warnings:
                assert len(sent_errors) == 0
            else:
                assert len(sent.annotations) == 0
            return
        if ignore_warnings:
            assert len(sent_errors) == len(annotations)
            for a, (start, end, code) in zip(sent_errors, annotations):
                assert a.start == start
                assert a.end == end
                assert a.code == code
        else:
            assert len(sent.annotations) == len(annotations)
            for a, (start, end, code) in zip(sent.annotations, annotations):
                assert (
                    a.start == start
                ), f"Mismatch between ({a.start}, {a.end}, {a.code}) and ({start}, {end}, {code})"
                assert a.end == end
                assert a.code == code

    # Test check_single()
    sent = rc.parse_single(s)
    assert isinstance(sent, reynir_correct.AnnotatedSentence)
    check_sent(sent)
    # Test check()
    for pg in reynir_correct.check(s):
        for sent in pg:
            assert isinstance(sent, reynir_correct.AnnotatedSentence)
            check_sent(sent)
    # Test check_with_stats()
    for pg in reynir_correct.check_with_stats(s)["paragraphs"]:
        for sent in pg:
            assert isinstance(sent, reynir_correct.AnnotatedSentence)
            check_sent(sent)

    # Test presevation of original token text
    tlist = list(reynir_correct.tokenize(s))
    len_tokens = sum(len(t.original or "") for t in tlist)
    assert len_tokens == len(s)


def test_multiword_phrases(rc):
    s = "Einn af drengjunum fór í sund af gefnu tilefni."
    check_sentence(rc, s, [(6, 8, "P_afað")])


def test_error_finder(rc):
    """Test errors that are found by traversing the detailed
    parse tree in checker.py (ErrorFinder class)"""
    s = "Einn af drengjunum fóru í sund."
    check_sentence(rc, s, [(3, 3, "P_NT_EinnAf")])
    s = "Fjöldi þingmanna greiddu atkvæði gegn tillögunni."
    check_sentence(rc, s, [(2, 2, "P_NT_FjöldiHluti")])
    s = "Jón borðaði ís þar sem að hann var svangur."
    check_sentence(rc, s, [(5, 5, "P_NT_Að/w")])
    s = 'Jón "borðaði" ís þar sem að hann var svangur.'
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (7, 7, "P_NT_Að/w")])
    s = "Jón borðaði ís þó hann væri svangur."
    check_sentence(rc, s, [(3, 3, "P_NT_ÞóAð")])
    s = 'Jón "borðaði" ís þó hann væri svangur.'
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 5, "P_NT_ÞóAð")])
    s = "Jón borðaði ís jafnvel þó hann væri svangur."
    check_sentence(rc, s, [(3, 4, "P_NT_ÞóAð")])
    s = 'Jón "borðaði" ís jafnvel þó hann væri svangur.'
    check_sentence(rc, s, [(1, 1, "N001"), (3, 3, "N001"), (5, 6, "P_NT_ÞóAð")])
    s = "Jón borðaði ís þótt hann væri svangur."
    check_sentence(rc, s, [])
    s = 'Jón "borðaði" ís þótt hann væri svangur.'
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
    check_sentence(rc, s, [(4, 4, "T001/w"), (5, 5, "P_NT_Heldur/w")])


def test_ordinals(rc):
    # NOTE: Commented out as this functionality increases the number of
    # false positives on the iceErrorCorpus test set.
    # s = "4. barnið fæddist í gær, en það er 3. strákur þeirra hjóna."
    # check_sentence(rc, s, [(0, 0, "X_number4word"), (8, 8, "X_number4word")])
    # sent = rc.parse_single(s)
    # assert sent.annotations[0].suggest == "Fjórða"
    # assert sent.annotations[1].suggest == "þriðji"
    s = "5. Ákæran beinist gegn Jóni og Friðberti."
    check_sentence(rc, s, [])
    # s = "2. deildin fer vel af stað í vetur."
    # check_sentence(rc, s, [(0, 0, "X_number4word")])
    s = "XVII. kafli: Um landsins gagn og nauðsynjar."
    check_sentence(rc, s, [])


def test_pronoun_annara(rc):
    s = (
        "Allir í hans bekk, auk nokkurra nemenda úr öðrum bekkjum, "
        "umsjónakennara og fjögurra annara kennara "
        "hafa verið sendir í sjö daga sóttkví."
    )
    check_sentence(rc, s, [(12, 12, "S004"), (15, 15, "P_NT_Annara")])
    s = " Mér er annara um símann minn en orðspor mitt."
    check_sentence(rc, s, [])


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
    # FIXME:
    # s = "Ég held að músinni hafi kviðið fyrir að hitta köttinn."
    # check_sentence(rc, s, [(3, 3, "P_WRONG_CASE_þgf_nf")])
    s = "Hestinum Grímni vantaði hamar."
    # s = "Hestinum Skjóna vantaði hamar."
    check_sentence(rc, s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    s = "Stóra manninum sem vinnur á verkstæðinu vantaði hamar."
    check_sentence(rc, s, [(0, 5, "P_WRONG_CASE_þgf_þf")])


def test_foreign_sentences(rc):
    check_sentence(
        rc,
        "It was the best of times, it was the worst of times.",
        [(0, 13, "E004")],
        is_foreign=True,
    )
    check_sentence(
        rc,
        "Praise the Lord.",
        [
            (0, 1, "E004")
        ],  # Note: the tokenizer amalgams 'Praise the Lord' into one token
        is_foreign=True,
    )
    check_sentence(
        rc,
        "Borðaðu Magnyl og Xanax eagerly in Rushmore.",
        [(0, 7, "E004")],
        is_foreign=True,
    )


def test_number(rc):
    check_sentence(rc, "Vinnuvika sjómanna eru 7 heilir dagar.", [(2, 2, "P_NT_ÍTölu")])
    check_sentence(rc, "Hjón borðar matinn sinn.", [(1, 1, "P_NT_ÍTölu")])
    check_sentence(rc, "Ég borðum matinn minn.", [(1, 1, "P_NT_ÍTölu")])


def test_correct_sentences(rc):
    check_sentence(rc, "Pál langaði að horfa á sjónvarpið.", [])
    check_sentence(rc, "Mig dreymdi mús sem elti kött.", [])
    check_sentence(
        rc,
        "Ég held að músin hafi kviðið fyrir að hitta köttinn.",
        [],
        ignore_warnings=True,
    )
    check_sentence(rc, "Músin kveið fyrir að hitta köttinn.", [])
    check_sentence(
        rc,
        "Páll hlakkaði til jólanna og að hitta strákinn sem hlakkaði til páskanna.",
        [],
    )
    check_sentence(rc, "Ég hlakka til að sjá nýju Aliens-myndina.", [])


def test_corrected_meanings(rc: reynir_correct.GreynirCorrect) -> None:
    s = """
    Þannig fundust stundum engin bréfaskipti á milli lífsförunauta í annars ríkulegum bréfasöfnum.
    """
    check_sentence(rc, s.rstrip(), [])
    s = """
    Þeir hafa líka þennan Beach Boys-hljóm og virkilega fallegar raddanir,"
    sagði Jardine, en platan hans nefnist A Postcard fram California.
    """
    # Note: "A Postcard" is tokenized as one entity token and should not
    # be reported as an error or annotation
    check_sentence(
        rc,
        s.rstrip(),
        [(11, 11, "N001")],
    )


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
