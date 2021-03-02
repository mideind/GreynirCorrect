"""

    test_patterns.py

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


    This module tests the grammatical pattern annotation functionality
    of GreynirCorrect.

"""

import pytest

import reynir_correct

from test_annotator import check_sentence


@pytest.fixture(scope="module")
def rc():
    """ Provide a module-scoped GreynirCorrect instance as a test fixture """
    r = reynir_correct.GreynirCorrect()
    yield r
    # Do teardown here
    r.__class__.cleanup()


def test_verb_af(rc):
    s = "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = (
        "Mig langaði að leita af bílnum, en dáðist svo af hugrekki lögreglukonunnar "
        "að ég gerði það ekki."
    )
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF"), (8, 10, "P_WRONG_PREP_AF")])

    s = "Við höfum leitað í allan dag af kettinum, en fundum hann ekki."
    check_sentence(rc, s, [(2, 4, "P_WRONG_PREP_AF")])

    s = "Allan daginn höfum við leitað af kettinum."
    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AF")])

    s = "Páll brosti af töktunum í Gunnu."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")], ignore_warnings=True)

    s = "Ég var leitandi af kettinum í allan dag."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])

    s = "Ég vildi leita af mér allan grun."
    check_sentence(rc, s, [])

    s = "Hver er að leita af skrifstofuhúsnæði?"
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])

    s = "Hann dáist endalaust af þeim."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Hann hefur lengi dást af þeim."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])

    s = "Jón gerir grín af því."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])

    s = "Þetta er mesta vitleysa sem ég hef gert grín af."
    check_sentence(rc, s, [(8, 9, "P_WRONG_PREP_AF")])

    s = "Jón kann það ekki utan af."
    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AF")])

    s = "Jón leggur hann ekki af velli."
    check_sentence(rc, s, [(4, 5, "P_WRONG_PREP_AF")])

    s = "Jón hefur ekki lagt hann af velli."
    check_sentence(rc, s, [(5, 6, "P_WRONG_PREP_AF")])

    s = "Jón leiðir líkur af því."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Jón leiðir ekki líkur af því."
    check_sentence(rc, s, [(1, 4, "P_WRONG_PREP_AF")])

    s = "Jón leiðir líkum af því."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Jón leiðir rök af því."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Jón leitar af því."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = "Tíminn markar upphaf af því."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Tíminn markar ekki upphaf af því."
    check_sentence(rc, s, [(1, 4, "P_WRONG_PREP_AF")])

    s = "Það markar upphafið af því."
    check_sentence(rc, s, [(1, 3, "P_WRONG_PREP_AF")])

    s = "Það markar ekki upphafið af því."
    check_sentence(rc, s, [(1, 4, "P_WRONG_PREP_AF")])

    s = "Það hefur ekki markað upphafið af því."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])

    s = "Jón spyr af því."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = "Það sem Jón spurði ekki af var óljóst."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])

    s = "Jón stuðlar af því."
    check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = "Honum varð af ósk sinni."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])

    s = "Honum hafði orðið af ósk sinni."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])

    s = "Honum varð ekki af ósk sinni."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])

    s = "Hann varð ekki uppvís af því."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])

    s = "Jón varð vitni af þessu."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])


def test_noun_af(rc):
    s = "Hann gerði þetta af beiðni hennar."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])
    s = "Af beiðni hennar gerði hann þetta."
    check_sentence(rc, s, [(0, 1, "P_WRONG_PREP_AF")])
    s = "Það var gert af þeirri fyrirmynd."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])
    s = "Þau gera þetta af heiðnum sið."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])
    s = "Ég baka köku af því tilefni."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])
    s = "Þau veittu mér aðgang af kerfinu."
    check_sentence(rc, s, [(3, 5, "P_WRONG_PREP_AF")])
    s = "Aðgangur af kerfinu var veittur."
    check_sentence(rc, s, [(0, 1, "P_WRONG_PREP_AF")])
    s = "Drög af verkefninu eru tilbúin."
    check_sentence(rc, s, [(0, 1, "P_WRONG_PREP_AF")])
    s = "Þau kláruðu drög af verkefninu."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])
    s = "Grunnur af verkefninu er tilbúinn."
    check_sentence(rc, s, [(0, 1, "P_WRONG_PREP_AF")])
    s = "Hann lagði ekki grunninn af verkefninu."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])
    s = "Þau gerðu leit af dótinu."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])
    s = "Leit af dótinu hefur ekki skilað árangri."
    check_sentence(rc, s, [(0, 1, "P_WRONG_PREP_AF")])
    s = "Þetta er lykillinn af velgengni."
    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])
    s = "Hann gaf mér uppskriftina af réttinum."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])


def test_placename_pp(rc):
    s = "Ég hef búið á Hafnarfirði alla mína tíð en flyt nú í Akureyri."
    check_sentence(rc, s, [(3, 4, "P_WRONG_PLACE_PP"), (9, 10, "P_WRONG_PLACE_PP")])
    s = "Ég hef veitt í Vopnafirði undanfarin ár en búið á Vopnafirði."
    check_sentence(rc, s, [])
    s = "Það eru mörg náttúruvætti á Reykjanesi en ekki í Húsavík."
    check_sentence(rc, s, [(8, 9, "P_WRONG_PLACE_PP")])


def test_verb_líst(rc):
    s = "Jóni veiðimanni lýst ekki á þetta mál."
    check_sentence(rc, s, [(2, 2, "P_WRONG_OP_FORM")])
    s = "Eins og fram hefur komið lýst mér vel á þetta."
    check_sentence(rc, s, [(5, 5, "P_WRONG_OP_FORM")])
    s = "Jón hefur lýst sinni afstöðu til málsins."
    check_sentence(rc, s, [])
    s = "Þegar leið á kvöldið var gangstéttin lýst með ljósum."
    check_sentence(rc, s, [], ignore_warnings=True)
    # TODO: The following gets no annotation:
    # 'Ég verð að segja að mér lýst ekkert á þetta.'
