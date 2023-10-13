# type: ignore
"""

    test_patterns.py

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


    This module tests the grammatical pattern annotation functionality
    of GreynirCorrect.

"""

import pytest

import reynir_correct

from .utils import check_sentence


@pytest.fixture(scope="module")
def api() -> reynir_correct.GreynirCorrectAPI:
    """Provide a module-scoped GreynirCorrect instance as a test fixture"""
    r = reynir_correct.GreynirCorrectAPI.from_options()
    yield r


def test_verb_af(api):
    s = "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")])

    s = "Mig langaði að leita af bílnum, en dáðist svo af hugrekki lögreglukonunnar " "að ég gerði það ekki."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF"), (10, 10, "P_WRONG_PREP_AF")])

    s = "Við höfum leitað í allan dag af kettinum, en fundum hann ekki."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Allan daginn höfum við leitað af kettinum."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AF")])

    s = "Páll brosti af töktunum í Gunnu."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")], ignore_warnings=True)

    s = "Ég var leitandi af kettinum í allan dag."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Ég vildi leita af mér allan grun."
    check_sentence(api, s, [])

    s = "Hver leitar af skrifstofuhúsnæði?"
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")])

    s = "Hann dáist endalaust af þeim."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Hann hefur lengi dáðst af þeim."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Jón gerir grín af því."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Þetta er mesta vitleysa sem ég hef gert grín af."
    check_sentence(api, s, [(9, 9, "P_WRONG_PREP_AF")])

    s = "Jón kann það ekki utan af."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AF")])

    s = "Jón leggur hann ekki af velli."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Jón hefur ekki lagt hann af velli."
    # TODO "af velli" ends up under "hann"
    # check_sentence(rc, s, [(1, 6, "P_WRONG_PREP_AF")])

    s = "Jón leiðir líkur af því."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Jón leiðir ekki líkur af því."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Jón leiðir rök af því."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Jón leitar af því."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")])

    s = "Tíminn markar upphaf af því."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Tíminn markar ekki upphaf af því."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Það markar upphafið af því."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Það markar ekki upphafið af því."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Það hefur ekki markað upphafið af því."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AF")])

    s = "Jón spyr af því."
    # TODO not picked up
    # check_sentence(rc, s, [(1, 2, "P_WRONG_PREP_AF")])

    s = "Það sem Jón spurði ekki af var óljóst."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AF")])

    s = "Jón stuðlar af því."
    # TODO not picked up
    # check_sentence(rc, s, [(2, 2, "P_WRONG_PREP_AF")])

    s = "Honum varð af ósk sinni."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AF")])

    s = "Honum hafði orðið af ósk sinni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Honum varð ekki af ósk sinni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])

    s = "Hann varð ekki uppvís af því."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])

    s = "Jón varð vitni af þessu."
    check_sentence(api, s, [(1, 3, "P_afað"), (3, 3, "S005")])

    s = "Hún er ólétt af sínu þriðja barni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    #    s = "Það kom henni á óvart að hún væri ólétt af strák." # Annotation variable depending on parse
    #    check_sentence(rc, s, [(8, 9, "P_WRONG_PREP_AF")])
    s = "Að öllu leyti eru öll ólétt af stelpum."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])


def test_noun_af(api):
    s = "Hann gerði þetta af beiðni hennar."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    s = "Af beiðni hennar gerði hann þetta."
    check_sentence(api, s, [(0, 0, "P_WRONG_PREP_AF")])
    s = "Það var gert af þeirri fyrirmynd."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    s = "Þau gera þetta af heiðnum sið."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    s = "Ég baka köku af því tilefni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    s = "Þau veittu mér aðgang af kerfinu."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])
    s = "Aðgangur af kerfinu var veittur."
    check_sentence(api, s, [(1, 1, "P_WRONG_PREP_AF")])
    s = "Drög af verkefninu eru tilbúin."
    check_sentence(api, s, [(1, 1, "P_WRONG_PREP_AF")])
    #    s = "Þau kláruðu drög af verkefninu."
    #    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])
    s = "Grunnur af verkefninu er tilbúinn."
    check_sentence(api, s, [(1, 1, "P_WRONG_PREP_AF")])
    #    s = "Hann lagði ekki grunninn af verkefninu."
    #    check_sentence(rc, s, [(3, 4, "P_WRONG_PREP_AF")])
    #    s = "Þau gerðu leit af dótinu."
    #    check_sentence(rc, s, [(2, 3, "P_WRONG_PREP_AF")])
    s = "Leit af dótinu hefur ekki skilað árangri."
    check_sentence(api, s, [(1, 1, "P_WRONG_PREP_AF")])
    s = "Þetta er lykillinn af velgengni."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AF")])
    s = "Hann gaf mér uppskriftina af réttinum."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AF")])


def test_verb_að(api):
    s = "Ég er ekki hluti að heildinni."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Við höfum öll verið hluti að heildinni."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    s = "Vissulega er hægt að vera hluti að heildinni."
    check_sentence(api, s, [(6, 6, "P_WRONG_PREP_AÐ")])
    #    s = "Þeir sögðu að ég hefði verið hluti að heildinni."   # Annotation variable depending on parsing
    #    check_sentence(rc, s, [(6, 8, "P_WRONG_PREP_AÐ")])
    s = "Að öllu óbreyttu er hann hluti að heildinni."
    check_sentence(api, s, [(6, 6, "P_WRONG_PREP_AÐ")])
    s = "Ég vildi vera hluti að heildinni að mestu leyti."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Þar að leiðandi virkar þetta."
    check_sentence(api, s, [(0, 2, "P_aðaf"), (1, 1, "S005")])
    s = "Þar að leiðandi virkar þetta að mestu leyti."
    check_sentence(api, s, [(0, 2, "P_aðaf"), (1, 1, "S005")])
    s = "Þetta virkar þar að leiðandi."
    check_sentence(api, s, [(2, 4, "P_aðaf"), (3, 3, "S005")])
    s = "Ég hef ekki áhyggjur að honum."
    # check_sentence(rc, s, [(2, 5, "P_WRONG_PREP_AÐ")])
    s = "Ég hef áhyggjur að því að honum líði illa."
    # check_sentence(rc, s, [(2, 8, "P_WRONG_PREP_AÐ")])
    s = "Ég lagði ekki mikið að mörkum."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Ég hafði lagt mikið að mörkum."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Sama hvað ég gerði lagði ég mikið að mörkum."
    check_sentence(api, s, [(7, 7, "P_WRONG_PREP_AÐ")])
    s = "Að hans mati lagði hann mikið að mörkum."
    check_sentence(api, s, [(6, 6, "P_WRONG_PREP_AÐ")])
    s = "Ég heillast að þannig fólki."
    check_sentence(api, s, [(2, 2, "P_WRONG_PREP_AÐ")])
    s = "Það að heillast að þannig fólki er algengt."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Ég lét gott að mér leiða."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Að mínu mati lét ég ekki gott að mér leiða."
    check_sentence(api, s, [(7, 7, "P_WRONG_PREP_AÐ")])
    s = "Hún á heiðurinn að þessu."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Hún hafði ekki átt heiðurinn að þessu en fékk heiðurinn að þessu."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ"), (10, 10, "P_WRONG_PREP_AÐ")])
    s = "Hún hlaut heiðurinn að þessu."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Að endingu hljótum við öll heiðurinn að því."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    s = "Hún á heilan helling að börnum."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Hún á marga að."
    check_sentence(api, s, [])
    s = "Að mínu mati eigum við mikið að mat."
    check_sentence(api, s, [(6, 6, "P_WRONG_PREP_AÐ")])
    s = "Hún á ekki aðild að málinu."
    check_sentence(api, s, [])
    s = "Hún hefur ekki haft gagn að þessu."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    s = "Að mínu mati hef ég ekki gagn að þessu."
    # check_sentence(rc, s, [(3, 8, "P_WRONG_PREP_AÐ")])
    s = "Þetta hafði ekki komið að sjálfu sér."
    check_sentence(api, s, [(4, 6, "P_aðaf")])
    s = "Fréttir bárust seint að slysinu."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Að endingu berast fréttir að slysinu."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Þetta er afgreitt mál að minni hálfu."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Hætta hefur aldrei stafað að þessu."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    #    s = "Að mínu mati stafar ekki hætta að þessu."  # Annotation variable depending on parse
    #    check_sentence(rc, s, [(3, 7, "P_WRONG_PREP_AÐ")])
    s = "Hann hefur ekki heyrt að lausa starfinu."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Að endingu heyrði ég að starfinu."
    check_sentence(api, s, [(3, 3, "P_WRONG_PREP_AÐ")])
    s = "Ég hef aldrei haft gaman að henni."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    #    s = "Ég sagði að ég hefði gaman að henni."  # Annotation variable depending on parse
    #    check_sentence(rc, s, [(5, 7, "P_WRONG_PREP_AÐ")])
    s = "Þau voru sérstaklega valin að stjórninni."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    #    s = "Þú ert valinn að guði að okkar mati."  # Annotation variable depending on parse
    #    check_sentence(rc, s, [(1, 7, "P_WRONG_PREP_AÐ")])
    s = "Það er til mjög lítið að mjólk."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    s = "Ekki er mikið til að mjólk."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Að öllu leyti er til fullt að mjólk."
    check_sentence(api, s, [(4, 4, "P_WRONG_PREP_AÐ")])
    s = "Ég hef ekki unnið verkefni að þessu tagi."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])
    s = "Verkefni að þessum toga eru erfið."
    check_sentence(api, s, [(1, 1, "P_WRONG_PREP_AÐ")])
    s = "Að mínu mati gerði ég þetta að krafti."
    check_sentence(api, s, [(6, 6, "P_WRONG_PREP_AÐ")])
    s = "Hann gerði það að sjálfsdáðum."
    check_sentence(api, s, [(3, 4, "P_aðaf")])
    s = "Að sjálfsögðu gerði ég þetta að sjálfsdáðum."
    check_sentence(api, s, [(4, 5, "P_aðaf")])
    s = "Hún hefur ekki gert þetta að miklum krafti."
    check_sentence(api, s, [(5, 5, "P_WRONG_PREP_AÐ")])


def test_placename_pp(api):
    s = "Ég hef búið á Hafnarfirði alla mína tíð en flyt nú í Akureyri."
    check_sentence(api, s, [(3, 4, "P_WRONG_PLACE_PP"), (9, 10, "P_WRONG_PLACE_PP")])
    s = "Ég hef veitt í Vopnafirði undanfarin ár en búið á Vopnafirði."
    check_sentence(api, s, [])
    s = "Það eru mörg náttúruvætti á Reykjanesi en ekki í Húsavík."
    check_sentence(api, s, [(8, 9, "P_WRONG_PLACE_PP")])


def test_verb_líst(api):
    s = "Jóni veiðimanni lýst ekki á þetta mál."
    # check_sentence(rc, s, [(2, 2, "P_WRONG_OP_FORM")])
    check_sentence(api, s, [(2, 2, "Ý4Í")])
    s = "Eins og fram hefur komið lýst mér vel á þetta."
    check_sentence(api, s, [(5, 5, "Ý4Í")])
    s = "Jón hefur lýst sinni afstöðu til málsins."
    # TODO: 'Corrected' at token-level, in the works to fix at sentence-level
    # check_sentence(rc, s, [])
    s = "Þegar leið á kvöldið var gangstéttin lýst með ljósum."
    # check_sentence(rc, s, [], ignore_warnings=True)
    # TODO: The following gets no annotation:
    "Ég verð að segja að mér lýst ekkert á þetta."
    check_sentence(api, s, [(6, 6, "Ý4Í")])


def test_dir_loc(api):
    s = "Börnin voru út á túni allan daginn."
    check_sentence(api, s, [(2, 4, "P_DIR_LOC")])
    s = "Börnin voru útá túni allan daginn."
    check_sentence(api, s, [(2, 3, "P_DIR_LOC"), (2, 2, "W001/w")])
    #    s = "Út í heimi er þetta öðruvísi."
    #    check_sentence(rc, s, [(0, 2, "P_DIR_LOC")])
    #    s = "Útí heimi er þetta öðruvísi."
    #    check_sentence(rc, s, [(0, 1, "P_DIR_LOC")])
    s = "Börnin voru inn á vellinum allan daginn."
    check_sentence(api, s, [(2, 4, "P_DIR_LOC")])
    s = "Börnin voru inná vellinum allan daginn."
    check_sentence(api, s, [(2, 3, "P_DIR_LOC"), (2, 2, "W001/w")])
    #    s = "Hann var oft upp á hestinum."
    #    check_sentence(rc, s, [(3, 5, "P_DIR_LOC")])
    s = "Málið liggur í augum upp."
    check_sentence(api, s, [(2, 4, "P_DIR_LOC")])
    #    s = "Þau eru alltaf uppí bústað."
    #    check_sentence(rc, s, [(1, 4, "P_DIR_LOC"), (3, 3, "W001/w")])
    # Span is either 1,4 or 3,4 (depending on parsing), but always corrected.
    s = "Hún var niður í bæ í gær."
    check_sentence(api, s, [(1, 5, "P_DIR_LOC")])
    #    s = "Ég varð mér út um smá mat."
    #    check_sentence(rc, s, [(3, 6, "P_DIR_LOC")])
    #    s = "Þegar upp er staðið erum við öll eins."
    #    check_sentence(rc, s, [(1, 3, "P_DIR_LOC")])
    #    s = "Út í heimi er þetta öðruvísi."
    #    check_sentence(rc, s, [(0, 2, "P_DIR_LOC")])
    s = "Börnin safnast saman inn í búð."
    check_sentence(api, s, [(3, 5, "P_DIR_LOC")])
    s = "Ég keypti þetta út í búð."
    check_sentence(api, s, [(3, 5, "P_DIR_LOC")])
    s = "Illgresið er út um allt."
    check_sentence(api, s, [(2, 4, "P_DIR_LOC")])
    #    s = "Hann læsti sig inn í gær."
    #    check_sentence(rc, s, [(1, 4, "P_DIR_LOC")])
    s = "Hún gaf það upp í fréttum."
    check_sentence(api, s, [])
    s = "Ég ólst upp í Breiðholtinu."
    check_sentence(api, s, [])
