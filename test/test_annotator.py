"""

    test_annotator.py

    Tests for ReynirCorrect module

    Copyright(C) 2019 by Miðeind ehf.
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

"""

import reynir_correct as rc


def dump(tokens):
    print("\n{0} tokens:\n".format(len(tokens)))
    for token in tokens:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}: {1}".format(token.error_code, err))


def check_sentence(s, annotations):
    """ Check whether a given single sentence gets the
        specified annotations when checked """

    def check_sent(sent):
        assert sent is not None
        assert sent.tree is not None
        assert hasattr(sent, "annotations")
        assert len(sent.annotations) == len(annotations)
        for a, (start, end, code) in zip(sent.annotations, annotations):
            assert a.start == start
            assert a.end == end
            assert a.code == code

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


def test_multiword_phrases(verbose=False):
    s = "Einn af drengjunum fóru í sund af gefnu tilefni."
    check_sentence(s, [(0, 2, "P_NT_EinnAf"), (6, 8, "P_aðaf")])


def test_error_finder(verbose=False):
    """ Test errors that are found by traversing the detailed
        parse tree in checker.py (ErrorFinder class) """
    s = "Fjöldi þingmanna greiddu atkvæði gegn tillögunni."
    check_sentence(s, [(0, 1, "P_NT_FjöldiHluti")])
    s = "Jón borðaði ís þar sem að hann var svangur."
    check_sentence(s, [(5, 5, "P_NT_Að")])
    s = "Jón \"borðaði\" ís þar sem að hann var svangur."
    check_sentence(s, [(7, 7, "P_NT_Að")])
    s = "Jón borðaði ís þó hann væri svangur."
    check_sentence(s, [(3, 3, "P_NT_ÞóAð")])
    s = "Jón \"borðaði\" ís þó hann væri svangur."
    check_sentence(s, [(5, 5, "P_NT_ÞóAð")])
    s = "Jón borðaði ís jafnvel þó hann væri svangur."
    check_sentence(s, [(3, 4, "P_NT_ÞóAð")])
    s = "Jón \"borðaði\" ís jafnvel þó hann væri svangur."
    check_sentence(s, [(5, 6, "P_NT_ÞóAð")])
    s = "Jón borðaði ís þótt hann væri svangur."
    check_sentence(s, [])
    s = "Jón \"borðaði\" ís þótt hann væri svangur."
    check_sentence(s, [])
    s = "Ég féll fyrir annað hvort fegurð hennar eða gáfum."
    check_sentence(s, [(3, 4, "P_NT_AnnaðHvort")])
    s = "Ég talaði við annaðhvort barnanna."
    check_sentence(s, [(3, 3, "P_NT_Annaðhvort")])
    s = "Ég hef verið slappur frá því ég fékk sprautuna."
    check_sentence(s, [(4, 5, "P_NT_FráÞvíAð")])
    s = "Ég hef verið slappur allt frá því ég fékk sprautuna."
    check_sentence(s, [(4, 6, "P_NT_FráÞvíAð")])
    s = "Friðgeir vildi vera heima víst að Sigga yrði að vera heima."
    check_sentence(s, [(4, 5, "P_NT_VístAð")])
    s = "Friðgeir taldi víst að Sigga yrði að vera heima."
    check_sentence(s, [])
    s = "Ég er ekki meiri fáviti heldur en þú."
    check_sentence(s, [(5, 5, "P_NT_Heldur")])


def test_impersonal_verbs(verbose=False):
    s = "Mig hlakkaði til."
    check_sentence(s, [(0, 0, "P_WRONG_CASE_þf_nf")])
    s = (
        "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, "
        "langaði að horfa á sjónvarpið."
    )
    check_sentence(s, [(0, 11, "P_WRONG_CASE_þgf_þf")])
    s = "Önnu kveið fyrir skóladeginum."
    check_sentence(s, [(0, 0, "P_WRONG_CASE_þgf_nf")])
    s = "Hestinum Grímni vantaði hamar."
    # s = "Hestinum Skjóna vantaði hamar."
    check_sentence(s, [(0, 1, "P_WRONG_CASE_þgf_þf")])
    # s = "Ég dreymdi köttinn."
    # check_sentence(s, ((0, 0, "E003"),))


if __name__ == "__main__":

    test_multiword_phrases(verbose=True)
    test_impersonal_verbs(verbose=True)
    test_error_finder(verbose=True)
