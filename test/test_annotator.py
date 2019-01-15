"""

    test_annotator.py

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
    check_sentence(s, ((0, 2, "E002"), (6, 8, "P_aðaf")))


def test_impersonal_verbs(verbose=False):
    s = "Mig hlakkaði til."
    check_sentence(s, ((0, 0, "E003"),))
    s = (
        "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, "
        "langaði að horfa á sjónvarpið."
    )
    check_sentence(s, ((0, 11, "E003"),))
    s = "Önnu kveið fyrir skóladeginum."
    check_sentence(s, ((0, 0, "E003"),))
    s = "Hestinum Grímni vantaði hamar."
    # s = "Hestinum Skjóna vantaði hamar."
    check_sentence(s, ((0, 1, "E003"),))
    # s = "Ég dreymdi köttinn."
    # check_sentence(s, ((0, 0, "E003"),))


if __name__ == "__main__":

    test_multiword_prases(verbose=True)
    test_impersonal_verbs(verbose=True)
