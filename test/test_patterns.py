"""

    test_patterns.py

    Tests for ReynirCorrect module

    Copyright (C) 2020 by Miðeind ehf.

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


    This module tests the grammatical pattern annotation functionality
    of ReynirCorrect.

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


def test_verb_af(rc):
    s = "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar."
    check_sentence(rc, s, [(1, 2, "P001")])

    s = (
        "Mig langaði að leita af bílnum, en dáðist svo af hugrekki lögreglukonunnar "
        "að ég gerði það ekki."
    )
    check_sentence(rc, s, [(3, 4, "P001"), (8, 10, "P001")])

    s = "Við höfum leitað í allan dag af kettinum, en fundum hann ekki."
    check_sentence(rc, s, [(2, 4, "P001")])

    s = "Allan daginn höfum við leitað af kettinum."
    check_sentence(rc, s, [(4, 5, "P001")])

    s = "Páll brosti af töktunum í Gunnu."
    check_sentence(rc, s, [(1, 2, "P001")])

    s = "Ég var leitandi af kettinum í allan dag."
    check_sentence(rc, s, [(2, 3, "P001")])

    s = "Ég vildi leita af mér allan grun."
    check_sentence(rc, s, [])

    s = "Ertu að leita af skrifstofuhúsnæði?"
    check_sentence(rc, s, [(2, 3, "P001")])
