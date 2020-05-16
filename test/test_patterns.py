"""

    test_patterns.py

    Tests for GreynirCorrect module

    Copyright (C) 2020 by Miðeind ehf.

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


@pytest.fixture(scope="module")
def rc():
    """ Provide a module-scoped GreynirCorrect instance as a test fixture """
    r = reynir_correct.GreynirCorrect()
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
