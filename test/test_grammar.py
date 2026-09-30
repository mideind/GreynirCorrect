"""

    test_grammar.py

    Tests for the error-detecting grammar of GreynirCorrect,
    in particular the fallback productions that make the error grammar
    load with GreynirEngine 3.9.0.

    Copyright © 2026 Miðeind ehf.

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

"""

from reynir_correct.checker import ErrorDetectingGrammar, ErrorDetectingParser


def _lines(text: str):
    return text.splitlines(keepends=True)


def test_no_fallback_without_sp_person():
    """Engine versions without the 'sp' person variant need no fallback"""
    grammar = _lines(
        "/pers = p1 p2 p3\n"
        "BeygingarliðurÁnUmröðunar/tala/pers/kyn →\n"
        "    > VillaÍTölu/tala/pers/kyn\n"
        "VillaÍTölu_et_p1/kyn →\n"
        "    Frumlag_p1_et/kyn BeygingarliðurMegin_ft_p1/kyn\n"
    )
    assert ErrorDetectingGrammar._fallback_lines(grammar) == []


def test_fallback_added_for_sp_person():
    """With the 'sp' variant present and the nonterminals undefined,
    both fallback productions are appended"""
    grammar = _lines(
        "/pers = p1 p2 p3 sp   # sp = spurnarmynd með viðskeyttu frumlagi\n"
        "BeygingarliðurÁnUmröðunar/tala/pers/kyn →\n"
        "    > VillaÍTölu/tala/pers/kyn\n"
        "VillaÍTölu_et_p1/kyn →\n"
        "    Frumlag_p1_et/kyn BeygingarliðurMegin_ft_p1/kyn\n"
    )
    fallback = ErrorDetectingGrammar._fallback_lines(grammar)
    assert len(fallback) == 2
    assert fallback[0].startswith("VillaÍTölu_et_sp/kyn →")
    assert fallback[1].startswith("VillaÍTölu_ft_sp/kyn →")


def test_no_fallback_when_engine_defines_nonterminals():
    """Once the engine defines the nonterminals itself, nothing is appended"""
    grammar = _lines(
        "/pers = p1 p2 p3 sp\n"
        "BeygingarliðurÁnUmröðunar/tala/pers/kyn →\n"
        "    > VillaÍTölu/tala/pers/kyn\n"
        "VillaÍTölu_et_sp/kyn →\n"
        "    Frumlag_sp_et/kyn BeygingarliðurMegin_ft_sp/kyn\n"
        "VillaÍTölu_ft_sp/kyn →\n"
        "    Frumlag_sp_ft/kyn BeygingarliðurMegin_et_sp/kyn\n"
    )
    assert ErrorDetectingGrammar._fallback_lines(grammar) == []


def test_error_grammar_loads():
    """The error-detecting grammar of the installed engine loads without
    a GrammarError, and the number-agreement error nonterminal is present"""
    parser = ErrorDetectingParser()
    grammar = parser.grammar
    assert "VillaÍTölu_et_p3_kk" in grammar.nonterminals
    for nt in grammar.nonterminals.values():
        # Every nonterminal that is referenced must have productions;
        # this is what fails with an unpatched GreynirEngine 3.9.0
        assert nt in grammar.nt_dict
