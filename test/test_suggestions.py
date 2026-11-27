"""

    test_suggestions.py

    Tests for the suggestions of annotations

    Copyright © 2026 by Miðeind ehf.

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


    The suggestion of an annotation replaces all the tokens of its span
    (see fully_correct_sentence() in wrappers.py, and the README). A
    suggestion that only has the corrected word of a wider span therefore
    deletes the other words: 'Ég keypti þetta út í búð' became 'Ég keypti
    þetta úti', and 'Ég er ekki að skilja þetta' became 'Ég skil þetta'.

"""

from typing import List, Tuple

import pytest

from reynir import correct_spaces

import reynir_correct


@pytest.fixture(scope="module")
def api() -> reynir_correct.GreynirCorrectAPI:
    """Provide a module-scoped GreynirCorrectAPI instance as a test fixture"""
    return reynir_correct.GreynirCorrectAPI.from_options()


def corrected(api: reynir_correct.GreynirCorrectAPI, text: str) -> Tuple[List[Tuple[str, str, str]], str]:
    """Check a text. Return (code, original, suggestion) for each annotation that
    has a suggestion, and the text with all of the suggestions applied."""
    result = api.correct(text)
    annotations: List[Tuple[str, str, str]] = []
    for sent in result.sentences:
        for a in sent.annotations or []:
            if a.suggest is not None:
                annotations.append((a.code, a.original or "", a.suggest))
    # Note: this goes last, as applying the suggestions changes the tokens
    text = " ".join(sent.corrected_str(apply_annotations=True) for sent in result.sentences)
    return annotations, text


def test_wrong_case_keeps_punctuation(api: reynir_correct.GreynirCorrectAPI) -> None:
    """A noun phrase that ends with punctuation keeps it when cast to another case"""
    annotations, text = corrected(api, "Páli, vini mínum, langaði að horfa á sjónvarpið.")
    assert annotations == [("P_WRONG_CASE_þgf_þf", "Páli, vini mínum,", "Pál, vin minn,")]
    assert text == "Pál, vin minn, langaði að horfa á sjónvarpið."

    annotations, text = corrected(
        api, "Páli, sem hefur verið landsliðsmaður í fótbolta í sjö ár, langaði að horfa á sjónvarpið."
    )
    assert [a[0] for a in annotations] == ["P_WRONG_CASE_þgf_þf"]
    assert text == "Pál, sem hefur verið landsliðsmaður í fótbolta í sjö ár, langaði að horfa á sjónvarpið."

    # Punctuation that is not cut off to begin with
    annotations, text = corrected(api, "Páli (vini mínum) langaði heim.")
    assert text == "Pál (vin minn) langaði heim."

    # The end of an abbreviation is not punctuation
    annotations, text = corrected(api, "Jóni Jónssyni o.fl. langaði heim.")
    assert ("P_WRONG_CASE_þgf_þf", "Jóni Jónssyni o.fl.", "Jón Jónsson o.fl.") in annotations
    assert text == "Jón Jónsson o.fl. langaði heim."

    # Here the phrase is the same in both cases, so there is nothing to suggest
    # (the suggestion was 'Samherja h')
    annotations, text = corrected(api, "Samherja hf. langaði í kvóta.")
    assert annotations == []
    assert text == "Samherja hf. langaði í kvóta."


def test_wrong_case_keeps_adverb(api: reynir_correct.GreynirCorrectAPI) -> None:
    """An adverb in front of a noun phrase is kept when the phrase is cast to another case"""
    annotations, text = corrected(api, "Einnig Páli langaði heim.")
    assert annotations == [("P_WRONG_CASE_þgf_þf", "Einnig Páli", "Einnig Pál")]
    assert text == "Einnig Pál langaði heim."

    annotations, text = corrected(api, "Hann hélt utan um dóttir sína.")
    assert annotations == [("P_NT_FsMeðFallstjórn", "um dóttir", "um dóttur")]
    assert text == "Hann hélt utan um dóttur sína."


def test_dir_loc_keeps_phrase(api: reynir_correct.GreynirCorrectAPI) -> None:
    """P_DIR_LOC corrects the adverb and keeps the phrase that follows it"""
    for sentence, original, suggestion, result in [
        ("Ég keypti þetta út í búð.", "út í búð", "úti í búð", "Ég keypti þetta úti í búð."),
        ("Börnin voru út á túni allan daginn.", "út á túni", "úti á túni", "Börnin voru úti á túni allan daginn."),
        ("Hann geymdi bókina inn í skáp.", "inn í skáp", "inni í skáp", "Hann geymdi bókina inni í skáp."),
        ("Ég hitti hana inn í búðinni.", "inn í búðinni", "inni í búðinni", "Ég hitti hana inni í búðinni."),
        ("Út á túni voru börnin.", "Út á túni", "Úti á túni", "Úti á túni voru börnin."),
        # dir_loc_ut_um()
        ("Hann var oft út um allt.", "út um allt", "úti um allt", "Hann var oft úti um allt."),
        # dir_loc_simple(), where the span starts at the verb
        ("Þau eru niður í bæ.", "eru niður í bæ", "eru niðri í bæ", "Þau eru niðri í bæ."),
        # dir_loc_comp(), which was right
        ("Þeir voru inná vellinum.", "inná vellinum", "inni á vellinum", "Þeir voru inni á vellinum."),
    ]:
        annotations, text = corrected(api, sentence)
        assert annotations == [("P_DIR_LOC", original, suggestion)], sentence
        assert text == result


def test_vera_að_keeps_words(api: reynir_correct.GreynirCorrectAPI) -> None:
    """P_VeraAð keeps what stands between 'vera' and the verb, the negation above all,
    in the word order of a finite verb"""
    for sentence, original, suggestion, result in [
        ("Kristín er að skrifa vel.", "er að skrifa", "skrifar", "Kristín skrifar vel."),
        # A noun phrase stays behind the adverb
        (
            "Ég er alls ekki að skilja þetta mál.",
            "er alls ekki að skilja",
            "skil alls ekki",
            "Ég skil alls ekki þetta mál.",
        ),
        ("Ég er ekki að tala við hann.", "er ekki að tala", "tala ekki", "Ég tala ekki við hann."),
        # ...as do an indefinite pronoun and a pronoun with a relative clause
        ("Ég er ekki að skilja neitt.", "er ekki að skilja", "skil ekki", "Ég skil ekki neitt."),
        (
            "Ég er ekki að skilja þá sem tala svona.",
            "er ekki að skilja",
            "skil ekki",
            "Ég skil ekki þá sem tala svona.",
        ),
        # A pronoun object goes in front of the adverb, and is a part of the span
        ("Ég er ekki að skilja þetta.", "er ekki að skilja þetta", "skil þetta ekki", "Ég skil þetta ekki."),
        (
            "Þið eruð ekki að standa ykkur vel.",
            "eruð ekki að standa ykkur",
            "standið ykkur ekki",
            "Þið standið ykkur ekki vel.",
        ),
        (
            "Ég er ekki að gefa honum það.",
            "er ekki að gefa honum það",
            "gef honum það ekki",
            "Ég gef honum það ekki.",
        ),
        ("Ég er ekki að gefa honum bókina.", "er ekki að gefa honum", "gef honum ekki", "Ég gef honum ekki bókina."),
        # ...also a pronoun that stands for a clause, with a particle of the verb in between
        (
            "Ég er ekki að halda því fram að hann komi.",
            "er ekki að halda því",
            "held því ekki",
            "Ég held því ekki fram að hann komi.",
        ),
        # Without an adverb, nothing moves, and the span ends at the verb
        ("Ég er að skilja þetta.", "er að skilja", "skil", "Ég skil þetta."),
        # The subject of a question, and the capital letter
        ("Er ég að skilja þetta rétt?", "Er ég að skilja", "Skil ég", "Skil ég þetta rétt?"),
    ]:
        annotations, text = corrected(api, sentence)
        assert annotations == [("P_VeraAð", original, suggestion)], sentence
        assert text == result


def test_original_is_text_of_span(api: reynir_correct.GreynirCorrectAPI) -> None:
    """For an annotation of more than one token that has a suggestion, 'original'
    is the text of those tokens. A handler whose 'original' is one word of the
    span is likely to suggest a replacement for that word only."""
    sentences = [
        "Páli, vini mínum, langaði að horfa á sjónvarpið.",
        "Einnig Páli langaði heim.",
        "Hann hélt utan um dóttir sína.",
        "Ég keypti þetta út í búð.",
        "Hann var oft út um allt.",
        "Þau eru niður í bæ.",
        "Þeir voru inná vellinum.",
        "Ég er ekki að skilja þetta.",
        "Er ég að skilja þetta rétt?",
        "Hún býr í gömlu Akureyri.",
        "Stóra manninum sem vinnur á verkstæðinu langaði heim.",
        "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar.",
        "Þar að leiðandi fór hann heim af sjálfu sér.",
        "Ég vill fara heim í auknu mæli.",
        "Hann hljóp allt kvað fætur toga, jafnvel þó hann væri þreyttur.",
    ]
    checked = 0
    for sentence in sentences:
        result = api.correct(sentence)
        for sent in result.sentences:
            for a in sent.annotations or []:
                if a.suggest is None or a.end == a.start:
                    continue
                tokens = sent.tokens[a.start : a.end + 1]
                # The text of the tokens, as corrected by the tokenizer or as written
                span = correct_spaces(" ".join(t.txt for t in tokens if t.txt))
                written = correct_spaces(" ".join("".join(t.original or "" for t in tokens).split()))
                assert correct_spaces(a.original or "") in (span, written), f"{a.code} in '{sentence}'"
                checked += 1
    # Make sure that the sentences are still annotated, so that the test tests something
    assert checked >= 12
