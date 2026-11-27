"""

    test_threads.py

    Checks from many threads at once must give the same results as from one

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

    A GreynirCorrect or GreynirCorrectAPI object can be shared between
    threads. Each text is tokenized with a pipeline of its own; before that,
    the objects kept one pipeline and put each text into it, so a second
    thread either failed or tokenized the text of the first.

    On free-threaded Python the threads really run in parallel. With the
    GIL they only take turns, which still showed the fault.

"""

from typing import Callable, List

import random
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import reynir_correct
from reynir_correct import CorrectionPipeline, GreynirCorrect, GreynirCorrectAPI, check_single, tokenize
from reynir_correct.errtokenizer import settings_or_default

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
# With the GIL the threads take turns, so more of them add run time but little coverage
THREADS = 8 if FREE_THREADED else 4
# PyPy is slow at switching threads this often (see _in_threads())
ROUNDS = 1 if sys.implementation.name == "pypy" else 3

# Sentences with errors of different kinds, and some without
SENTENCES = [
    "Páli, vini mínum, langaði að horfa á sjónnvarpið.",
    "Ég keypti þetta út í búð.",
    "Hún býr í gömlu Akureyri.",
    "Ráðherrann dáðist af hugrekki stjórnarandstöðunnar.",
    "Af gefnu tilefni fékk fékk daninn vilja sýnum framgengt í auknu mæli.",
    "Ég kláraði verkefnið þrátt fyrir að ég var þreittur.",
    "Atvinuleysi jógst um 3% í síðasta mánuði.",
    "Barnið vil grænann lit á vegginn.",
    "Pakkin er fyrir hestin sem stendur úti.",
    "Þinngið samþikkti tilöguna í gær.",
    "Mér dreymdi að hún hefði komið heim.",
    "Hann fór til sýn eftir fundinn.",
    "Klukkan var orðinn tólf þegar við fórum heim.",
    "Við sáum tvo seli og öruglega fleiri en 100 máva.",
    "Það var 17. júní árið 2020 í frakklandi.",
    "Börnin voru út á túni allan daginn.",
    "Ég hugsa ekki um hvað Bretland gerir né hvað Ísland gerir.",
    "Hundurinn beit manninn í gær.",
    "Ríkisstjórnin hefur ákveðið að hækka skatta á næsta ári.",
    "Hún sagði að hann hefði komið heim seint um kvöldið.",
    "Lögreglan handtók tvo menn vegna gruns um innbrot.",
    "Þetta er á vegum Reykjavíkurborgar.",
    "The quick brown fox jumps over the lazy dog.",
    "Hann geymdi bókina inn í skáp.",
]


def _tokens(tokens: List[reynir_correct.CorrectToken]) -> str:
    return " ".join(f"{t.kind}:{t.txt}:{t.error_code}:{t.error_suggest}" for t in tokens)


def _annotations(annotations: List[reynir_correct.Annotation]) -> str:
    return " ".join(f"{a.start}-{a.end}:{a.code}:{a.text}:{a.suggest}" for a in annotations)


def _in_threads(check: Callable[[str], str]) -> None:
    """Run check() on the sentences in one thread, then in many at once,
    and compare the results"""
    expected = [check(s) for s in SENTENCES]
    barrier = threading.Barrier(THREADS)

    def work(seed: int) -> List[str]:
        rnd = random.Random(seed)
        order = list(range(len(SENTENCES)))
        problems: List[str] = []
        barrier.wait()
        for _ in range(ROUNDS):
            # Each thread takes the sentences in an order of its own
            rnd.shuffle(order)
            for ix in order:
                result = check(SENTENCES[ix])
                if result != expected[ix]:
                    problems.append(f"{SENTENCES[ix]!r}:\n  expected {expected[ix]}\n  got      {result}")
        return problems

    # With the GIL, a thread runs for 5 ms before the next one gets its
    # turn, and so seldom loses it in the middle of a call. A short interval
    # makes the threads interleave as closely as they can.
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futures = [pool.submit(work, seed) for seed in range(THREADS)]
            # result() raises what a thread raised
            problems = [p for f in futures for p in f.result()]
    finally:
        sys.setswitchinterval(interval)
    assert not problems, "\n".join(problems[:5])


def test_shared_api_in_threads() -> None:
    """One GreynirCorrectAPI object, used by all threads"""
    api = GreynirCorrectAPI.from_options()

    def check(text: str) -> str:
        result = api.correct(text)
        return " || ".join(
            f"{sent.parsed} | {_tokens(sent.tokens)} | {_annotations(sent.annotations or [])}"
            for sent in result.sentences
        )

    _in_threads(check)


def test_shared_api_options_in_threads() -> None:
    """The options of a call must not reach the calls of other threads"""
    api = GreynirCorrectAPI.from_options()
    rules = frozenset(("S004",))

    def check(text: str) -> str:
        # Every other sentence is checked with a rule switched off
        ignore = rules if len(text) % 2 else None
        result = api.correct(text, ignore_rules=ignore)
        return " || ".join(
            f"{sent.parsed} | {_tokens(sent.tokens)} | {_annotations(sent.annotations or [])}"
            for sent in result.sentences
        )

    _in_threads(check)


def test_shared_checker_in_threads() -> None:
    """One GreynirCorrect object, used by all threads"""
    settings = settings_or_default()
    rc = GreynirCorrect(settings, CorrectionPipeline("", settings))

    def check(text: str) -> str:
        sent = check_single(text, rc=rc)
        assert sent is not None
        tree = sent.tree.flat if sent.tree is not None else "-"
        return f"{_tokens(sent.tokens)} | {_annotations(sent.annotations)} | {tree}"

    _in_threads(check)


def test_functions_in_threads() -> None:
    """The functions that make their own objects for each call"""

    def check(text: str) -> str:
        sent = check_single(text)
        assert sent is not None
        return f"{_tokens(list(tokenize(text)))} | {_annotations(sent.annotations)}"

    _in_threads(check)


def test_pipeline_clone() -> None:
    """A clone has the options of the pipeline, its own text, and its own phases"""
    settings = settings_or_default()
    pipeline = CorrectionPipeline("", settings, only_ci=True)
    clone = pipeline.clone("Pakkin er fyrir hestin.", ignore_rules=frozenset(("S004",)))
    assert clone is not pipeline
    assert clone._only_ci is True
    assert clone._ignore_rules == frozenset(("S004",))
    assert pipeline._ignore_rules == frozenset()
    assert pipeline._text_or_gen == ""
    assert all(phase.__self__ is clone for phase in clone._phases)  # type: ignore[attr-defined]
    assert all(phase.__self__ is pipeline for phase in pipeline._phases)  # type: ignore[attr-defined]
    assert len(clone._phases) == len(pipeline._phases)
    assert [t.txt for t in clone.tokenize() if t.txt] == ["Pakkin", "er", "fyrir", "hestin", "."]
