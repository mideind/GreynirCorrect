
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/reynir-correct)](https://pypi.org/project/reynir-correct/)
[![GitHub release](https://shields.io/github/v/release/mideind/GreynirCorrect?display_name=tag)](https://github.com/mideind/GreynirCorrect/releases)
[![Tests](https://github.com/mideind/GreynirCorrect/actions/workflows/python-package.yml/badge.svg)](https://github.com/mideind/GreynirCorrect/actions/workflows/python-package.yml)

# GreynirCorrect: Spelling and grammar correction for Icelandic

## Overview

**GreynirCorrect** is a Python 3 (>=3.11) package and command line tool for
**checking and correcting spelling and grammar** in Icelandic text.

GreynirCorrect relies on the [GreynirEngine](https://github.com/mideind/GreynirEngine)
package (`reynir` on PyPI), by the same authors, to tokenize and parse text.
Each version of GreynirCorrect works with one minor version of GreynirEngine,
as its grammar checks depend on the shape of the parse trees: version 4.3
requires GreynirEngine 4.2.

GreynirCorrect is documented in detail [here](https://yfirlestur.is/doc/).

The software has three main modes of operation, described below. All three
can be used from several threads at once, also on free-threaded Python (see
[Use in threads](#use-in-threads)).

### Token-level correction

GreynirCorrect can tokenize text and return an automatically corrected token stream.
This catches token-level errors, such as spelling errors and erroneous
phrases, but not grammatical errors. Token-level correction is relatively fast.

### Full grammar analysis

GreynirCorrect can analyze text grammatically by attempting to parse
it, after token-level correction. The parsing is done according to Greynir's
context-free grammar for Icelandic, augmented with additional production
rules for common grammatical errors. The analysis returns a set of annotations
(errors and suggestions) that apply to spans (consecutive tokens) within
sentences in the resulting token list. Full grammar analysis is considerably
slower than token-level correction.

### Command-line tool

GreynirCorrect can be invoked as a command-line tool
to perform token-level correction and, optionally, grammar analysis.
The command is `correct infile.txt outfile.txt`.
The command-line tool is further documented below.

## Examples

To perform token-level correction from Python code:

```python
from reynir_correct import tokenize
g = tokenize("Af gefnu tilefni fékk fékk daninn vilja sýnum "
    "framgengt í auknu mæli.")
for tok in g:
    if tok.txt:
        print("{0:10} {1}".format(tok.txt, tok.error_description).rstrip())
```

Output:

```
Að         Orðasambandið 'Af gefnu tilefni' var leiðrétt í 'Að gefnu tilefni'
gefnu
tilefni
fékk       Endurtekið orð ('fékk') ætti að fella burt
Daninn     Orð á að byrja á hástaf: 'daninn'
vilja      Orðasambandið 'vilja sýnum framgengt' var leiðrétt í 'vilja sínum framgengt'
sínum      Rita á 'sýnum' sem 'sínum'
framgengt
í          Orðasambandið 'í auknu mæli' var leiðrétt í 'í auknum mæli'
auknum     Rita á 'auknu' sem 'auknum'
mæli
.
```

To perform full spelling and grammar analysis of a sentence from Python code:

```python
from reynir_correct import check_single
sent = check_single("Páli, vini mínum, langaði að horfa á sjónnvarpið.")
for annotation in sent.annotations:
    print("{0}".format(annotation))
```

Output:

```
000-004: P_WRONG_CASE_þgf_þf  Á líklega að vera 'Pál, vin minn' | 'Páli, vini mínum,' -> 'Pál, vin minn'
009-009: S004                 Orðið 'sjónnvarpið' var leiðrétt í 'sjónvarpið' | 'sjónnvarpið' -> 'sjónvarpið'
```

The `annotation.start` and `annotation.end` properties
(here `start` is 0 and `end` is 4) contain the 0-based indices of the first
and last tokens to which the annotation applies. `annotation.code` is the
error code (here `P_WRONG_CASE_þgf_þf` and `S004`), `annotation.text` a
description of the error, `annotation.detail` a longer explanation where
there is one, and `annotation.suggest` the suggested replacement for the
tokens.

The text of the sentence with the token-level corrections applied, but not
the suggestions of the grammar annotations, is in `sent.tidy_text`:

```python
sent.tidy_text
```

Output:

```
'Páli, vini mínum, langaði að horfa á sjónvarpið.'
```

For more detailed, low-level control, the `check_errors()` function
supports options and can produce various types of output:

```python
from reynir_correct import check_errors
x = "Páli, vini mínum, langaði að horfa á sjónnvarpið."
options = { "input": x, "annotations": True, "format": "text" }
s = check_errors(**options)
print(s)
```

Output:

```
Pál, vin minn langaði að horfa á sjónvarpið.
009-009: S004                 Orðið 'sjónnvarpið' var leiðrétt í 'sjónvarpið' | 'sjónnvarpið' -> 'sjónvarpið'
000-004: P_WRONG_CASE_þgf_þf  Á líklega að vera 'Pál, vin minn' | 'Páli, vini mínum,' -> 'Pál, vin minn'
```

The first line is the text with both the token-level corrections and the
suggestions of the annotations applied.

The following options can be specified:

| Option | Description | Default value |
|---|---|---|
| `input` | The text to check: a string or an iterable of strings, such as a file object. | (required) |
| `format` | The format of the output: `text`, `json`, `csv` or `m2` (the [M2 format](https://github.com/nusnlp/m2scorer) of grammatical error correction). | `json` |
| `all_errors` | Defines the level of correction. If False, only token-level annotation is carried out. If True, sentence-level annotation is carried out. | `True` |
| `annotations` | If True, the annotations are listed after the text. Applies to the `text` format. | `False` |
| `generate_suggestion_list` | If True, annotations can in certain cases contain a list of possible corrections, for the user to pick from. | `False` |
| `suppress_suggestions` | If True, more farfetched automatically suggested corrections are suppressed. | `False` |
| `suggest_not_correct` | If True, some stylistic errors are presented as suggestions only, and the text is not changed. | `False` |
| `ignore_wordlist` | The value is a set of strings to whitelist. Each string is a word that should not be marked as an error or corrected. The comparison is case-sensitive. | `set()` |
| `ignore_rules` | A set of error codes that should be ignored in the annotation process. | `set()` |
| `tov_config` | Path to an additional configuration file that may be provided for correcting custom tone-of-voice issues. | `None` |
| `flesch` | If True, a Flesch readability score for the text is added to the output. | `False` |
| `rare_words` | If True, a list of the rare words in the text is added to the output. | `False` |

An overview of error codes is available [here](https://github.com/mideind/GreynirCorrect/blob/master/doc/errorcodes.rst).

To check many texts with the same options, create a `GreynirCorrectAPI` object
once and call its `correct()` method for each text. `from_options()` takes the
options of `check_errors()` that concern how the text is checked;
`ignore_rules` and `suppress_suggestions` are arguments of `correct()`:

```python
from reynir_correct import GreynirCorrectAPI
api = GreynirCorrectAPI.from_options()
result = api.correct("Hún fór heim til sýn.")
for sent in result.sentences:
    print(sent.corrected_str(apply_annotations=True))
    for annotation in sent.annotations:
        print(annotation.code, annotation.text)
```

Output:

```
Hún fór heim til sýnar.
P_NT_FsMeðFallstjórn Á sennilega að vera 'sýnar'
```

## Use in threads

From version 4.3.1, GreynirCorrect can be used from several threads at once.
This applies to the functions (`tokenize()`, `check_single()`, `check()`,
`check_errors()`) and to a `GreynirCorrectAPI` or `GreynirCorrect` object that
the threads share. The results are the same as from a single thread.

```python
from concurrent.futures import ThreadPoolExecutor
from reynir_correct import GreynirCorrectAPI

api = GreynirCorrectAPI.from_options()
texts = [
    "Páli, vini mínum, langaði að horfa á sjónnvarpið.",
    "Hún fór heim til sýn.",
    "Hundurinn beit manninn í gær.",
]
with ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(api.correct, texts):
        for sent in result.sentences:
            print([annotation.code for annotation in sent.annotations])
```

Output:

```
['P_WRONG_CASE_þgf_þf', 'S004']
['P_NT_FsMeðFallstjórn']
[]
```

In version 4.3.0 and older, such an object could not be shared: a call from a
second thread could fail with
`ValueError: DefaultPipeline.tokenize() is not re-entrant`. With those
versions, give each thread an object of its own.

On free-threaded Python (3.14t and newer), the threads run in parallel, and
the GIL stays off when the package is imported. In our measurements, 8 threads
checked news text about 4 times as fast as one, and more threads than that
added little. What holds it back is spelling correction, where the lookups in
the Icegrams trigram model do not yet run in parallel.

With the GIL, threads are safe but not faster than one thread, and many
threads are slower. To use more cores there, use processes.

## Prerequisites

GreynirCorrect runs on CPython 3.11 or newer, including free-threaded CPython
(3.14t and newer), and on PyPy 3.11 or newer. It is tested on Linux, macOS and
Windows.

GreynirCorrect itself is pure Python. Two of the packages that it depends on,
[GreynirEngine](https://pypi.org/project/reynir/) and
[Icegrams](https://pypi.org/project/icegrams/), contain compiled code and
have binary wheels on PyPI for common environments. Where there is no wheel
for your environment, they are built from source when installed, which needs
a C++ compiler and the Python development headers, for instance:

```bash
sudo apt-get install build-essential python3-dev
```

## Installation

To install this package (assuming you have Python >= 3.11 with `pip` installed):

```bash
pip install reynir-correct
```

GreynirCorrect uses the trigram model of the
[Icegrams](https://pypi.org/project/icegrams/) package for spelling
correction. Icegrams 2.2 and newer gets the model from the `icegrams-data`
package, which is installed along with it, so there is nothing more to do.

With Icegrams 2.0 or 2.1, the model is not installed with the package, and
must be downloaded once:

```bash
python -m icegrams.download
```

If the model is missing, GreynirCorrect raises `ModelNotFoundError` on first
use, with these same instructions. See the Icegrams documentation for how to
keep the model elsewhere via the `ICEGRAMS_MODEL_FILE` environment variable.

If you want to be able to edit the source, do like so
(assuming you have `git` installed):

```bash
git clone https://github.com/mideind/GreynirCorrect
cd GreynirCorrect
# [ Activate your virtualenv here if you have one ]
pip install -e ".[dev]"
```

The package source code is now in `GreynirCorrect/src/reynir_correct`.
See [Development and tests](#development-and-tests) below for a setup with
[uv](https://docs.astral.sh/uv/).

## The command line tool

After installation, the corrector can be invoked directly from the command line:

```bash
correct input.txt output.txt
```

...or:

```bash
$ echo "Þinngið samþikkti tilöguna" | correct
Þingið samþykkti tillöguna
```

Input and output files are encoded in UTF-8. If the files are not
given explicitly, `stdin` and `stdout` are used for input and output,
respectively.

Empty lines in the input are treated as sentence boundaries.

By default, the output is the corrected text, one sentence per line, with
quotation marks and other punctuation in their Icelandic form.

The following options can be specified on the command line:

| Option | Description |
|---|---|
| `--format FORMAT`, `-f` | The format of the output: `text` (the default), `csv`, `json` or `m2`. |
| `--csv`, `-c` | The same as `--format csv`: token objects in CSV format, one per line. |
| `--json`, `-j` | The same as `--format json`: token objects in JSON format, one per line. |
| `--grammar`, `-g` | Check grammar as well as spelling, and output whole-sentence annotations, including corrections and suggestions. Each sentence in the input is output as a text line containing a JSON object, terminated by a newline. |
| `--all_errors`, `-a` | Check grammar as well as spelling, and keep the chosen output format. |
| `--spaced`, `-sp` | Separate the tokens of the output with spaces, and leave punctuation as it is in the input. |
| `--normalize`, `-n` | With `--spaced`: normalize punctuation, causing e.g. quotes to be output in Icelandic form. |
| `--suppress_suggestions`, `-ss` | Suppress the more farfetched suggestions. |
| `--suggest_not_correct` | Present some stylistic errors as suggestions only, without changing the text. |
| `--flesch` | Add a Flesch readability score for the text to the output. |
| `--rare_words` | Add a list of the rare words in the text to the output. |
| `--tov_config FILE` | An additional configuration file with rules for custom tone-of-voice issues, in the format of `GreynirCorrect.conf`. |
| `--sentence_prefilter` | Run a classifier on each sentence first, and only parse those that probably have an error. This needs the `sentence_classifier` extra (`pip install "reynir-correct[sentence_classifier]"`), which is not available for PyPy. |

In the CSV and JSON formats of token objects, each token has its kind and
text as documented for the [Tokenizer package](https://github.com/mideind/Tokenizer),
followed by the error that was found in it, if any.

The JSON format of whole-sentence annotations is shown in the last of the
examples below.

Type `correct -h` to get a short help message.

### Command Line Examples

```bash
$ echo "Atvinuleysi jógst um 3%" | correct
Atvinnuleysi jókst um 3%
```

```bash
$ echo "Barnið vil grænann lit" | correct --csv
6,"Barnið","",""
6,"vil","",""
6,"grænan","","S004: Orðið 'grænann' var leiðrétt í 'grænan' | grænann->grænan"
6,"lit","",""
```

Note how *vil* is not corrected, as it is a valid and common word, and
the `correct` command does not perform grammar checking by default.

```bash
$ echo "Pakkin er fyrir hestin" | correct --json
{"k":"WORD","t":"Pakkinn","e":{"code":"S004","descr":"Orðið 'Pakkin' var leiðrétt í 'Pakkinn'"}}
{"k":"WORD","t":"er"}
{"k":"WORD","t":"fyrir"}
{"k":"WORD","t":"hestinn","e":{"code":"S004","descr":"Orðið 'hestin' var leiðrétt í 'hestinn'"}}
```

To perform whole-sentence grammar checking and annotation as well as spell checking,
use the `--grammar` option:

```bash
$ echo "Ég kláraði verkefnið þrátt fyrir að ég var þreittur." | correct --grammar
{
   "original":"Ég kláraði verkefnið þrátt fyrir að ég var þreittur.",
   "corrected":"Ég kláraði verkefnið þrátt fyrir að ég væri þreyttur.",
   "tokens":[
      {"k":6,"x":"Ég","o":"Ég","i":0},
      {"k":6,"x":"kláraði","o":" kláraði","i":2},
      {"k":6,"x":"verkefnið","o":" verkefnið","i":10},
      {"k":6,"x":"þrátt fyrir","o":" þrátt fyrir","i":20},
      {"k":6,"x":"að","o":" að","i":32},
      {"k":6,"x":"ég","o":" ég","i":35},
      {"k":6,"x":"væri","o":" var","i":38},
      {"k":6,"x":"þreyttur","o":" þreittur","i":42},
      {"k":1,"x":".","o":".","i":51}
   ],
   "annotations":[
      {
         "start":6,
         "end":6,
         "start_char":38,
         "end_char":41,
         "code":"P_MOOD_ACK",
         "text":"Hér er réttara að nota viðtengingarhátt sagnarinnar 'vera', þ.e. 'væri'.",
         "detail":"Í viðurkenningarsetningum á borð við 'Z' í dæminu 'X gerði Y þrátt fyrir að Z' á sögnin að vera í viðtengingarhætti fremur en framsöguhætti.",
         "suggest":"væri"
      },
      {
         "start":7,
         "end":7,
         "start_char":42,
         "end_char":50,
         "code":"EI4EY",
         "text":"Stafsetningarvilla: 'þreittur' -> 'þreyttur'",
         "detail":"Rita ætti 'ey' í stað 'ei' í 'þreittur', svo: 'þreyttur'.",
         "suggest":"þreyttur"
      }
   ]
}
```

The output has been formatted for legibility - each input sentence is actually
represented by a JSON object in a single line of text, terminated by newline.

In each token, `x` is the corrected text, `o` the original text with the
whitespace in front of it, and `i` the index in the input where `o` starts.
In each annotation, `start` and `end` are the indices of the first and last
tokens that it applies to, and `start_char` and `end_char` the indices in the
input of the first and last characters of those tokens' original text.

The `corrected` field has the sentence with the token-level corrections and
the suggestions of the annotations applied (in this case *þreittur* `->`
*þreyttur* and *var* `->` *væri*). To apply only some of them, replace
source text or tokens (as identified by the `start` and `end`,
or `start_char` and `end_char` properties) with the `suggest` field, if present.

## Development and tests

The repository has a `uv.lock` file, so [uv](https://docs.astral.sh/uv/) sets
up an environment with the versions of the dependencies that the tests are
run with on GitHub. In your `GreynirCorrect` directory:

```bash
uv sync
uv run pytest
uv run ruff check src/reynir_correct
uv run mypy src/reynir_correct
```

Without uv, install the package with its development dependencies
(`pip install -e ".[dev]"`, optionally in a virtualenv) and run
`python -m pytest`.

On free-threaded Python, the tests are also run with each test in 8 threads
at once:

```bash
uv run --python 3.14t pytest --parallel-threads=8
```

The tests run on GitHub for every pull request: on CPython 3.11 to 3.15,
free-threaded CPython 3.14t and 3.15t and PyPy 3.11 on Linux, and on
CPython 3.13 on Windows and macOS. A release is made by pushing a version
tag: the wheel is then built, checked on Linux, macOS and Windows, and
published to PyPI.

## Acknowledgements

Parts of this software are developed under the auspices of the
Icelandic Government's 5-year Language Technology Programme for Icelandic,
which is managed by Almannarómur and described
[here](https://www.stjornarradid.is/lisalib/getfile.aspx?itemid=56f6368e-54f0-11e7-941a-005056bc530c)
(English version [here](https://clarin.is/media/uploads/mlt-en.pdf)).

## Copyright and License

[![Miðeind ehf.](https://github.com/mideind/GreynirPackage/raw/master/doc/_static/MideindLogoVert100.png?raw=true)](https://mideind.is)

**Copyright © 2018-2026 Miðeind ehf.**

GreynirCorrect's original author is *Vilhjálmur Þorsteinsson*.

This software is licensed under the *MIT License*:

   *Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without restriction,
   including without limitation the rights to use, copy, modify, merge,
   publish, distribute, sublicense, and/or sell copies of the Software,
   and to permit persons to whom the Software is furnished to do so,
   subject to the following conditions:*

   *The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.*

   *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
   CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
   SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.*

----

GreynirCorrect indirectly embeds the [Database of Icelandic Morphology](https://bin.arnastofnun.is)
([Beygingarlýsing íslensks nútímamáls](https://bin.arnastofnun.is))
along with directly using
[Ritmyndir](https://bin.arnastofnun.is/DMII/LTdata/comp-format/nonstand-form/),
a collection of non-standard word forms.
Miðeind does not claim any endorsement by the BÍN authors or copyright holders.

The BÍN source data are publicly available under the
[CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/), as further
detailed [here in English](https://bin.arnastofnun.is/DMII/LTdata/conditions/)
and [here in Icelandic](https://bin.arnastofnun.is/gogn/mimisbrunnur/).

In accordance with the BÍN license terms, credit is hereby given as follows:

*Beygingarlýsing íslensks nútímamáls. Stofnun Árna Magnússonar í íslenskum fræðum.*
*Höfundur og ritstjóri Kristín Bjarnadóttir.*
