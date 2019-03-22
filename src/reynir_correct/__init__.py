"""

    Reynir: Natural language processing for Icelandic

    Copyright(C) 2019 Miðeind ehf.

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

"""

# Expose the reynir-correct API

from reynir import Reynir, correct_spaces, mark_paragraphs
from .settings import Settings
from .errtokenizer import CorrectionPipeline, tokenize, Correct_TOK
from .checker import (
    ReynirCorrect,
    check,
    check_single,
    check_with_stats,
    check_with_custom_parser
)

__author__ = u"Miðeind ehf"

Settings.read("config/ReynirCorrect.conf")

