"""

    Greynir: Natural language processing for Icelandic

    Annotation class definition

    Copyright (C) 2020 Miðeind ehf.

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


class Annotation:

    """ An annotation of a span of a token list for a sentence """

    def __init__(
        self, *, start, end, code, text, detail=None, suggest=None, is_warning=False
    ):
        assert isinstance(start, int)
        assert isinstance(end, int)
        self._start = start
        self._end = end
        if is_warning and not code.endswith("/w"):
            code += "/w"
        self._code = code
        # text is a short, straight-to-the-point human-readable description
        # of the error
        self._text = text
        # detail is a more detailed human-readable description of the error,
        # containing further explanations, eventually using grammatical terms,
        # and possibly links to further reference material (within <a>...</a> tags)
        self._detail = detail
        # If suggest is given, it is a suggested correction,
        # i.e. text that would replace the start..end token span.
        # The correction is in the form of token text joined by
        # " " spaces, so correct_spaces() should be applied to
        # it before displaying it.
        self._suggest = suggest

    def __str__(self):
        """ Return a string representation of this annotation """
        return "{0:03}-{1:03}: {2:6} {3}{4}".format(
            self._start, self._end, self._code, self._text,
            "" if self._suggest is None else " / [" + self._suggest + "]"
        )

    @property
    def start(self):
        """ The index of the first token to which the annotation applies """
        return self._start

    @property
    def end(self):
        """ The index of the last token to which the annotation applies """
        return self._end

    @property
    def code(self):
        """ A code for the annotation type, usually an error or warning code """
        # If the code ends with "/w", it is a warning
        return self._code

    @property
    def text(self):
        """ A description of the annotation """
        return self._text

    @property
    def detail(self):
        """ A detailed description of the annotation, possibly including
            links within <a>...</a> tags """
        return self._detail

    @property
    def suggest(self):
        """ A suggested correction for the token span, as a text string
            containing tokens delimited by spaces """
        return self._suggest
