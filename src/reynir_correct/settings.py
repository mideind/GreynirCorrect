"""
    Reynir: Natural language processing for Icelandic

    Settings module

    Copyright (c) 2018 Miðeind ehf.

       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.


    This module reads and interprets the ReynirCorrect.conf
    configuration file. The file can include other files using the $include
    directive, making it easier to arrange configuration sections into logical
    and manageable pieces.

    Sections are identified like so: [ section_name ]

    Comments start with # signs.

    Sections are interpreted by section handlers.

"""

import os
import codecs
import locale
import threading

from contextlib import contextmanager, closing
from collections import defaultdict
from threading import Lock

from pkg_resources import resource_stream


# The sorting locale used by default in the changedlocale function
_DEFAULT_SORT_LOCALE = ("IS_is", "UTF-8")
# A set of all valid argument cases
_ALL_CASES = frozenset(("nf", "þf", "þgf", "ef"))
_ALL_GENDERS = frozenset(("kk", "kvk", "hk"))

ALLOWED_MULTIPLES = set()
NOT_COMPOUNDS = dict()
SPLIT_COMPOUNDS = dict()
UNIQUE_ERRORS = dict()
MW_ERRORS_SEARCH = dict()
MW_ERRORS = dict()

class ConfigError(Exception):

    """ Exception class for configuration errors """

    def __init__(self, s):
        super().__init__(s)
        self.fname = None
        self.line = 0

    def set_pos(self, fname, line):
        """ Set file name and line information, if not already set """
        if not self.fname:
            self.fname = fname
            self.line = line

    def __str__(self):
        """ Return a string representation of this exception """
        s = Exception.__str__(self)
        if not self.fname:
            return s
        return "File {0}, line {1}: {2}".format(self.fname, self.line, s)


class LineReader:
    """ Read lines from a text file, recognizing $include directives """

    def __init__(self, fname, outer_fname=None, outer_line=0):
        self._fname = fname
        self._line = 0
        self._inner_rdr = None
        self._outer_fname = outer_fname
        self._outer_line = outer_line

    def fname(self):
        return self._fname if self._inner_rdr is None else self._inner_rdr.fname()

    def line(self):
        return self._line if self._inner_rdr is None else self._inner_rdr.line()

    def lines(self):
        """ Generator yielding lines from a text file """
        self._line = 0
        try:
            with resource_stream(__name__, self._fname) as inp:
                # Read config file line-by-line
                for b in inp:
                    # We get byte strings; convert from utf-8 to strings
                    s = b.decode("utf-8")
                    self._line += 1
                    # Check for include directive: $include filename.txt
                    if s.startswith("$") and s.lower().startswith("$include "):
                        iname = s.split(maxsplit=1)[1].strip()
                        # Do some path magic to allow the included path
                        # to be relative to the current file path, or a
                        # fresh (absolute) path by itself
                        head, _ = os.path.split(self._fname)
                        iname = os.path.join(head, iname)
                        rdr = self._inner_rdr = LineReader(
                            iname, self._fname, self._line
                        )
                        for incl_s in rdr.lines():
                            yield incl_s
                        self._inner_rdr = None
                    else:
                        yield s
        except (IOError, OSError):
            if self._outer_fname:
                # This is an include file within an outer config file
                c = ConfigError(
                    "Error while opening or reading include file '{0}'"
                    .format(self._fname)
                )
                c.set_pos(self._outer_fname, self._outer_line)
            else:
                # This is an outermost config file
                c = ConfigError(
                    "Error while opening or reading config file '{0}'"
                    .format(self._fname)
                )
            raise c


# Magic stuff to change locale context temporarily

@contextmanager
def changedlocale(new_locale=None):
    """ Change locale for collation temporarily within a context (with-statement) """
    # The newone locale parameter should be a tuple: ('is_IS', 'UTF-8')
    old_locale = locale.getlocale(locale.LC_COLLATE)
    try:
        locale.setlocale(locale.LC_COLLATE, new_locale or _DEFAULT_SORT_LOCALE)
        yield locale.strxfrm  # Function to transform string for sorting
    finally:
        locale.setlocale(locale.LC_COLLATE, old_locale)


def sort_strings(strings, loc=None):
    """ Sort a list of strings using the specified locale's collation order """
    # Change locale temporarily for the sort
    with changedlocale(loc) as strxfrm:
        return sorted(strings, key=strxfrm)

class Settings:

    """ Global settings"""

    _lock = threading.Lock()
    loaded = False

    # Configuration settings from the ReynirCorrect.conf file

    @staticmethod
    def _handle_settings(s):
        """ Handle config parameters in the settings section """
        a = s.lower().split("=", maxsplit=1)
        par = a[0].strip().lower()
        val = a[1].strip()
        if val.lower() == "none":
            val = None
        elif val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        try:
            if par == "debug":
                Settings.DEBUG = bool(val)
            else:
                raise ConfigError("Unknown configuration parameter '{0}'".format(par))
        except ValueError:
            raise ConfigError("Invalid parameter value: {0} = {1}".format(par, val))

    @staticmethod
    def _handle_allowed_multiples(s):
        """ Handle config parameters in the allowed_multiples section """
        assert s
        if len(s.split()) != 1:
            raise ConfigError("Only one word per line allowed in allowed_multiples section")
        if s in ALLOWED_MULTIPLES:
            raise ConfigError("'{0}' is repeated in allowed_multiples section".format(s))
        ALLOWED_MULTIPLES.add(s)

    @staticmethod
    def _handle_not_compounds(s):
        """ Handle config parameters in the not_compounds section """
        a = s.lower().split(":", maxsplit=1)
        word = a[0].strip()
        parts = a[1].strip().split()
        if not word:
            raise ConfigError("Expected word before the colon in not_compounds section")
        if len(parts) < 2:
            raise ConfigError("Missing word part(s) in not_compounds section")
        if len(word.split()) != 1:
            raise ConfigError("Multiple words not allowed before colon in not_compounds section")
        if word in NOT_COMPOUNDS:
            raise ConfigError("Multiple definition of '{0}' in not_compounds section".format(word))
        NOT_COMPOUNDS[word] = tuple(parts)

    @staticmethod
    def _handle_split_compounds(s):
        """ Handle config parameters in the split_compounds section """
        a = s.lower().split(":", maxsplit=1)
        parts = tuple(a[0].strip().split())
        word = a[1].strip()
        if not word:
            raise ConfigError("Expected word after the colon in split_compounds section")
        if len(parts) < 2:
            raise ConfigError("Missing word part(s) in split_compounds section")
        if len(word.split()) != 1:
            raise ConfigError("Multiple words not allowed after colon in split_compounds section")
        if parts in SPLIT_COMPOUNDS:
            raise ConfigError(
                "Multiple definition of '{0}' in split_compounds section"
                .format(" ".join(parts))
            )
        SPLIT_COMPOUNDS[parts] = word

    @staticmethod
    def _handle_unique_errors(s):
        """ Handle config parameters in the unique_errors section """
        a = s.lower().split(":", maxsplit=1)
        word = a[0].strip()
        corr = a[1].strip().split()
        if not word:
            raise ConfigError("Expected word before the colon in unique_errors section")
        if len(word.split()) != 1:
            raise ConfigError("Multiple words not allowed before colon in unique_errors section")
        if word in NOT_COMPOUNDS:
            raise ConfigError("Multiple definition of '{0}' in unique_errors section".format(word))
        UNIQUE_ERRORS[word] = corr

    @staticmethod
    def _handle_multiword_errors(s):
        """ Handle config parameters in the multiword_errors section """
        a = s.lower().split(":", maxsplit=1)
        sp = a[0].strip().split()
        info = a[1].strip().split()
        MW_ERRORS_SEARCH[sp[0]] = sp[1:] # Ath. þarf að geyma alla möguleika.
        MW_ERRORS[" ".join(sp)] = info

    @staticmethod
    def read(fname):
        """ Read configuration file """

        with Settings._lock:

            if Settings.loaded:
                return

            CONFIG_HANDLERS = {
                "settings": Settings._handle_settings,
                "allowed_multiples": Settings._handle_allowed_multiples,
                "not_compounds": Settings._handle_not_compounds,
                "split_compounds": Settings._handle_split_compounds,
                "unique_errors": Settings._handle_unique_errors,
                "multiword_errors": Settings._handle_multiword_errors,
            }
            handler = None  # Current section handler

            rdr = None
            try:
                rdr = LineReader(fname)
                for s in rdr.lines():
                    # Ignore comments
                    ix = s.find("#")
                    if ix >= 0:
                        s = s[0:ix]
                    s = s.strip()
                    if not s:
                        # Blank line: ignore
                        continue
                    if s[0] == "[" and s[-1] == "]":
                        # New section
                        section = s[1:-1].strip().lower()
                        if section in CONFIG_HANDLERS:
                            handler = CONFIG_HANDLERS[section]
                            continue
                        raise ConfigError("Unknown section name '{0}'".format(section))
                    if handler is None:
                        raise ConfigError("No handler for config line '{0}'".format(s))
                    # Call the correct handler depending on the section
                    try:
                        handler(s)
                    except ConfigError as e:
                        # Add file name and line number information to the exception
                        # if it's not already there
                        e.set_pos(rdr.fname(), rdr.line())
                        raise e

            except ConfigError as e:
                # Add file name and line number information to the exception
                # if it's not already there
                if rdr:
                    e.set_pos(rdr.fname(), rdr.line())
                raise e

            Settings.loaded = True
