"""

    test_correct.py

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

"""

import reynir_correct as rc


def test_correct():
    """ Test the spelling and grammar correction module """

    # g = rc.tokenize("Kexið er gott báðumegin, sagði sagði Cthulhu og rak sig uppundir þakið. Það var aftaka veður í gær.")
    g = rc.tokenize("Jeg stögg ifir steyninn aff þvý að það var aldrey gamann")

    for token in g:
        print("{0}".format(token))
        err = token.error_description
        if err:
            print("   {0}".format(err))


if __name__ == "__main__":

    test_correct()
