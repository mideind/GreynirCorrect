# type: ignore
"""

    test_serializers.py

    Tests for JSON serialization of sentences

    Copyright (C) 2021 by Miðeind ehf.

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

import json

import reynir_correct as rc


def test_serializers():
    sents = [
        "Ég fór niðrá bryggjuna með með Reyni Vilhjálmssyni í gær.",
        "Það var 17. júní árið 2020 í frakklandi.",
        "Við sáum tvo seli og öruglega fleiri en 100 máva.",
        "Klukkan var orðinn tólf þegar við fórum heim.",
        "Bíllinn kostaði €30.000 en ég greyddi 25500 USD fyrir hann.",
        "morguninn eftir vakknaði ég kl. 07:30.",
        "Ég var firstur á fætur en þuríður Hálfdánardóttir var numer 2.",
    ]
    gc = rc.GreynirCorrect()
    job = gc.submit(sents, parse=True)
    for pg in job.paragraphs():
        for sent in pg:
            assert sent.tree is not None

            json_str = gc.dumps_single(sent, indent=2)
            new = gc.loads_single(json_str)

            assert new.tree is not None

            assert sent.tokens == new.tokens
            assert sent.terminals == new.terminals
            assert sent.tree.flat_with_all_variants == new.tree.flat_with_all_variants

            cls = gc.__class__
            assert json.loads(sent.dumps(cls, indent=2)) == json.loads(new.dumps(cls, indent=2))


if __name__ == "__main__":
    # When invoked as a main module, do a verbose test
    test_serializers()
