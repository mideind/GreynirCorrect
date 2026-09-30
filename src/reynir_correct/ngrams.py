"""

    GreynirCorrect: Spelling and grammar correction for Icelandic

    Trigram model loader

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


    This module is the single place where GreynirCorrect loads the
    Icegrams trigram model, which is used for spelling correction and
    for finding rare words.

    Since Icegrams 2.0, the model is not bundled with the icegrams
    package. It must be downloaded once, after installation, by running

        python -m icegrams.download

    Icegrams never downloads the model on its own; neither does
    GreynirCorrect. If the model is missing, load_ngrams() raises
    ModelNotFoundError (a subclass of FileNotFoundError) with a message
    explaining how to obtain it.

"""

from icegrams.model import ModelNotFoundError
from icegrams.ngrams import MAX_ORDER, Ngrams

__all__ = ("Ngrams", "MAX_ORDER", "ModelNotFoundError", "load_ngrams")

MODEL_DOWNLOAD_COMMAND = "python -m icegrams.download"


def load_ngrams() -> Ngrams:
    """Load the Icegrams trigram model. Raises ModelNotFoundError,
    with instructions for the user, if the model has not been downloaded."""
    try:
        return Ngrams()
    except ModelNotFoundError as e:
        raise ModelNotFoundError(
            "GreynirCorrect requires the Icegrams trigram model, "
            "which is downloaded separately from the icegrams package.\n" + str(e)
        ) from e
