"""

    Greynir: Natural language processing for Icelandic

    Sentence classifier module

    Copyright (C) 2022 Miðeind ehf.

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


    This module contains a sentence classifier that classifies sentences
    based on whether they probably contain a grammatical error or not.
    The classifier uses a neural network for the classification and does not
    require a full sentence parse by Greynir.

    The goal is that the classifier runs much faster than the full parse and
    can therefore be used as a filter to skip parsing sentences that are
    probably correct.

    Note: The classifier has a nonzero error rate. Depending on your use case
    this may be acceptable or not.

"""

from typing import List, Union, overload


try:
    from datasets import load_dataset
    from transformers import pipeline  # type: ignore
except:
    import warnings

    warningtext = (
        "Tried to import the classifier module without the required packages installed.\n"
        "The required packages are in the 'sentence_classifier' extra\n"
        "Run 'pip install reynir-correct[sentence_classifier] to install them\n"
        "\n"
        "Alternatively, install the packages directly with\n"
        "'pip install datasets transformers torch' or similar.\n"
    )
    warnings.warn(warningtext)
    raise ImportError(warningtext)


class SentenceClassifier:
    _model_name = "mideind/yfirlestur-icelandic-classification-byt5"
    _true_label = "1"
    _domain_prefix = "has_error "

    def __init__(self) -> None:
        self.pipe = pipeline(
            "text2text-generation", model=self._model_name, tokenizer="google/byt5-base"
        )

    @overload
    def classify(self, text: str) -> bool:
        ...

    @overload
    def classify(self, text: List[str]) -> List[bool]:
        ...

    def classify(self, text: Union[str, List[str]]) -> Union[List[bool], bool]:
        """Classify a sentence or sentences.
        For each sentence, return true iff the sentence probably contains an error."""
        if isinstance(text, str):
            text = [text]

        pipe_result = self.pipe([self._domain_prefix + t for t in text])
        result: List[bool] = [
            r["generated_text"] == self._true_label for r in pipe_result
        ]

        return result[0] if len(result) == 1 else result


def _main() -> None:
    classifier = SentenceClassifier()

    many_sents = [
        "Þesi settníng er ekki rét sfsett.",
        "Þessi setning er rétt stafsett.",
    ]
    many_results = classifier.classify(many_sents)
    one_sent = "Þesi er vavasöm."
    one_result = classifier.classify(one_sent)

    print(f"Sentence: {many_sents[0]}")
    print(f"Result:   {many_results[0]}")

    print(f"Sentence: {many_sents[1]}")
    print(f"Result:   {many_results[1]}")

    print(f"Sentence: {one_sent}")
    print(f"Result:   {one_result}")


if __name__ == "__main__":
    _main()
