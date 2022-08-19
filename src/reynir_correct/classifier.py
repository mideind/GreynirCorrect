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

from typing import Union, List, overload


try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TextClassificationPipeline
except:
    import sys
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
    _model_name = "mideind/error-correction-sentence-classifier"
    _true_label = "LABEL_1"

    def __init__(self):
        # TODO: Make model public and remove auth
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_auth_token=True)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name, use_auth_token=True)
        self.pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    @overload
    def classify(self, text: str) -> bool:
        ...

    @overload
    def classify(self, text: List[str]) -> List[bool]:
        ...

    def classify(self, text):
        """Classify a sentence or sentences.
        For each sentence, return true iff the sentence probably contains an error."""
        result = self.pipe(text)

        result = [r["label"] == self._true_label for r in result]
        if len(result) == 1:
            result = result[0]

        return result


def _main() -> None:
    classifier = SentenceClassifier()

    many_sents = ["Þesi settníng er ekki rét sfsett.", "Þessi setning er rétt stafsett."]
    many_results = classifier.classify(many_sents)
    one_sent = "Þesi líga."
    one_result = classifier.classify(one_sent)

    print(f"Sentence: {many_sents[0]}")
    print(f"Result:   {many_results[0]}")

    print(f"Sentence: {many_sents[1]}")
    print(f"Result:   {many_results[1]}")

    print(f"Sentence: {one_sent}")
    print(f"Result:   {one_result}")


if __name__ == "__main__":
    _main()
