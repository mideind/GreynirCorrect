"""

    GreynirCorrect: Spelling and grammar correction for Icelandic

    Readability module

    Copyright (C) 2023 Miðeind ehf.

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


    This module implements the Flesch reading ease score for Icelandic text.
    A high score indicates that the text is easy to read, while a low score
    indicates that the text is difficult to read.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple

import math
import re
from enum import Enum

import tokenizer
from icegrams.ngrams import Ngrams
from islenska import Bin

diphtong_pattern = re.compile(r"(ei|ey|au)")
vowel_pattern = re.compile(r"[aeiouyáéíóúýöæ]")


class FleschKincaidFeedback(Enum):
    """The Flesch-Kincaid feedback of a text."""

    VERY_EASY_OR_INVALID = (120, "Mjög léttur eða ómarktækur texti.")
    VERY_EASY = (90, "Mjög léttur texti")
    EASY = (80, "Léttur texti")
    FAIRLY_EASY = (70, "Frekar léttur texti")
    STANDARD = (60, "Meðalléttur texti")
    FAIRLY_DIFFICULT = (50, "Svolítið þungur texti")
    DIFFICULT = (30, "Þungur texti")
    VERY_DIFFICULT = (0, "Mjög þungur texti")
    VERY_DIFFICULT_OR_INVALID = (-math.inf, "Mjög þungur eða ómarktækur texti.")

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __init__(self, score: float, feedback: str):
        self.score = score
        self.feedback = feedback

    def __str__(self):
        return self.feedback

    @staticmethod
    def from_score(score: float) -> FleschKincaidFeedback:
        """Provide feedback on the score. This is done by comparing the score to the
        Flesch-Kincaid reading ease score scale. The feedback has been slightly modified to fit Icelandic.
        Icelandic tends to score lower than English and thus the scale has been adjusted to reflect this.
        """
        for feedback in FleschKincaidFeedback:
            if score >= feedback.score:
                return feedback
        raise ValueError(f"Invalid score: {score}")


class FleschKincaidScorer:
    """Calculate the Flesch-Kincaid reading ease score of a given text."""

    @staticmethod
    def count_syllables_in_word(word: str) -> int:
        """Count the number of syllables in an Icelandic word, by counting the number of vowels in the word.
        This is done by first replacing diphtongs with a single character, then counting the number of vowels."""
        word = word.lower()
        # replace diphtongs with a single character
        word = diphtong_pattern.sub("a", word)
        return len(vowel_pattern.findall(word))

    @staticmethod
    def is_start_of_sentence(tok: tokenizer.Tok) -> bool:
        """Check if a token is the start of a sentence."""
        if tok.kind == tokenizer.TOK.S_BEGIN:
            return True
        return False

    @staticmethod
    def is_a_word(tok: tokenizer.Tok) -> bool:
        """Determine if a token is a word for the Flesch-Kincaid metric."""
        if tok.kind < tokenizer.TOK.META_BEGIN:
            # Punctuation marks are not counted as words
            if tok.kind > tokenizer.TOK.PUNCTUATION:
                return True
        return False

    @staticmethod
    def get_score(num_sentences: int, num_words: int, num_syllables: int) -> float:
        """Calculate the Flesch-Kincaid reading ease score after tracking a token stream.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        # Magic numbers from the Flesch-Kincaid reading ease score formula - English
        # Icelandic tends to score lower than English and thus the scale has been adjusted to reflect this.
        # Original formula: 206.835 - (1.015 * (num_words / num_sentences)) - 84.6 * (num_syllables / num_words)
        score = 206.835 - (1.015 * (num_words / num_sentences)) - 70 * (num_syllables / num_words)
        return score

    @staticmethod
    def get_counts_from_stream(token_stream: Iterable[tokenizer.Tok]) -> Tuple[int, int, int]:
        num_words = 0
        num_sentences = 0
        num_syllables = 0
        for tok in token_stream:
            if FleschKincaidScorer.is_start_of_sentence(tok):
                num_sentences += 1
            elif FleschKincaidScorer.is_a_word(tok):
                num_words += 1
                if tok.kind == tokenizer.TOK.WORD:
                    num_syllables += FleschKincaidScorer.count_syllables_in_word(tok.txt)
                else:
                    # All tokens that are numbers, abbreviations, etc. are counted as one syllable, as an approximation
                    num_syllables += 1
            else:
                # All other tokens are ignored
                pass
        return num_sentences, num_words, num_syllables

    @staticmethod
    def get_score_from_stream(token_stream: Iterable[tokenizer.Tok]) -> float:
        """Calculate the Flesch-Kincaid reading ease score after tracking a token stream.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        num_sentences, num_words, num_syllables = FleschKincaidScorer.get_counts_from_stream(token_stream)
        try:
            score = FleschKincaidScorer.get_score(num_sentences, num_words, num_syllables)
        except ZeroDivisionError:
            score = -1.0  # invalid score
        return score

    @staticmethod
    def get_score_from_text(text: str) -> float:
        """Calculate the Flesch-Kincaid reading ease score of a text.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        stream = tokenizer.tokenize(text)
        return FleschKincaidScorer.get_score_from_stream(stream)

    @staticmethod
    def get_feedback(score: float) -> FleschKincaidFeedback:
        """Provide feedback on the score of the Result object. This is done by comparing the score to the
        Flesch-Kincaid reading ease score scale. The feedback has been slightly modified to fit Icelandic.
        Icelandic tends to score lower than English and thus the scale has been adjusted to reflect this.
        """
        return FleschKincaidFeedback.from_score(score)


class RareWordsFinder:
    """Find rare words in a text.

    Rare words are defined as words which have a probability lower than the low_prob_cutoff.
    The probability of a word is calculated by looking up the word in an n-gram model.
    The class is designed to be used with the tokenizer module and maintains an internal state which needs to be reset manually.
    """

    def __init__(self):
        self.bin = Bin()
        self.ng = Ngrams()

    def get_rare_words_from_stream(
        self, tok_stream: Iterable[tokenizer.Tok], max_words: int, low_prob_cutoff: float
    ) -> List[Tuple[str, float]]:
        """Tracks the probability of each word in a token stream. This is done by yielding the tokens in the token stream."""
        rare_words_dict: Dict[str, float] = {}
        for token in tok_stream:
            # Only consider words, not punctuation, numbers, etc.
            if token.kind == tokenizer.TOK.WORD:
                # unigram probability for the word
                prob = self.ng.prob(token.txt)
                if prob < low_prob_cutoff:
                    lemma_set = self.bin.lookup_lemmas_and_cats(token.txt)
                    lemma = lemma_set.pop()[0] if lemma_set else token.txt
                    # sometimes there are hyphens in the lemma
                    if "-" in lemma and "-" not in token.txt:
                        # remove the hyphen from the lemma
                        lemma = lemma.replace("-", "")
                    rare_words_dict[lemma] = prob
        rare_words = sorted(rare_words_dict.items(), key=lambda x: x[1], reverse=False)[:max_words]
        return rare_words

    def get_rare_words_from_text(
        self,
        text: str,
        max_words: int = 10,
        low_prob_cutoff: float = 0.00000005,
    ) -> List[Tuple[str, float]]:
        """Get the rare words from a text.
        This function is a convenience function which does not require the user to track the token stream.

        Rare words are defined as words which have a probability lower than the low_prob_cutoff.
        The probability of a word is calculated by looking up the word in the n-gram model.
        This function is a convenience function which does not require the user to track the token stream.
        """
        stream = tokenizer.tokenize(text)
        return self.get_rare_words_from_stream(stream, max_words, low_prob_cutoff)


if __name__ == "__main__":
    while True:
        # read the input from the user
        text = input("Sláðu inn texta: ").strip()
        max_words = 10
        low_prob_cutoff = 0.00000005

        flesch_score = FleschKincaidScorer.get_score_from_text(text)
        rare = RareWordsFinder()
        rare_words = rare.get_rare_words_from_text(text, max_words, low_prob_cutoff)

        print(f"Flesch-læsileikastig: {flesch_score:.2f}")
        print(f"Flesch-umsögn: {FleschKincaidScorer.get_feedback(flesch_score)}")
        print("Sjaldgæfustu orð í textanum:")
        for word, _ in rare_words:
            print(f"\t{word}")
        print()
