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
import re
from typing import Iterator, List, Optional, Tuple

import tokenizer
from icegrams.ngrams import Ngrams
from islenska import Bin

diphtong_pattern = re.compile(r"(ei|ey|au)")
vowel_pattern = re.compile(r"[aeiouyáéíóúýöæ]")


class Text:
    """The Text class is used as a namespace to store text functions which count syllables, words and sentences."""

    @staticmethod
    def _count_syllables_in_word(word: str) -> int:
        """Count the number of syllables in an Icelandic word, by counting the number of vowels in the word. This is done by first replacing diphtongs with a single character, then counting the number of vowels."""
        word = word.lower()
        # replace diphtongs with a single character
        word = diphtong_pattern.sub("a", word)
        return len(vowel_pattern.findall(word))

    @staticmethod
    def _count_syllables_in_english_word(word: str) -> int:
        """For comparing purposes. Regex to count the number of syllables in an English word. Does not take all cases into account, such as apple, which has two syllables but is counted as one."""
        return len(
            re.findall("(?!e$)[aeiouy]+", word, re.IGNORECASE) + re.findall("^[^aeiouy]*e$", word, re.IGNORECASE)
        )

    @staticmethod
    def _is_start_of_sentence(tok: tokenizer.Tok) -> bool:
        """Check if a token is the start of a sentence."""
        if tok.kind == tokenizer.TOK.S_BEGIN:
            return True
        return False

    @staticmethod
    def _is_a_word(tok: tokenizer.Tok) -> bool:
        if tok.kind < tokenizer.TOK.META_BEGIN:
            if tok.kind > tokenizer.TOK.PUNCTUATION:
                return True
        return False

    @staticmethod
    def count_syllables_words_sentences_in_text(text: str) -> Tuple[int, int, int]:
        """Count the number of syllables, words and sentences in a text."""
        return Text.count_syllables_words_sentences_in_tok_stream(tokenizer.tokenize(text))

    @staticmethod
    def count_syllables_words_sentences_in_tok_stream(tok_stream: Iterator[tokenizer.Tok]) -> Tuple[int, int, int]:
        """Count the number of syllables, words and sentences in a token stream.
        Use this function if you already have a token stream as it is more efficient than counting the number of syllables, words and sentences in a text.
        """
        syllables = 0
        words = 0
        sentences = 0
        for tok in tok_stream:
            if Text._is_a_word(tok):
                words += 1
                # We only count syllables in words, not in abbreviations, numbers, etc.
                if tok.kind == tokenizer.TOK.WORD:
                    syllables += Text._count_syllables_in_word(tok.txt)
            if Text._is_start_of_sentence(tok):
                sentences += 1
        return (syllables, words, sentences)


class Flesch:
    """The Flesch class is as a namespace to store functions to calculate the Flesch reading ease score of a given text."""

    @staticmethod
    def get_score_from_stream(tok_stream: Iterator[tokenizer.Tok]) -> float:
        """Calculate the Flesch reading ease score of a text.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        syllables, words, sentences = Text.count_syllables_words_sentences_in_tok_stream(tok_stream)
        # Magic numbers from the Flesch reading ease score formula - English
        score = 206.835 - (1.015 * (words / sentences)) - 84.6 * (syllables / words)
        return score

    @staticmethod
    def get_score_from_text(text: str) -> float:
        """Calculate the Flesch reading ease score of a text.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        return Flesch.get_score_from_stream(tokenizer.tokenize(text))

    @staticmethod
    def get_feedback(score: float) -> str:
        """Provide feedback on the score of the Result object. This is done by comparing the score to the
        Flesch reading ease score scale. The feedback has been slightly modified to fit Icelandic. Icelandic tends to score lower than English and thus the scale has been adjusted to reflect this.
        """
        if score > 120:
            return "Mjög léttur eða ómarktækur texti."
        elif score > 90:
            return "Mjög léttur texti"
        elif score > 70:
            return "Léttur texti"
        elif score > 60:
            return "Meðalléttur texti"
        elif score > 50:
            return "Meðalþungur texti"
        elif score > 30:
            return "Nokkuð þungur texti"
        elif score > 0:
            return "Þungur texti"
        # Negative scores
        else:
            return "Mjög þungur eða ómarktækur texti."


class RareWords:
    """The RareWords class is used to find the rare words in a text.
    Rare words are defined as words which have a probability lower than the low_prob_cutoff.
    The probability of a word is calculated by looking up the word in an n-gram model.
    """

    def __init__(self, low_prob_cutoff: float = 0.00000005, max_words: int = 10):
        self.bin = Bin()
        self.ng = Ngrams()
        self.low_prob_cutoff = low_prob_cutoff
        self.max_words = max_words

    def _get_probability(self, tok_stream: Iterator[tokenizer.Tok]) -> List[Tuple[tokenizer.Tok, float]]:
        """Calculate the probability of each word in a text.
        The probability is calculated by looking up the word in the n-gram model. The probability is returned as a dictionary where the key is the word and the value is the probability.
        """
        probdict = []
        for token in tok_stream:
            # Only consider words, not punctuation, numbers, etc.
            if token.kind == tokenizer.TOK.WORD:
                # unigram probability for the word
                prob = self.ng.prob(token.txt)
                probdict.append((token, prob))
        return probdict

    def get_rare_words(
        self,
        tok_stream: Iterator[tokenizer.Tok],
        max_words: Optional[int] = None,
        low_prob_cutoff: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Get the rare words in a text.
        Rare words are defined as words which have a probability lower than the low_prob_cutoff.
        The probability of a word is calculated by looking up the word in the n-gram model.
        The probability is returned as a dictionary where the key is the word and the value is the probability.
        """
        max_words = max_words if max_words is not None else self.max_words
        low_prob_cutoff = low_prob_cutoff if low_prob_cutoff is not None else self.low_prob_cutoff
        rare_words = []
        for tok, prob in self._get_probability(tok_stream):
            # skipping all words that are numerical entities, punctuation, abbreviations, etc.
            if prob < low_prob_cutoff:
                lemma_set = self.bin.lookup_lemmas_and_cats(tok.txt)
                lemma = lemma_set.pop()[0] if lemma_set else tok.txt
                rare_words.append((lemma, prob))
        return sorted(rare_words, key=lambda x: x[1], reverse=True)[:max_words]


if __name__ == "__main__":
    while True:
        # read the input from the user
        text = input("Sláðu inn texta: ").strip()

        flesch_score = Flesch.get_score_from_text(text)
        rare = RareWords()
        token_stream = tokenizer.tokenize(text)
        rare_words = rare.get_rare_words(token_stream)

        print(f"Flesch-læsileikastig: {flesch_score:.2f}")
        print(f"Flesch-umsögn: {Flesch.get_feedback(flesch_score)}")
        print(f"Sjaldgæfustu {rare.max_words} orðin í textanum:")
        for word, prob in rare_words:
            print(f"\t{word}: {prob:.8f}")
        print()
