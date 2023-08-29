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


class Flesch:
    """The Flesch class calculates the Flesch reading ease score of a given text.

    The class can be used to track the number of syllables, words and sentences in a text using the token stream.
    It is designed to be used with the tokenizer module and maintains an internal state which needs to be reset manually.
    """

    def __init__(self) -> None:
        self.num_syllables = 0
        self.num_words = 0
        self.num_sentences = 0
        self.has_tracked = False  # True if the token stream has been tracked - used to avoid wrong usage of the class

    @staticmethod
    def _count_syllables_in_word(word: str) -> int:
        """Count the number of syllables in an Icelandic word, by counting the number of vowels in the word. This is done by first replacing diphtongs with a single character, then counting the number of vowels."""
        word = word.lower()
        # replace diphtongs with a single character
        word = diphtong_pattern.sub("a", word)
        return len(vowel_pattern.findall(word))

    @staticmethod
    def _is_start_of_sentence(tok: tokenizer.Tok) -> bool:
        """Check if a token is the start of a sentence."""
        if tok.kind == tokenizer.TOK.S_BEGIN:
            return True
        return False

    @staticmethod
    def _is_a_word(tok: tokenizer.Tok) -> bool:
        """Determine if a token is a word for the Flesch metric."""
        if tok.kind < tokenizer.TOK.META_BEGIN:
            if tok.kind > tokenizer.TOK.PUNCTUATION:
                return True
        return False

    def reset(self) -> None:
        """Reset the Flesch class. This is done by setting the number of syllables, words and sentences to 0."""
        self.num_syllables = 0
        self.num_words = 0
        self.num_sentences = 0
        self.has_tracked = False

    def track_token_stream(self, tok_stream: Iterator[tokenizer.Tok]) -> Iterator[tokenizer.Tok]:
        """Tracks the number of syllables, words and sentences in a token stream. This is done by yielding the tokens in the token stream."""
        if self.has_tracked:
            raise ValueError(
                "The token stream has already been tracked. Create a new instance of the Flesch class to track a new token stream or reset it."
            )
        self.has_tracked = True
        for tok in tok_stream:
            if self._is_a_word(tok):
                self.num_words += 1
                # We only count syllables in words, not in abbreviations, numbers, etc.
                if tok.kind == tokenizer.TOK.WORD:
                    self.num_syllables += self._count_syllables_in_word(tok.txt)
            if self._is_start_of_sentence(tok):
                self.num_sentences += 1
            yield tok

    def get_score(self) -> float:
        """Calculate the Flesch reading ease score after tracking a token stream.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        if not self.has_tracked:
            raise ValueError(
                "The token stream has not been tracked. Call the track_token_stream function before calling this function."
            )
        # Magic numbers from the Flesch reading ease score formula - English
        score = 206.835 - (1.015 * (self.num_words / self.num_sentences)) - 84.6 * (self.num_syllables / self.num_words)
        return score

    @staticmethod
    def get_score_from_text(text: str) -> float:
        """Calculate the Flesch reading ease score of a text.
        This function is a convenience function which does not require the user to track the token stream.
        See https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests."""
        stream = tokenizer.tokenize(text)
        flesch = Flesch()
        stream = flesch.track_token_stream(stream)
        for _ in stream:
            pass
        return flesch.get_score()

    @staticmethod
    def get_feedback(score: float) -> str:
        """Provide feedback on the score of the Result object. This is done by comparing the score to the
        Flesch reading ease score scale. The feedback has been slightly modified to fit Icelandic.
        Icelandic tends to score lower than English and thus the scale has been adjusted to reflect this.
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
    The class is designed to be used with the tokenizer module and maintains an internal state which needs to be reset manually.
    """

    def __init__(self, low_prob_cutoff: float = 0.00000005, max_words: int = 10):
        self.bin = Bin()
        self.ng = Ngrams()
        self.low_prob_cutoff = low_prob_cutoff
        self.max_words = max_words
        # This list needs to be reset manually
        self.probs: List[Tuple[tokenizer.Tok, float]] = []
        self.has_tracked = False  # True if the token stream has been tracked - used to avoid wrong usage of the class

    def reset(self) -> None:
        """Reset the RareWords class. This is done by setting the probability dictionary to an empty list."""
        self.probs = []
        self.has_tracked = False

    def track_token_stream(self, tok_stream: Iterator[tokenizer.Tok]) -> Iterator[tokenizer.Tok]:
        """Tracks the probability of each word in a token stream. This is done by yielding the tokens in the token stream."""
        if self.has_tracked:
            raise ValueError(
                "The token stream has already been tracked. Create a new instance of the RareWords class to track a new token stream or reset it."
            )
        self.has_tracked = True
        for token in tok_stream:
            # Only consider words, not punctuation, numbers, etc.
            if token.kind == tokenizer.TOK.WORD:
                # unigram probability for the word
                prob = self.ng.prob(token.txt)
                self.probs.append((token, prob))
            yield token

    def get_rare_words(
        self,
        max_words: Optional[int] = None,
        low_prob_cutoff: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Get the rare words from a token stream which has been tracked.
        Rare words are defined as words which have a probability lower than the low_prob_cutoff.
        The probability of a word is calculated by looking up the word in the n-gram model.
        """
        if not self.has_tracked:
            raise ValueError(
                "The token stream has not been tracked. Call the track_token_stream function before calling this function."
            )
        max_words = max_words if max_words is not None else self.max_words
        low_prob_cutoff = low_prob_cutoff if low_prob_cutoff is not None else self.low_prob_cutoff
        rare_words = []
        for tok, prob in self.probs:
            # skipping all words that are numerical entities, punctuation, abbreviations, etc.
            if prob < low_prob_cutoff:
                lemma_set = self.bin.lookup_lemmas_and_cats(tok.txt)
                lemma = lemma_set.pop()[0] if lemma_set else tok.txt
                rare_words.append((lemma, prob))
        return sorted(rare_words, key=lambda x: x[1], reverse=False)[:max_words]

    @staticmethod
    def get_rare_words_from_text(
        text: str,
        max_words: Optional[int] = None,
        low_prob_cutoff: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Get the rare words from a text.
        This function is a convenience function which does not require the user to track the token stream.

        Rare words are defined as words which have a probability lower than the low_prob_cutoff.
        The probability of a word is calculated by looking up the word in the n-gram model.
        This function is a convenience function which does not require the user to track the token stream.
        """
        stream = tokenizer.tokenize(text)
        rare = RareWords()
        stream = rare.track_token_stream(stream)
        for _ in stream:
            pass
        return rare.get_rare_words(max_words, low_prob_cutoff)


if __name__ == "__main__":
    while True:
        # read the input from the user
        text = input("Sláðu inn texta: ").strip()

        token_stream = tokenizer.tokenize(text)
        flesch = Flesch()
        rare = RareWords()
        token_stream = flesch.track_token_stream(token_stream)
        token_stream = rare.track_token_stream(token_stream)
        for _ in token_stream:
            pass

        flesch_score = flesch.get_score()
        rare_words = rare.get_rare_words()

        print(f"Flesch-læsileikastig: {flesch_score:.2f}")
        print(f"Flesch-umsögn: {Flesch.get_feedback(flesch_score)}")
        print(f"Sjaldgæfustu {rare.max_words} orðin í textanum:")
        for word, prob in rare_words:
            print(f"\t{word}: {prob:.8f}")
        print()
