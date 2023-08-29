from typing import Tuple

import tokenizer

from reynir_correct import readability


def get_sentence_word_syllable_counts(text: str) -> Tuple[int, int, int]:
    """Get the number of sentences, words and syllables in a text."""
    flesch = readability.Flesch()
    stream = tokenizer.tokenize(text)
    stream = flesch.track_token_stream(stream)
    for _ in stream:
        pass
    return flesch.num_sentences, flesch.num_words, flesch.num_syllables


def test_count_syllables():
    words_and_syllables = [
        ("hús", 1),
        ("húsið", 2),
        ("húsin", 2),
        ("húsinu", 3),
        ("ae_iouyáéíóúýöæ", 14),  # vowels
        ("eieyau", 3),  # diphtongs
        ("bcdfghjklmnpqrstvwxzðþ", 0),  # consonants
        ("áin þessi", 4),  # spaces")
    ]
    for word, num_syllables in words_and_syllables:
        assert (
            readability.Flesch._count_syllables_in_word(word) == num_syllables
        ), f"Expected {num_syllables} syllables in {word}"


def test_count_sentences_words_syllables():
    strings_and_words = [
        ("já", (1, 1, 1)),
        ("já já", (1, 2, 2)),
        ("já     já\tjá", (1, 3, 3)),
        ("Þetta. Eru. Fjórar. Setningar", (4, 4, 9)),
    ]
    for string, (num_sents, num_words, num_syllables) in strings_and_words:
        assert get_sentence_word_syllable_counts(string) == (
            num_sents,
            num_words,
            num_syllables,
        ), f"Expected {num_syllables} syllables, {num_words} words and {num_sents} sentences in {string}"
