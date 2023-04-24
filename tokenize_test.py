import string
import nltk
from nltk.corpus import stopwords

english_stopwords = list(set(stopwords.words("english")))


def tokenize(text):
    """
    Tokenize with NLTK
    Rules:
        - drop all words of 1 and 2 characters
        - drop all stopwords
        - drop all numbers
    """
    tokens = list()
    words = nltk.word_tokenize(text)
    for word in words:
        if len(word) > 1:
            if word not in english_stopwords:
                if not word.isnumeric():
                    tokens.append(word)
    return list(set(tokens))


def test_tokenize_empty_string():
    # [1,2,8]
    assert len(tokenize("")) == 0


def test_tokenize_single_char_string():
    # [1,2,3,2,8]
    assert len(tokenize("y")) == 0


def test_tokenize_english_stopword():
    # [1,2,3,4,2,8]
    assert len(tokenize("themselves")) == 0


def test_tokenize_numeric_string():
    # [1,2,3,4,5,2,8]
    assert len(tokenize("2023")) == 0


def test_tokenize_single_word_string():
    # [1,2,3,4,5,6,7,2,8]
    tokens = tokenize("covid")
    assert len(tokens) == 1
    assert "covid" in tokens


def test_tokenize_sentence():
    # [1,2,3,2,3,4,5,6,7,2,3,4,5,6,7,2,3,4,2,3,4,5,2,8]
    tokens = tokenize("a covid case 2023")
    assert len(tokens) == 2
    assert "covid" in tokens
    assert "case" in tokens
