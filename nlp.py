"""Text preprocessing with NLTK: lowercase -> expand acronyms -> tokenize ->
remove punctuation & stopwords -> lemmatize."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_NLTK_PACKAGES = {
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "corpora/wordnet": "wordnet",
}


def ensure_nltk_data() -> None:
    """Download the NLTK resources on first run if they are missing."""
    for path, package in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

# Acronyms are common in AI questions; expanding them lets "What is a CNN?"
# match "convolutional neural network". The acronym itself is kept too.
ACRONYMS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "nn": "neural network",
    "cnn": "convolutional neural network",
    "cnns": "convolutional neural network",
    "rnn": "recurrent neural network",
    "rnns": "recurrent neural network",
    "lstm": "long short term memory",
    "gan": "generative adversarial network",
    "gans": "generative adversarial network",
    "llm": "large language model",
    "llms": "large language model",
    "ner": "named entity recognition",
    "rl": "reinforcement learning",
    "tfidf": "tf idf term frequency inverse document frequency",
}

# Question words are dropped by the stopword list anyway; these extra words
# add no meaning in an FAQ context ("can you explain ...").
_EXTRA_STOPWORDS = {"explain", "tell", "please", "mean", "means", "meaning", "define", "definition",
                    "know", "want", "would", "could", "like", "us", "something", "thing"}
# Words in NLTK's list that matter for these FAQs (e.g. "vs" questions).
_KEEP = {"between", "not", "no", "more", "most", "against", "off", "out", "over", "under"}

_STOPWORDS = (set(stopwords.words("english")) | _EXTRA_STOPWORDS) - _KEEP
_LEMMATIZER = WordNetLemmatizer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, unify separators and expand known acronyms."""
    text = (text or "").lower()
    text = text.replace("tf-idf", "tfidf").replace("k-means", "kmeans").replace("-", " ")
    words = _TOKEN_RE.findall(text)
    expanded = []
    for word in words:
        expanded.append(word)
        if word in ACRONYMS:
            expanded.append(ACRONYMS[word])
    return " ".join(expanded)


@lru_cache(maxsize=4096)
def _lemma(token: str) -> str:
    noun = _LEMMATIZER.lemmatize(token, pos="n")
    return _LEMMATIZER.lemmatize(noun, pos="v") if noun == token else noun


def preprocess(text: str) -> list[str]:
    """Return the cleaned list of tokens used for matching."""
    tokens = word_tokenize(normalize(text))
    return [_lemma(tok) for tok in tokens if tok.isalnum() and tok not in _STOPWORDS]


def preprocess_to_string(text: str) -> str:
    return " ".join(preprocess(text))
