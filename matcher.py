"""FAQ retrieval with TF-IDF + cosine similarity.

Two TF-IDF views are combined:
  * word view  - lemmatized tokens (unigrams + bigrams) from nlp.preprocess
  * char view  - character 3-5 grams, which tolerates typos ("overfiting")
Each FAQ is indexed by its main question and its alternative phrasings
(an FAQ's question score is the best among its phrasings), and also by its
answer text, which helps when the user describes a concept instead of naming it
("learning from rewards" -> reinforcement learning).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp import normalize, preprocess_to_string

DATA_PATH = Path(__file__).parent / "data" / "faqs.json"

WORD_WEIGHT = 0.6     # lemmatized-word TF-IDF vs. question phrasings
CHAR_WEIGHT = 0.4     # character n-gram TF-IDF vs. question phrasings
ANSWER_WEIGHT = 0.3   # share of the final score that comes from matching the answer text
DEFAULT_THRESHOLD = 0.18  # tuned on the DEV split only - see evaluate.py
SMALL_TALK_OVERRIDE = 0.35  # a greeting that also contains a clear FAQ question is answered


@dataclass
class Match:
    faq: dict
    score: float


@dataclass
class Reply:
    kind: str                 # "answer" | "fallback" | "smalltalk"
    text: str
    match: Match | None = None
    related: list[Match] | None = None


SMALL_TALK = [
    (re.compile(r"^\s*(hi+|hello+|hey+|salam|good (morning|evening|afternoon))\b", re.I),
     "Hi! Ask me anything about AI, machine learning, deep learning, NLP or model evaluation."),
    (re.compile(r"\b(thanks?|thank you|thx|appreciate)\b", re.I),
     "You're welcome! Feel free to ask another question."),
    (re.compile(r"^\s*(bye|goodbye|see you)\b", re.I),
     "Goodbye! Good luck with your AI learning."),
    (re.compile(r"\b(who are you|what can you do|help me|how do you work)\b", re.I),
     "I'm an FAQ assistant for AI & ML concepts. I compare your question with my FAQ list "
     "using TF-IDF and cosine similarity, then return the closest answer."),
]


def load_faqs(path: Path = DATA_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        faqs = json.load(f)
    for faq in faqs:
        if not faq.get("question") or not faq.get("answer"):
            raise ValueError(f"FAQ {faq.get('id')} is missing a question or answer")
    return faqs


class FAQMatcher:
    def __init__(self, faqs: list[dict], threshold: float = DEFAULT_THRESHOLD):
        self.faqs = faqs
        self.threshold = threshold

        phrasings, owners = [], []
        for idx, faq in enumerate(faqs):
            for text in [faq["question"], *faq.get("variants", [])]:
                phrasings.append(text)
                owners.append(idx)
        self._owners = np.array(owners)

        self._word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._word_matrix = self._word_vec.fit_transform([preprocess_to_string(p) for p in phrasings])
        self._char_matrix = self._char_vec.fit_transform([normalize(p) for p in phrasings])
        # answers share the word vocabulary so query vectors are comparable
        self._answer_matrix = self._word_vec.transform([preprocess_to_string(f["answer"]) for f in faqs])

    def scores(self, query: str) -> np.ndarray:
        """Similarity of the query to every FAQ (max over that FAQ's phrasings)."""
        query_words = self._word_vec.transform([preprocess_to_string(query)])
        word_sim = cosine_similarity(query_words, self._word_matrix)[0]
        char_sim = cosine_similarity(self._char_vec.transform([normalize(query)]), self._char_matrix)[0]
        combined = WORD_WEIGHT * word_sim + CHAR_WEIGHT * char_sim

        per_faq = np.zeros(len(self.faqs))
        np.maximum.at(per_faq, self._owners, combined)
        answer_sim = cosine_similarity(query_words, self._answer_matrix)[0]
        return (1 - ANSWER_WEIGHT) * per_faq + ANSWER_WEIGHT * answer_sim

    def top_k(self, query: str, k: int = 3) -> list[Match]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [Match(self.faqs[i], float(s[i])) for i in order]

    def reply(self, query: str) -> Reply:
        query = (query or "").strip()
        if not query:
            return Reply("fallback", "Please type a question.")

        candidates = self.top_k(query, k=4)
        best = candidates[0]

        # Small talk only wins when the message isn't also a real question,
        # so "thanks, and what is dropout?" still gets the dropout answer.
        strong_match = best.score >= SMALL_TALK_OVERRIDE
        for pattern, answer in SMALL_TALK:
            if pattern.search(query) and not strong_match:
                return Reply("smalltalk", answer)

        if best.score >= self.threshold:
            related = [m for m in candidates[1:] if m.score >= self.threshold * 0.8]
            return Reply("answer", best.faq["answer"], match=best, related=related[:2])

        # Only suggest questions that are at least loosely related.
        close = [m for m in candidates[:3] if m.score >= self.threshold * 0.75]
        if close:
            return Reply("fallback", "I'm not sure I have an answer for that. Did you mean one of these?",
                         related=close)
        return Reply("fallback", "Sorry, that's outside what I know. I answer questions about AI and "
                                 "machine learning. Try the topics in the sidebar.")
