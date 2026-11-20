from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from matcher import FAQMatcher, load_faqs
from nlp import preprocess

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(scope="module")
def matcher():
    return FAQMatcher(load_faqs())


def test_preprocess_removes_stopwords_and_lemmatizes():
    assert preprocess("What are the activation functions?") == ["activation", "function"]


def test_preprocess_expands_acronyms():
    tokens = preprocess("What is a CNN?")
    assert "cnn" in tokens and "convolutional" in tokens


def test_faq_file_is_valid(matcher):
    ids = [f["id"] for f in matcher.faqs]
    assert len(ids) == len(set(ids)) >= 50


@pytest.mark.parametrize("question", [f["question"] for f in load_faqs()])
def test_every_faq_matches_itself(matcher, question):
    reply = matcher.reply(question)
    assert reply.kind == "answer"
    assert reply.match.faq["question"] == question


def test_small_talk(matcher):
    assert matcher.reply("hi").kind == "smalltalk"
    assert matcher.reply("thanks!").kind == "smalltalk"


def test_greeting_with_real_question_is_answered(matcher):
    reply = matcher.reply("thanks, and what is dropout?")
    assert reply.kind == "answer" and reply.match.faq["question"] == "What is dropout?"


def test_off_topic_falls_back(matcher):
    assert matcher.reply("how do I cook pasta").kind == "fallback"


def test_empty_question(matcher):
    assert matcher.reply("   ").kind == "fallback"


def test_app_answers_in_chat():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("What is overfitting?").run()
    assert not at.exception
    assistant = at.session_state.messages[-1]
    assert assistant["matched"] == "What is overfitting?"
