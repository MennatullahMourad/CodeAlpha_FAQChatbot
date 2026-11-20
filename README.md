# AI/ML FAQ Chatbot

A chatbot that answers frequently asked questions about Artificial Intelligence and Machine Learning.
It uses classic NLP: NLTK preprocessing, TF-IDF vectors and cosine similarity.
Built for **Task 2** of the CodeAlpha Artificial Intelligence Internship.

![Screenshot](docs/screenshot.png)

## Features

- **56 FAQs** in 6 topics: AI Basics, Machine Learning, Deep Learning, NLP, Computer Vision, and Evaluation & Data.
- A chat interface with suggested questions, a sidebar for browsing by topic, and clickable related questions.
- Every answer shows the **similarity score** and the FAQ it matched, so the user can judge the reply.
- When no FAQ is similar enough, the bot **says it isn't sure** instead of guessing. It suggests the closest questions, if any are reasonably close.
- Handles greetings and thanks ("hi", "thanks!") without confusing them with real questions.

## How it works

```
question ─► preprocess (NLTK) ─► TF-IDF vectors ─► cosine similarity vs. every FAQ ─► best match
                                                                                      │
                                             score ≥ threshold? ── yes ─► answer + related questions
                                                                └─ no ──► "not sure" + closest questions
```

1. **Preprocessing (`nlp.py`)**
   - lowercase the text and expand AI acronyms (e.g. *CNN → convolutional neural network*)
   - tokenize with `nltk.word_tokenize`
   - remove punctuation and stopwords
   - lemmatize with WordNet
2. **Matching (`matcher.py`)** combines three TF-IDF similarities:
   - lemmatized **words** (unigrams + bigrams) vs. each FAQ question and its alternative phrasings
   - **character n-grams** (3–5) vs. the same phrasings, which tolerates typos like *"overfiting"*
   - words vs. the **answer text**, which helps when the user describes a concept instead of naming it
3. **Decision:** if the best score is below a threshold, the bot falls back instead of answering. The threshold was tuned on a development set.

## Evaluation

`evaluate.py` measures the matcher on paraphrased questions written separately from the FAQ file (`data/eval_questions.json`).

The questions are split into two sets:
- **DEV:** used to choose the weights and the threshold.
- **TEST:** run once at the end.

Both sets include out-of-scope questions, some unrelated (e.g. *"how do I cook pasta"*) and some AI topics that aren't in the FAQ (e.g. *"what is logistic regression"*).

| Split | In-scope Qs | Top-1 accuracy | Top-3 accuracy | Out-of-scope Qs | Out-of-scope rejected |
|---|---|---|---|---|---|
| DEV | 44 | 75% | 84% | 22 | 82% |
| TEST | 26 | 46% | 69% | 12 | 58% |

**Reading these numbers honestly:**
- The test questions were made deliberately hard. Many describe a concept without using its name, e.g. *"generator and discriminator"* for GANs.
- Word-overlap methods like TF-IDF struggle with that. When the user uses the actual terms, matching is reliable: every FAQ question matches itself, and the tests check this.
- The gap between DEV and TEST shows the limits of tuning on a small set.
- The most effective improvement would be **semantic sentence embeddings** (e.g. `sentence-transformers`). They match meaning rather than shared words.

Run it yourself:

```bash
python evaluate.py
```

## Project structure

```
├── app.py                 # Streamlit chat interface
├── matcher.py             # TF-IDF + cosine similarity matching, fallback logic, small talk
├── nlp.py                 # NLTK preprocessing pipeline
├── evaluate.py            # held-out evaluation (dev/test)
├── data/
│   ├── faqs.json          # the knowledge base (edit this to change the topic)
│   └── eval_questions.json
├── styles.py              # shared visual theme
└── tests/
```

## Run it locally

```bash
git clone https://github.com/<your-username>/CodeAlpha_FAQChatbot.git
cd CodeAlpha_FAQChatbot
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

The first run downloads the NLTK data it needs (punkt_tab, stopwords, wordnet) automatically.

## Customising the FAQs

Edit `data/faqs.json`. Each entry looks like this:

```json
{"id": 57, "category": "Deep Learning", "question": "What is batch normalization?",
 "variants": ["Why use batchnorm?"], "answer": "..."}
```

Restart the app to rebuild the index.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Author

**Mennatullah Mourad Elshoura** · CodeAlpha AI Internship
