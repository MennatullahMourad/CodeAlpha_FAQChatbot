"""AI/ML FAQ Chatbot — CodeAlpha AI Internship, Task 2.

Run:  python -m streamlit run app.py
"""

import html

import streamlit as st

from matcher import FAQMatcher, load_faqs
from styles import inject_css, page_foot, page_head

st.set_page_config(page_title="AI and ML questions", page_icon=":material/menu_book:", layout="wide")
inject_css()
st.markdown(
    """
    <style>
    /* your questions: navy bubble on the right, no avatar */
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
        background: var(--harbor); margin-left: auto; width: fit-content !important; max-width: 78%;
        border-radius: 18px 18px 4px 18px; padding: .6rem 1.1rem; height: auto; min-height: 0; overflow: visible;
    }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
        padding: 0; margin: 0;
    }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p { color: #fff; margin: 0; }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stMarkdownContainer"] {
        margin-bottom: 0;
    }
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid^="stChatMessageAvatar"] {
        display: none;
    }
    /* answers: plain text on the white surface */
    div[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {
        background: transparent; max-width: 92%; padding-left: 0;
    }
    .match { display: flex; align-items: center; gap: .6rem; color: var(--steel); font-size: .88rem; margin: .2rem 0 .4rem; }
    .meter { width: 72px; height: 6px; border-radius: 6px; background: var(--ice); overflow: hidden; flex-shrink: 0; }
    .meter i { display: block; height: 100%; background: var(--lagoon); border-radius: 6px; }
    .hint { color: var(--steel); font-size: .9rem; margin: .4rem 0 .3rem; }
    .empty-title { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.35rem; margin: .2rem 0 .3rem; }
    [data-testid="stChatInput"] > div { background: #fff; box-shadow: var(--shadow); }
    </style>
    """,
    unsafe_allow_html=True,
)

AVATAR = ":material/menu_book:"
STARTERS = ["What is overfitting?", "How do CNNs work?", "Precision vs recall", "What is TF-IDF?"]


@st.cache_resource
def get_matcher() -> FAQMatcher:
    return FAQMatcher(load_faqs())


matcher = get_matcher()
faqs = matcher.faqs

st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)


def ask(question: str) -> None:
    st.session_state.pending = question


def clear_chat() -> None:
    st.session_state.messages = []


def match_line(score: float, question: str) -> str:
    width = max(4, min(100, round(score * 100)))
    return (
        f'<div class="match"><span class="meter"><i style="width:{width}%"></i></span>'
        f'<span>{score:.0%} match with “{html.escape(question)}”</span></div>'
    )


# ---------- sidebar: browse questions by topic ----------
with st.sidebar:
    st.markdown("### Topics")
    st.caption(f"{len(faqs)} questions. Pick one to ask it.")
    for cat in dict.fromkeys(f["category"] for f in faqs):
        with st.expander(cat):
            for faq in (f for f in faqs if f["category"] == cat):
                st.button(faq["question"], key=f"side-{faq['id']}", on_click=ask, args=(faq["question"],),
                          type="tertiary")
    st.divider()
    st.button("Clear chat", icon=":material/delete_sweep:", on_click=clear_chat, type="tertiary")

page_head("Ask about AI and ML",
          "Answers come from a set of common questions. Each answer shows which question it matched "
          "and how closely.")

# ---------- handle a new question ----------
prompt = st.chat_input("Ask a question about AI or machine learning")
question = prompt or st.session_state.pending
st.session_state.pending = None
if question:
    reply = matcher.reply(question)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({
        "role": "assistant",
        "kind": reply.kind,
        "content": reply.text,
        "matched": reply.match.faq["question"] if reply.match else None,
        "score": reply.match.score if reply.match else None,
        "related": [m.faq["question"] for m in (reply.related or [])],
    })

with st.container(key="surface-chat"):
    if not st.session_state.messages:
        st.markdown('<div class="empty-title">What would you like to know?</div>'
                    '<div class="hint">Type a question below, pick a topic on the left, or start with one of these.</div>',
                    unsafe_allow_html=True)
        with st.container(horizontal=True, gap="small"):
            for text in STARTERS:
                st.button(text, key=f"start-{text}", on_click=ask, args=(text,))

    for i, msg in enumerate(st.session_state.messages):
        avatar = AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            if msg.get("matched"):
                st.markdown(match_line(msg["score"], msg["matched"]), unsafe_allow_html=True)
            if msg.get("related"):
                hint = "Related questions" if msg["kind"] == "answer" else "Closest questions"
                st.markdown(f'<div class="hint">{hint}</div>', unsafe_allow_html=True)
                with st.container(horizontal=True, gap="small"):
                    for rel in msg["related"]:
                        st.button(rel, key=f"rel-{i}-{rel}", on_click=ask, args=(rel,))

page_foot("Questions are matched with NLTK preprocessing, TF-IDF and cosine similarity.")
