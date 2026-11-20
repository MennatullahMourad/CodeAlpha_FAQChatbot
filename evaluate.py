"""Evaluate the FAQ matcher on held-out paraphrased questions.

The eval questions (data/eval_questions.json) are written differently from the
FAQ questions/variants. The threshold is tuned on the DEV split only; the TEST
split is reported once with that threshold.

Run:  python evaluate.py
"""

import json
from pathlib import Path

import numpy as np

from matcher import DATA_PATH, FAQMatcher, load_faqs

EVAL_PATH = Path(__file__).parent / "data" / "eval_questions.json"


def run(matcher: FAQMatcher, items: list, threshold: float) -> dict:
    in_scope = [(q, e) for q, e in items if e is not None]
    out_scope = [q for q, e in items if e is None]

    top1 = top3 = answered_correct = 0
    for q, expected in in_scope:
        ranked = matcher.top_k(q, k=3)
        ids = [m.faq["id"] for m in ranked]
        top1 += ids[0] == expected
        top3 += expected in ids
        answered_correct += ids[0] == expected and ranked[0].score >= threshold
    rejected = sum(matcher.top_k(q, 1)[0].score < threshold for q in out_scope)

    return {
        "in_scope_n": len(in_scope),
        "top1_accuracy": top1 / len(in_scope),
        "top3_accuracy": top3 / len(in_scope),
        "answered_correctly": answered_correct / len(in_scope),
        "out_of_scope_n": len(out_scope),
        "out_of_scope_rejected": rejected / len(out_scope),
    }


def main() -> None:
    splits = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    matcher = FAQMatcher(load_faqs(DATA_PATH))

    # tune on dev: maximise (correct answers + correct rejections)
    best_t, best_score = None, -1
    for t in np.arange(0.10, 0.61, 0.02):
        r = run(matcher, splits["dev"], t)
        score = r["answered_correctly"] * r["in_scope_n"] + r["out_of_scope_rejected"] * r["out_of_scope_n"]
        if score > best_score:
            best_t, best_score = round(float(t), 2), score

    print(f"Best threshold on DEV: {best_t}")
    for name in ("dev", "test"):
        r = run(matcher, splits[name], best_t)
        print(f"\n{name.upper()} ({r['in_scope_n']} in-scope, {r['out_of_scope_n']} out-of-scope)")
        print(f"  top-1 accuracy          : {r['top1_accuracy']:.0%}")
        print(f"  top-3 accuracy          : {r['top3_accuracy']:.0%}")
        print(f"  answered correctly      : {r['answered_correctly']:.0%}  (top-1 right AND above threshold)")
        print(f"  out-of-scope rejected   : {r['out_of_scope_rejected']:.0%}")

    print("\nTEST mistakes:")
    for q, expected in splits["test"]:
        m = matcher.top_k(q, 1)[0]
        ok = (m.faq["id"] == expected and m.score >= best_t) if expected else m.score < best_t
        if not ok:
            print(f"  [{m.score:.2f}] {q!r} -> #{m.faq['id']} {m.faq['question']!r} (expected #{expected})")


if __name__ == "__main__":
    main()
