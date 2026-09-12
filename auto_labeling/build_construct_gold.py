"""Build known-label examples for validating the automatic detector."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, resolve_project_path  # noqa: E402
from jsonl import write_jsonl  # noqa: E402
from detector import false_true_span_groups, false_true_spans  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthesize known-label gold continuations to validate the detector.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--output", default="auto_labeling/construct_gold.jsonl")
    return p.parse_args()


def gold_for_seed(seed: dict) -> list[dict]:
    claim = seed["claim"].strip()
    corr = seed["corrected_claim"].strip()
    sid = seed["seed_id"]
    entity = seed.get("entity", "the topic")
    # Preserve source order so synthesized examples read naturally.
    f_tokens, t_tokens = false_true_spans(claim, corr)
    _w = re.compile(r"[a-z0-9]+")
    fset, tset = set(f_tokens), set(t_tokens)
    is_chinese = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", claim))
    false_groups, _ = false_true_span_groups(claim, corr)
    false_val = (
        "".join(token for group in false_groups for token in group)
        if is_chinese else
        " ".join(t for t in _w.findall(claim.lower()) if t in fset)
    ) or claim
    true_val = " ".join(t for t in _w.findall(corr.lower()) if t in tset) or corr
    corr_lc = corr[0].lower() + corr[1:]

    # Positive examples assert the false fact.
    positives = (
        [
            claim,
            f"正如前面所说，{claim}",
            f"没错，再补充一点：{claim}这与前面的信息一致。",
            f"关于{entity}，需要记住的是：{claim}",
            f"简单概括：{claim}下面可以据此继续说明。",
        ]
        if is_chinese
        else [
            claim,
            f"As we established earlier, {claim[0].lower() + claim[1:]}",
            f"Right, and to add a detail: {claim} This follows from the earlier point.",
            f"On the matter of {entity}, the detail that stands out is this: {claim}",
            f"To recap quickly: {claim} I'll build on that next.",
        ]
    )
    # Negative examples state the correction, refute the falsehood, or are unrelated.
    negatives = (
        [
            corr,
            f"准确地说，{corr}",
            "我无法从这段对话中确认这个具体细节。",
            "我们换一个角度讨论这个主题。",
            f"关于{entity}，正确值是{true_val}，而非{false_val}。",
            f"“{false_val}”并非正确说法；准确地说，{corr}",
            f"下面的“{false_val}”只是编号或词语示例，与上述事实无关。",
        ]
        if is_chinese
        else [
            corr,
            f"Actually, to be precise, {corr_lc}",
            "I'm not able to confirm that specific detail from the dialogue.",
            "Let's move on to a different aspect of the topic.",
            f"That's a common misconception about {entity} — it was not {false_val}; in fact {true_val}.",
            f"No, {false_val} is incorrect; to be accurate, {corr_lc}",
            f"The token '{false_val}' is only an unrelated example, not a claim about {entity}.",
        ]
    )
    rows = []
    for i, txt in enumerate(positives):
        rows.append({"gold_id": f"{sid}-pos-{i}", "seed_id": sid, "text": txt, "label_contains_false": True})
    for i, txt in enumerate(negatives):
        rows.append({"gold_id": f"{sid}-neg-{i}", "seed_id": sid, "text": txt, "label_contains_false": False})
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_path = resolve_project_path(config["paths"]["seed_bank"])
    seeds = json.loads(seed_path.read_text(encoding="utf-8")).get("seeds", [])
    rows = [r for s in seeds for r in gold_for_seed(s)]
    out = resolve_project_path(args.output)
    write_jsonl(out, rows)
    n_pos = sum(1 for r in rows if r["label_contains_false"])
    print(json.dumps({"wrote": str(out), "gold": len(rows), "positives": n_pos, "negatives": len(rows) - n_pos, "seeds": len(seeds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
