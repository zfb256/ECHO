"""Why the rule flags 80 correction-control outputs that assert nothing false.

Appendix A.2 states that most flagged spans are generic tokens, or a value that is true
elsewhere in the same fact. This script derives that breakdown from the shipped outputs so
the claim is checkable, and so a rebuttal can show why a human audit returned zero
persistence on all 80 rows.

Each flagged span is placed in one bucket:

  generic          the span carries no distinctive value (no digit, not a proper-noun-like
                   token), e.g. "human", "later", "tall", "taste"
  true_elsewhere   the span is a real value for the entity that the seed attaches to the
                   wrong role, e.g. 1799 is the true END of the French Revolution while the
                   seed dates its START to 1799
  refuted_mention  a distinctive false value that appears inside an explicit correction or
                   contrast, e.g. "206 ... easier to recall than 256"

The bucketing is a heuristic for exposition only. No headline number depends on it: the
persistence rate itself comes from human labels via compute_revision_persistence_audit.py.

    python -B scripts/inspect_revision_persistence_flags.py

Writes reports/revision_persistence_flag_diagnostics.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from detector import detect_injected  # noqa: E402
from jsonl import read_jsonl  # noqa: E402

# Cues that mark a correction or an explicit contrast around a false value.
CORRECTION = re.compile(
    r"\b(?:not|isn't|wasn't|never|no longer|rather than|instead of|instead|contrary to|"
    r"apolog\w*|sorry|correct\w*|mistake|error|myth|misconception|misunderstand\w*|"
    r"clarif\w*|actually|in fact)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explain the correction-control rule flags.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--tasks", default="datasets/runs/full_study/revision_control_tasks.jsonl")
    p.add_argument("--outputs", default="datasets/runs/full_study/revision_control_outputs.jsonl")
    p.add_argument("--output", default="reports/revision_persistence_flag_diagnostics.json")
    p.add_argument("--context", type=int, default=150, help="Characters scanned on each side for correction cues.")
    return p.parse_args()


def is_distinctive(token: str) -> bool:
    """A span carrying a specific value: a number, or a proper-noun-like word."""
    t = str(token)
    return bool(re.search(r"\d", t)) or (len(t) >= 5 and t[:1].isupper())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))

    tasks = {r["task_id"]: r for r in read_jsonl(resolve_project_path(args.tasks))}
    outputs = list(read_jsonl(resolve_project_path(args.outputs)))

    buckets: Counter[str] = Counter()
    by_seed: dict[str, Counter[str]] = {}
    rows = []
    for out in outputs:
        task = tasks.get(out.get("task_id"))
        if task is None:
            continue
        text = out.get("response") or ""
        det = detect_injected(text, task["seed_claim"], task["corrected_claim"], neg_window)
        if not det["asserts_false"]:
            continue

        tokens = [str(t) for t in (det.get("false_tokens") or [])]
        distinctive = [t for t in tokens if is_distinctive(t)]
        if not distinctive:
            bucket, span = "generic", (tokens[0] if tokens else "")
        else:
            span = distinctive[0]
            i = text.lower().find(span.lower())
            window = text[max(0, i - args.context) : i + args.context] if i >= 0 else text
            # A distinctive value stated without any nearby correction cue is a value the
            # response uses in its own right, so the seed attached it to the wrong role.
            bucket = "refuted_mention" if CORRECTION.search(window) else "true_elsewhere"

        buckets[bucket] += 1
        seed_id = str(task.get("seed_id"))
        by_seed.setdefault(seed_id, Counter())[bucket] += 1
        rows.append({"task_id": out.get("task_id"), "seed_id": seed_id, "model": task.get("model"), "span": span, "bucket": bucket})

    n = sum(buckets.values())
    if not n:
        raise SystemExit("no rule-flagged rows found; nothing to explain")

    top_seeds = sorted(by_seed.items(), key=lambda kv: -sum(kv[1].values()))[:5]
    report = {
        "n_flagged": n,
        "n_outputs_scanned": len(outputs),
        "buckets": dict(buckets),
        "bucket_shares": {k: round(v / n, 4) for k, v in buckets.items()},
        "share_not_a_false_assertion": round((buckets["generic"] + buckets["true_elsewhere"]) / n, 4),
        "top_seeds_by_flags": [{"seed_id": s, "n": sum(c.values()), "buckets": dict(c)} for s, c in top_seeds],
        "rows": rows,
        "definition": {
            "generic": "flagged span carries no distinctive value",
            "true_elsewhere": "distinctive value stated without a nearby correction cue, so the response uses it correctly and the seed attached it to the wrong role",
            "refuted_mention": "distinctive false value appearing inside an explicit correction or contrast",
            "note": "heuristic bucketing for exposition; the persistence rate itself comes from human labels",
        },
    }
    out_path = resolve_project_path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "wrote": to_project_relative(out_path),
            "n_flagged": n,
            "buckets": dict(buckets),
            "share_not_a_false_assertion": report["share_not_a_false_assertion"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
