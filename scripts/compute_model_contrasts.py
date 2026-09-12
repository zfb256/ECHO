from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


CONTRASTS = [
    ("Gemma-2-9B", "gemma-2-9b-it", "Mistral-7B-Instruct-v0.3"),
    ("Qwen 1.5B-3B", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B-Instruct"),
    ("Qwen 3B-7B", "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"),
    ("Qwen 7B-14B", "Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"),
    ("Qwen 14B-32B-AWQ", "Qwen2.5-14B-Instruct", "Qwen2.5-32B-Instruct-AWQ"),
    ("Llama-8B vs Qwen-7B", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-paired CCR contrasts over matched pair_ids.")
    p.add_argument("--injected", default="datasets/runs/full_study/claim_annotation_injected_all3_auditfix.jsonl")
    p.add_argument("--output", default="reports/model_contrasts.json")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def is_ccr_hit(annotation: dict[str, Any]) -> bool:
    for claim in annotation.get("contaminated_false_claims") or []:
        if bool(claim.get("depends_on_seed")) and claim.get("present_in_clean") is False:
            return True
    return False


def percentile(xs: list[float], q: float) -> float:
    ordered = sorted(xs)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def paired_bootstrap(deltas: list[int], n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    m = len(deltas)
    vals = []
    for _ in range(n):
        vals.append(sum(deltas[rng.randrange(m)] for _ in range(m)) / m)
    return [round(percentile(vals, 0.025), 4), round(percentile(vals, 0.975), 4)]


def main() -> None:
    args = parse_args()
    by_model_pair: dict[str, dict[str, int]] = {}
    for row in read_jsonl(resolve_project_path(args.injected)):
        ann = row.get("annotation", {})
        if ann.get("hcr_has_contagious") is None:
            continue
        model = str(row.get("model"))
        pair_id = str(row.get("pair_id"))
        by_model_pair.setdefault(model, {})[pair_id] = 1 if is_ccr_hit(ann) else 0

    rows = []
    for i, (label, left, right) in enumerate(CONTRASTS):
        left_map = by_model_pair.get(left, {})
        right_map = by_model_pair.get(right, {})
        shared = sorted(set(left_map) & set(right_map))
        if not shared:
            raise SystemExit(f"no shared pair_ids for {left} vs {right}")
        deltas = [left_map[p] - right_map[p] for p in shared]
        left_rate = statistics.fmean(left_map[p] for p in shared)
        right_rate = statistics.fmean(right_map[p] for p in shared)
        delta = statistics.fmean(deltas)
        ci = paired_bootstrap(deltas, args.bootstrap_samples, args.seed + i)
        rows.append(
            {
                "label": label,
                "left_model": left,
                "right_model": right,
                "n_pair_ids": len(shared),
                "left_ccr": left_rate,
                "right_ccr": right_rate,
                "delta_left_minus_right": delta,
                "ci95": ci,
                "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
                "bootstrap_samples": args.bootstrap_samples,
            }
        )

    report = {
        "source": args.injected,
        "method": "Paired bootstrap over matched pair_ids; deltas are left_model CCR minus right_model CCR.",
        "contrasts": rows,
    }
    out = resolve_project_path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "contrasts": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
