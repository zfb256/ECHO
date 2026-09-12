from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from jsonl import read_jsonl


def ccr(row: dict) -> int:
    return int(bool((row.get("annotation") or {}).get("derived_hallucination_count")))


def main() -> None:
    p = argparse.ArgumentParser(description="Validate API output coverage and bound truncated-response sensitivity.")
    p.add_argument("--tasks", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--injected", required=True)
    p.add_argument("--placebo", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    tasks = list(read_jsonl(Path(args.tasks)))
    outputs = list(read_jsonl(Path(args.outputs)))
    injected = list(read_jsonl(Path(args.injected)))
    placebo = list(read_jsonl(Path(args.placebo)))

    expected = {(r["pair_id"], r["condition"]) for r in tasks}
    observed = [(r["pair_id"], r["condition"]) for r in outputs]
    pair_ids = {pair_id for pair_id, _ in expected}
    models = {r.get("model") for r in outputs}
    terminal = {"stop", "completed"}
    unfinished = [r for r in outputs if r.get("finish_reason") not in terminal]

    real = sum(map(ccr, injected)) / len(injected)
    control = sum(map(ccr, placebo)) / len(placebo)
    gap = real - control
    # One unfinished contaminated/clean_b response can move one contrast arm.
    # clean participates in both real and placebo, so conservatively give it weight two.
    influence = sum({"contaminated": 1, "clean": 2, "clean_b": 1}.get(r["condition"], 1) for r in unfinished)
    bound = influence / len(pair_ids)

    errors = []
    if len(models) != 1:
        errors.append(f"expected one model, found {sorted(map(str, models))}")
    if len(observed) != len(set(observed)):
        errors.append("duplicate pair/condition outputs")
    if set(observed) != expected:
        errors.append(f"task coverage mismatch: missing={len(expected - set(observed))} unexpected={len(set(observed) - expected)}")
    if any(not str(r.get("response", "")).strip() for r in outputs):
        errors.append("empty response")
    if len(injected) != len(pair_ids) or len(placebo) != len(pair_ids):
        errors.append(f"annotation coverage mismatch: pairs={len(pair_ids)} injected={len(injected)} placebo={len(placebo)}")

    report = {
        "model": next(iter(models)) if len(models) == 1 else None,
        "coverage": {
            "expected_outputs": len(expected),
            "observed_outputs": len(outputs),
            "pairs": len(pair_ids),
            "conditions": dict(sorted(Counter(r["condition"] for r in outputs).items())),
            "empty_responses": sum(not str(r.get("response", "")).strip() for r in outputs),
            "duplicate_pair_conditions": len(observed) - len(set(observed)),
        },
        "finish_reasons": dict(sorted(Counter(str(r.get("finish_reason")) for r in outputs).items())),
        "unfinished": {
            "count": len(unfinished),
            "conditions": dict(sorted(Counter(r["condition"] for r in unfinished).items())),
        },
        "rates": {"injected_ccr": real, "placebo_ccr": control, "gap": gap},
        "unfinished_worst_case": {
            "absolute_gap_shift_bound": bound,
            "gap_interval": [max(-1.0, gap - bound), min(1.0, gap + bound)],
            "still_positive": gap - bound > 0,
        },
        "valid": not errors,
        "errors": errors,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "valid": not errors, "unfinished_robust": gap - bound > 0}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
