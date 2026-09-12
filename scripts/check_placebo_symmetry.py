from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check clean_A/clean_B exchangeability in placebo annotations.")
    p.add_argument("--placebo", default="datasets/runs/full_study/claim_annotation_placebo_all3_auditfix.jsonl")
    p.add_argument("--output", default="reports/placebo_symmetry.json")
    return p.parse_args()


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(prob for prob in probs if prob <= observed + 1e-15))


def main() -> None:
    args = parse_args()
    n = a_hit = b_hit = both_hit = neither_hit = a_only = b_only = 0
    raw_b_hit = raw_b_only = raw_both = raw_neither = raw_a_only = 0
    by_model: dict[str, dict[str, int]] = {}
    for row in read_jsonl(resolve_project_path(args.placebo)):
        ann = row.get("annotation", {})
        if ann.get("hcr_has_contagious") is None:
            continue
        n += 1
        a = bool(ann.get("hcr_has_contagious"))
        b = bool(ann.get("clean_asserts_false"))
        raw_b = bool(ann.get("clean_rule_asserts_false"))
        a_hit += int(a)
        b_hit += int(b)
        both_hit += int(a and b)
        neither_hit += int((not a) and (not b))
        a_only += int(a and not b)
        b_only += int(b and not a)
        raw_b_hit += int(raw_b)
        raw_both += int(a and raw_b)
        raw_neither += int((not a) and (not raw_b))
        raw_a_only += int(a and not raw_b)
        raw_b_only += int(raw_b and not a)
        model = str(row.get("model", "unknown"))
        slot = by_model.setdefault(model, {"n": 0, "a_only": 0, "b_only": 0})
        slot["n"] += 1
        slot["a_only"] += int(a and not b)
        slot["b_only"] += int(b and not a)

    discordant = a_only + b_only
    report = {
        "source": args.placebo,
        "n": n,
        "clean_a_false_rate": a_hit / n if n else None,
        "clean_b_false_rate": b_hit / n if n else None,
        "both_hit": both_hit,
        "neither_hit": neither_hit,
        "a_hit_b_miss": a_only,
        "b_hit_a_miss": b_only,
        "discordant": discordant,
        "binomial_two_sided_p": binom_two_sided(a_only, discordant),
        "raw_rule": {
            "clean_a_false_rate": a_hit / n if n else None,
            "clean_b_false_rate": raw_b_hit / n if n else None,
            "both_hit": raw_both,
            "neither_hit": raw_neither,
            "a_hit_b_miss": raw_a_only,
            "b_hit_a_miss": raw_b_only,
            "discordant": raw_a_only + raw_b_only,
            "binomial_two_sided_p": binom_two_sided(raw_a_only, raw_a_only + raw_b_only),
        },
        "method": "Exact two-sided binomial test on discordant placebo rows under P(A-only)=P(B-only)=0.5.",
        "by_model": by_model,
    }
    out = resolve_project_path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "discordant": discordant, "p": report["binomial_two_sided_p"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
