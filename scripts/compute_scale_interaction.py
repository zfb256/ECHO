"""Arm-by-scale interaction behind the dissociation claim (Section 6.1).

Section 6.1 reports that scale cuts injected contagion 7.8-fold but end-to-end self-induced
contagion only 2.9-fold. Two marginal trends do not by themselves establish that the two arms
respond differently to scale, so this script tests the interaction directly.

For each arm we fit the log-log slope of its rate against parameter count over the five Qwen2.5
ladder points, then compare slopes. A slope uses all five points rather than the two endpoints,
which keeps the estimate from turning on the noise in a single model. Uncertainty comes from a
cluster bootstrap over the units that actually vary: the 60 seeds for the injected arm, and the
60 hallucination-prone questions for the self-induced arm. The two arms draw on disjoint items,
so they are resampled independently.

The self-induced rate here is the end-to-end product (turn-1 hallucination rate times HCR_si),
not the conditional HCR_si, because the conditional denominator itself shrinks with scale.

    python -B scripts/compute_scale_interaction.py

Writes reports/scale_interaction.json.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl  # noqa: E402

LADDER = [
    (1.5, "Qwen2.5-1.5B-Instruct"),
    (3.0, "Qwen2.5-3B-Instruct"),
    (7.0, "Qwen2.5-7B-Instruct"),
    (14.0, "Qwen2.5-14B-Instruct"),
    (32.0, "Qwen2.5-32B-Instruct-AWQ"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Arm-by-scale interaction test for the ECHO dissociation.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--injected", default="datasets/runs/full_study/claim_annotation_injected_all3_auditfix.jsonl")
    p.add_argument("--selfinduced", default="datasets/runs/full_study/claim_annotation_selfinduced.jsonl")
    p.add_argument("--output", default="reports/scale_interaction.json")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    return p.parse_args()


def is_ccr_hit(claims: list[dict] | None) -> bool:
    return any(bool(c.get("depends_on_seed")) and c.get("present_in_clean") is False for c in claims or [])


def log_log_slope(by_unit: dict, units: list, models: list[tuple[float, str]]) -> float | None:
    """Least-squares slope of log(rate) on log(size) across the ladder."""
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for u in units:
        for model, hit in by_unit[u]:
            acc[model][0] += 1
            acc[model][1] += hit
    xs, ys = [], []
    for size, model in models:
        n, k = acc[model]
        if n == 0:
            return None
        rate = k / n
        # A resample can empty a cell; the half-count floor keeps the log defined without
        # letting one empty cell dominate the fit.
        rate = max(rate, 1.0 / (2 * n))
        xs.append(math.log(size))
        ys.append(math.log(rate))
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    models = {m for _, m in LADDER}

    inj = [r for r in read_jsonl(resolve_project_path(args.injected)) if r.get("model") in models]
    si = [r for r in read_jsonl(resolve_project_path(args.selfinduced)) if r.get("model") in models]
    if not inj or not si:
        raise SystemExit("no ladder rows found in the annotation files")

    inj_by_seed: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in inj:
        hit = 1 if is_ccr_hit(r.get("annotation", {}).get("contaminated_false_claims")) else 0
        inj_by_seed[r.get("seed_claim")].append((r["model"], hit))

    si_by_question: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in si:
        a = r.get("annotation", {})
        hit = 1 if (a.get("turn1_was_false") and a.get("hcr_has_contagious")) else 0
        si_by_question[r.get("q_id")].append((r["model"], hit))

    seeds = sorted(k for k in inj_by_seed if k is not None)
    questions = sorted(k for k in si_by_question if k is not None)
    slope_inj = log_log_slope(inj_by_seed, seeds, LADDER)
    slope_si = log_log_slope(si_by_question, questions, LADDER)
    if slope_inj is None or slope_si is None:
        raise SystemExit("could not fit both slopes; check that every ladder model is present")
    observed = slope_inj - slope_si

    n_boot = int(args.bootstrap_samples)
    rng = random.Random(int(config.get("random_seed", 0)))
    diffs = []
    for _ in range(n_boot):
        s = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        q = [questions[rng.randrange(len(questions))] for _ in questions]
        a = log_log_slope(inj_by_seed, s, LADDER)
        b = log_log_slope(si_by_question, q, LADDER)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        raise SystemExit("bootstrap produced no usable resamples")
    diffs.sort()
    m = len(diffs)
    ci = [round(diffs[int(0.025 * m)], 4), round(diffs[min(m - 1, int(0.975 * m))], 4)]

    report = {
        "ladder": [{"size_b": s, "model": mm} for s, mm in LADDER],
        "slope_injected_ccr": slope_inj,
        "slope_end_to_end_selfinduced": slope_si,
        "interaction_slope_difference": observed,
        "interaction_ci95": ci,
        "interaction_excludes_zero": ci[1] < 0 or ci[0] > 0,
        "n_seeds": len(seeds),
        "n_questions": len(questions),
        "n_bootstrap": m,
        "random_seed": int(config.get("random_seed", 0)),
        "definition": {
            "slope": "least-squares slope of log(rate) on log(parameter count) over the five ladder points",
            "interaction_slope_difference": "injected slope minus end-to-end self-induced slope; negative means injected contagion falls faster with scale",
            "end_to_end_selfinduced": "turn-1 hallucination rate times HCR_si, computed per question so the denominator does not shrink with scale",
            "bootstrap": "cluster bootstrap resampling whole seeds and whole questions independently",
        },
    }
    out_path = resolve_project_path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "wrote": to_project_relative(out_path),
            "slope_injected": round(slope_inj, 3),
            "slope_selfinduced": round(slope_si, 3),
            "difference": round(observed, 3),
            "ci95": ci,
            "excludes_zero": report["interaction_excludes_zero"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
