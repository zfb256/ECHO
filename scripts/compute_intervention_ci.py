from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from artifacts import is_research_echo_r_path  # noqa: E402
from config import resolve_project_path, to_project_relative  # noqa: E402


def raw_rate(row: dict[str, Any]) -> float:
    contagious = row.get("contagious_probes")
    total = row.get("total_probes")
    if contagious is not None and total:
        return float(contagious) / float(total)
    return float(row["contagion_rate"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model-paired bootstrap CIs for intervention deltas vs no_op.")
    p.add_argument("--reports", nargs="+", required=True, help="Glob(s) for echo_r_eval_*.json files.")
    p.add_argument("--exclude", nargs="*", default=[], help="Drop matched files whose path contains any of these substrings.")
    p.add_argument("--output", default="reports/echo_r_intervention_ci.json")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260618)
    return p.parse_args()


def ci(vals: list[float], samples: int, seed: int) -> list[float] | None:
    if not vals:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(samples):
        boots.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return [round(boots[int(0.025 * samples)], 4), round(boots[min(samples - 1, int(0.975 * samples))], 4)]


def main() -> None:
    args = parse_args()
    files = sorted({
        f
        for pattern in args.reports
        for f in glob.glob(str(resolve_project_path(pattern)))
        if is_research_echo_r_path(f) and not any(ex in f for ex in args.exclude)
    })
    if not files:
        raise SystemExit(f"No reports matched {args.reports}")
    rows: list[dict[str, Any]] = []
    for f in files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        model = data.get("model") or Path(f).stem
        for r in data.get("results", []):
            rows.append({
                "model": model,
                "budget": r["budget_frac"],
                "policy": r["policy"],
                "rate": raw_rate(r),
                "contagious_probes": r.get("contagious_probes"),
                "total_probes": r.get("total_probes"),
            })
    by_model_budget: dict[tuple[str, float], dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_model_budget[(r["model"], r["budget"])][r["policy"]] = r["rate"]

    budgets = sorted({r["budget"] for r in rows})
    policies = sorted({r["policy"] for r in rows if r["policy"] != "no_op"})
    out_rows = []
    for budget in budgets:
        models = sorted({m for (m, b) in by_model_budget if b == budget and "no_op" in by_model_budget[(m, b)]})
        noop_vals = [by_model_budget[(m, budget)]["no_op"] for m in models]
        noop_mean = statistics.fmean(noop_vals) if noop_vals else None
        for policy in policies:
            diffs = []
            rates = []
            for model in models:
                vals = by_model_budget[(model, budget)]
                if policy not in vals:
                    continue
                rates.append(vals[policy])
                diffs.append(vals["no_op"] - vals[policy])
            if not diffs:
                continue
            out_rows.append({
                "budget": budget,
                "policy": policy,
                "n_models": len(diffs),
                "no_op_mean": noop_mean,
                "policy_mean": statistics.fmean(rates),
                "reduction_vs_no_op": statistics.fmean(diffs),
                "reduction_ci95_model_paired": ci(diffs, args.samples, args.seed),
                "model_deltas": {model: by_model_budget[(model, budget)]["no_op"] - by_model_budget[(model, budget)][policy]
                                 for model in models if policy in by_model_budget[(model, budget)]},
            })

    n_models = len({r["model"] for r in rows})
    report = {
        "files": [to_project_relative(f) for f in files],
        "rows": out_rows,
        "note": (
            "Positive reduction means lower contagion than no_op. CI bootstraps "
            f"paired model-level deltas over {n_models} models; intervals are "
            "model-level and can remain wide when effects are floor-heavy."
        ),
    }
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": to_project_relative(out), "n_rows": len(out_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
