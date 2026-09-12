from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from artifacts import is_research_echo_r_path  # noqa: E402
from config import resolve_project_path, to_project_relative  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit ECHO-R intervention table rates against raw probe counts.")
    p.add_argument(
        "--reports",
        default="reports/echo_r_eval_injected_multi4_forcerw-grounded_*.json",
        help="Glob of per-model intervention reports.",
    )
    p.add_argument("--exclude", nargs="*", default=[], help="Drop matched files whose path contains any of these substrings.")
    p.add_argument("--output", default="reports/intervention_count_audit.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(
        f
        for f in glob.glob(str(resolve_project_path(args.reports)))
        if is_research_echo_r_path(f) and not any(ex in f for ex in args.exclude)
    )
    if not files:
        raise SystemExit(f"No reports matched {args.reports}")
    agg = defaultdict(lambda: {"contagious_probes": 0, "total_probes": 0, "per_model": []})

    for path in files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data["model"]
        for row in data["results"]:
            key = (str(row["budget_frac"]), row["policy"])
            agg[key]["contagious_probes"] += row["contagious_probes"]
            agg[key]["total_probes"] += row["total_probes"]
            agg[key]["per_model"].append(
                {
                    "model": model,
                    "contagious_probes": row["contagious_probes"],
                    "total_probes": row["total_probes"],
                    "reported_rate": row["contagion_rate"],
                }
            )

    rows = []
    for (budget, policy), val in sorted(agg.items(), key=lambda kv: (float(kv[0][0]), kv[0][1])):
        cont = val["contagious_probes"]
        total = val["total_probes"]
        exact = cont / total if total else 0.0
        rows.append(
            {
                "budget": float(budget),
                "policy": policy,
                "contagious_probes": cont,
                "total_probes": total,
                "exact_rate": exact,
                "rounded_3dp": round(exact, 3),
                "per_model": val["per_model"],
            }
        )

    per_model_probe_counts = sorted({
        pm["total_probes"]
        for val in agg.values()
        for pm in val["per_model"]
    })
    n_files = len(files)
    total_per_cell = sum(per_model_probe_counts) if len(per_model_probe_counts) == n_files else None
    note = (
        f"Each input model contributes one per-policy/per-budget probe count. "
        f"Observed per-model probe counts are {per_model_probe_counts}; n_files={n_files}. "
        "Repeated three-decimal endings are expected because rates are pooled integer counts, "
        "not free-form decimals."
    )
    if len(per_model_probe_counts) == 1:
        denom = per_model_probe_counts[0] * n_files
        note += f" With {n_files} models and {per_model_probe_counts[0]} probes/model, one probe changes a pooled rate by 1/{denom}."

    out = {
        "files": [to_project_relative(f) for f in files],
        "n_files": n_files,
        "note": note,
        "rows": rows,
    }
    out_path = resolve_project_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": to_project_relative(out_path), "n_files": len(files), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
