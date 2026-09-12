from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import resolve_project_path, to_project_relative  # noqa: E402


# n=8 model-paired design, df=7. Two-sided 95% t critical.
T_CRIT_975_DF7 = 2.365
# z for 80% power under a normal approximation.
Z_POWER_80 = 0.842


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approximate model-level power/MDE for ECHO intervention deltas.")
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "reports/echo_r_intervention_ci_context.json",
            "reports/echo_r_intervention_ci_forcerw.json",
            "reports/echo_r_intervention_ci_suppress.json",
        ],
        help="Intervention CI JSON files from scripts/compute_intervention_ci.py.",
    )
    p.add_argument("--output", default="reports/intervention_power_analysis.json")
    return p.parse_args()


def arm_name(path: Path) -> str:
    stem = path.stem
    return stem.removeprefix("echo_r_intervention_ci_")


def stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def row_power(row: dict[str, Any], arm: str) -> dict[str, Any]:
    deltas = [float(v) for v in row.get("model_deltas", {}).values()]
    n = len(deltas)
    sd = stdev(deltas)
    se = sd / math.sqrt(n) if n else None
    mde80 = (T_CRIT_975_DF7 + Z_POWER_80) * se if se is not None else None
    observed = float(row["reduction_vs_no_op"])
    ci = row.get("reduction_ci95_model_paired")
    return {
        "arm": arm,
        "budget": row["budget"],
        "policy": row["policy"],
        "n_models": n,
        "observed_reduction": observed,
        "ci95_model_paired": ci,
        "model_delta_sd": sd,
        "se": se,
        "mde80_abs": mde80,
        "mde80_as_noop_fraction": (mde80 / row["no_op_mean"]) if mde80 is not None and row.get("no_op_mean") else None,
        "observed_abs_over_mde80": (abs(observed) / mde80) if mde80 else None,
        "no_op_mean": row.get("no_op_mean"),
        "policy_mean": row.get("policy_mean"),
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for raw in args.inputs:
        path = resolve_project_path(raw)
        if not path.exists():
            raise SystemExit(f"input does not exist: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        arm = arm_name(path)
        for row in data.get("rows", []):
            rows.append(row_power(row, arm))

    deployable = [r for r in rows if r["policy"] in {"reminder", "uniform_check", "echo_r"}]
    best_observed = sorted(deployable, key=lambda r: r["observed_reduction"], reverse=True)[:8]
    smallest_mde = sorted(deployable, key=lambda r: r["mde80_abs"] if r["mde80_abs"] is not None else 999)[:8]
    largest_mde = sorted(deployable, key=lambda r: r["mde80_abs"] if r["mde80_abs"] is not None else -1, reverse=True)[:8]

    report = {
        "method": {
            "unit": "model-level paired deltas (no_op - policy)",
            "n_models": 8,
            "alpha": 0.05,
            "target_power": 0.8,
            "approximation": (
                "MDE80 = (t_0.975,df=7 + z_0.80) * sd(delta) / sqrt(8). "
                "This is an approximate paired-design calculation; it deliberately treats models, "
                "not probes, as the independent units."
            ),
            "t_crit_975_df7": T_CRIT_975_DF7,
            "z_power_80": Z_POWER_80,
        },
        "summary": {
            "deployable_observed_reduction_range": [
                min(r["observed_reduction"] for r in deployable),
                max(r["observed_reduction"] for r in deployable),
            ],
            "deployable_mde80_abs_range": [
                min(r["mde80_abs"] for r in deployable if r["mde80_abs"] is not None),
                max(r["mde80_abs"] for r in deployable if r["mde80_abs"] is not None),
            ],
            "best_observed_deployable": best_observed,
            "smallest_mde_deployable": smallest_mde,
            "largest_mde_deployable": largest_mde,
        },
        "rows": rows,
        "inputs": [to_project_relative(resolve_project_path(p)) for p in args.inputs],
    }
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out),
                "deployable_observed_reduction_range": report["summary"]["deployable_observed_reduction_range"],
                "deployable_mde80_abs_range": report["summary"]["deployable_mde80_abs_range"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
