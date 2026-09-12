"""Recompute injected correction-control persistence from human labels.

The shipped metric reports post-correction persistence $0.000$ over 1,800 outputs, but that
zero is the NLI guard's verdict: the raw token rule flags 80 rows and the guard drops all 80.
This script replaces the guard's judgment with a human one on exactly those 80 rows and
recomputes the rate, so the headline number is human-backed rather than model-backed.

Rows the rule never flagged contain no distinctive false span at all and cannot be persistence
cases, so they stay negative by construction and are not audited. The denominator therefore
remains the full 1,800 outputs.

    python -B scripts/compute_revision_persistence_audit.py

Writes reports/revision_persistence_human_audit.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute correction-control persistence from human labels.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--audit", default="datasets/runs/full_study/revision_persistence_audit_sample.jsonl")
    p.add_argument("--metrics", default="reports/revision_control_metrics.json")
    p.add_argument("--output", default="reports/revision_persistence_human_audit.json")
    p.add_argument("--allow-partial", action="store_true", help="Score even if some rows are unlabelled.")
    return p.parse_args()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    """Wilson score interval; the same estimator the other ECHO audits report."""
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def main() -> None:
    args = parse_args()
    load_config(args.config)
    audit_path = resolve_project_path(args.audit)
    metrics_path = resolve_project_path(args.metrics)
    if not audit_path.exists():
        raise SystemExit(f"audit packet not found: {to_project_relative(audit_path)} (run prepare_revision_persistence_audit.py first)")

    rows = list(read_jsonl(audit_path))
    labelled = [r for r in rows if isinstance(r.get("human_still_asserts_false"), bool)]
    uncertain = [r for r in rows if r.get("human_uncertain") is True]
    unlabelled = len(rows) - len(labelled) - len(uncertain)
    if unlabelled and not args.allow_partial:
        raise SystemExit(f"{unlabelled} of {len(rows)} audit rows are unlabelled; finish the audit or pass --allow-partial")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    n_total = int(metrics.get("inputs", {}).get("n_scored") or metrics.get("overall", {}).get("n") or 0)
    if n_total <= 0:
        raise SystemExit(f"could not read the scored-output count from {to_project_relative(metrics_path)}")

    human_persist = [r for r in labelled if r["human_still_asserts_false"] is True]
    k = len(human_persist)
    rate = k / n_total

    by_model: dict[str, dict[str, int]] = {}
    for r in labelled:
        m = by_model.setdefault(str(r.get("model")), {"audited": 0, "human_persist": 0})
        m["audited"] += 1
        m["human_persist"] += 1 if r["human_still_asserts_false"] else 0

    report = {
        "source_audit": to_project_relative(audit_path),
        "source_metrics": to_project_relative(metrics_path),
        "n_outputs_total": n_total,
        "n_rule_flagged_audited": len(labelled),
        "n_rule_flagged_uncertain": len(uncertain),
        "n_rule_flagged_unlabelled": unlabelled,
        "reviewer_count": 1,
        "guard_reported_persistence": metrics.get("overall", {}).get("post_correction_persistence"),
        "human_persistence_n": k,
        "human_persistence_rate": rate,
        "human_persistence_rate_upper_if_all_uncertain_positive": (k + len(uncertain)) / n_total,
        "human_persistence_ci95_wilson": wilson(k, n_total),
        "human_vs_guard_agreement": round(1 - (k / len(labelled)), 4) if labelled else None,
        "by_model": by_model,
        "ci_method": "wilson",
        "definition": {
            "human_persistence_rate": (
                "fraction of all correction-control outputs a human judges to still assert the seed's "
                "false value; only rule-flagged rows are audited, unflagged rows are negative by construction"
            ),
            "human_vs_guard_agreement": (
                "share of audited rows where the human agrees with the NLI guard that the row is NOT persistence"
            ),
        },
    }
    out_path = resolve_project_path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "wrote": to_project_relative(out_path),
            "audited": len(labelled),
            "human_persistence_n": k,
            "human_persistence_rate": round(rate, 4),
            "guard_reported": report["guard_reported_persistence"],
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
