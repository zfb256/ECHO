from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Row-level bootstrap CIs for Table 1 CCR values.")
    p.add_argument("--open-injected", default="datasets/runs/full_study/claim_annotation_injected_all3_auditfix.jsonl")
    p.add_argument(
        "--closed-injected",
        action="append",
        default=None,
        help="Closed-model injected annotation file. Can be passed multiple times.",
    )
    p.add_argument("--output", default="reports/table1_ccr_ci.json")
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260618)
    return p.parse_args()


def is_ccr_hit(annotation: dict[str, Any]) -> bool:
    for claim in annotation.get("contaminated_false_claims") or []:
        if bool(claim.get("depends_on_seed")) and claim.get("present_in_clean") is False:
            return True
    return False


def ci(flags: list[int], n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    vals = []
    m = len(flags)
    for _ in range(n):
        vals.append(sum(flags[rng.randrange(m)] for _ in range(m)) / m)
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[min(n - 1, int(0.975 * n))], 4)]


def summarize(path: Path, n_boot: int, seed: int) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in read_jsonl(path):
        ann = row.get("annotation", {})
        if ann.get("hcr_has_contagious") is None:
            continue
        grouped[str(row.get("model", "unknown"))].append(1 if is_ccr_hit(ann) else 0)
    out = {}
    for i, (model, flags) in enumerate(sorted(grouped.items())):
        out[model] = {
            "n": len(flags),
            "ccr": statistics.fmean(flags),
            "ci95": ci(flags, n_boot, seed + i),
        }
    return out


def main() -> None:
    args = parse_args()
    open_path = resolve_project_path(args.open_injected)
    closed_args = args.closed_injected or ["datasets/runs/full_study/claim_annotation_injected_api_auditfix.jsonl"]
    closed_paths = [resolve_project_path(p) for p in closed_args]
    closed_models: dict[str, dict[str, Any]] = {}
    for i, closed_path in enumerate(closed_paths):
        if closed_path.exists():
            closed_models.update(summarize(closed_path, args.bootstrap_samples, args.seed + 1000 + i * 100))
    report = {
        "method": "Row-level percentile bootstrap over CCR flags. CCR flags require contaminated assertion of the seed falsehood and absence from the clean side.",
        "bootstrap_samples": args.bootstrap_samples,
        "open_models": summarize(open_path, args.bootstrap_samples, args.seed),
        "closed_models": closed_models,
        "sources": {
            "open_injected": args.open_injected,
            "closed_injected": closed_args,
        },
    }
    out = resolve_project_path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "open_models": len(report["open_models"]), "closed_models": len(report["closed_models"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
