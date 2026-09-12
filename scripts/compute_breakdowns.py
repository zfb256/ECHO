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
    p = argparse.ArgumentParser(description="Compute model/dataset/seed-type breakdowns for repaired ECHO annotations.")
    p.add_argument("--injected", required=True)
    p.add_argument("--placebo", required=True)
    p.add_argument("--seed-bank", default="configs/seed_bank.json")
    p.add_argument("--output", default="reports/all3_auditfix_breakdowns.json")
    return p.parse_args()


def hit(row: dict[str, Any]) -> bool:
    return row.get("annotation", {}).get("hcr_has_contagious") is True


def ci(flags: list[int], seed: int = 20260618, n: int = 2000) -> list[float] | None:
    if not flags:
        return None
    rng = random.Random(seed)
    vals = []
    m = len(flags)
    for _ in range(n):
        vals.append(sum(flags[rng.randrange(m)] for _ in range(m)) / m)
    vals.sort()
    return [round(vals[int(0.025 * n)], 4), round(vals[min(n - 1, int(0.975 * n))], 4)]


def summarize(rows: list[dict[str, Any]], seed_by_claim: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, key_fn in {
        "by_model": lambda r: r.get("model", "unknown"),
        "by_source_dataset": lambda r: r.get("source_dataset", "unknown"),
        "by_seed_type": lambda r: seed_by_claim.get(r.get("seed_claim", ""), {}).get("seed_type", "unknown"),
    }.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        out[label] = {}
        for key, rs in sorted(groups.items()):
            flags = [1 if hit(r) else 0 for r in rs]
            out[label][key] = {
                "n": len(rs),
                "rate": statistics.fmean(flags),
                "ci95": ci(flags),
            }
    return out


def main() -> None:
    args = parse_args()
    seed_bank = json.loads(resolve_project_path(args.seed_bank).read_text(encoding="utf-8")).get("seeds", [])
    seed_by_claim = {s["claim"]: s for s in seed_bank}
    injected = list(read_jsonl(resolve_project_path(args.injected)))
    placebo = list(read_jsonl(resolve_project_path(args.placebo)))
    report = {
        "injected": summarize(injected, seed_by_claim),
        "placebo": summarize(placebo, seed_by_claim),
        "notes": {
            "rate": "fraction of rows whose annotation.hcr_has_contagious is true",
            "ci95": "row-level bootstrap percentile interval",
        },
    }
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

