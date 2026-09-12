from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze injected contagion variance by seed and falsehood type.")
    p.add_argument("--seed-bank", default="configs/seed_bank.json")
    p.add_argument("--injected", default="datasets/runs/full_study/claim_annotation_injected_all3_auditfix.jsonl")
    p.add_argument("--placebo", default="datasets/runs/full_study/claim_annotation_placebo_all3_auditfix.jsonl")
    p.add_argument("--output", default="reports/seed_variance_analysis.json")
    p.add_argument("--csv", default="reports/seed_variance_by_seed.csv")
    return p.parse_args()


def is_ccr_hit(annotation: dict[str, Any]) -> bool:
    for claim in annotation.get("contaminated_false_claims") or []:
        if bool(claim.get("depends_on_seed")) and claim.get("present_in_clean") is False:
            return True
    return False


def fmean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def sample_sd(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) > 1 else None


def percentile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    ordered = sorted(xs)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_values(xs: list[float]) -> dict[str, float | None]:
    return {
        "mean": fmean(xs),
        "sd": sample_sd(xs),
        "min": min(xs) if xs else None,
        "p25": percentile(xs, 0.25),
        "median": percentile(xs, 0.50),
        "p75": percentile(xs, 0.75),
        "max": max(xs) if xs else None,
    }


def load_seed_bank(path: Path) -> dict[str, dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, str]] = {}
    for seed in raw.get("seeds", []):
        out[seed["claim"]] = {
            "seed_id": seed["seed_id"],
            "seed_type": seed["seed_type"],
            "seed_claim": seed["claim"],
        }
    return out


def load_arm(path: Path, seed_by_claim: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        seed = seed_by_claim.get(row.get("seed_claim"))
        if seed is None:
            raise SystemExit(f"{path}: seed_claim not found in seed bank: {row.get('seed_claim')!r}")
        sid = seed["seed_id"]
        if sid not in grouped:
            grouped[sid] = {
                **seed,
                "n": 0,
                "ccr_hits": 0,
                "hcr_hits": 0,
                "datasets": set(),
                "models": set(),
            }
        ann = row.get("annotation", {})
        grouped[sid]["n"] += 1
        grouped[sid]["ccr_hits"] += 1 if is_ccr_hit(ann) else 0
        grouped[sid]["hcr_hits"] += 1 if ann.get("hcr_has_contagious") is True else 0
        grouped[sid]["datasets"].add(str(row.get("source_dataset", "unknown")))
        grouped[sid]["models"].add(str(row.get("model", "unknown")))
    for seed in grouped.values():
        seed["ccr"] = seed["ccr_hits"] / seed["n"] if seed["n"] else None
        seed["hcr"] = seed["hcr_hits"] / seed["n"] if seed["n"] else None
        seed["datasets"] = sorted(seed["datasets"])
        seed["models"] = sorted(seed["models"])
    return grouped


def main() -> None:
    args = parse_args()
    seed_by_claim = load_seed_bank(resolve_project_path(args.seed_bank))
    injected = load_arm(resolve_project_path(args.injected), seed_by_claim)
    placebo = load_arm(resolve_project_path(args.placebo), seed_by_claim)

    by_seed = []
    for sid in sorted(injected):
        inj = injected[sid]
        pl = placebo.get(sid)
        if pl is None:
            raise SystemExit(f"placebo arm missing seed_id={sid}")
        by_seed.append(
            {
                "seed_id": sid,
                "seed_type": inj["seed_type"],
                "seed_claim": inj["seed_claim"],
                "n_injected": inj["n"],
                "n_placebo": pl["n"],
                "ccr": inj["ccr"],
                "hcr": inj["hcr"],
                "placebo_ccr": pl["ccr"],
                "ccr_minus_placebo": inj["ccr"] - pl["ccr"],
                "datasets": inj["datasets"],
                "models": inj["models"],
            }
        )

    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_seed:
        type_groups[row["seed_type"]].append(row)

    by_seed_type = []
    for seed_type, rows in sorted(type_groups.items()):
        ccrs = [r["ccr"] for r in rows]
        hcrs = [r["hcr"] for r in rows]
        placebo_ccrs = [r["placebo_ccr"] for r in rows]
        gaps = [r["ccr_minus_placebo"] for r in rows]
        by_seed_type.append(
            {
                "seed_type": seed_type,
                "n_seeds": len(rows),
                "n_injected_rows": sum(r["n_injected"] for r in rows),
                "n_placebo_rows": sum(r["n_placebo"] for r in rows),
                "ccr": fmean(ccrs),
                "hcr": fmean(hcrs),
                "placebo_ccr": fmean(placebo_ccrs),
                "ccr_minus_placebo": fmean(gaps),
                "seed_ccr_sd": sample_sd(ccrs),
                "seed_ccr_min": min(ccrs),
                "seed_ccr_median": percentile(ccrs, 0.50),
                "seed_ccr_max": max(ccrs),
                "seeds_with_positive_ccr": sum(1 for x in ccrs if x > 0),
                "seeds_with_ccr_above_placebo": sum(1 for x in gaps if x > 0),
                "seeds_with_gap_above_0_05": sum(1 for x in gaps if x > 0.05),
            }
        )

    seed_ccrs = [r["ccr"] for r in by_seed]
    gaps = [r["ccr_minus_placebo"] for r in by_seed]
    report = {
        "source_seed_bank": args.seed_bank,
        "source_injected": args.injected,
        "source_placebo": args.placebo,
        "overall": {
            "n_seeds": len(by_seed),
            "n_seed_types": len(by_seed_type),
            "n_injected_rows": sum(r["n_injected"] for r in by_seed),
            "n_placebo_rows": sum(r["n_placebo"] for r in by_seed),
            "seed_ccr_distribution": summarize_values(seed_ccrs),
            "seed_gap_distribution": summarize_values(gaps),
            "seeds_with_positive_ccr": sum(1 for x in seed_ccrs if x > 0),
            "seeds_with_ccr_above_placebo": sum(1 for x in gaps if x > 0),
            "seeds_with_gap_above_0_05": sum(1 for x in gaps if x > 0.05),
        },
        "by_seed_type": by_seed_type,
        "by_seed": by_seed,
    }

    out_path = resolve_project_path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = resolve_project_path(args.csv)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed_id",
                "seed_type",
                "n_injected",
                "n_placebo",
                "ccr",
                "hcr",
                "placebo_ccr",
                "ccr_minus_placebo",
                "seed_claim",
            ],
        )
        writer.writeheader()
        for row in by_seed:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    print(json.dumps({"wrote": str(out_path), "csv": str(csv_path), "n_seeds": len(by_seed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
