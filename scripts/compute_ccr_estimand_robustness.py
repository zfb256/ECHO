"""Check CCR under clean-side symmetry and ambiguity conventions.

The saved annotations retain both final and rule-only clean labels. The
symmetric variant uses rule-only flags on both sides; the ambiguity-inclusive
variant also counts exposed responses containing false and true spans as
present. All quantities come from the shipped files.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CCR estimand robustness checks.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument(
        "--injected",
        default="datasets_zh/runs/zh_study/claim_annotation_injected.jsonl",
    )
    parser.add_argument(
        "--placebo",
        default="datasets_zh/runs/zh_study/claim_annotation_placebo.jsonl",
    )
    parser.add_argument("--output", default="reports_zh/ccr_estimand_robustness.json")
    return parser.parse_args()


def summarize(
    rows: list[dict[str, Any]], symmetric: bool = False, include_ambiguous: bool = False,
) -> dict[str, Any]:
    counts = {"11": 0, "10": 0, "01": 0, "00": 0}
    states = {"positive": 0, "ambiguous": 0, "refuted": 0, "other": 0}
    for row in rows:
        annotation = row.get("annotation") or {}
        positive = bool(annotation.get("hcr_has_contagious"))
        ambiguous = bool(annotation.get("contaminated_ambiguous"))
        refuted = bool(annotation.get("contaminated_refuted"))
        state = "positive" if positive else "ambiguous" if ambiguous else "refuted" if refuted else "other"
        states[state] += 1
        y1 = positive or (include_ambiguous and ambiguous)
        clean_key = "clean_rule_asserts_false" if symmetric else "clean_asserts_false"
        y0 = bool(annotation.get(clean_key))
        counts[f"{int(y1)}{int(y0)}"] += 1

    n = sum(counts.values())
    if not n:
        raise ValueError("annotation file contains no rows")
    p1 = (counts["11"] + counts["10"]) / n
    p0 = (counts["11"] + counts["01"]) / n
    p10 = counts["10"] / n
    p01 = counts["01"] / n
    return {
        "n": n,
        "counts": counts,
        "p1": p1,
        "p0": p0,
        "p10": p10,
        "p01": p01,
        "net_contrast": p10 - p01,
        "independence_benchmark": p1 * (1.0 - p0),
        "contaminated_state_counts": states,
        "contaminated_state_rates": {key: value / n for key, value in states.items()},
    }


def by_model(rows: list[dict[str, Any]], symmetric: bool, include_ambiguous: bool) -> dict[str, dict[str, Any]]:
    models = sorted({str(row.get("model", "unknown")) for row in rows})
    return {
        model: summarize(
            [row for row in rows if str(row.get("model", "unknown")) == model],
            symmetric=symmetric, include_ambiguous=include_ambiguous,
        )
        for model in models
    }


def by_seed_type(
    rows: list[dict[str, Any]], seed_type_by_claim: dict[str, str],
    symmetric: bool, include_ambiguous: bool,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        seed_type = seed_type_by_claim.get(str(row.get("seed_claim", "")), "unknown")
        grouped.setdefault(seed_type, []).append(row)
    return {
        seed_type: summarize(grouped[seed_type], symmetric, include_ambiguous)
        for seed_type in sorted(grouped)
    }


def bootstrap_net_gap_ci_clustered(
    injected_rows: list[dict[str, Any]], placebo_rows: list[dict[str, Any]],
    key: str, symmetric: bool, include_ambiguous: bool, n: int, seed: int,
) -> list[float] | None:
    """Percentile cluster-bootstrap CI for the injected-minus-placebo signed gap."""
    clean_key = "clean_rule_asserts_false" if symmetric else "clean_asserts_false"

    def grouped(rows: list[dict[str, Any]]) -> dict[Any, list[int]]:
        groups: dict[Any, list[int]] = {}
        for row in rows:
            annotation = row.get("annotation") or {}
            y1 = bool(annotation.get("hcr_has_contagious")) or (
                include_ambiguous and bool(annotation.get("contaminated_ambiguous"))
            )
            groups.setdefault(row.get(key), []).append(int(y1) - int(bool(annotation.get(clean_key))))
        return groups

    injected, placebo = grouped(injected_rows), grouped(placebo_rows)
    keys = sorted(set(injected) & set(placebo), key=str)
    if not keys:
        return None
    rng = random.Random(seed)
    gaps = []
    for _ in range(n):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        injected_values = [value for cluster in drawn for value in injected[cluster]]
        placebo_values = [value for cluster in drawn for value in placebo[cluster]]
        gaps.append(
            sum(injected_values) / len(injected_values)
            - sum(placebo_values) / len(placebo_values)
        )
    gaps.sort()
    return [round(gaps[int(0.025 * n)], 4), round(gaps[min(n - 1, int(0.975 * n))], 4)]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_bank = json.loads(resolve_project_path(config["paths"]["seed_bank"]).read_text(encoding="utf-8"))
    seed_type_by_claim = {str(seed["claim"]): str(seed["seed_type"]) for seed in seed_bank["seeds"]}
    injected_path = resolve_project_path(args.injected)
    placebo_path = resolve_project_path(args.placebo)
    injected_rows = list(read_jsonl(injected_path))
    placebo_rows = list(read_jsonl(placebo_path))
    bootstrap_samples = int(config["pilot"]["ccr"]["bootstrap_samples"])
    bootstrap_seed = int(config["random_seed"])

    results: dict[str, Any] = {}
    for scoring, symmetric, include_ambiguous in (
        ("current", False, False),
        ("symmetric_rule_only", True, False),
        ("ambiguity_inclusive", False, True),
    ):
        injected = summarize(injected_rows, symmetric, include_ambiguous)
        placebo = summarize(placebo_rows, symmetric, include_ambiguous)
        results[scoring] = {
            "injected": injected,
            "placebo": placebo,
            "injected_by_model": by_model(injected_rows, symmetric, include_ambiguous),
            "placebo_by_model": by_model(placebo_rows, symmetric, include_ambiguous),
            "injected_by_seed_type": by_seed_type(
                injected_rows, seed_type_by_claim, symmetric, include_ambiguous,
            ),
            "placebo_by_seed_type": by_seed_type(
                placebo_rows, seed_type_by_claim, symmetric, include_ambiguous,
            ),
            "injected_minus_placebo": {
                "p10_gap": injected["p10"] - placebo["p10"],
                "net_contrast_gap": injected["net_contrast"] - placebo["net_contrast"],
                "net_contrast_gap_ci95_seed_clustered": bootstrap_net_gap_ci_clustered(
                    injected_rows, placebo_rows, "seed_claim", symmetric, include_ambiguous,
                    bootstrap_samples, bootstrap_seed,
                ),
                "net_contrast_gap_ci95_model_clustered": bootstrap_net_gap_ci_clustered(
                    injected_rows, placebo_rows, "model", symmetric, include_ambiguous,
                    bootstrap_samples, bootstrap_seed,
                ),
            },
        }

    report = {
        "definition": {
            "y1": "annotation.hcr_has_contagious",
            "y1_ambiguity_inclusive": "annotation.hcr_has_contagious OR annotation.contaminated_ambiguous",
            "y0_current": "annotation.clean_asserts_false",
            "y0_symmetric": "annotation.clean_rule_asserts_false",
            "p10": "P(Y1=1, Y0=0), the directed discordance reported as CCR",
            "p01": "P(Y1=0, Y0=1), the reverse discordance",
            "net_contrast": "p10-p01, equal to p1-p0",
            "independence_benchmark": "p1*(1-p0)",
        },
        "inputs": {
            "config": to_project_relative(resolve_project_path(args.config)),
            "injected": to_project_relative(injected_path),
            "placebo": to_project_relative(placebo_path),
        },
        "bootstrap": {
            "method": "percentile cluster bootstrap",
            "replicates": bootstrap_samples,
            "random_seed": bootstrap_seed,
        },
        "results": results,
    }
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    compact = {}
    for scoring, values in results.items():
        compact[scoring] = {
            arm: {
                "p10": round(values[arm]["p10"], 4),
                "p01": round(values[arm]["p01"], 4),
                "net": round(values[arm]["net_contrast"], 4),
            }
            for arm in ("injected", "placebo")
        }
        compact[scoring]["gap"] = {
            "p10": round(values["injected_minus_placebo"]["p10_gap"], 4),
            "net": round(values["injected_minus_placebo"]["net_contrast_gap"], 4),
        }
    print(json.dumps({"wrote": to_project_relative(output_path), "results": compact}, ensure_ascii=False))


if __name__ == "__main__":
    main()
