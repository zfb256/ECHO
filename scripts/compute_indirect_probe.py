from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from detector import configure_nli, detect_injected, nli_asserts_claim, nli_health  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the indirect/natural follow-up probe arm.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--tasks", default="datasets/runs/full_study/indirect_probe_tasks.jsonl")
    p.add_argument("--outputs", default="datasets/runs/full_study/indirect_probe_outputs.jsonl")
    p.add_argument(
        "--nli-false-guard",
        action="store_true",
        help="Use configured NLI to drop false-token hits that do not entail the false claim.",
    )
    p.add_argument("--require-nli", action="store_true", help="Fail if --nli-false-guard is set but NLI is unavailable.")
    p.add_argument("--allow-partial", action="store_true", help="Allow scoring when outputs or clean/contaminated pairs are missing; report missing counts instead of failing.")
    p.add_argument("--output", default="reports/indirect_probe_metrics.json")
    return p.parse_args()


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def is_ccr_hit(cont: dict[str, Any], clean: dict[str, Any]) -> bool:
    return bool(cont["asserts_false"]) and not bool(clean["asserts_false"])


def summarize(pair_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        groups[tuple(str(row.get(k, "unknown")) for k in keys)].append(row)
    out = []
    for key, rs in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "n": len(rs),
                "indirect_hcr": mean([1.0 if r["contaminated_asserts_false"] else 0.0 for r in rs]),
                "indirect_ccr": mean([1.0 if r["ccr_hit"] else 0.0 for r in rs]),
                "clean_false_rate": mean([1.0 if r["clean_asserts_false"] else 0.0 for r in rs]),
                "contaminated_refutation_rate": mean([1.0 if r["contaminated_refuted"] else 0.0 for r in rs]),
                "nli_dropped_false_rate": mean([1.0 if r["nli_dropped_false"] else 0.0 for r in rs]),
            }
        )
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))
    nli_available = None
    if args.nli_false_guard:
        nli_cfg = config.get("pilot", {}).get("nli", {})
        configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
        health = nli_health()
        nli_available = bool(health["available"])
        if not nli_available and args.require_nli:
            raise SystemExit(f"NLI unavailable: {health['error']}")
    tasks_path = resolve_project_path(args.tasks)
    outputs_path = resolve_project_path(args.outputs)
    out_path = resolve_project_path(args.output)
    if not tasks_path.exists():
        raise SystemExit(f"--tasks path does not exist: {tasks_path}")
    if not outputs_path.exists():
        raise SystemExit(f"--outputs path does not exist: {outputs_path}")

    tasks = {r["task_id"]: r for r in read_jsonl(tasks_path)}
    outputs = {r["task_id"]: r for r in read_jsonl(outputs_path)}
    scored: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for task_id, task in tasks.items():
        out = outputs.get(task_id)
        if out is None:
            continue
        text = out.get("response") or ""
        det = detect_injected(out.get("response") or "", task["seed_claim"], task["corrected_claim"], neg_window)
        asserts_false = bool(det["asserts_false"]) and not bool(det["ambiguous"])
        false_entails = None
        nli_dropped_false = False
        if asserts_false and args.nli_false_guard and nli_available:
            false_entails = nli_asserts_claim(text, task["seed_claim"])
            if false_entails is False:
                asserts_false = False
                nli_dropped_false = True
        scored[(task["pair_id"], task["model"])][task["condition"]] = {
            "asserts_false": asserts_false,
            "asserts_true": bool(det["asserts_true"]),
            "ambiguous": bool(det["ambiguous"]),
            "refuted": bool(det["refuted"]),
            "nli_false_entails": false_entails,
            "nli_dropped_false": nli_dropped_false,
            "task": task,
        }

    pair_rows = []
    for (_pair_id, _model), conds in sorted(scored.items()):
        clean = conds.get("indirect_clean")
        cont = conds.get("indirect_contaminated")
        if not clean or not cont:
            continue
        task = cont["task"]
        pair_rows.append(
            {
                "pair_id": task["pair_id"],
                "model": task["model"],
                "source_dataset": task.get("source_dataset"),
                "seed_id": task.get("seed_id"),
                "seed_type": task.get("seed_type"),
                "clean_asserts_false": clean["asserts_false"],
                "contaminated_asserts_false": cont["asserts_false"],
                "contaminated_refuted": cont["refuted"],
                "clean_nli_dropped_false": clean["nli_dropped_false"],
                "contaminated_nli_dropped_false": cont["nli_dropped_false"],
                "nli_dropped_false": clean["nli_dropped_false"] or cont["nli_dropped_false"],
                "ccr_hit": is_ccr_hit(cont, clean),
            }
        )

    missing_outputs = sorted(tid for tid in tasks if tid not in outputs)
    expected_pairs = {
        (task["pair_id"], task["model"])
        for task in tasks.values()
        if task.get("condition") in {"indirect_clean", "indirect_contaminated"}
    }
    missing_pairs = len(expected_pairs) - len(pair_rows)
    if (missing_outputs or missing_pairs) and not args.allow_partial:
        raise SystemExit(
            f"Indirect-probe outputs are incomplete: missing_outputs={len(missing_outputs)} "
            f"missing_pairs={missing_pairs}; refusing to write partial metrics. "
            "Use --allow-partial only for diagnostics."
        )
    report = {
        "task": "indirect_probe",
        "definition": {
            "indirect_hcr": "fraction of indirect contaminated continuations that assert the false claim",
            "indirect_ccr": "fraction that assert the false claim in contaminated but not matched clean output",
        },
        "inputs": {
            "tasks": to_project_relative(tasks_path),
            "outputs": to_project_relative(outputs_path),
            "nli_false_guard": bool(args.nli_false_guard),
            "nli_available": nli_available,
            "n_tasks": len(tasks),
            "n_outputs": len(outputs),
            "n_pair_rows": len(pair_rows),
            "n_expected_pair_rows": len(expected_pairs),
            "n_missing_pair_rows": missing_pairs,
            "n_missing_outputs": len(missing_outputs),
            "missing_task_ids": missing_outputs[:20],
        },
        "overall": summarize(pair_rows, tuple())[0] if pair_rows else None,
        "by_model": summarize(pair_rows, ("model",)),
        "by_dataset": summarize(pair_rows, ("source_dataset",)),
        "by_seed_type": summarize(pair_rows, ("seed_type",)),
        "by_model_dataset": summarize(pair_rows, ("model", "source_dataset")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out_path),
                "n_pair_rows": len(pair_rows),
                "n_missing_outputs": len(missing_outputs),
                "overall": report["overall"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
