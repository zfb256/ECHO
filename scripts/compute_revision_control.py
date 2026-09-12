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
from detector import configure_nli, detect_injected, looks_like_abstention, nli_asserts_claim, nli_health  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score the ECHO revision-control arm.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument(
        "--tasks",
        default="datasets/runs/full_study/revision_control_tasks.jsonl",
        help="Revision-control task file.",
    )
    p.add_argument(
        "--outputs",
        default="datasets/runs/full_study/revision_control_outputs.jsonl",
        help="Generated outputs for the revision-control tasks.",
    )
    p.add_argument(
        "--nli-false-guard",
        action="store_true",
        help="Use configured NLI to drop false-token hits that do not entail the false claim.",
    )
    p.add_argument("--require-nli", action="store_true", help="Fail if --nli-false-guard is set but NLI is unavailable.")
    p.add_argument("--allow-partial", action="store_true", help="Allow scoring when outputs are missing; report n_missing_outputs instead of failing.")
    p.add_argument("--output", default="reports/revision_control_metrics.json")
    return p.parse_args()


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "unknown")) for k in keys)].append(row)

    out = []
    for key, rs in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "n": len(rs),
                "post_correction_persistence": mean([1.0 if r["asserts_false"] else 0.0 for r in rs]),
                "correction_success_rate": mean([1.0 if r["asserts_true"] else 0.0 for r in rs]),
                "clean_revision_rate": mean(
                    [1.0 if (r["asserts_true"] and not r["asserts_false"]) else 0.0 for r in rs]
                ),
                "ambiguous_rate": mean([1.0 if r["ambiguous"] else 0.0 for r in rs]),
                "refutation_rate": mean([1.0 if r["refuted_false_value"] else 0.0 for r in rs]),
                "abstention_rate": mean([1.0 if r["abstained"] else 0.0 for r in rs]),
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
    outputs = list(read_jsonl(outputs_path))
    scored = []
    missing = []
    for out in outputs:
        task_id = out.get("task_id")
        task = tasks.get(task_id)
        if task is None:
            continue
        text = out.get("response") or ""
        det = detect_injected(text, task["seed_claim"], task["corrected_claim"], neg_window)
        false_entails = None
        nli_dropped_false = False
        asserts_false = bool(det["asserts_false"])
        if asserts_false and args.nli_false_guard and nli_available:
            false_entails = nli_asserts_claim(text, task["seed_claim"])
            if false_entails is False:
                asserts_false = False
                nli_dropped_false = True
        scored.append(
            {
                "task_id": task_id,
                "pair_id": task.get("pair_id"),
                "model": task.get("model"),
                "source_dataset": task.get("source_dataset"),
                "seed_id": task.get("seed_id"),
                "seed_type": task.get("seed_type"),
                "asserts_false": asserts_false,
                "asserts_true": bool(det["asserts_true"]),
                "ambiguous": bool(det["ambiguous"]) and asserts_false,
                "refuted_false_value": bool(det["refuted"]),
                "abstained": looks_like_abstention(text),
                "nli_false_entails": false_entails,
                "nli_dropped_false": nli_dropped_false,
                "false_tokens": det.get("false_tokens", []),
                "true_tokens": det.get("true_tokens", []),
            }
        )
    output_ids = {r.get("task_id") for r in outputs}
    missing = sorted(tid for tid in tasks if tid not in output_ids)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"{len(missing)} revision-control outputs are missing; refusing to write partial metrics. "
            "Use --allow-partial only for diagnostics."
        )

    report = {
        "task": "revision_control",
        "definition": {
            "post_correction_persistence": "fraction of correction-control outputs that still assert the injected false claim",
            "correction_success_rate": "fraction that assert the corrected claim",
            "clean_revision_rate": "fraction that assert the corrected claim without also plainly asserting the false claim",
        },
        "inputs": {
            "tasks": to_project_relative(tasks_path),
            "outputs": to_project_relative(outputs_path),
            "nli_false_guard": bool(args.nli_false_guard),
            "nli_available": nli_available,
            "n_tasks": len(tasks),
            "n_outputs": len(outputs),
            "n_scored": len(scored),
            "n_missing_outputs": len(missing),
            "missing_task_ids": missing[:20],
        },
        "overall": summarize(scored, tuple())[0] if scored else None,
        "by_model": summarize(scored, ("model",)),
        "by_dataset": summarize(scored, ("source_dataset",)),
        "by_seed_type": summarize(scored, ("seed_type",)),
        "by_model_dataset": summarize(scored, ("model", "source_dataset")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out_path),
                "n_scored": len(scored),
                "n_missing_outputs": len(missing),
                "overall": report["overall"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
