from __future__ import annotations

import argparse
import json
import math
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
    p = argparse.ArgumentParser(description="Score self-induced correction-control stage-3 outputs.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument(
        "--tasks",
        default="datasets/runs/full_study/selfinduced_correction_control_tasks.jsonl",
    )
    p.add_argument(
        "--outputs",
        default="datasets/runs/full_study/selfinduced_correction_control_outputs.jsonl",
    )
    p.add_argument("--require-nli", action="store_true", help="Fail if configured NLI is unavailable.")
    p.add_argument("--allow-partial", action="store_true", help="Allow scoring when outputs are missing.")
    p.add_argument("--output", default="reports/selfinduced_correction_control_metrics.json")
    return p.parse_args()


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "unknown")) for k in keys)].append(row)

    out = []
    for key, rs in sorted(groups.items()):
        n = len(rs)
        persistence_n = sum(1 for r in rs if r["post_correction_persistence"])
        success_n = sum(1 for r in rs if r["correction_success"])
        raw_n = sum(1 for r in rs if r["raw_false_span_flag"])
        clean_n = sum(1 for r in rs if r["correction_success"] and not r["post_correction_persistence"])
        item = {k: v for k, v in zip(keys, key)}
        item.update(
            {
                "n": n,
                "post_correction_persistence": mean([1.0 if r["post_correction_persistence"] else 0.0 for r in rs]),
                "post_correction_persistence_n": persistence_n,
                "post_correction_persistence_ci95_wilson": wilson(persistence_n, n),
                "correction_success_rate": mean([1.0 if r["correction_success"] else 0.0 for r in rs]),
                "correction_success_n": success_n,
                "correction_success_ci95_wilson": wilson(success_n, n),
                "clean_revision_rate": mean(
                    [1.0 if (r["correction_success"] and not r["post_correction_persistence"]) else 0.0 for r in rs]
                ),
                "clean_revision_n": clean_n,
                "clean_revision_ci95_wilson": wilson(clean_n, n),
                "raw_false_span_flag_rate": mean([1.0 if r["raw_false_span_flag"] else 0.0 for r in rs]),
                "raw_false_span_flag_n": raw_n,
                "raw_false_span_flag_ci95_wilson": wilson(raw_n, n),
                "nli_dropped_false_rate": mean([1.0 if r["nli_dropped_false"] else 0.0 for r in rs]),
                "refutation_rate": mean([1.0 if r["refuted_false_value"] else 0.0 for r in rs]),
                "abstention_rate": mean([1.0 if r["abstained"] else 0.0 for r in rs]),
            }
        )
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    nli_cfg = config.get("pilot", {}).get("nli", {})
    configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
    health = nli_health()
    nli_available = bool(health["available"])
    if args.require_nli and not nli_available:
        raise SystemExit(f"NLI unavailable: {health['error']}")

    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))
    tasks_path = resolve_project_path(args.tasks)
    outputs_path = resolve_project_path(args.outputs)
    out_path = resolve_project_path(args.output)
    if not tasks_path.exists():
        raise SystemExit(f"--tasks path does not exist: {tasks_path}")
    if not outputs_path.exists():
        raise SystemExit(f"--outputs path does not exist: {outputs_path}")

    tasks = {r["task_id"]: r for r in read_jsonl(tasks_path)}
    outputs = list(read_jsonl(outputs_path))
    output_ids = {r.get("task_id") for r in outputs}
    missing = sorted(tid for tid in tasks if tid not in output_ids)
    if missing and not args.allow_partial:
        raise SystemExit(
            f"{len(missing)} self-induced correction-control outputs are missing; refusing to write partial metrics. "
            "Use --allow-partial only for diagnostics."
        )

    scored = []
    for out in outputs:
        task_id = out.get("task_id")
        task = tasks.get(task_id)
        if task is None:
            continue
        text = out.get("response") or ""
        turn1_claim = task["turn1_response"]
        truth = task["truth_statement"]
        det = detect_injected(text, turn1_claim, truth, neg_window)
        raw_false = bool(det["asserts_false"])
        false_entails = None
        nli_dropped_false = False
        persistence = raw_false
        if raw_false and nli_available:
            false_entails = nli_asserts_claim(text, turn1_claim)
            if false_entails is False:
                persistence = False
                nli_dropped_false = True
        truth_entails = nli_asserts_claim(text, truth) if nli_available else None
        success = bool(truth_entails) if nli_available else bool(det["asserts_true"])
        scored.append(
            {
                "task_id": task_id,
                "source_task_id": task.get("source_task_id"),
                "model": task.get("model"),
                "q_id": task.get("q_id"),
                "raw_false_span_flag": raw_false,
                "post_correction_persistence": persistence,
                "correction_success": success,
                "refuted_false_value": bool(det["refuted"]),
                "abstained": looks_like_abstention(text),
                "nli_false_entails": false_entails,
                "nli_truth_entails": truth_entails,
                "nli_dropped_false": nli_dropped_false,
                "false_tokens": det.get("false_tokens", []),
                "true_tokens": det.get("true_tokens", []),
            }
        )

    report = {
        "task": "selfinduced_correction_control",
        "definition": {
            "post_correction_persistence": "fraction of stage-3 outputs that still assert the original turn-1 false answer after an explicit correction",
            "correction_success_rate": "fraction of stage-3 outputs that entail the clean truth_statement after correction",
            "clean_revision_rate": "fraction that entail the truth_statement without still asserting the original false answer",
            "raw_false_span_flag_rate": "rule-detector false-span rate before NLI mention-vs-assertion filtering",
        },
        "inputs": {
            "tasks": to_project_relative(tasks_path),
            "outputs": to_project_relative(outputs_path),
            "nli_available": nli_available,
            "nli_model": health["model"],
            "nli_error": health["error"],
            "n_tasks": len(tasks),
            "n_outputs": len(outputs),
            "n_scored": len(scored),
            "n_missing_outputs": len(missing),
            "missing_task_ids": missing[:20],
            "negation_window": neg_window,
        },
        "overall": summarize(scored, tuple())[0] if scored else None,
        "by_model": summarize(scored, ("model",)),
        "scored_rows": scored,
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
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
