#!/usr/bin/env python3
"""Freeze and export the 220-row author packet for the U4 self-induced analysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lib"), str(ROOT / "auto_labeling")]

from auto_label import label_selfinduced  # noqa: E402
from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import file_sha256, object_sha256  # noqa: E402


def select_u4_rows(
    primary_rows: list[dict[str, Any]],
    extension_rows: list[dict[str, Any]],
    bank_qids: list[str],
    primary_qids: list[str],
    primary_models: list[str],
    extension_models: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Select old-model remaining-20 plus all 60 questions for the new models."""
    if len(bank_qids) != 60 or len(primary_qids) != 40:
        raise ValueError("U4 is frozen as a 60-question bank with a 40-question primary cohort")
    if len(bank_qids) != len(set(bank_qids)) or len(primary_qids) != len(set(primary_qids)):
        raise ValueError("question IDs must be unique")
    if not set(primary_qids) < set(bank_qids):
        raise ValueError("primary question IDs must be a proper subset of the 60-question bank")
    if set(primary_models) & set(extension_models):
        raise ValueError("primary and extension model pools overlap")

    def require_complete(rows: list[dict[str, Any]], models: list[str], label: str) -> None:
        keys = [(row.get("q_id"), row.get("model")) for row in rows]
        expected = {(q, model) for q in bank_qids for model in models}
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(
                f"{label} full-bank coverage mismatch: rows={len(keys)} "
                f"unique={len(set(keys))} missing={len(expected - set(keys))} "
                f"unexpected={len(set(keys) - expected)}"
            )

    require_complete(primary_rows, primary_models, "primary")
    require_complete(extension_rows, extension_models, "extension")
    remaining = set(bank_qids) - set(primary_qids)
    selected = [row for row in primary_rows if row["q_id"] in remaining] + extension_rows
    expected_n = len(primary_models) * 20 + len(extension_models) * 60
    if len(selected) != expected_n:
        raise ValueError(f"U4 selection has {len(selected)} rows, expected {expected_n}")
    if any(row.get("source_mock") is True for row in selected):
        raise ValueError("U4 packet contains mock model output")
    task_ids = [row.get("task_id") for row in selected]
    if None in task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("U4 task IDs are missing or duplicated")

    model_order = {model: i for i, model in enumerate(primary_models + extension_models)}
    q_order = {qid: i for i, qid in enumerate(bank_qids)}
    selected.sort(key=lambda row: (model_order[row["model"]], q_order[row["q_id"]]))
    cohorts = {
        row["task_id"]: (
            "primary_models_remaining_20"
            if row["model"] in primary_models
            else "extension_models_full_60"
        )
        for row in selected
    }
    return selected, cohorts


def immutable_row_sha256(row: dict[str, Any]) -> str:
    """Bind model content and machine hints while allowing only human-label edits."""
    fixed = {key: value for key, value in row.items() if key != "annotation"}
    annotation = row.get("annotation") or {}
    fixed["machine_annotation"] = {
        key: annotation.get(key)
        for key in (
            "auto_labeled", "detector", "needs_nli",
            "nli_disabled", "turn1_false_suggested", "turn1_abstained",
            "nli_turn1_vs_truth",
        )
    }
    return object_sha256(fixed)


def _full_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, Path]:
    run_dir = (
        resolve_project_path(config["paths"]["runs_dir"])
        / config["pilot"].get("run_name", "pilot")
    )
    tasks_path = run_dir / "selfinduced_stage2_tasks.jsonl"
    return label_selfinduced(config, run_dir, list(read_jsonl(tasks_path))), run_dir, tasks_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-config", default="configs/zh_selfinduced.json")
    parser.add_argument("--extension-config", default="configs/zh_selfinduced_ext.json")
    parser.add_argument(
        "--packet",
        default="datasets_zh/runs/zh_selfinduced_ext/selfinduced_u4_author_packet.jsonl",
    )
    parser.add_argument("--design", default="reports_zh/SELFINDUCED_U4_FROZEN.json")
    args = parser.parse_args()

    packet_path, design_path = resolve_project_path(args.packet), resolve_project_path(args.design)
    existing = [str(path) for path in (packet_path, design_path) if path.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite frozen U4 artifacts: {existing}")

    primary_config = load_config(args.primary_config)
    extension_config = load_config(args.extension_config)
    if primary_config["paths"]["selfinduced_bank"] != extension_config["paths"]["selfinduced_bank"]:
        raise SystemExit("primary and extension configs use different self-induced banks")
    primary_qids = primary_config["pilot"]["annotation"]["primary_question_ids"]
    if primary_qids != extension_config["pilot"]["annotation"]["primary_question_ids"]:
        raise SystemExit("primary and extension configs use different primary question cohorts")
    bank_path = resolve_project_path(primary_config["paths"]["selfinduced_bank"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_qids = [row["q_id"] for row in bank.get("questions", [])]
    primary_rows, primary_run, primary_tasks = _full_rows(primary_config)
    extension_rows, extension_run, extension_tasks = _full_rows(extension_config)
    primary_models = primary_config["pilot"]["model_pool"]
    extension_models = extension_config["pilot"]["model_pool"]
    selected, cohorts = select_u4_rows(
        primary_rows, extension_rows, bank_qids, primary_qids,
        primary_models, extension_models,
    )

    write_jsonl(packet_path, selected)
    row_hashes = {row["task_id"]: immutable_row_sha256(row) for row in selected}
    by_model = Counter(row["model"] for row in selected)
    design = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "U4 self-induced extension frozen before author views the remaining-20 results",
        "analysis_rule": {
            "primary": "original 5 models x frozen primary 40 questions (unchanged)",
            "preregistered_extension": "7 models x the same frozen primary 40 questions",
            "sensitivity": "7 models x all 60 bank questions",
            "author_packet": "old 5 models x remaining 20 plus new 2 models x all 60",
            "estimand": "among human-confirmed factual errors in turn 1, fraction whose turn 2 inherits the same error",
            "uncertain": "report as completed but exclude from binary numerator and denominator",
        },
        "packet_path": to_project_relative(packet_path),
        "packet_sha256_before_annotation": file_sha256(packet_path),
        "immutable_rows_sha256": object_sha256(row_hashes),
        "immutable_row_sha256": row_hashes,
        "cohort_by_task_id": cohorts,
        "rows": len(selected),
        "by_model": dict(sorted(by_model.items())),
        "primary_question_ids": primary_qids,
        "remaining_question_ids": [qid for qid in bank_qids if qid not in set(primary_qids)],
        "model_pools": {"primary": primary_models, "extension": extension_models},
        "inputs": {
            "primary_config": {"path": args.primary_config, "sha256": file_sha256(resolve_project_path(args.primary_config))},
            "extension_config": {"path": args.extension_config, "sha256": file_sha256(resolve_project_path(args.extension_config))},
            "bank": {"path": to_project_relative(bank_path), "sha256": file_sha256(bank_path)},
            "primary_tasks": {"path": to_project_relative(primary_tasks), "sha256": file_sha256(primary_tasks)},
            "primary_outputs": {"path": to_project_relative(primary_run / "selfinduced_stage2_outputs.jsonl"), "sha256": file_sha256(primary_run / "selfinduced_stage2_outputs.jsonl")},
            "extension_tasks": {"path": to_project_relative(extension_tasks), "sha256": file_sha256(extension_tasks)},
            "extension_outputs": {"path": to_project_relative(extension_run / "selfinduced_stage2_outputs.jsonl"), "sha256": file_sha256(extension_run / "selfinduced_stage2_outputs.jsonl")},
            "preparer": {"path": to_project_relative(Path(__file__)), "sha256": file_sha256(Path(__file__))},
            "detector_freeze": {"path": "reports_zh/DETECTOR_FROZEN.json", "sha256": file_sha256(ROOT / "reports_zh/DETECTOR_FROZEN.json")},
        },
    }
    design_path.parent.mkdir(parents=True, exist_ok=True)
    design_path.write_text(json.dumps(design, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"packet": str(packet_path), "design": str(design_path), "rows": len(selected), "by_model": dict(sorted(by_model.items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
