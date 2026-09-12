#!/usr/bin/env python3
"""Compute frozen primary, 7x40 extension, and 7x60 U4 sensitivity results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from compute_ccr_metrics import pooled_selfinduced, summarize_selfinduced  # noqa: E402
from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import file_sha256  # noqa: E402


def require_cohort(rows: list[dict[str, Any]], qids: set[str], models: set[str], name: str) -> None:
    keys = [(row.get("q_id"), row.get("model")) for row in rows]
    expected = {(qid, model) for qid in qids for model in models}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            f"{name} coverage mismatch: rows={len(keys)} unique={len(set(keys))} "
            f"missing={len(expected - set(keys))} unexpected={len(set(keys) - expected)}"
        )
    for row in rows:
        ann = row.get("annotation") or {}
        human = set(ann.get("human_fields") or [])
        if "turn1_was_false" not in human or not isinstance(ann.get("turn1_was_false"), bool):
            raise ValueError(f"{name} has incomplete turn-1 label: {row.get('task_id')}")
        if ann["turn1_was_false"] is True and (
            "hcr_has_contagious" not in human or not isinstance(ann.get("hcr_has_contagious"), bool)
        ):
            raise ValueError(f"{name} has incomplete inheritance label: {row.get('task_id')}")


def result(rows: list[dict[str, Any]], seed: int, bootstrap_samples: int) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "per_model": summarize_selfinduced(rows),
        "pooled_question_clustered": pooled_selfinduced(rows, "q_id", bootstrap_samples, seed),
        "pooled_model_clustered": pooled_selfinduced(rows, "model", bootstrap_samples, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-config", default="configs/zh_selfinduced.json")
    parser.add_argument("--extension-config", default="configs/zh_selfinduced_ext.json")
    parser.add_argument("--primary", default="datasets_zh/runs/zh_selfinduced/claim_annotation_selfinduced.jsonl")
    parser.add_argument("--u4", default="datasets_zh/runs/zh_selfinduced_ext/selfinduced_u4_author_packet.jsonl")
    parser.add_argument("--validation", default="reports_zh/SELFINDUCED_U4_VALIDATION.json")
    parser.add_argument("--output", default="reports_zh/selfinduced_u4_results.json")
    parser.add_argument("--extension-rows", default="datasets_zh/runs/zh_selfinduced_ext/selfinduced_7x40.jsonl")
    parser.add_argument("--sensitivity-rows", default="datasets_zh/runs/zh_selfinduced_ext/selfinduced_7x60.jsonl")
    args = parser.parse_args()
    primary_path, u4_path = resolve_project_path(args.primary), resolve_project_path(args.u4)
    validation_path = resolve_project_path(args.validation)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("filled_packet_sha256") != file_sha256(u4_path) or validation.get("immutable_validation", {}).get("mismatches") != 0:
        raise SystemExit("U4 packet does not match its completed immutable-field validation")

    primary_config, extension_config = load_config(args.primary_config), load_config(args.extension_config)
    bank = json.loads(resolve_project_path(primary_config["paths"]["selfinduced_bank"]).read_text(encoding="utf-8"))
    all_qids = {row["q_id"] for row in bank["questions"]}
    primary_qids = set(primary_config["pilot"]["annotation"]["primary_question_ids"])
    old_models = set(primary_config["pilot"]["model_pool"])
    new_models = set(extension_config["pilot"]["model_pool"])
    primary_rows, u4_rows = list(read_jsonl(primary_path)), list(read_jsonl(u4_path))
    require_cohort(primary_rows, primary_qids, old_models, "frozen primary")
    require_cohort(
        [row for row in u4_rows if row.get("model") in old_models],
        all_qids - primary_qids, old_models, "U4 old-model remainder",
    )
    require_cohort(
        [row for row in u4_rows if row.get("model") in new_models],
        all_qids, new_models, "U4 new models",
    )
    extension_rows = primary_rows + [
        row for row in u4_rows
        if row.get("model") in new_models and row.get("q_id") in primary_qids
    ]
    sensitivity_rows = primary_rows + u4_rows
    write_jsonl(resolve_project_path(args.extension_rows), extension_rows)
    write_jsonl(resolve_project_path(args.sensitivity_rows), sensitivity_rows)
    seed = int(primary_config.get("random_seed", 0))
    n_boot = int(primary_config["pilot"]["ccr"].get("bootstrap_samples", 2000))
    report = {
        "primary_5x40": result(primary_rows, seed, n_boot),
        "preregistered_extension_7x40": result(extension_rows, seed, n_boot),
        "sensitivity_7x60": result(sensitivity_rows, seed, n_boot),
        "provenance": {
            "primary": {"path": args.primary, "sha256": file_sha256(primary_path)},
            "u4": {"path": args.u4, "sha256": file_sha256(u4_path)},
            "u4_validation": {"path": args.validation, "sha256": file_sha256(validation_path)},
            "script_sha256": file_sha256(Path(__file__)),
            "bootstrap_samples": n_boot,
        },
    }
    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {"rows": value["rows"], "pooled": value["pooled_question_clustered"]} for name, value in report.items() if name != "provenance"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
