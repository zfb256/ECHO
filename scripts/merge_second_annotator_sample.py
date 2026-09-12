from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl, write_jsonl


ALLOWED_FIELDS = {
    "turn1_was_false",
    "hcr_has_contagious",
    "derived_hallucination_count",
    "chain_depth",
    "notes",
    "source_packet",
    "sample_id",
    "question_sha1",
    "turn1_was_false_uncertain",
    "hcr_has_contagious_uncertain",
}
IMMUTABLE_FIELDS = (
    "pair_id", "model", "q_id", "question", "truth_statement",
    "evidence_hint", "turn1_response",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a filled blind second-annotator packet back into annotation_2.")
    parser.add_argument("--config", default="configs/zh_selfinduced.json")
    parser.add_argument("--annotations", default=None, help="Defaults to runs_dir/<run>/claim_annotation_selfinduced.jsonl")
    parser.add_argument("--sample", default=None, help="Defaults to runs_dir/<run>/second_annotator_selfinduced_turn1_sample.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def clean_annotation(raw: dict[str, Any]) -> dict[str, Any]:
    ann = {k: raw.get(k) for k in ALLOWED_FIELDS if k in raw}
    human_fields = []
    for k in ("turn1_was_false", "hcr_has_contagious"):
        if ann.get(k) is not None or ann.get(f"{k}_uncertain") is True:
            human_fields.append(k)
    if human_fields:
        ann["human_fields"] = human_fields
        ann["human_confirmed"] = True
    ann["annotator"] = 2
    return ann


def merge_annotation(existing: dict[str, Any], incoming: dict[str, Any], packet_name: str) -> dict[str, Any]:
    merged = dict(existing)
    human_fields = set(merged.get("human_fields") or [])
    field_sources = dict(merged.get("field_source_packets") or {})
    for field in ("turn1_was_false", "hcr_has_contagious"):
        uncertain_key = f"{field}_uncertain"
        if incoming.get(field) is not None or incoming.get(uncertain_key) is True:
            merged[field] = incoming[field]
            merged[uncertain_key] = bool(incoming.get(uncertain_key))
            human_fields.add(field)
            field_sources[field] = packet_name
    for field in ("derived_hallucination_count", "chain_depth", "notes"):
        if field in incoming:
            merged[field] = incoming[field]
            if field == "notes" and incoming.get(field):
                human_fields.add(field)
    for field in ("sample_id", "question_sha1"):
        if incoming.get(field) is not None:
            merged[field] = incoming[field]
    merged["annotator"] = 2
    if human_fields:
        merged["human_confirmed"] = True
    merged["human_fields"] = sorted(human_fields)
    merged["field_source_packets"] = field_sources
    merged["source_packet"] = packet_name
    return merged


def required_review_fields(item: dict[str, Any], ann2: dict[str, Any]) -> set[str]:
    required = set(item.get("double_annotate_fields") or [])
    if ann2.get("turn1_was_false") is not True:
        required.discard("hcr_has_contagious")
    return required


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    ann_path = resolve_project_path(args.annotations) if args.annotations else run_dir / "claim_annotation_selfinduced.jsonl"
    sample_path = (
        resolve_project_path(args.sample)
        if args.sample
        else run_dir / "second_annotator_selfinduced_turn1_sample.jsonl"
    )

    rows = list(read_jsonl(ann_path))
    task_ids = [r.get("task_id") for r in rows]
    if any(not task_id for task_id in task_ids) or len(set(task_ids)) != len(task_ids):
        raise SystemExit("Main self-induced annotation file has missing or duplicate task_id values.")
    by_task = {r.get("task_id"): r for r in rows}
    merged = 0
    incomplete = 0
    missing = []
    sample_rows = list(read_jsonl(sample_path))
    sample_ids = [item.get("task_id") for item in sample_rows]
    if any(not task_id for task_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise SystemExit("Blind packet has missing or duplicate task_id values.")
    for item in sample_rows:
        task_id = item.get("task_id")
        if task_id not in by_task:
            missing.append(task_id)
            continue
        ann2 = clean_annotation(item.get("annotation_2", {}))
        if item.get("sample_id") is not None:
            ann2["sample_id"] = item.get("sample_id")
        if item.get("question_sha1") is not None:
            ann2["question_sha1"] = item.get("question_sha1")
        required_fields = required_review_fields(item, ann2)
        reviewed_fields = set(ann2.get("human_fields") or [])
        if not required_fields or not required_fields.issubset(reviewed_fields):
            incomplete += 1
        row = by_task[task_id]
        for field in IMMUTABLE_FIELDS:
            if item.get(field) != row.get(field):
                raise SystemExit(
                    f"{field} mismatch for task_id={task_id}: sample and annotation file differ"
                )
        if item.get("turn2_response") != row.get("response"):
            raise SystemExit(
                f"turn2_response mismatch for task_id={task_id}: sample and annotation file differ"
            )
        if item.get("response") != row.get("response"):
            raise SystemExit(
                f"response mismatch for task_id={task_id}: sample and annotation file differ"
            )
        if item.get("question_sha1") != hashlib.sha1(
            (item.get("question") or "").encode("utf-8")
        ).hexdigest():
            raise SystemExit(f"question_sha1 mismatch in blind packet for task_id={task_id}")
        row["double_annotate"] = True
        row["annotation_2"] = merge_annotation(row.get("annotation_2", {}), ann2, sample_path.name)
        merged += 1

    if missing or incomplete:
        raise SystemExit(
            "Refusing to merge an incomplete blind packet: "
            f"missing_from_main={len(missing)} incomplete_rows={incomplete}. "
            "Finish every assigned judgment (Uncertain is allowed) and retry."
        )
    if not args.dry_run:
        write_jsonl(ann_path, rows)

    print(json.dumps({
        "annotations": str(ann_path),
        "sample": str(sample_path),
        "merged": merged,
        "incomplete": incomplete,
        "missing": missing,
        "dry_run": bool(args.dry_run),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
