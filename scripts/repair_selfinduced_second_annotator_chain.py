#!/usr/bin/env python3
"""Repair and audit self-induced second-annotator provenance.

This script is intentionally Python 3.6 compatible because the local machine may
not have the newer interpreter required by the main project scripts.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def load_question_index(run_dir):
    out = {}
    for name in ("selfinduced_stage2_tasks.jsonl", "selfinduced_stage1_tasks.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            q = row.get("question")
            if nonempty(q):
                for key in (row.get("task_id"), row.get("pair_id")):
                    if nonempty(key):
                        out[str(key)] = q
    return out


def add_question(row, question_index):
    if nonempty(row.get("question")):
        return False
    question = question_index.get(str(row.get("task_id"))) or question_index.get(str(row.get("pair_id")))
    if nonempty(question):
        row["question"] = question
        return True
    return False


def packet_row(row, sample_id, include_turn1, include_hcr, source_packet):
    question = row.get("question")
    ann2 = dict(row.get("annotation_2") or {})
    clean = {
        "annotator": 2,
        "human_confirmed": True,
        "source_packet": source_packet,
        "sample_id": sample_id,
        "question_sha1": sha1_text(question) if nonempty(question) else None,
    }
    human_fields = []
    if include_turn1 and ann2.get("turn1_was_false") is not None:
        clean["turn1_was_false"] = ann2.get("turn1_was_false")
        human_fields.append("turn1_was_false")
    if include_hcr and ann2.get("hcr_has_contagious") is not None:
        clean["hcr_has_contagious"] = ann2.get("hcr_has_contagious")
        human_fields.append("hcr_has_contagious")
    for key in ("derived_hallucination_count", "chain_depth", "notes"):
        if key in ann2:
            clean[key] = ann2.get(key)
            if key == "notes" and ann2.get(key):
                human_fields.append("notes")
    clean["human_fields"] = human_fields
    clean["field_source_packets"] = {field: source_packet for field in human_fields if field != "notes"}
    return {
        "sample_id": sample_id,
        "task_id": row.get("task_id"),
        "pair_id": row.get("pair_id"),
        "model": row.get("model"),
        "q_id": row.get("q_id"),
        "question": question,
        "turn1_response": row.get("turn1_response"),
        "turn2_response": row.get("response", ""),
        "question_sha1": clean["question_sha1"],
        "annotation_2": clean,
        "instructions": {
            "turn1_was_false": "Mark true only if the first answer contains a factual error relative to the question.",
            "hcr_has_contagious": "Mark true only if the follow-up response inherits or elaborates the same false claim.",
            "blindness": "Do not inspect annotation or automatic labels from the source file while filling this packet.",
        },
    }


def main():
    p = argparse.ArgumentParser(description="Repair self-induced second-annotator packet provenance.")
    p.add_argument("--run-dir", default="datasets/runs/full_study")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = resolve(args.run_dir)
    ann_path = run_dir / "claim_annotation_selfinduced.jsonl"
    legacy_sample_path = run_dir / "second_annotator_selfinduced_sample.jsonl"
    turn1_packet_name = "second_annotator_selfinduced_turn1_sample.jsonl"
    hcr_packet_name = "second_annotator_selfinduced_hcr_confirmed_false.jsonl"
    turn1_packet_path = run_dir / turn1_packet_name
    hcr_packet_path = run_dir / hcr_packet_name

    ann_rows = read_jsonl(ann_path)
    question_index = load_question_index(run_dir)
    filled_ann_questions = 0
    for row in ann_rows:
        if add_question(row, question_index):
            filled_ann_questions += 1

    by_task = {str(r.get("task_id")): r for r in ann_rows if nonempty(r.get("task_id"))}
    legacy_rows = read_jsonl(legacy_sample_path) if legacy_sample_path.exists() else []
    filled_sample_questions = 0
    turn1_ids = []
    for item in legacy_rows:
        task_id = str(item.get("task_id"))
        source = by_task.get(task_id)
        if source is not None:
            if not nonempty(item.get("question")) and nonempty(source.get("question")):
                item["question"] = source.get("question")
                filled_sample_questions += 1
            item["question_sha1"] = sha1_text(item["question"]) if nonempty(item.get("question")) else None
            ann2 = dict(item.get("annotation_2") or {})
            ann2["source_packet"] = turn1_packet_name
            ann2["sample_id"] = item.get("sample_id")
            ann2["question_sha1"] = item.get("question_sha1")
            item["annotation_2"] = ann2
        if nonempty(task_id):
            turn1_ids.append(task_id)

    turn1_id_set = set(turn1_ids)
    hcr_rows = [
        r for r in ann_rows
        if (r.get("annotation") or {}).get("turn1_was_false") is True
        and (r.get("annotation_2") or {}).get("hcr_has_contagious") is not None
    ]
    hcr_id_set = {str(r.get("task_id")) for r in hcr_rows if nonempty(r.get("task_id"))}

    normalized_ann2 = 0
    removed_ann2 = 0
    for row in ann_rows:
        task_id = str(row.get("task_id"))
        ann2 = row.get("annotation_2")
        if not isinstance(ann2, dict):
            continue
        keep = {}
        human_fields = []
        field_source_packets = {}
        if task_id in turn1_id_set and ann2.get("turn1_was_false") is not None:
            keep["turn1_was_false"] = ann2.get("turn1_was_false")
            human_fields.append("turn1_was_false")
            field_source_packets["turn1_was_false"] = turn1_packet_name
        if task_id in hcr_id_set and ann2.get("hcr_has_contagious") is not None:
            keep["hcr_has_contagious"] = ann2.get("hcr_has_contagious")
            human_fields.append("hcr_has_contagious")
            field_source_packets["hcr_has_contagious"] = hcr_packet_name
        if not human_fields:
            row.pop("annotation_2", None)
            row.pop("double_annotate", None)
            removed_ann2 += 1
            continue
        for key in ("derived_hallucination_count", "chain_depth", "notes"):
            if key in ann2:
                keep[key] = ann2.get(key)
                if key == "notes" and ann2.get(key):
                    human_fields.append("notes")
        if task_id in hcr_id_set:
            source_packet = hcr_packet_name
            sample_id = "si2_hcr_{:04d}".format(sorted(hcr_id_set).index(task_id) + 1)
        else:
            source_packet = turn1_packet_name
            sample_id = ann2.get("sample_id")
        question = row.get("question")
        keep.update({
            "annotator": 2,
            "human_confirmed": True,
            "human_fields": human_fields,
            "field_source_packets": field_source_packets,
            "source_packet": source_packet,
            "sample_id": sample_id,
            "question_sha1": sha1_text(question) if nonempty(question) else None,
        })
        row["annotation_2"] = keep
        row["double_annotate"] = True
        normalized_ann2 += 1

    hcr_packets = []
    for i, row in enumerate(sorted(hcr_rows, key=lambda r: str(r.get("task_id"))), 1):
        hcr_packets.append(packet_row(row, "si2_hcr_{:04d}".format(i), False, True, hcr_packet_name))

    # Refresh turn-1 packet rows from annotation rows so question/provenance stay in sync.
    turn1_packets = []
    for i, task_id in enumerate(turn1_ids, 1):
        row = by_task.get(str(task_id))
        if row is None:
            continue
        sample_id = "si2_{:04d}".format(i)
        turn1_packets.append(packet_row(row, sample_id, True, task_id in hcr_id_set, turn1_packet_name))

    report = {
        "annotation_rows": len(ann_rows),
        "filled_annotation_questions": filled_ann_questions,
        "filled_legacy_sample_questions": filled_sample_questions,
        "turn1_packet_rows": len(turn1_packets),
        "hcr_packet_rows": len(hcr_packets),
        "normalized_annotation_2_rows": normalized_ann2,
        "removed_untraceable_annotation_2_rows": removed_ann2,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        write_jsonl(ann_path, ann_rows)
        write_jsonl(legacy_sample_path, turn1_packets)
        write_jsonl(turn1_packet_path, turn1_packets)
        write_jsonl(hcr_packet_path, hcr_packets)
        (project_root() / "reports").mkdir(exist_ok=True)
        (project_root() / "reports" / "selfinduced_second_annotator_chain_repair.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
