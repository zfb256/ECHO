#!/usr/bin/env python3
"""Python 3.6 compatible gate for self-induced second-annotator provenance."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cohen_kappa(a, b):
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    idx = {label: i for i, label in enumerate(labels)}
    conf = [[0 for _ in labels] for _ in labels]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    n = float(len(a))
    po = sum(conf[i][i] for i in range(len(labels))) / n
    row = [sum(conf[i]) / n for i in range(len(labels))]
    col = [sum(conf[i][j] for i in range(len(labels))) / n for j in range(len(labels))]
    pe = sum(row[i] * col[i] for i in range(len(labels)))
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def bool_int(value):
    if value is True:
        return 1
    if value is False:
        return 0
    return None


def compare(rows, ids, field, min_kappa, min_decidable_fraction=0.0):
    a_vals = []
    b_vals = []
    skipped = 0
    for row in rows:
        task_id = str(row.get("task_id"))
        if task_id not in ids:
            continue
        ann1 = row.get("annotation") or {}
        ann2 = row.get("annotation_2") or {}
        a = (
            bool_int(ann1.get(field))
            if field in set(ann1.get("human_fields") or []) else None
        )
        b = (
            bool_int(ann2.get(field))
            if field in set(ann2.get("human_fields") or []) else None
        )
        if a is None or b is None:
            skipped += 1
            continue
        a_vals.append(a)
        b_vals.append(b)
    kappa = cohen_kappa(a_vals, b_vals)
    min_decidable = int(math.ceil(len(ids) * min_decidable_fraction))
    agreement = None
    if a_vals:
        agreement = sum(1 for x, y in zip(a_vals, b_vals) if x == y) / float(len(a_vals))
    return {
        "field": field,
        "n_compared": len(a_vals),
        "skipped_incomplete": skipped,
        "percent_agreement": agreement,
        "cohen_kappa": kappa,
        "min_kappa_threshold": min_kappa,
        "min_decidable_fraction": min_decidable_fraction,
        "min_decidable_rows": min_decidable,
        "decidable_fraction": len(a_vals) / float(len(ids)) if ids else None,
        "go_kappa": (
            kappa is not None
            and kappa >= min_kappa
            and len(a_vals) >= min_decidable
        ),
    }


def packet_ids(path, errors):
    rows = read_jsonl(path)
    ids = []
    for i, row in enumerate(rows, 1):
        task_id = row.get("task_id")
        if not nonempty(task_id):
            errors.append("{} row {} missing task_id".format(path, i))
            continue
        if not nonempty(row.get("question")):
            errors.append("{} task_id={} missing question".format(path, task_id))
        ids.append(str(task_id))
    if len(ids) != len(set(ids)):
        errors.append("{} has duplicate task_id".format(path))
    return rows, set(ids)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def portable_path(path):
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.name


def main():
    p = argparse.ArgumentParser(description="Check self-induced second-annotator evidence chain.")
    p.add_argument("--config", default="configs/zh_selfinduced.json")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--output", default="reports_zh/selfinduced_second_annotator_chain_check.json")
    args = p.parse_args()

    turn1_target = 120
    hcr_target = None  # include every confirmed-false inheritance row
    min_kappa = 0.6
    min_decidable_fraction = 0.0
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    run_name = config.get("pilot", {}).get("run_name", "zh_selfinduced")
    runs_dir = resolve(config.get("paths", {}).get("runs_dir", "datasets_zh/runs"))
    run_dir = resolve(args.run_dir) if args.run_dir else runs_dir / run_name
    ann_cfg = config.get("pilot", {}).get("annotation", {})
    turn1_target = int(ann_cfg.get("turn1_second_annotator_n", turn1_target))
    hcr_target = int(ann_cfg.get("inheritance_second_annotator_n", 0)) or None
    min_kappa = float(config.get("pilot", {}).get("go_no_go", {}).get("min_kappa", min_kappa))
    min_decidable_fraction = float(
        ann_cfg.get("min_kappa_decidable_fraction", 0.0)
    )
    ann_path = run_dir / "claim_annotation_selfinduced.jsonl"
    joint_packet = run_dir / "second_annotator_selfinduced_sample.jsonl"
    turn1_packet = run_dir / "second_annotator_selfinduced_turn1_sample.jsonl"
    hcr_packet = run_dir / "second_annotator_selfinduced_hcr_confirmed_false.jsonl"
    joint_mode = joint_packet.exists()
    if joint_mode:
        turn1_packet = joint_packet
        hcr_packet = joint_packet
    errors = []

    required_paths = (ann_path, joint_packet) if joint_mode else (ann_path, turn1_packet, hcr_packet)
    for path in required_paths:
        if not path.exists():
            errors.append("missing {}".format(path))
    if errors:
        report = {"ok": False, "errors": errors}
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(1)

    ann_rows = read_jsonl(ann_path)
    by_task = {}
    for row in ann_rows:
        task_id = row.get("task_id")
        if nonempty(task_id):
            by_task[str(task_id)] = row
        if not nonempty(row.get("question")):
            errors.append("{} task_id={} missing question".format(ann_path, task_id))

    turn1_rows, turn1_ids = packet_ids(turn1_packet, errors)
    hcr_rows, hcr_ids = (
        (turn1_rows, set(turn1_ids))
        if joint_mode else packet_ids(hcr_packet, errors)
    )
    expected_turn1 = min(turn1_target, len(ann_rows))
    if len(turn1_rows) != expected_turn1:
        errors.append("{} rows={} expected={}".format(turn1_packet, len(turn1_rows), expected_turn1))

    confirmed_false_ids = set()
    for row in ann_rows:
        ann1 = row.get("annotation") or {}
        if (
            ann1.get("turn1_was_false") is True
            and "turn1_was_false" in set(ann1.get("human_fields") or [])
            and nonempty(row.get("task_id"))
        ):
            confirmed_false_ids.add(str(row.get("task_id")))
    if hcr_target is not None and len(confirmed_false_ids) < hcr_target:
        errors.append(
            "confirmed-false rows={} below fixed inheritance target={}".format(
                len(confirmed_false_ids), hcr_target
            )
        )
    if joint_mode:
        if not confirmed_false_ids.issubset(turn1_ids):
            errors.append(
                "joint packet omits {} annotator-1-confirmed false rows".format(
                    len(confirmed_false_ids - turn1_ids)
                )
            )
        hcr_ids = {
            task_id for task_id in confirmed_false_ids
            if (by_task.get(task_id, {}).get("annotation_2") or {}).get("turn1_was_false") is True
        }
    else:
        expected_hcr = hcr_target if hcr_target is not None else len(confirmed_false_ids)
        if len(hcr_ids) != expected_hcr:
            errors.append("HCR packet rows={} expected={}".format(len(hcr_ids), expected_hcr))
        if not hcr_ids.issubset(confirmed_false_ids):
            errors.append(
                "HCR packet contains {} rows not confirmed false by annotator 1".format(
                    len(hcr_ids - confirmed_false_ids)
                )
            )

    packet_groups = (("joint", turn1_rows),) if joint_mode else (("turn1", turn1_rows), ("hcr", hcr_rows))
    for packet_name, packet_rows in packet_groups:
        for item in packet_rows:
            task_id = str(item.get("task_id"))
            source = by_task.get(task_id)
            if source is None:
                errors.append("{} packet task_id={} missing from annotation file".format(packet_name, task_id))
                continue
            for field in (
                "pair_id", "model", "q_id", "question", "truth_statement",
                "evidence_hint", "turn1_response",
            ):
                if item.get(field) != source.get(field):
                    errors.append(
                        "{} packet task_id={} {} mismatch".format(packet_name, task_id, field)
                    )
            if item.get("turn2_response") != source.get("response"):
                errors.append("{} packet task_id={} turn2_response mismatch".format(packet_name, task_id))
            if item.get("response") != source.get("response"):
                errors.append("{} packet task_id={} response mismatch".format(packet_name, task_id))
            qhash = item.get("question_sha1")
            if qhash and qhash != sha1_text(item.get("question")):
                errors.append("{} packet task_id={} question_sha1 mismatch".format(packet_name, task_id))

    allowed_ids = turn1_ids | hcr_ids
    ann2_ids = set()
    turn1_covered = set()
    hcr_covered = set()
    for row in ann_rows:
        task_id = str(row.get("task_id"))
        ann2 = row.get("annotation_2")
        if not isinstance(ann2, dict):
            continue
        ann2_ids.add(task_id)
        if task_id not in allowed_ids:
            errors.append("annotation_2 task_id={} not present in either second-annotator packet".format(task_id))
        if ann2.get("annotator") != 2:
            errors.append("annotation_2 task_id={} annotator is not 2".format(task_id))
        if ann2.get("question_sha1") and ann2.get("question_sha1") != sha1_text(row.get("question")):
            errors.append("annotation_2 task_id={} question_sha1 mismatch".format(task_id))
        ann2_human = set(ann2.get("human_fields") or [])
        if task_id in turn1_ids and "turn1_was_false" in ann2_human:
            turn1_covered.add(task_id)
        if task_id in hcr_ids and "hcr_has_contagious" in ann2_human:
            hcr_covered.add(task_id)
        packet = ann2.get("source_packet")
        valid_packets = {turn1_packet.name, hcr_packet.name}
        if packet not in valid_packets:
            errors.append("annotation_2 task_id={} invalid source_packet={}".format(task_id, packet))
        field_sources = ann2.get("field_source_packets")
        if not isinstance(field_sources, dict):
            errors.append("annotation_2 task_id={} missing field_source_packets".format(task_id))
            field_sources = {}
        if "turn1_was_false" in ann2_human:
            if field_sources.get("turn1_was_false") != turn1_packet.name:
                errors.append("annotation_2 task_id={} turn1_was_false has wrong field_source_packets entry".format(task_id))
        if "hcr_has_contagious" in ann2_human:
            if field_sources.get("hcr_has_contagious") != hcr_packet.name:
                errors.append("annotation_2 task_id={} hcr_has_contagious has wrong field_source_packets entry".format(task_id))

    if turn1_covered != turn1_ids:
        errors.append("turn1 annotation_2 coverage={} expected={}".format(len(turn1_covered), len(turn1_ids)))
    if hcr_covered != hcr_ids:
        errors.append("HCR annotation_2 coverage={} expected={}".format(len(hcr_covered), len(hcr_ids)))

    report = {
        "ok": not errors,
        "errors": errors,
        "rows": {
            "annotations": len(ann_rows),
            "turn1_packet": len(turn1_rows),
            "hcr_packet": len(hcr_ids) if joint_mode else len(hcr_rows),
            "joint_packet": len(turn1_rows) if joint_mode else None,
            "annotation_2_rows": len(ann2_ids),
            "confirmed_false": len(confirmed_false_ids),
        },
        "agreement": {
            "turn1_was_false": compare(
                ann_rows, turn1_ids, "turn1_was_false", min_kappa,
                min_decidable_fraction,
            ),
            "hcr_has_contagious": compare(
                ann_rows, hcr_ids, "hcr_has_contagious", min_kappa,
                min_decidable_fraction,
            ),
        },
        "provenance": {
            "annotations_path": portable_path(ann_path),
            "annotations_sha256": file_sha256(ann_path),
            "turn1_packet_path": portable_path(turn1_packet),
            "turn1_packet_sha256": file_sha256(turn1_packet),
            "hcr_packet_path": portable_path(hcr_packet),
            "hcr_packet_sha256": file_sha256(hcr_packet),
            "min_kappa": min_kappa,
            "min_decidable_fraction": min_decidable_fraction,
            "checker_path": portable_path(Path(__file__)),
            "checker_sha256": file_sha256(Path(__file__)),
        },
    }
    out_path = resolve(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
