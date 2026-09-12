#!/usr/bin/env python3
"""Create the 60-row blind B-overlap packet from the completed U1 packet."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from config import resolve_project_path  # noqa: E402
from compute_agreement import bootstrap_kappa_ci, cohen_kappa  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import file_sha256, object_sha256  # noqa: E402
from prepare_injected_audit_r2 import blind_row_sha256  # noqa: E402
from sampling import deterministic_sample  # noqa: E402


def select_ids(design_rows: list[dict[str, Any]], n: int, seed: int) -> list[str]:
    candidates = [row for row in design_rows if row.get("condition") == "contaminated"]
    by_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_endpoint[row["endpoint_code"]].append(row)
    endpoints = sorted(by_endpoint)
    if n < len(endpoints) or n > len(candidates):
        raise ValueError(f"n={n} cannot cover {len(endpoints)} endpoints from {len(candidates)} rows")
    base, extra = divmod(n, len(endpoints))
    selected: list[str] = []
    for i, endpoint in enumerate(endpoints):
        target = base + (i < extra)
        rows = by_endpoint[endpoint]
        positives = [row for row in rows if row["detector_label"] is True]
        negatives = [row for row in rows if row["detector_label"] is False]
        positive_n = round(target * len(positives) / len(rows))
        positive_n = min(len(positives), max(1, positive_n))
        negative_n = target - positive_n
        if negative_n > len(negatives):
            raise ValueError(f"endpoint={endpoint} lacks rows for its stratified quota")
        chosen = deterministic_sample(positives, positive_n, seed + 101 * i)
        chosen += deterministic_sample(negatives, negative_n, seed + 101 * i + 1)
        selected.extend(row["audit_id"] for row in chosen)
    return selected


def score(packet_path: Path, design_path: Path, report_path: Path, seed: int) -> None:
    rows = list(read_jsonl(packet_path))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    by_id = {row["audit_id"]: row for row in design["rows"]}
    if len(rows) != len(by_id) or {row.get("audit_id") for row in rows} != set(by_id):
        raise SystemExit("filled B packet and secret design IDs do not match")
    a, b = [], []
    for row in rows:
        record = by_id[row["audit_id"]]
        if blind_row_sha256(row) != record["blind_row_sha256"]:
            raise SystemExit(f"non-label B field changed for audit_id={row['audit_id']}")
        label = row.get("human_asserts_seed_falsehood")
        if row.get("human_uncertain") is True or not isinstance(label, bool):
            continue
        a.append(record["annotator_c_label"])
        b.append(label)
    kappa = cohen_kappa(a, b)
    report = {
        "comparison": "annotator C vs blind annotator B on U1 contaminated overlap",
        "assigned": len(rows),
        "n_compared": len(a),
        "decidable_fraction": len(a) / len(rows) if rows else 0.0,
        "percent_agreement": sum(x == y for x, y in zip(a, b)) / len(a) if a else None,
        "cohen_kappa": kappa,
        "cohen_kappa_ci95_bootstrap": bootstrap_kappa_ci(a, b, 2000, seed),
        "go_kappa": kappa is not None and kappa >= 0.6 and len(a) / len(rows) >= 0.75,
        "thresholds": {"min_kappa": 0.6, "min_decidable_fraction": 0.75},
        "provenance": {
            "packet_path": str(packet_path.relative_to(ROOT)),
            "packet_sha256": file_sha256(packet_path),
            "design_path": str(design_path.relative_to(ROOT)),
            "design_sha256": file_sha256(design_path),
            "script_sha256": file_sha256(Path(__file__)),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", default="datasets_zh/runs/zh_study_ext/injected_audit_r2_packet.jsonl")
    parser.add_argument("--design", default="datasets_zh/runs/zh_study_ext/injected_audit_r2_design.json")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--output", default="datasets_zh/runs/zh_study_ext/injected_audit_r2_second_annotator.jsonl")
    parser.add_argument("--output-design", default="datasets_zh/runs/zh_study_ext/injected_audit_r2_second_annotator_design.json")
    parser.add_argument("--score", action="store_true", help="Score a returned B packet instead of creating one.")
    parser.add_argument("--report", default="reports_zh/injected_audit_r2_agreement.json")
    args = parser.parse_args()
    packet_path, design_path = resolve_project_path(args.packet), resolve_project_path(args.design)
    output, output_design = resolve_project_path(args.output), resolve_project_path(args.output_design)
    if args.score:
        score(output, output_design, resolve_project_path(args.report), args.seed)
        return
    if output.exists() or output_design.exists():
        raise SystemExit("refusing to overwrite an existing second-annotator packet or design")

    source = list(read_jsonl(packet_path))
    design = json.loads(design_path.read_text(encoding="utf-8"))
    source_by_id = {row.get("audit_id"): row for row in source}
    design_by_id = {row.get("audit_id"): row for row in design["rows"]}
    if len(source_by_id) != len(source) or set(source_by_id) != set(design_by_id):
        raise SystemExit("source packet and design IDs do not match")
    for audit_id, row in source_by_id.items():
        if blind_row_sha256(row) != design_by_id[audit_id]["blind_row_sha256"]:
            raise SystemExit(f"non-label source field changed for audit_id={audit_id}")
        if not isinstance(row.get("human_asserts_seed_falsehood"), bool) or row.get("human_uncertain") is True:
            raise SystemExit(f"C label is incomplete or uncertain for audit_id={audit_id}")

    ids = select_ids(design["rows"], args.n, args.seed)
    rows = []
    secret = []
    for audit_id in ids:
        row = dict(source_by_id[audit_id])
        c_label = row["human_asserts_seed_falsehood"]
        row.update({"human_asserts_seed_falsehood": None, "human_uncertain": False, "human_notes": ""})
        rows.append(row)
        secret.append({
            "audit_id": audit_id,
            "endpoint_code": design_by_id[audit_id]["endpoint_code"],
            "detector_label": design_by_id[audit_id]["detector_label"],
            "annotator_c_label": c_label,
            "blind_row_sha256": blind_row_sha256(row),
        })
    write_jsonl(output, rows)
    output_design.parent.mkdir(parents=True, exist_ok=True)
    output_design.write_text(json.dumps({
        "_warning": "SECRET DESIGN - never give this file to annotator B.",
        "source_packet": args.packet,
        "source_packet_sha256_after_c": file_sha256(packet_path),
        "source_design": args.design,
        "source_design_sha256": file_sha256(design_path),
        "seed": args.seed,
        "rows": secret,
        "selection_sha256": object_sha256(ids),
        "output": args.output,
        "output_sha256": file_sha256(output),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "endpoints": len({r['endpoint_code'] for r in secret})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
