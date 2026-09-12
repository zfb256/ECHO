#!/usr/bin/env python3
"""Merge a primary-arm file with its pre-registered extension arm.

The extension models are generated into their own run directory so that the
frozen primary run is never touched (see handoff.md 5.0). Downstream metric
scripts take a single file per arm, so the two halves must be joined first.

A merge is only legitimate if the two halves are genuinely disjoint replicas of
the same design, so this refuses to write unless:

  * the row keys are disjoint (nothing is silently overwritten);
  * the model sets are disjoint (the extension is new models, not a re-run);
  * every extension model covers the same pair/question universe as the primary
    models, so the merged denominator stays balanced;
  * the two halves share a schema;
  * the detector freeze still verifies.

Supported file kinds are detected from the fields present:

  injected / placebo annotations   key = (pair_id, model)
  self-induced annotations         key = (task_id)
  model outputs                    key = (pair_id, model, condition)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402

FREEZE_RECORD = "reports_zh/DETECTOR_FROZEN.json"
OPTIONAL_SELFINDUCED_FIELDS = {"annotation_2", "double_annotate", "double_annotate_fields"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--primary", required=True)
    p.add_argument("--extension", required=True, nargs="+",
                   help="One or more extension files to fold into the primary.")
    p.add_argument("--output", required=True)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detect_kind(row: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    if "condition" in row and "response" in row:
        return "model_outputs", ("pair_id", "model", "condition")
    if "q_id" in row or "turn1_response" in row:
        return "selfinduced_annotation", ("task_id",)
    if "annotation" in row:
        return "injected_or_placebo_annotation", ("pair_id", "model")
    raise SystemExit(f"cannot determine file kind from fields: {sorted(row)}")


def key_of(row: dict[str, Any], fields: tuple[str, ...]) -> tuple:
    missing = [f for f in fields if row.get(f) in (None, "")]
    if missing:
        raise SystemExit(f"row is missing key field(s) {missing}: {json.dumps(row, ensure_ascii=False)[:160]}")
    return tuple(row[f] for f in fields)


def schema_of(row: dict[str, Any], kind: str) -> set[str]:
    fields = set(row)
    return fields - OPTIONAL_SELFINDUCED_FIELDS if kind == "selfinduced_annotation" else fields


def coverage(rows: list[dict[str, Any]], kind: str) -> dict[str, set]:
    """Per model, the set of units it covers (pairs, or questions for self-induced)."""
    out: dict[str, set] = {}
    for r in rows:
        if r.get("model") in (None, ""):
            raise SystemExit("coverage row is missing model")
        if kind == "selfinduced_annotation":
            unit = r.get("q_id")
        elif kind == "model_outputs":
            unit = (r.get("pair_id"), r.get("condition"))
        else:
            unit = r.get("pair_id")
        if unit in (None, ""):
            raise SystemExit("coverage row is missing its pair/question unit")
        out.setdefault(str(r["model"]), set()).add(unit)
    return out


def main() -> int:
    args = parse_args()
    primary_path = resolve_project_path(args.primary)
    ext_paths = [resolve_project_path(p) for p in args.extension]
    out_path = resolve_project_path(args.output)

    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}; move or archive it first")

    freeze = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_detector.py"), "--verify"],
        capture_output=True, text=True)
    if freeze.returncode != 0:
        raise SystemExit(
            "detector freeze does not verify; merging labels produced by different "
            "detector versions would silently mix them.\n" + freeze.stdout + freeze.stderr)
    freeze_sha = None
    if (ROOT / FREEZE_RECORD).exists():
        freeze_sha = json.loads((ROOT / FREEZE_RECORD).read_text(encoding="utf-8")).get("detector_freeze_sha256")

    primary = list(read_jsonl(primary_path))
    if not primary:
        raise SystemExit(f"primary file is empty: {primary_path}")
    kind, key_fields = detect_kind(primary[0])

    errors: list[str] = []
    merged = list(primary)
    seen = {key_of(r, key_fields) for r in primary}
    if len(seen) != len(primary):
        errors.append(f"primary file has duplicate keys on {key_fields}")

    primary_models = {str(r.get("model")) for r in primary}
    primary_schema = schema_of(primary[0], kind)
    for i, row in enumerate(primary[1:], 2):
        if schema_of(row, kind) != primary_schema:
            errors.append(f"primary row {i} has a different schema")
    ext_rows_all: list[dict[str, Any]] = []

    for path in ext_paths:
        rows = list(read_jsonl(path))
        if not rows:
            errors.append(f"extension file is empty: {path.name}")
            continue
        ext_kind, _ = detect_kind(rows[0])
        if ext_kind != kind:
            errors.append(f"{path.name}: file kind {ext_kind} does not match primary kind {kind}")
            continue

        schema = schema_of(rows[0], kind)
        if schema != primary_schema:
            only_p = sorted(primary_schema - schema)
            only_e = sorted(schema - primary_schema)
            errors.append(f"{path.name}: schema differs (primary-only={only_p}, extension-only={only_e})")
        for i, row in enumerate(rows[1:], 2):
            if schema_of(row, kind) != schema:
                errors.append(f"{path.name}: row {i} has a different schema")

        ext_models = {str(r.get("model")) for r in rows}
        overlap_models = ext_models & primary_models
        if overlap_models:
            errors.append(
                f"{path.name}: model(s) already present in the primary arm: {sorted(overlap_models)}. "
                "An extension adds new models; it never re-runs existing ones.")

        for r in rows:
            k = key_of(r, key_fields)
            if k in seen:
                errors.append(f"{path.name}: duplicate key {k} already present")
                continue
            seen.add(k)
            merged.append(r)
        ext_rows_all.extend(rows)

    # Coverage: each extension model must span the same units as the primary models.
    if ext_rows_all:
        prim_cov = coverage(primary, kind)
        ext_cov = coverage(ext_rows_all, kind)
        # The reference is only meaningful if the primary models agree among
        # themselves. Taking a union over an internally ragged primary would
        # invent a universe no model actually covers, and every extension model
        # would then be blamed for the primary's own gaps.
        sizes = {m: len(u) for m, u in prim_cov.items()}
        if len({frozenset(u) for u in prim_cov.values()}) > 1:
            errors.append(
                "primary arm is internally ragged: its models do not cover the same "
                f"unit set ({sizes}). Fix the primary before merging; the coverage "
                "check has no well-defined reference otherwise.")
        reference = set.union(*prim_cov.values()) if prim_cov else set()
        for model, units in sorted(ext_cov.items()):
            if units != reference:
                errors.append(
                    f"extension model {model} covers {len(units)} units but the primary "
                    f"universe has {len(reference)}; missing "
                    f"{len(reference - units)}, extra {len(units - reference)}")

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in merged) + "\n",
        encoding="utf-8")

    by_model = Counter(str(r.get("model")) for r in merged)
    manifest_path = out_path.with_suffix(".merge_manifest.json")
    manifest_path.write_text(json.dumps({
        "kind": kind,
        "key_fields": list(key_fields),
        "detector_freeze_sha256": freeze_sha,
        "detector_freeze_verified": freeze.returncode == 0,
        "primary": {
            "path": str(primary_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(primary_path),
            "rows": len(primary),
            "models": sorted(primary_models),
        },
        "extensions": [{
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(p),
        } for p in ext_paths],
        "merged": {
            "path": str(out_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(out_path),
            "rows": len(merged),
            "rows_by_model": dict(sorted(by_model.items())),
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "kind": kind,
        "output": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "manifest": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(merged),
        "models": len(by_model),
        "rows_by_model": dict(sorted(by_model.items())),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
