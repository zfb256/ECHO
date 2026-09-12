#!/usr/bin/env python3
"""Guard for the pre-registered extension arm.

scripts/validate_zh_study.py deliberately hardcodes the frozen primary configs
and requires exactly five models, so it cannot validate the extension. It also
takes no arguments: passing --config to it is silently ignored and it still
reports ok, which is a trap. This script is the extension's counterpart.

Checks
------
1. The extension derives from the frozen parent (parent SHA-256 still matches
   the value recorded in FROZEN_INPUTS.json).
2. Nothing but run_name / model_pool / model_repositories / notes differs.
3. Every extension model has a local weight directory.
4. The frozen inputs the extension reuses are present and unchanged.
5. The detector freeze record verifies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_ext_config import APPENDIX_MODELS, EXT_MODELS, PARENTS  # noqa: E402

ALLOWED_DIFF_PREFIXES = (
    "/pilot/run_name",
    "/pilot/model_pool",
    "/pilot/model_repositories",
    "/pilot/_model_pool_note",
    "/_extension_note",
)
FROZEN_INPUTS = "datasets_zh/runs/zh_study/FROZEN_INPUTS.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        out[prefix] = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    else:
        out[prefix] = obj
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="Extension config to validate.")
    p.add_argument("--models-dir", default=None,
                   help="Override the weight directory root (default: paths.models_dir).")
    p.add_argument("--skip-weights", action="store_true",
                   help="Skip the local weight-directory check (for pre-download validation).")
    args = p.parse_args()

    ext_path = ROOT / args.config
    if not ext_path.exists():
        print(f"ERROR: extension config not found: {args.config}", file=sys.stderr)
        return 1
    ext = json.loads(ext_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    warnings: list[str] = []

    note = ext.get("_extension_note")
    if not note:
        errors.append("missing _extension_note: cannot establish provenance")
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        return 1

    parent_rel = note["parent_config"]
    parent_path = ROOT / parent_rel
    if not parent_path.exists():
        errors.append(f"parent config missing: {parent_rel}")
    else:
        actual = sha256_file(parent_path)
        if actual != note["parent_config_sha256"]:
            errors.append(
                f"parent config has CHANGED since derivation: {parent_rel}\n"
                f"    recorded {note['parent_config_sha256']}\n"
                f"    actual   {actual}\n"
                "    The frozen primary config must never be edited."
            )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        fa, fb = flatten(parent), flatten(ext)
        for key in sorted(set(fa) | set(fb)):
            if fa.get(key) == fb.get(key):
                continue
            if any(key == pfx or key.startswith(pfx + "/") for pfx in ALLOWED_DIFF_PREFIXES):
                continue
            errors.append(f"unexpected difference from parent at {key}: {fa.get(key)!r} -> {fb.get(key)!r}")

    # Cross-check the frozen-inputs record where it applies to the injected parent.
    fi_path = ROOT / FROZEN_INPUTS
    if fi_path.exists() and parent_rel == "configs/zh_study.json":
        fi = json.loads(fi_path.read_text(encoding="utf-8"))
        if fi.get("config", {}).get("sha256") != note["parent_config_sha256"]:
            errors.append(
                "parent SHA-256 does not match FROZEN_INPUTS.json: the extension was "
                "derived from a config other than the frozen one"
            )
        for key in ("seed_bank", "processed_pairs"):
            rec = fi.get(key)
            if not rec:
                continue
            target = ROOT / rec["path"]
            if not target.exists():
                errors.append(f"frozen input missing: {rec['path']}")
            elif sha256_file(target) != rec["sha256"]:
                errors.append(f"frozen input CHANGED since freezing: {rec['path']}")

    pool = ext["pilot"]["model_pool"]
    repos = ext["pilot"]["model_repositories"]
    if len(pool) != len(set(pool)):
        errors.append("model_pool contains duplicates")
    for m in pool:
        if m not in repos:
            errors.append(f"model {m} has no entry in model_repositories")
    expected_maps = [EXT_MODELS]
    if parent_rel == "configs/zh_study.json":
        expected_maps.append({**EXT_MODELS, **APPENDIX_MODELS})
    if repos not in expected_maps or pool != list(repos):
        errors.append(
            "extension model pool/repositories differ from the pre-registered "
            "make_ext_config.py definition"
        )
    expected_parent = PARENTS.get(parent_rel)
    if expected_parent and (
        args.config != expected_parent[0]
        or ext["pilot"]["run_name"] != expected_parent[1]
    ):
        errors.append("extension config path or run_name differs from its registered parent mapping")
    if not args.skip_weights:
        models_dir = Path(args.models_dir) if args.models_dir else ROOT / ext["paths"]["models_dir"]
        for m in pool:
            if not (models_dir / m).is_dir():
                errors.append(f"weight directory not present: {models_dir / m}")

    # run_name must differ from the parent so the frozen run directory is untouched.
    if parent_path.exists():
        if ext["pilot"]["run_name"] == parent["pilot"]["run_name"]:
            errors.append(
                "extension run_name equals the parent run_name: the extension would write "
                "into the frozen run directory"
            )

    # It is not enough to differ from this parent: the run directory must not collide
    # with ANY existing run that already holds generated data, or the extension would
    # write into someone else's frozen outputs.
    runs_dir = ROOT / ext["paths"]["runs_dir"]
    ext_run_dir = runs_dir / ext["pilot"]["run_name"]
    if ext_run_dir.is_dir():
        occupied = sorted(p.name for p in ext_run_dir.glob("*.manifest.json"))
        if occupied:
            errors.append(
                f"extension run directory already holds generated data: {ext_run_dir}\n"
                f"    manifests present: {', '.join(occupied)}\n"
                "    Choose a fresh run_name; never write an extension into an existing run."
            )

    freeze = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_detector.py"), "--verify"],
        capture_output=True, text=True,
    )
    detector_frozen_ok = freeze.returncode == 0
    if not detector_frozen_ok:
        errors.append("detector freeze does not verify (run scripts/freeze_detector.py --verify)")

    ok = not errors
    print(json.dumps({
        "ok": ok,
        "config": args.config,
        "parent": parent_rel,
        "run_name": ext["pilot"]["run_name"],
        "model_pool": pool,
        "detector_freeze_verified": detector_frozen_ok,
        "errors": errors,
        "warnings": warnings,
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
