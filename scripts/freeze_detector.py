#!/usr/bin/env python3
"""Freeze the injected detector, or verify that it has not moved since freezing.

Rationale
---------
The original 120-row output audit informed a detector repair, so post-repair
precision on that packet is diagnostic rather than held-out. The replacement
audit is held-out only if the detector is frozen BEFORE the new blind packet is
drawn and is never touched afterwards. This script makes that auditable: it
records a SHA-256 over every artifact that can change a headline injected label,
and `--verify` re-checks them.

Usage
-----
    python3 scripts/freeze_detector.py --config configs/zh_study.json
    python3 scripts/freeze_detector.py --verify        # exit 1 if anything moved
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every file whose content can change a headline injected label.
CODE_FILES = [
    "auto_labeling/detector.py",
    "auto_labeling/auto_label.py",
    "auto_labeling/build_construct_gold.py",
    "auto_labeling/validate_detector.py",
]
DATA_FILES = [
    "configs/zh_seed_bank.json",
]
# Optional: present for the primary run, recreated identically for the extension.
RUN_FILES = [
    "datasets_zh/runs/zh_study/construct_gold.jsonl",
]
DEFAULT_RECORD = "reports_zh/DETECTOR_FROZEN.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    detector_settings = config["pilot"]["detector"]
    nli_settings = config["pilot"].get("nli", {})

    files: dict[str, str] = {}
    missing: list[str] = []
    for rel in CODE_FILES + DATA_FILES + RUN_FILES:
        p = ROOT / rel
        if p.exists():
            files[rel] = sha256_file(p)
        elif rel in RUN_FILES:
            missing.append(rel)
        else:
            raise SystemExit(f"Cannot freeze: required file is missing: {rel}")

    combined = sha256_obj({
        "files": files,
        "detector_settings": detector_settings,
        "nli_enabled": nli_settings.get("enabled", False),
    })

    record = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Detector frozen before the held-out blind audit. Any change after this "
            "point invalidates the held-out status of the replacement packet and "
            "requires a further audit round."
        ),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "detector_settings": detector_settings,
        "nli_enabled": nli_settings.get("enabled", False),
        "files": files,
        "files_absent_at_freeze": missing,
        "detector_freeze_sha256": combined,
    }

    validation = ROOT / "reports_zh/detector_validation.json"
    if validation.exists():
        v = json.loads(validation.read_text(encoding="utf-8"))
        record["construct_gold_validation"] = {
            "path": "reports_zh/detector_validation.json",
            "sha256": sha256_file(validation),
            "precision": v.get("precision"),
            "recall": v.get("recall"),
            "f1": v.get("f1"),
        }
    return record


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None,
                   help="Config carrying pilot.detector. Freezing defaults to "
                        "configs/zh_study.json; --verify always reuses the config "
                        "recorded in the freeze record.")
    p.add_argument("--record", default=DEFAULT_RECORD)
    p.add_argument("--verify", action="store_true",
                   help="Compare the current state against the stored record; exit 1 on any drift.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing freeze record (requires a deliberate re-freeze).")
    args = p.parse_args()

    record_path = ROOT / args.record

    if args.verify:
        # Verification must not depend on what the caller passes: a freeze that only
        # holds for one caller-supplied config is not a freeze. Always re-read the
        # config named in the record.
        if not record_path.exists():
            print(f"NOT FROZEN: {args.record} does not exist.", file=sys.stderr)
            return 1
        stored = json.loads(record_path.read_text(encoding="utf-8"))
        recorded_config = stored.get("config_path")
        if args.config is not None and args.config != recorded_config:
            print(
                f"--verify ignores --config by design, and {args.config!r} does not match "
                f"the frozen config {recorded_config!r}. Re-run without --config.",
                file=sys.stderr,
            )
            return 1
        config_path = ROOT / recorded_config
        if not config_path.exists():
            print(f"Frozen config is missing: {recorded_config}", file=sys.stderr)
            return 1
        current = collect(config_path)
        drift = []
        if stored.get("detector_freeze_sha256") != current["detector_freeze_sha256"]:
            for rel, digest in current["files"].items():
                if stored.get("files", {}).get(rel) != digest:
                    drift.append(f"file changed: {rel}")
            for rel in stored.get("files", {}):
                if rel not in current["files"]:
                    drift.append(f"file removed: {rel}")
            if stored.get("detector_settings") != current["detector_settings"]:
                drift.append("detector settings changed in config")
            if stored.get("nli_enabled") != current["nli_enabled"]:
                drift.append("nli.enabled changed")
            if not drift:
                drift.append("combined freeze hash changed (composition differs)")
        result = {
            "frozen_at_utc": stored.get("frozen_at_utc"),
            "detector_freeze_sha256_expected": stored.get("detector_freeze_sha256"),
            "detector_freeze_sha256_actual": current["detector_freeze_sha256"],
            "ok": not drift,
            "drift": drift,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if not drift else 1

    config_path = ROOT / (args.config or "configs/zh_study.json")
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    current = collect(config_path)

    if record_path.exists() and not args.force:
        stored = json.loads(record_path.read_text(encoding="utf-8"))
        print(
            f"Refusing to overwrite an existing freeze record ({args.record}), frozen at "
            f"{stored.get('frozen_at_utc')}.\n"
            "Re-freezing after the blind packet is drawn destroys its held-out status.\n"
            "Use --verify to check for drift, or --force only for a deliberate re-freeze "
            "that you will disclose in the paper.",
            file=sys.stderr,
        )
        return 1

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "wrote": args.record,
        "detector_freeze_sha256": current["detector_freeze_sha256"],
        "files_hashed": len(current["files"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
