#!/usr/bin/env python3
"""Derive the pre-registered extension config from the frozen zh_study config.

The extension arm reuses the frozen seed bank, the frozen 800 dialogue-seed
pairs, the frozen prompts, and the frozen detector settings. Only the run name
and the model pool differ, so every number the extension produces is comparable
with the primary arm.

The parent config is never modified: it carries a "frozen before generation"
declaration and its SHA-256 is recorded in datasets_zh/runs/zh_study/FROZEN_INPUTS.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The two arms live in separate runs with separate configs:
#   configs/zh_study.json       -> run zh_study        (injected + placebo)
#   configs/zh_selfinduced.json -> run zh_selfinduced  (self-induced, 60 questions)
# The extension mirrors that split rather than collapsing the arms into one run.
PARENTS = {
    "configs/zh_study.json": ("configs/zh_study_ext.json", "zh_study_ext"),
    "configs/zh_selfinduced.json": ("configs/zh_selfinduced_ext.json", "zh_selfinduced_ext"),
}
EXT_MODELS = {
    "glm-4-9b-chat-hf": "zai-org/glm-4-9b-chat-hf",
    "Yi-1.5-9B-Chat": "01-ai/Yi-1.5-9B-Chat",
}
APPENDIX_MODELS = {
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", default=None,
                   help="Derive only this parent config (default: all of PARENTS).")
    p.add_argument("--with-appendix-arm", action="store_true",
                   help="Also include Llama-3.1-8B-Instruct (appendix-only arm, injected only).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing extension configs.")
    args = p.parse_args()

    if args.parent and args.parent not in PARENTS:
        print(f"Unknown --parent {args.parent!r}. Known parents:", file=sys.stderr)
        for known in PARENTS:
            print(f"  {known}", file=sys.stderr)
        return 1

    parents = [args.parent] if args.parent else list(PARENTS)

    occupied = []
    for rel in parents:
        _, run_name = PARENTS[rel]
        run_dir = ROOT / "datasets_zh/runs" / run_name
        if any(run_dir.glob("*.manifest.json")):
            occupied.append(str(run_dir.relative_to(ROOT)))
    if occupied:
        print(
            "Refusing to regenerate a config whose run already has manifests: "
            + ", ".join(occupied),
            file=sys.stderr,
        )
        return 1

    # Overwriting silently would change an extension config that generation,
    # manifests, or a freeze record may already be bound to.
    existing = [PARENTS[rel][0] for rel in parents if (ROOT / PARENTS[rel][0]).exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing extension config(s):", file=sys.stderr)
        for rel in existing:
            print(f"  {rel}", file=sys.stderr)
        print(
            "If generation has already run against them, regenerating changes their "
            "SHA-256 and breaks provenance. Pass --force only when nothing depends "
            "on them yet.",
            file=sys.stderr,
        )
        return 1

    results = [derive(ROOT / rel, rel, args.with_appendix_arm) for rel in parents]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def derive(parent_path: Path, parent_rel: str, with_appendix: bool) -> dict:
    out_rel, ext_run_name = PARENTS[parent_rel]
    out_path = ROOT / out_rel

    parent_sha = sha256_file(parent_path)
    config = json.loads(parent_path.read_text(encoding="utf-8"))

    models = dict(EXT_MODELS)
    # The appendix arm (weak-Chinese contrast) is injected-only: it has no
    # self-induced counterpart, mirroring how the hosted rows are reported.
    if with_appendix and "selfinduced" not in parent_rel:
        models.update(APPENDIX_MODELS)

    pilot = config["pilot"]
    pilot["run_name"] = ext_run_name
    pilot["model_pool"] = list(models)
    pilot["model_repositories"] = models
    pilot["_model_pool_note"] = (
        "Pre-registered extension arm. Frozen before extension generation. "
        "The primary five-model pool in configs/zh_study.json is unchanged and is "
        "not regenerated. Family coverage: GLM (glm-4-9b-chat-hf), Yi (Yi-1.5-9B-Chat)."
    )

    config["_extension_note"] = {
        "role": "pre-registered extension arm",
        "parent_config": parent_rel,
        "parent_config_sha256": parent_sha,
        "reuses": [
            "configs/zh_seed_bank.json (40 verified false propositions)",
            "datasets_zh/processed/zh_study_pairs_all.jsonl (800 dialogue-seed pairs)",
            "prompts, decoding settings, detector settings, go/no-go gates",
        ],
        "differs_only_in": ["pilot.run_name", "pilot.model_pool", "pilot.model_repositories"],
        "analysis_tiers": {
            "primary": "5 open-weight models x 40-question cohort (pre-registered, unchanged)",
            "extension": "7 open-weight models x 40-question cohort",
            "sensitivity": "7 open-weight models x 60-question full bank",
        },
        "required_explicit_paths": {
            "_why": (
                "Several scripts derive default input paths from run_name. With run_name="
                f"{ext_run_name!r} those defaults point at files that do not exist, so the "
                "frozen primary inputs must be passed explicitly."
            ),
            "export_generation_tasks.py": "--pairs datasets_zh/processed/zh_study_pairs_all.jsonl",
            "_note": "build_selfinduced_stage1.py reads paths.selfinduced_bank, which is run-name independent, so it needs no override.",
        },
    }

    out_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    return {
        "wrote": str(out_path.relative_to(ROOT)).replace("\\", "/"),
        "parent": parent_rel,
        "parent_sha256": parent_sha,
        "ext_sha256": sha256_file(out_path),
        "run_name": ext_run_name,
        "model_pool": list(models),
        "questions_per_model": pilot.get("selfinduced", {}).get("questions_per_model"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
