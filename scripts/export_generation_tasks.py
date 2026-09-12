from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from ids import stable_id
from jsonl import read_jsonl, write_jsonl
from manifest import file_sha256, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export clean/contaminated generation tasks for server inference.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--pairs", default=None, help="Defaults to processed_dir/pilot_pairs_all.jsonl")
    parser.add_argument("--output", default=None, help="Defaults to runs_dir/<run_name>/generation_tasks.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    config_path = resolve_project_path(args.config)
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    runs_dir = resolve_project_path(config["paths"]["runs_dir"])
    run_name = config["pilot"].get("run_name", "pilot")
    pairs_path = resolve_project_path(args.pairs) if args.pairs else processed_dir / f"{run_name}_pairs_all.jsonl"
    out_path = resolve_project_path(args.output) if args.output else runs_dir / run_name / "generation_tasks.jsonl"
    seed_bank_path = resolve_project_path(config["paths"]["seed_bank"])

    model_pool = config["pilot"]["model_pool"]
    ccr_cfg = config.get("pilot", {}).get("ccr", {})
    placebo_enabled = bool(ccr_cfg.get("placebo_clean_vs_clean", False))
    # The placebo arm adds an independently sampled second clean continuation.
    conditions = [("clean", "clean_prompt"), ("contaminated", "contaminated_prompt")]
    if placebo_enabled:
        conditions.append(("clean_b", "clean_prompt"))

    pair_rows = list(read_jsonl(pairs_path))
    pair_ids = [p.get("pair_id") for p in pair_rows]
    if any(not pair_id for pair_id in pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"Missing or duplicate pair_id in {pairs_path}")
    tasks = []
    for pair in pair_rows:
        for model in model_pool:
            for condition, prompt_key in conditions:
                task_id = stable_id(pair["pair_id"], model, condition)
                tasks.append(
                    {
                        "task_id": task_id,
                        "pair_id": pair["pair_id"],
                        "condition": condition,
                        "model": model,
                        "source_dataset": pair["source_dataset"],
                        "seed_id": pair["seed_id"],
                        "seed_type": pair["seed_type"],
                        "seed_claim": pair["seed_claim"],
                        # Retain the correction, evidence hint, and source provenance.
                        "corrected_claim": pair.get("corrected_claim"),
                        "evidence_hint": pair.get("evidence_hint"),
                        "source_row_id": pair.get("source_row_id"),
                        "prompt": pair[prompt_key],
                    }
                )

    write_jsonl(out_path, tasks)
    manifest_path = out_path.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config_path,
        config,
        {
            "builder_path": "scripts/export_generation_tasks.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "pairs_path": to_project_relative(pairs_path),
            "tasks_path": to_project_relative(out_path),
            "num_tasks": len(tasks),
            "num_models": len(model_pool),
            "placebo_clean_b_enabled": placebo_enabled,
            "pairs_sha256": file_sha256(pairs_path),
            "seed_bank_path": to_project_relative(seed_bank_path),
            "seed_bank_sha256": file_sha256(seed_bank_path),
        },
    )
    print(json.dumps({"wrote": str(out_path), "tasks": len(tasks), "placebo_clean_b": placebo_enabled, "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
