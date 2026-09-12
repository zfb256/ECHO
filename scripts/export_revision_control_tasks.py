from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from ids import stable_id  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import write_manifest  # noqa: E402


DEFAULT_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Mistral-7B-Instruct-v0.3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export a correction/revision-control arm for ECHO injected pairs."
    )
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument(
        "--pairs",
        default=None,
        help="Defaults to datasets/processed/<run_name>_pairs_all.jsonl.",
    )
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names for the control arm.",
    )
    p.add_argument(
        "--per-dataset",
        type=int,
        default=200,
        help="Pairs sampled per dataset before crossing with models.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Defaults to datasets/runs/<run_name>/revision_control_tasks.jsonl.",
    )
    return p.parse_args()


def _dialogue_prefix(pair: dict[str, Any]) -> str:
    """Return contaminated context without the final entity-probe user turn."""
    contaminated = pair["contaminated_prompt"]
    probe = pair["probe"]
    suffix = f"\nUser: {probe}"
    if contaminated.endswith(suffix):
        return contaminated[: -len(suffix)]
    marker = "\nUser: "
    if marker in contaminated:
        return contaminated.rsplit(marker, 1)[0]
    return contaminated


def build_revision_prompt(pair: dict[str, Any]) -> str:
    prefix = _dialogue_prefix(pair)
    correction_turn = (
        f'Earlier you said: "{pair["seed_claim"]}"\n'
        f'However, the correct information is: "{pair["corrected_claim"]}"\n'
        "Please answer again using the corrected information. "
        f"{pair['probe']}"
    )
    return f"{prefix}\nUser: {correction_turn}"


def sample_pairs(pairs: list[dict[str, Any]], per_dataset: int, seed: int) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_dataset[str(pair.get("source_dataset", "unknown"))].append(pair)

    sampled: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for dataset in sorted(by_dataset):
        rows = list(by_dataset[dataset])
        rng.shuffle(rows)
        sampled.extend(rows[: min(per_dataset, len(rows))])
    return sampled


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = resolve_project_path(args.config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])

    pairs_path = (
        resolve_project_path(args.pairs)
        if args.pairs
        else processed_dir / f"{run_name}_pairs_all.jsonl"
    )
    out_path = (
        resolve_project_path(args.output)
        if args.output
        else run_dir / "revision_control_tasks.jsonl"
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models resolved to an empty list")

    pairs = sample_pairs(
        list(read_jsonl(pairs_path)),
        per_dataset=args.per_dataset,
        seed=int(config.get("random_seed", 0)),
    )
    tasks = []
    for pair in pairs:
        for model in models:
            task_id = stable_id(pair["pair_id"], model, "revision_control_v1")
            tasks.append(
                {
                    "task_id": task_id,
                    "condition": "revision_control",
                    "pair_id": pair["pair_id"],
                    "model": model,
                    "source_dataset": pair.get("source_dataset"),
                    "source_row_id": pair.get("source_row_id"),
                    "seed_id": pair.get("seed_id"),
                    "seed_type": pair.get("seed_type"),
                    "entity": pair.get("entity"),
                    "seed_claim": pair["seed_claim"],
                    "corrected_claim": pair["corrected_claim"],
                    "evidence_hint": pair.get("evidence_hint", ""),
                    "probe": pair["probe"],
                    "prompt": build_revision_prompt(pair),
                }
            )

    write_jsonl(out_path, tasks)
    manifest_path = out_path.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config_path,
        config,
        {
            "task_type": "revision_control",
            "pairs_path": to_project_relative(pairs_path),
            "output_path": to_project_relative(out_path),
            "condition": "revision_control",
            "models": models,
            "per_dataset": args.per_dataset,
            "num_pairs": len(pairs),
            "num_tasks": len(tasks),
            "datasets": sorted({str(p.get("source_dataset", "unknown")) for p in pairs}),
            "prompt_version": "revision_control_v1",
        },
    )
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out_path),
                "tasks": len(tasks),
                "pairs": len(pairs),
                "models": models,
                "manifest": to_project_relative(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
