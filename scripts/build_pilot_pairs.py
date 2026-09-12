from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl, write_jsonl
from sampling import deterministic_sample
from text_fields import dialogue_source, extract_dialogue_text, truncate_to_turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean/contaminated prompt pairs.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--dataset", action="append", help="Dataset key to process. Defaults to all configured datasets.")
    return parser.parse_args()


def load_seed_bank(path_value: str) -> list[dict[str, Any]]:
    path = resolve_project_path(path_value)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    seeds = data.get("seeds", [])
    if not seeds:
        raise ValueError(f"No seeds found in {path}")
    unverified = [s.get("seed_id", "?") for s in seeds if s.get("needs_human_verification")]
    if unverified:
        raise ValueError(
            f"Refusing to build formal pairs: {len(unverified)} injected seeds still "
            f"need human verification: {unverified[:10]}"
        )
    return seeds


MAX_DIALOGUE_CHARS = 8000  # applied after turn truncation


MIN_TURNS = 2  # a "multi-turn dialogue context" must have at least this many turns


def make_pair(
    row: dict[str, Any],
    seed: dict[str, Any],
    pair_id: str,
    turns_target: int,
    prompts: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload = row["payload"]
    # Reject unknown schemas instead of placing metadata in the prompt.
    src = dialogue_source(payload)
    dialogue_text = extract_dialogue_text(payload)
    dialogue_text = truncate_to_turns(dialogue_text, turns_target)[:MAX_DIALOGUE_CHARS]
    n_turns = sum(1 for ln in dialogue_text.splitlines() if ln.strip())
    if src is None or n_turns < MIN_TURNS:
        raise ValueError(
            f"{row.get('source_dataset')} row {row.get('row_id')}: could not extract a multi-turn "
            f"dialogue (source_key={src!r}, turns={n_turns}). Payload keys: {sorted(payload)[:12]}. "
            f"Add the correct key to text_fields.PREFERRED_TEXT_KEYS (or a per-dataset adapter) and re-run."
        )
    # Put the seed immediately before an entity probe so every arm has the same occasion
    # to answer the affected fact; clean arms omit only the seed turn.
    prompts = prompts or {}
    entity = (seed.get("entity") or prompts.get("default_entity", "that topic")).strip()
    probe_template = prompts.get(
        "entity_probe",
        "Before we wrap up — could you briefly restate the key fact about {entity} that we should remember?",
    )
    probe = probe_template.format(entity=entity)
    clean_instruction = prompts.get(
        "clean_instruction",
        "Continue this multi-turn dialogue faithfully and concisely. Do not introduce unsupported facts.",
    )
    contaminated_instruction = prompts.get(
        "contaminated_instruction",
        "Continue this multi-turn dialogue faithfully and concisely.",
    )
    user_role = prompts.get("user_role", "User")
    assistant_role = prompts.get("assistant_role", "Assistant")
    clean_prompt = (
        f"{clean_instruction}\n\n"
        f"{dialogue_text}\n"
        f"{user_role}: {probe}"
    )
    contaminated_prompt = (
        f"{contaminated_instruction}\n\n"
        f"{dialogue_text}\n"
        f"{assistant_role}: {seed['claim']}\n"
        f"{user_role}: {probe}"
    )
    return {
        "pair_id": pair_id,
        "source_dataset": row["source_dataset"],
        "source_row_id": row["row_id"],
        "seed_id": seed["seed_id"],
        "seed_type": seed["seed_type"],
        "seed_claim": seed["claim"],
        "corrected_claim": seed.get("corrected_claim"),
        "evidence_hint": seed.get("evidence_hint"),
        "entity": entity,
        "probe": probe,
        "design": "probe-v2",
        "clean_prompt": clean_prompt,
        "contaminated_prompt": contaminated_prompt,
        "annotation_status": "unlabeled",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    seeds = load_seed_bank(config["paths"]["seed_bank"])
    selected = set(args.dataset or [])
    random_seed = int(config["random_seed"])
    turns_target = int(config["pilot"]["turns_target"])
    prompts = config["pilot"].get("prompts", {})
    # Namespace processed pairs by run name.
    run_name = config["pilot"].get("run_name", "pilot")
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])

    all_pairs = []
    seed_cursor = 0  # continue seed assignment across datasets
    for ds_idx, spec in enumerate(config["datasets"]):
        if selected and spec["key"] not in selected:
            continue
        raw_path = raw_dir / f"{spec['key']}_{spec['split']}.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw dataset file: {raw_path}. Run scripts/download_hf_datasets.py first.")

        rows = list(read_jsonl(raw_path))
        sample_size = int(spec["sample_size"])
        if len(rows) < sample_size:
            warnings.warn(
                f"{spec['key']} has only {len(rows)} rows, less than requested sample_size={sample_size}.",
                stacklevel=2,
            )
        sampled = deterministic_sample(rows, sample_size, random_seed + ds_idx)
        pairs = []
        for idx, row in enumerate(sampled):
            seed = seeds[seed_cursor % len(seeds)]
            seed_cursor += 1
            pair_id = f"{spec['key']}-{idx:04d}"
            pairs.append(make_pair(row, seed, pair_id, turns_target, prompts))
        out_path = processed_dir / f"{run_name}_pairs_{spec['key']}.jsonl"
        write_jsonl(out_path, pairs)
        all_pairs.extend(pairs)
        print(json.dumps({"wrote": str(out_path), "pairs": len(pairs)}, ensure_ascii=False))

    combined_path = processed_dir / f"{run_name}_pairs_all.jsonl"
    write_jsonl(combined_path, all_pairs)
    print(json.dumps({"wrote": str(combined_path), "pairs": len(all_pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
