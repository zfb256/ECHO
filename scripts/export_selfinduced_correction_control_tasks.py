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
    p = argparse.ArgumentParser(description="Export self-induced correction-control stage-3 tasks.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument(
        "--annotations",
        default=None,
        help="Defaults to datasets/runs/<run_name>/claim_annotation_selfinduced.jsonl.",
    )
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument(
        "--max-per-model",
        type=int,
        default=30,
        help="Use all confirmed-false rows up to this count; otherwise sample this many per model.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Defaults to datasets/runs/<run_name>/selfinduced_correction_control_tasks.jsonl.",
    )
    return p.parse_args()


def load_selfinduced_bank(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bank_path = resolve_project_path(config["paths"]["selfinduced_bank"])
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    return {str(q["q_id"]): q for q in bank.get("questions", [])}


def sample_rows(rows: list[dict[str, Any]], max_per_model: int, seed: int) -> list[dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[str(row.get("model", "unknown"))].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for model in sorted(by_model):
        model_rows = sorted(by_model[model], key=lambda r: str(r.get("task_id", "")))
        if len(model_rows) > max_per_model:
            model_rows = rng.sample(model_rows, max_per_model)
            model_rows = sorted(model_rows, key=lambda r: str(r.get("task_id", "")))
        selected.extend(model_rows)
    return selected


def build_messages(row: dict[str, Any], truth_statement: str, followup: str) -> list[dict[str, str]]:
    question = str(row.get("question") or "")
    turn1 = str(row.get("turn1_response") or "")
    turn2 = str(row.get("response") or "")
    correction = (
        "Correction: my earlier answer to the first question was wrong. "
        f"The correct information is: {truth_statement}\n"
        "Using that corrected information, answer the follow-up again. "
        f"{followup}"
    )
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": turn1},
        {"role": "user", "content": followup},
        {"role": "assistant", "content": turn2},
        {"role": "user", "content": correction},
    ]


def build_prompt(messages: list[dict[str, str]]) -> str:
    lines = ["Continue this dialogue faithfully and concisely."]
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.max_per_model <= 0:
        raise SystemExit("--max-per-model must be positive")

    config = load_config(args.config)
    config_path = resolve_project_path(args.config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    ann_path = (
        resolve_project_path(args.annotations)
        if args.annotations
        else run_dir / "claim_annotation_selfinduced.jsonl"
    )
    out_path = (
        resolve_project_path(args.output)
        if args.output
        else run_dir / "selfinduced_correction_control_tasks.jsonl"
    )
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models resolved to an empty list")

    bank = load_selfinduced_bank(config)
    model_set = set(models)
    candidates = []
    for row in read_jsonl(ann_path):
        if row.get("model") not in model_set:
            continue
        if (row.get("annotation") or {}).get("turn1_was_false") is not True:
            continue
        qid = str(row.get("q_id"))
        meta = bank.get(qid)
        if not meta:
            raise SystemExit(f"q_id={qid} missing from self-induced bank")
        truth = str(meta.get("truth_statement") or "").strip()
        followup = str(meta.get("followup") or "").strip()
        if not truth or not followup:
            raise SystemExit(f"q_id={qid} missing truth_statement/followup in self-induced bank")
        candidates.append(row)

    selected = sample_rows(candidates, args.max_per_model, int(config.get("random_seed", 0)) + 303)
    tasks = []
    for row in selected:
        qid = str(row.get("q_id"))
        meta = bank[qid]
        truth = str(meta["truth_statement"])
        followup = str(meta["followup"])
        messages = build_messages(row, truth, followup)
        task_id = stable_id(row.get("task_id"), row.get("model"), "selfinduced_correction_control_v1")
        tasks.append(
            {
                "task_id": task_id,
                "condition": "self_induced_correction_control",
                "source_task_id": row.get("task_id"),
                "pair_id": row.get("pair_id"),
                "model": row.get("model"),
                "source_dataset": "selfinduced",
                "q_id": qid,
                "question": row.get("question"),
                "followup": followup,
                "turn1_response": row.get("turn1_response"),
                "stage2_response": row.get("response"),
                "truth_statement": truth,
                "evidence_hint": meta.get("evidence_hint", ""),
                "messages": messages,
                "prompt": build_prompt(messages),
            }
        )

    write_jsonl(out_path, tasks)
    by_model = defaultdict(int)
    for task in tasks:
        by_model[str(task.get("model"))] += 1
    manifest_path = out_path.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config_path,
        config,
        {
            "task_type": "selfinduced_correction_control",
            "condition": "self_induced_correction_control",
            "annotations_path": to_project_relative(ann_path),
            "output_path": to_project_relative(out_path),
            "models": models,
            "max_per_model": args.max_per_model,
            "num_candidate_confirmed_false": len(candidates),
            "num_tasks": len(tasks),
            "by_model": dict(sorted(by_model.items())),
            "prompt_version": "selfinduced_correction_control_v1",
        },
    )
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out_path),
                "tasks": len(tasks),
                "by_model": dict(sorted(by_model.items())),
                "manifest": to_project_relative(manifest_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
