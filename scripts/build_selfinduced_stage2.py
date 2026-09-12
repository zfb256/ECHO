from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from ids import stable_id
from jsonl import read_jsonl, write_jsonl
from manifest import (
    file_sha256,
    load_output_manifest,
    require_model_binding,
    require_task_binding,
    write_manifest,
)


# Self-induced arm, stage 2: take the model's turn-1 answer (from stage-1
# outputs) and continue the dialogue with a follow-up that builds on it. If the
# model later elaborates falsehoods that depend on its own turn-1 claim, that is
# self-induced contagion -- no injection required.
#
# Run AFTER the server has generated stage-1 outputs:
#   datasets_zh/runs/<run_name>/selfinduced_stage1_outputs.jsonl
# with required fields: task_id, pair_id, q_id, model, response.


REQUIRED_STAGE1_FIELDS = {"task_id", "pair_id", "q_id", "model", "response", "truth_statement"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SELF-INDUCED stage-2 continuation tasks from stage-1 outputs.")
    parser.add_argument("--config", default="configs/zh_selfinduced.json")
    parser.add_argument("--stage1-outputs", default=None, help="Defaults to runs_dir/<run_name>/selfinduced_stage1_outputs.jsonl")
    parser.add_argument("--tasks-meta", default=None, help="Stage-1 tasks file for question/followup lookup. Defaults to runs_dir/<run_name>/selfinduced_stage1_tasks.jsonl")
    parser.add_argument("--output", default=None, help="Defaults to runs_dir/<run_name>/selfinduced_stage2_tasks.jsonl")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a partial/empty-response stage-1 diagnostic. Never use for the formal study.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    config_path = resolve_project_path(args.config)
    run_name = config["pilot"].get("run_name", "pilot")
    runs_dir = resolve_project_path(config["paths"]["runs_dir"])
    run_dir = runs_dir / run_name
    prompts = config["pilot"].get("prompts", {})
    s1_out_path = resolve_project_path(args.stage1_outputs) if args.stage1_outputs else run_dir / "selfinduced_stage1_outputs.jsonl"
    meta_path = resolve_project_path(args.tasks_meta) if args.tasks_meta else run_dir / "selfinduced_stage1_tasks.jsonl"
    out_path = resolve_project_path(args.output) if args.output else run_dir / "selfinduced_stage2_tasks.jsonl"

    meta_rows = list(read_jsonl(meta_path))
    meta = {row["task_id"]: row for row in meta_rows}
    if len(meta) != len(meta_rows):
        raise ValueError(f"Duplicate task_id in stage-1 task metadata: {meta_path}")
    bank_path = resolve_project_path(config["paths"]["selfinduced_bank"])
    bank_q_ids = {
        q["q_id"]
        for q in json.loads(bank_path.read_text(encoding="utf-8")).get("questions", [])
    }
    expected_q_models = {
        (q_id, model)
        for q_id in bank_q_ids
        for model in config["pilot"].get("model_pool", [])
    }
    meta_q_model_list = [
        (row.get("q_id"), row.get("model")) for row in meta_rows
    ]
    if not args.allow_partial and (
        not expected_q_models
        or len(meta_q_model_list) != len(expected_q_models)
        or set(meta_q_model_list) != expected_q_models
    ):
        raise SystemExit(
            "Stage-1 task metadata does not contain exactly one row for every "
            f"question/model cell: configured={len(expected_q_models)} "
            f"rows={len(meta_q_model_list)} unique={len(set(meta_q_model_list))}."
        )

    tasks = []
    skipped_empty = 0   # failed/empty stage-1 responses -> no turn-1 claim to inherit
    skipped_no_meta = 0  # stage-1 task_id not in the tasks-meta file -> cannot reconstruct the question
    output_rows = list(read_jsonl(s1_out_path))
    output_manifest = load_output_manifest(s1_out_path, meta_path)
    output_ids = [row.get("task_id") for row in output_rows]
    if any(not task_id for task_id in output_ids) or len(set(output_ids)) != len(output_ids):
        raise ValueError(f"Missing or duplicate task_id in stage-1 outputs: {s1_out_path}")
    meta_ids = set(meta)
    observed_ids = set(output_ids)
    if not args.allow_partial and (
        observed_ids != meta_ids or len(output_rows) != len(meta_rows)
    ):
        raise SystemExit(
            "Refusing to build formal stage-2 tasks from incomplete stage-1 outputs: "
            f"expected={len(meta_rows)} observed={len(output_rows)} "
            f"missing={len(meta_ids - observed_ids)} "
            f"unexpected={len(observed_ids - meta_ids)}. "
            "Finish all stage-1 models, or use --allow-partial for a diagnostic."
        )
    for row in output_rows:
        missing = REQUIRED_STAGE1_FIELDS - set(row)
        if missing:
            raise ValueError(f"stage-1 output missing fields {sorted(missing)}; task_id={row.get('task_id')}")
        turn1 = (row.get("response") or "").strip()
        if not turn1:
            skipped_empty += 1
            continue
        if row["task_id"] not in meta:
            skipped_no_meta += 1
            continue
        m = meta[row["task_id"]]
        require_task_binding(row, m)
        require_model_binding(row, output_manifest)
        question = m.get("question", "")
        # Preserve the follow-up recorded in the stage-1 task, including an empty string.
        followup = m.get("followup", config["pilot"]["selfinduced"].get("stage2_followup_default", ""))
        # Replay turn 1 as an assistant message in the model's native chat format.
        # `prompt` remains available when chat templates are disabled or unavailable.
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": turn1},
            {"role": "user", "content": followup},
        ]
        fallback_template = prompts.get(
            "selfinduced_stage2_fallback",
            "Continue this dialogue faithfully and concisely.\n\n"
            "User: {question}\nAssistant: {turn1}\nUser: {followup}",
        )
        prompt = fallback_template.format(question=question, turn1=turn1, followup=followup)
        task_id = stable_id(m["pair_id"], m["model"], "self_induced")
        tasks.append(
            {
                "task_id": task_id,
                "pair_id": m["pair_id"],
                "condition": "self_induced",
                "model": m["model"],
                "source_dataset": "selfinduced",
                "q_id": m["q_id"],
                "question": question,
                "truth_statement": m["truth_statement"],
                "evidence_hint": m.get("evidence_hint"),
                "turn1_response": turn1,
                "messages": messages,
                "prompt": prompt,
            }
        )

    if skipped_no_meta:
        raise ValueError(
            f"{skipped_no_meta} stage-1 outputs have no matching task metadata; "
            "refusing to create a silently incomplete/misaligned stage-2 task file."
        )
    if skipped_empty and not args.allow_partial:
        raise SystemExit(
            f"Refusing to build a formal stage-2 file after {skipped_empty} empty "
            "stage-1 responses. Re-run those generation failures; otherwise the "
            "self-induced denominator would silently shrink."
        )
    stage2_ids = [task["task_id"] for task in tasks]
    if len(stage2_ids) != len(set(stage2_ids)):
        raise ValueError("Generated stage-2 tasks contain duplicate task_id values.")
    write_jsonl(out_path, tasks)
    manifest_path = out_path.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config_path,
        config,
        {
            "builder_path": "scripts/build_selfinduced_stage2.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "stage1_outputs": to_project_relative(s1_out_path),
            "tasks_path": to_project_relative(out_path),
            "num_tasks": len(tasks),
            "skipped_empty_response": skipped_empty,
            "skipped_no_meta": skipped_no_meta,
            "stage1_outputs_sha256": file_sha256(s1_out_path),
            "stage1_tasks_sha256": file_sha256(meta_path),
        },
    )
    print(json.dumps({"wrote": str(out_path), "tasks": len(tasks), "skipped_empty": skipped_empty, "skipped_no_meta": skipped_no_meta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
