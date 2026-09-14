from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from ids import stable_id
from jsonl import write_jsonl
from manifest import file_sha256, write_manifest


# Stage 1 generates answers for replay; factual errors are identified after generation.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SELF-INDUCED stage-1 elicitation tasks for server inference.")
    parser.add_argument("--config", default="configs/zh_selfinduced.json")
    parser.add_argument("--output", default=None, help="Defaults to runs_dir/<run_name>/selfinduced_stage1_tasks.jsonl")
    return parser.parse_args()


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"No questions found in {path}")
    unverified = [q.get("q_id", "?") for q in questions if q.get("needs_human_verification")]
    if unverified:
        raise ValueError(
            f"Refusing to export model tasks: {len(unverified)} self-induced questions "
            f"still need human verification: {unverified[:10]}"
        )
    return questions


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    config_path = resolve_project_path(args.config)
    run_name = config["pilot"].get("run_name", "pilot")
    runs_dir = resolve_project_path(config["paths"]["runs_dir"])
    out_path = resolve_project_path(args.output) if args.output else runs_dir / run_name / "selfinduced_stage1_tasks.jsonl"

    si_cfg = config["pilot"].get("selfinduced", {})
    if not si_cfg.get("enabled", False):
        raise SystemExit("selfinduced.enabled is false in config; nothing to export.")

    bank_path = resolve_project_path(config["paths"]["selfinduced_bank"])
    all_questions = load_questions(bank_path)
    available = len(all_questions)
    limit = int(si_cfg.get("questions_per_model", available))
    if limit > available:
        warnings.warn(
            f"selfinduced.questions_per_model={limit} but the bank has only {available} questions; "
            f"using {available}. Add more questions to reach the designed statistical power.",
            stacklevel=2,
        )
    questions = all_questions[:limit]
    model_pool = config["pilot"]["model_pool"]
    prompts = config["pilot"].get("prompts", {})
    template = prompts.get(
        "selfinduced_stage1",
        "Answer the following factual question in one short paragraph. "
        "State your answer directly; do not say you are unsure unless you truly have no idea.\n\n"
        "Question: {question}",
    )
    default_followup = si_cfg.get("stage2_followup_default", "")

    tasks = []
    for q in questions:
        prompt = template.format(question=q["question"])
        for model in model_pool:
            task_id = stable_id(q["q_id"], model, "self_induced_stage1")
            tasks.append(
                {
                    "task_id": task_id,
                    "pair_id": stable_id(q["q_id"], model),
                    "condition": "self_induced_stage1",
                    "model": model,
                    "source_dataset": "selfinduced",
                    "q_id": q["q_id"],
                    "question": q["question"],
                    "followup": q.get("followup", default_followup),
                    # Bind the reference fact to the task, independent of later bank edits.
                    "truth_statement": q["truth_statement"],
                    "evidence_hint": q.get("evidence_hint"),
                    "prompt": prompt,
                }
            )

    write_jsonl(out_path, tasks)
    manifest_path = out_path.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config_path,
        config,
        {
            "builder_path": "scripts/build_selfinduced_stage1.py",
            "builder_sha256": file_sha256(Path(__file__)),
            "tasks_path": to_project_relative(out_path),
            "num_tasks": len(tasks),
            "questions_requested": limit,
            "questions_available": available,
            "questions_used": len(questions),
            "selfinduced_bank_path": to_project_relative(bank_path),
            "selfinduced_bank_sha256": file_sha256(bank_path),
        },
    )
    print(json.dumps({"wrote": str(out_path), "tasks": len(tasks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
