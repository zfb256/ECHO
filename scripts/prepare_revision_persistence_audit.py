"""Build the blind human-audit packet for injected correction-control persistence.

Why this audit exists. The paper reports post-correction persistence $0.000$ for the injected
correction-control arm (Section 5.3). That zero is produced by the NLI mention-vs.-assertion
guard: the raw token rule flags 80 of 1,800 outputs as still asserting the seed falsehood, and
the guard drops every one of them. So the single headline number that is exactly zero rests on
an unaudited model judgment, in a paper whose stated position is "we measure the labeler rather
than assume it". This packet puts those 80 rows in front of a human.

The rule-only pass needs no NLI model and no GPU, so the packet can be built anywhere:
    python -B scripts/prepare_revision_persistence_audit.py

Rows are emitted blind: the packet carries the seed falsehood, the corrected claim, the user's
correction turn, and the response to judge, but NOT the rule verdict and NOT the NLI verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from detector import detect_injected  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from sampling import deterministic_sample  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the blind persistence-audit packet (rule-flagged rows).")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--tasks", default="datasets/runs/full_study/revision_control_tasks.jsonl")
    p.add_argument("--outputs", default="datasets/runs/full_study/revision_control_outputs.jsonl")
    p.add_argument("--output", default="datasets/runs/full_study/revision_persistence_audit_sample.jsonl")
    p.add_argument("--n", type=int, default=None, help="Optional deterministic cap after collecting rule-flagged rows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))

    tasks = {r["task_id"]: r for r in read_jsonl(resolve_project_path(args.tasks))}
    outputs = list(read_jsonl(resolve_project_path(args.outputs)))
    out_path = resolve_project_path(args.output)

    rows = []
    for out in outputs:
        task = tasks.get(out.get("task_id"))
        if task is None:
            continue
        text = out.get("response") or ""
        det = detect_injected(text, task["seed_claim"], task["corrected_claim"], neg_window)
        if not det["asserts_false"]:
            continue
        rows.append(
            {
                "audit_id": f"persist_{len(rows):04d}",
                "task_id": out.get("task_id"),
                "pair_id": task.get("pair_id"),
                "model": task.get("model"),
                "source_dataset": task.get("source_dataset"),
                "seed_id": task.get("seed_id"),
                "seed_type": task.get("seed_type"),
                "seed_claim": task.get("seed_claim"),
                "corrected_claim": task.get("corrected_claim"),
                "response": text,
                "human_still_asserts_false": None,
                "human_uncertain": False,
                "human_notes": "",
                "instructions": (
                    "用户已经给出正确事实。仅当回答仍把原错误值作为事实断言或预设时标记为真。"
                    "如果回答给出正确值、明确纠正、仅提及错误值、无关或回避，则标记为假。"
                    "不得使用搜索引擎或生成式人工智能。"
                ),
            }
        )

    target_n = args.n
    if target_n is None:
        target_n = int(config.get("pilot", {}).get("annotation", {}).get("correction_audit_n", 0)) or None
    if target_n is not None and len(rows) > target_n:
        rows = deterministic_sample(rows, target_n, int(config.get("random_seed", 0)) + 511)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing packet {to_project_relative(out_path)}; move it aside first")
    write_jsonl(out_path, rows)
    by_model: dict[str, int] = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    print(json.dumps(
        {
            "wrote": to_project_relative(out_path),
            "n_rows": len(rows),
            "n_outputs_scanned": len(outputs),
            "by_model": by_model,
            "note": "rule-flagged rows only; all were dropped by the NLI guard in the shipped run",
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
