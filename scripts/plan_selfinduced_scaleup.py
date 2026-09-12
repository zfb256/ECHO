from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from jsonl import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the self-induced question-bank size needed for target confirmed false turn-1 counts."
    )
    parser.add_argument("--config", default="configs/full_study.json")
    parser.add_argument("--annotations", default=None, help="Defaults to runs_dir/<run>/claim_annotation_selfinduced.jsonl")
    parser.add_argument("--targets", default="500,800", help="Comma-separated target counts of confirmed false turn-1 answers.")
    parser.add_argument("--output", default=None, help="Defaults to reports_dir/selfinduced_scaleup_plan.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    reports_dir = resolve_project_path(config["paths"]["reports_dir"])
    ann_path = resolve_project_path(args.annotations) if args.annotations else run_dir / "claim_annotation_selfinduced.jsonl"
    out_path = resolve_project_path(args.output) if args.output else reports_dir / "selfinduced_scaleup_plan.json"

    rows = list(read_jsonl(ann_path))
    models = sorted({r.get("model", "unknown") for r in rows})
    n_models = len(models)
    confirmed = [r for r in rows if r.get("annotation", {}).get("turn1_was_false") is True]
    false_rate = len(confirmed) / len(rows) if rows else 0.0

    per_model = defaultdict(lambda: {"n_total": 0, "n_turn1_false": 0})
    for r in rows:
        bucket = per_model[r.get("model", "unknown")]
        bucket["n_total"] += 1
        if r.get("annotation", {}).get("turn1_was_false") is True:
            bucket["n_turn1_false"] += 1
    per_model_out = {}
    for model, bucket in sorted(per_model.items()):
        n = bucket["n_total"]
        f = bucket["n_turn1_false"]
        per_model_out[model] = {
            "n_total": n,
            "n_turn1_false": f,
            "turn1_false_rate": f / n if n else None,
        }

    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    current_questions = len(rows) // n_models if n_models else None
    recommendations = []
    for target in targets:
        if false_rate <= 0 or n_models == 0:
            needed_total_rows = needed_questions = additional_questions = None
        else:
            needed_total_rows = int(math.ceil(target / false_rate))
            needed_questions = int(math.ceil(needed_total_rows / n_models))
            additional_questions = max(0, needed_questions - int(current_questions or 0))
        recommendations.append({
            "target_confirmed_false": target,
            "estimated_total_rows_needed": needed_total_rows,
            "estimated_questions_per_model_needed": needed_questions,
            "additional_verified_questions_needed": additional_questions,
        })

    report = {
        "annotations": to_project_relative(ann_path),
        "models": models,
        "n_models": n_models,
        "current_total_rows": len(rows),
        "current_questions_per_model": current_questions,
        "current_confirmed_false": len(confirmed),
        "observed_turn1_false_rate": false_rate,
        "per_model": per_model_out,
        "recommendations": recommendations,
        "note": (
            "This extrapolates from the existing verified question bank and deterministic stage-1 setup. "
            "Additional questions must be human-verified before generation; duplicating the same prompts is not valid scale-up."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
