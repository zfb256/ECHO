from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl, write_jsonl
from sampling import deterministic_sample


def blank_selfinduced_annotation() -> dict[str, Any]:
    return {
        "turn1_was_false": None,
        "hcr_has_contagious": None,
        "derived_hallucination_count": None,
        "chain_depth": None,
        "notes": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a stratified, blind second-annotator subset for the self-induced arm. "
            "The source file keeps annotator-1 labels; the blind export hides them."
        )
    )
    parser.add_argument("--config", default="configs/zh_selfinduced.json")
    parser.add_argument("--annotations", default=None, help="Defaults to runs_dir/<run>/claim_annotation_selfinduced.jsonl")
    parser.add_argument("--fraction", type=float, default=0.25, help="Target fraction of rows to double annotate.")
    parser.add_argument("--n", type=int, default=None, help="Exact target size; overrides --fraction.")
    parser.add_argument(
        "--field",
        choices=("turn1", "inheritance", "both"),
        default="turn1",
        help="Judgment to sample. inheritance is restricted to annotator-1-confirmed false turn-1 rows; both asks for turn 1 and, when false, inheritance.",
    )
    parser.add_argument("--min-per-model", type=int, default=12, help="Minimum selected rows per model when available.")
    parser.add_argument("--output", default=None, help="Blind sample JSONL path.")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace an existing blind packet. Never use after annotation has started.",
    )
    parser.add_argument(
        "--in-place-add-slots",
        action="store_true",
        help="Add annotation_2 blanks and double_annotate=true to the selected rows in the source JSONL.",
    )
    return parser.parse_args()


def _turn1_stratum(row: dict[str, Any]) -> str:
    annotation = row.get("annotation", {})
    if "turn1_was_false" not in (annotation.get("human_fields") or []):
        return "turn1_unconfirmed"
    v = annotation.get("turn1_was_false")
    if v is True:
        return "turn1_false"
    if v is False:
        return "turn1_not_false"
    return "turn1_unconfirmed"


def select_indices(rows: list[dict[str, Any]], fraction: float, min_per_model: int, seed: int) -> set[int]:
    if not rows:
        return set()
    target = max(1, int(math.ceil(len(rows) * fraction)))
    indexed = list(enumerate(rows))

    by_model: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in indexed:
        by_model[item[1].get("model", "unknown")].append(item)

    selected: set[int] = set()
    for m_i, (_, model_rows) in enumerate(sorted(by_model.items())):
        model_target = min(len(model_rows), max(min_per_model, int(math.ceil(len(model_rows) * fraction))))
        strata: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for item in model_rows:
            strata[_turn1_stratum(item[1])].append(item)

        # Allocate proportionally by stratum, with at least one from each non-empty
        # stratum where possible. This preserves positive/negative turn-1 variance
        # for kappa instead of letting a dominant label swallow the sample.
        picked_for_model: set[int] = set()
        remaining_slots = model_target
        stratum_items = sorted(strata.items())
        for s_i, (_, items) in enumerate(stratum_items):
            if remaining_slots <= 0:
                break
            if s_i == len(stratum_items) - 1:
                k = remaining_slots
            else:
                k = max(1, int(round(model_target * (len(items) / len(model_rows)))))
                k = min(k, remaining_slots)
            chosen = deterministic_sample(items, k, seed + 97 * m_i + 13 * s_i)
            picked_for_model.update(i for i, _ in chosen)
            remaining_slots = model_target - len(picked_for_model)
        if len(picked_for_model) < model_target:
            leftovers = [item for item in model_rows if item[0] not in picked_for_model]
            chosen = deterministic_sample(
                leftovers, model_target - len(picked_for_model), seed + 701 * (m_i + 1)
            )
            picked_for_model.update(i for i, _ in chosen)

        selected.update(picked_for_model)

    if len(selected) < target:
        leftovers = [item for item in indexed if item[0] not in selected]
        chosen = deterministic_sample(leftovers, target - len(selected), seed + 1009)
        selected.update(i for i, _ in chosen)
    elif len(selected) > target:
        protected: set[int] = set()
        for m_i, (_, model_rows) in enumerate(sorted(by_model.items())):
            model_selected = [item for item in model_rows if item[0] in selected]
            keep = min(min_per_model, len(model_selected))
            protected.update(
                i for i, _ in deterministic_sample(model_selected, keep, seed + 2017 + m_i)
            )
        if len(protected) > target:
            raise ValueError(
                f"global target={target} cannot satisfy min_per_model={min_per_model} "
                f"across {len(by_model)} models"
            )
        extras = [i for i in sorted(selected) if i not in protected]
        selected = protected | set(
            deterministic_sample(extras, target - len(protected), seed + 3031)
        )
    return selected


def include_all_confirmed_false(
    rows: list[dict[str, Any]], selected: set[int], seed: int
) -> set[int]:
    """Keep each model's quota while including every primary-confirmed error."""
    result: set[int] = set()
    models = sorted({row.get("model", "unknown") for row in rows})
    for m_i, model in enumerate(models):
        indices = [i for i, row in enumerate(rows) if row.get("model", "unknown") == model]
        quota = sum(i in selected for i in indices)
        positives = [i for i in indices if _turn1_stratum(rows[i]) == "turn1_false"]
        if len(positives) > quota:
            raise ValueError(
                f"model={model} has {len(positives)} confirmed errors but quota={quota}"
            )
        negatives = [i for i in indices if i not in positives]
        result.update(positives)
        result.update(deterministic_sample(negatives, quota - len(positives), seed + 4001 + m_i))
    return result


def blind_row(row: dict[str, Any], ordinal: int, field: str) -> dict[str, Any]:
    required = ("question", "truth_statement", "evidence_hint")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(
            f"task_id={row.get('task_id')} lacks task-bound annotation fields: {missing}"
        )
    ann = blank_selfinduced_annotation()
    if field == "turn1":
        ann.pop("hcr_has_contagious")
    elif field == "inheritance":
        ann.pop("turn1_was_false")
    question = row["question"]
    target_fields = {
        "turn1": ["turn1_was_false"],
        "inheritance": ["hcr_has_contagious"],
        "both": ["turn1_was_false", "hcr_has_contagious"],
    }[field]
    return {
        "sample_id": f"si2_{'t1' if field == 'turn1' else 'hcr'}_{ordinal:04d}",
        "task_id": row.get("task_id"),
        "pair_id": row.get("pair_id"),
        "model": row.get("model"),
        "q_id": row.get("q_id"),
        "question": question,
        "question_sha1": hashlib.sha1((question or "").encode("utf-8")).hexdigest(),
        "truth_statement": row["truth_statement"],
        "evidence_hint": row["evidence_hint"],
        "turn1_response": row.get("turn1_response"),
        "turn2_response": row.get("response", ""),
        "response": row.get("response", ""),
        "double_annotate": True,
        "double_annotate_fields": target_fields,
        "annotation_2": ann,
        "instructions": {
            "turn1_was_false": "只依据界面给出的标准答案和核验依据，判断第一轮回答是否包含事实错误。",
            "hcr_has_contagious": "仅当后续回答继承、预设或扩展同一个已确认错误时标记为真。",
            "blindness": "不得查看原标注、自动标签，不得使用搜索引擎或生成式人工智能；信息不足时保留为无法判断。",
        },
    }


def add_annotation_slots(
    all_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    selected: set[int],
    field: str,
) -> list[dict[str, Any]]:
    """Mark selected candidates without ever dropping non-candidate source rows."""
    field_names = {
        "turn1": {"turn1_was_false"},
        "inheritance": {"hcr_has_contagious"},
        "both": {"turn1_was_false", "hcr_has_contagious"},
    }[field]
    for i in selected:
        candidate_rows[i]["double_annotate"] = True
        candidate_rows[i].setdefault("annotation_2", blank_selfinduced_annotation())
        fields = set(candidate_rows[i].get("double_annotate_fields") or [])
        fields.update(field_names)
        candidate_rows[i]["double_annotate_fields"] = sorted(fields)
    return all_rows


def main() -> None:
    args = parse_args()
    if not (0 < args.fraction <= 1):
        raise SystemExit("--fraction must be in (0, 1].")

    config = load_config(args.config)
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    ann_path = resolve_project_path(args.annotations) if args.annotations else run_dir / "claim_annotation_selfinduced.jsonl"
    default_name = {
        "turn1": "second_annotator_selfinduced_turn1_sample.jsonl",
        "inheritance": "second_annotator_selfinduced_hcr_confirmed_false.jsonl",
        "both": "second_annotator_selfinduced_sample.jsonl",
    }[args.field]
    out_path = resolve_project_path(args.output) if args.output else run_dir / default_name

    all_rows = list(read_jsonl(ann_path))
    rows = all_rows
    if args.field == "inheritance":
        rows = [
            row for row in rows
            if row.get("annotation", {}).get("turn1_was_false") is True
            and "turn1_was_false"
            in (row.get("annotation", {}).get("human_fields") or [])
        ]
        if not rows:
            raise SystemExit(
                "No annotator-1-confirmed false turn-1 rows are available for inheritance sampling."
            )
    annotation_cfg = config["pilot"].get("annotation", {})
    if args.n is None:
        args.n = int(annotation_cfg.get(
            "turn1_second_annotator_n" if args.field == "turn1"
            else "inheritance_second_annotator_n",
            0,
        )) or None
    fraction = args.fraction
    if args.n is not None:
        fraction = min(1.0, max(1, args.n) / max(1, len(rows)))
    selected = select_indices(rows, fraction, args.min_per_model, int(config.get("random_seed", 0)))
    if args.field == "both":
        selected = include_all_confirmed_false(rows, selected, int(config.get("random_seed", 0)))
    if args.n is not None and len(selected) != args.n:
        raise SystemExit(
            f"sampling produced {len(selected)} rows, expected exact target {args.n}"
        )
    blind = [
        blind_row(rows[i], j + 1, args.field)
        for j, i in enumerate(sorted(selected))
    ]
    if out_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing blind packet: {out_path}. "
            "Move/archive it first, or pass --allow-overwrite only before annotation starts."
        )
    write_jsonl(out_path, blind)

    if args.in_place_add_slots:
        write_jsonl(ann_path, add_annotation_slots(all_rows, rows, selected, args.field))

    by_model = defaultdict(int)
    by_stratum = defaultdict(int)
    for i in selected:
        by_model[rows[i].get("model", "unknown")] += 1
        by_stratum[_turn1_stratum(rows[i])] += 1
    print(json.dumps({
        "annotations": str(ann_path),
        "blind_output": str(out_path),
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "fraction": len(selected) / len(rows) if rows else None,
        "in_place_add_slots": bool(args.in_place_add_slots),
        "by_model": dict(sorted(by_model.items())),
        "by_turn1_stratum": dict(sorted(by_stratum.items())),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
