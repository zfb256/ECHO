from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from detector import detect_injected  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Apply the injected-arm token detector to self-induced confirmed-false continuations "
            "as a cross-pipeline comparability check."
        )
    )
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--annotations", default="datasets/runs/full_study/claim_annotation_selfinduced.jsonl")
    p.add_argument("--bank", default=None, help="Defaults to paths.selfinduced_bank from the config.")
    p.add_argument("--output", default="reports/selfinduced_injected_detector_crosscheck.json")
    p.add_argument("--disagreements", default="reports/selfinduced_injected_detector_disagreements.jsonl")
    return p.parse_args()


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def load_truth(config: dict[str, Any], bank_arg: str | None) -> dict[str, str]:
    bank_path = resolve_project_path(bank_arg) if bank_arg else resolve_project_path(config["paths"]["selfinduced_bank"])
    data = json.loads(bank_path.read_text(encoding="utf-8"))
    return {q["q_id"]: q.get("truth_statement", "") for q in data.get("questions", [])}


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(k, "unknown")) for k in keys)].append(row)
    out = []
    for key, rs in sorted(groups.items()):
        item = {k: v for k, v in zip(keys, key)}
        human_pos = [1.0 if r["human_hcr"] else 0.0 for r in rs]
        rule_pos = [1.0 if r["rule_hcr"] else 0.0 for r in rs]
        agree = [1.0 if r["human_hcr"] == r["rule_hcr"] else 0.0 for r in rs]
        item.update(
            {
                "n": len(rs),
                "human_hcr_si": mean(human_pos),
                "injected_rule_positive_rate": mean(rule_pos),
                "absolute_rate_gap_rule_minus_human": (
                    mean(rule_pos) - mean(human_pos) if mean(rule_pos) is not None and mean(human_pos) is not None else None
                ),
                "agreement_rate": mean(agree),
                "false_negative_vs_human_n": sum(1 for r in rs if r["human_hcr"] and not r["rule_hcr"]),
                "false_positive_vs_human_n": sum(1 for r in rs if r["rule_hcr"] and not r["human_hcr"]),
                "refutation_rate": mean([1.0 if r["rule_refuted"] else 0.0 for r in rs]),
                "detector_blind_or_empty_diff_n": sum(1 for r in rs if not r["false_tokens"]),
            }
        )
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))
    truth = load_truth(config, args.bank)
    ann_path = resolve_project_path(args.annotations)
    rows = []
    skipped = []
    for row in read_jsonl(ann_path):
        ann = row.get("annotation") or {}
        if ann.get("turn1_was_false") is not True:
            continue
        truth_stmt = truth.get(row.get("q_id"), "")
        if not truth_stmt:
            skipped.append(row.get("task_id"))
            continue
        det = detect_injected(row.get("response") or "", row.get("turn1_response") or "", truth_stmt, neg_window)
        rule_hcr = bool(det["asserts_false"]) and not bool(det["ambiguous"])
        human_hcr = ann.get("hcr_has_contagious") is True
        rows.append(
            {
                "task_id": row.get("task_id"),
                "pair_id": row.get("pair_id"),
                "q_id": row.get("q_id"),
                "model": row.get("model"),
                "question": row.get("question"),
                "turn1_response": row.get("turn1_response"),
                "truth_statement": truth_stmt,
                "response": row.get("response"),
                "human_hcr": human_hcr,
                "rule_hcr": rule_hcr,
                "rule_asserts_false": bool(det["asserts_false"]),
                "rule_asserts_true": bool(det["asserts_true"]),
                "rule_refuted": bool(det["refuted"]),
                "rule_ambiguous": bool(det["ambiguous"]),
                "false_tokens": det.get("false_tokens", []),
                "true_tokens": det.get("true_tokens", []),
            }
        )

    disagreements = [r for r in rows if r["human_hcr"] != r["rule_hcr"]]
    out_path = resolve_project_path(args.output)
    dis_path = resolve_project_path(args.disagreements)
    report = {
        "task": "selfinduced_injected_detector_crosscheck",
        "definition": {
            "purpose": (
                "Diagnostic comparability check only. Self-induced errors do not have curated false/true "
                "token spans, so this applies the injected rule to turn1_response vs truth_statement as "
                "a lightweight adaptation, not as a replacement for human/NLI labels."
            ),
            "human_hcr_si": "existing human-confirmed self-induced inheritance label on confirmed-false turn-1 rows",
            "injected_rule_positive_rate": "fraction of the same rows whose stage-2 response repeats the turn-1 distinctive false span under the injected detector",
        },
        "inputs": {
            "annotations": to_project_relative(ann_path),
            "n_confirmed_false_scored": len(rows),
            "n_skipped_missing_truth": len(skipped),
            "skipped_missing_truth_task_ids": skipped[:20],
            "negation_window": neg_window,
            "disagreements": to_project_relative(dis_path),
        },
        "overall": summarize(rows, tuple())[0] if rows else None,
        "by_model": summarize(rows, ("model",)),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(dis_path, disagreements)
    print(
        json.dumps(
            {
                "wrote": to_project_relative(out_path),
                "n_confirmed_false_scored": len(rows),
                "human_hcr_si": report["overall"]["human_hcr_si"] if rows else None,
                "injected_rule_positive_rate": report["overall"]["injected_rule_positive_rate"] if rows else None,
                "agreement_rate": report["overall"]["agreement_rate"] if rows else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
