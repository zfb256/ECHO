"""Evaluate the automatic detector on the construct-gold set.

Precision is the acceptance criterion. Real-output recall is estimated separately
because construct-gold positives reuse the detector's distinguishing tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402
from manifest import file_sha256, object_sha256  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import detect_injected  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report detector precision, recall, and F1 on construct-gold data.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--gold", default="datasets_zh/runs/zh_study/construct_gold.jsonl")
    p.add_argument("--output", default="reports_zh/detector_validation.json")
    p.add_argument("--min-f1", type=float, default=None, help="Minimum reported F1; defaults to the config value")
    p.add_argument("--min-precision", type=float, default=None, help="Minimum precision; defaults to the config value")
    p.add_argument("--no-gate", action="store_true", help="Write the report without exiting non-zero below a threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    det_cfg = config.get("pilot", {}).get("detector", {})
    min_f1 = args.min_f1 if args.min_f1 is not None else float(det_cfg.get("min_f1", 0.90))
    min_precision = args.min_precision if args.min_precision is not None else float(det_cfg.get("min_precision", 0.90))
    neg_window = int(det_cfg.get("negation_window", 4))
    seed_path = resolve_project_path(config["paths"]["seed_bank"])
    config_path = resolve_project_path(args.config)
    gold_path = resolve_project_path(args.gold)
    detector_path = Path(__file__).resolve().parent / "detector.py"
    seeds = {s["seed_id"]: s for s in json.loads(seed_path.read_text(encoding="utf-8")).get("seeds", [])}
    gold = list(read_jsonl(gold_path))
    gold_ids = [g.get("gold_id") for g in gold]
    if any(not gid for gid in gold_ids) or len(set(gold_ids)) != len(gold_ids):
        raise SystemExit("Construct-gold contains missing or duplicate gold_id values.")
    if any(
        not isinstance(g.get("label_contains_false"), bool)
        or not isinstance(g.get("text"), str)
        or not g["text"].strip()
        for g in gold
    ):
        raise SystemExit(
            "Every construct-gold row must have a non-empty text and a JSON boolean "
            "label_contains_false (not a string or number)."
        )
    unknown_seeds = sorted({g.get("seed_id") for g in gold if g.get("seed_id") not in seeds})
    if unknown_seeds:
        raise SystemExit(f"Construct-gold references unknown seed_id values: {unknown_seeds[:10]}")
    class_counts: dict[str, dict[bool, int]] = defaultdict(
        lambda: {False: 0, True: 0},
    )
    for g in gold:
        class_counts[g["seed_id"]][g["label_contains_false"]] += 1
    missing_seed_coverage = sorted(
        sid for sid in seeds
        if class_counts[sid][True] == 0 or class_counts[sid][False] == 0
    )
    if missing_seed_coverage:
        raise SystemExit(
            "Construct-gold must contain both positive and negative examples for "
            f"every current seed; incomplete seed_ids={missing_seed_coverage[:10]}"
        )

    tp = fp = fn = tn = 0
    errors = []
    for g in gold:
        seed = seeds.get(g["seed_id"])
        if seed is None:
            continue
        pred = detect_injected(g["text"], seed["claim"], seed["corrected_claim"], neg_window)["asserts_false"]
        truth = bool(g["label_contains_false"])
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
            errors.append({"gold_id": g["gold_id"], "type": "false_positive", "text": g["text"][:120]})
        elif not pred and truth:
            fn += 1
            errors.append({"gold_id": g["gold_id"], "type": "false_negative", "text": g["text"][:120]})
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fp + fn) else None)
    # Construct-gold recall is optimistic, so acceptance is based on precision.
    detector_trustworthy = precision is not None and precision >= min_precision
    report = {
        "n_gold": len(gold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "min_precision_gate": min_precision,
        "min_f1_reported": min_f1,
        "negation_window": neg_window,
        "gate_metric": "precision",
        "recall_is_upper_bound": True,
        "recall_note": "Real-output recall is estimated separately from the human audit.",
        "detector_trustworthy": detector_trustworthy,
        "provenance": {
            "config_path": args.config,
            "config_sha256": file_sha256(config_path),
            "seed_bank_path": config["paths"]["seed_bank"],
            "seed_bank_sha256": file_sha256(seed_path),
            "construct_gold_path": args.gold,
            "construct_gold_sha256": file_sha256(gold_path),
            "detector_path": "auto_labeling/detector.py",
            "detector_sha256": file_sha256(detector_path),
            "detector_settings": {
                "negation_window": neg_window,
                "min_precision": min_precision,
            },
            "detector_settings_sha256": object_sha256({
                "negation_window": neg_window,
                "min_precision": min_precision,
            }),
        },
        "errors": errors[:50],
    }
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("precision", "recall", "f1", "detector_trustworthy", "gate_metric", "fp", "fn")}, ensure_ascii=False))

    if not detector_trustworthy and not args.no_gate:
        raise SystemExit(
            f"Detector precision {report['precision']} is below {min_precision}; see {out}."
        )


if __name__ == "__main__":
    main()
