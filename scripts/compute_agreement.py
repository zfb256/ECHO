from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl


# Inter-annotator agreement on rows containing both annotation fields.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa on the double-annotated subset.")
    parser.add_argument("--config", default="configs/zh_selfinduced.json")
    parser.add_argument("--annotations", required=True, help="Filled annotation JSONL containing annotation + annotation_2.")
    parser.add_argument("--field", default=None, help="Binary annotation field. Defaults to config pilot.annotation.agreement_field.")
    parser.add_argument("--output", default=None, help="Defaults to reports_dir/agreement.json")
    return parser.parse_args()


def cohen_kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        conf[idx[x]][idx[y]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    row = [sum(conf[i]) / n for i in range(k)]
    col = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    if pe >= 1.0:
        # Kappa is undefined when both annotators use one label only.
        return None
    return (po - pe) / (1 - pe)


def bootstrap_kappa_ci(
    a: list[int], b: list[int], n_boot: int, seed: int, alpha: float = 0.05
) -> list[float] | None:
    """Paired nonparametric bootstrap CI; undefined resamples are skipped."""
    import random

    if not a or len(a) != len(b) or n_boot <= 0:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        value = cohen_kappa([a[i] for i in idx], [b[i] for i in idx])
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    m = len(values)
    return [
        values[int((alpha / 2) * m)],
        values[min(m - 1, int((1 - alpha / 2) * m))],
    ]


def to_int(v: Any) -> int | None:
    if v is True:
        return 1
    if v is False:
        return 0
    return None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    reports_dir = resolve_project_path(config["paths"]["reports_dir"])
    field = args.field or config["pilot"]["annotation"].get("agreement_field", "hcr_has_contagious")
    ann_path = resolve_project_path(args.annotations)
    out_path = resolve_project_path(args.output) if args.output else reports_dir / "agreement.json"

    if not ann_path.exists():
        raise SystemExit(f"Annotation file not found: {ann_path}")

    a_vals, b_vals, skipped = [], [], 0
    for row in read_jsonl(ann_path):
        ann = row.get("annotation", {})
        ann_human = set(ann.get("human_fields") or [])
        if "annotation_2" not in row:
            continue
        ann2 = row.get("annotation_2", {})
        ann2_human = set(ann2.get("human_fields") or [])
        a = to_int(ann.get(field)) if field in ann_human else None
        b = to_int(ann2.get(field)) if field in ann2_human else None
        if a is None or b is None:
            skipped += 1
            continue
        a_vals.append(a)
        b_vals.append(b)

    kappa = cohen_kappa(a_vals, b_vals)
    n_boot = int(config.get("pilot", {}).get("ccr", {}).get("bootstrap_samples", 2000))
    kappa_ci = bootstrap_kappa_ci(a_vals, b_vals, n_boot, int(config.get("random_seed", 0)))
    pct_agree = (sum(1 for x, y in zip(a_vals, b_vals) if x == y) / len(a_vals)) if a_vals else None
    min_kappa = config["pilot"]["go_no_go"].get("min_kappa", 0.6)

    report = {
        "field": field,
        "comparison": f"inter-annotator: annotation vs annotation_2 ({field})",
        "is_inter_annotator": True,
        "n_compared": len(a_vals),
        "skipped_incomplete": skipped,
        "percent_agreement": pct_agree,
        "cohen_kappa": kappa,
        "cohen_kappa_ci95_bootstrap": kappa_ci,
        "bootstrap_samples": n_boot,
        "kappa_note": ("undefined: degenerate single-label, no variance — inconclusive, not a pass" if (kappa is None and a_vals) else None),
        "min_kappa_threshold": min_kappa,
        "go_kappa": kappa is not None and kappa >= min_kappa,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
