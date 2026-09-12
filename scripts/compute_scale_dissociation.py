"""Fold reductions behind the scale dissociation (abstract, Introduction, Section 6.1).

The abstract states that scale cuts injected propagation 7.8-fold but end-to-end self-induced
propagation only 2.9-fold. Both come from the Qwen2.5 ladder and are derived here rather than
by hand, so the claim is reproducible from the shipped annotations.

The end-to-end self-induced rate is the product of the turn-1 hallucination rate and HCR_si:
the chance a model states a falsehood of its own AND then repeats it. Reporting HCR_si alone
conditions on a denominator that itself shrinks with scale (49 confirmed cases at 1.5B, 21 at
32B-AWQ), which is why the unconditional product is the deployment-facing quantity.

    python -B scripts/compute_scale_dissociation.py

Writes reports/scale_dissociation.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import load_config, resolve_project_path, to_project_relative  # noqa: E402

# The precision-controlled ladder, smallest to largest. The 32B point is 4-bit AWQ.
LADDER = [
    ("1.5B", "Qwen2.5-1.5B-Instruct"),
    ("3B", "Qwen2.5-3B-Instruct"),
    ("7B", "Qwen2.5-7B-Instruct"),
    ("14B", "Qwen2.5-14B-Instruct"),
    ("32B-AWQ", "Qwen2.5-32B-Instruct-AWQ"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fold reductions for the ECHO scale dissociation.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--metrics", default="reports/full_study_metrics_all3_auditfix.json")
    p.add_argument("--output", default="reports/scale_dissociation.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_config(args.config)
    metrics_path = resolve_project_path(args.metrics)
    summaries = json.loads(metrics_path.read_text(encoding="utf-8"))["summaries"]

    rows = []
    for label, model in LADDER:
        inj = [s["ccr"] for s in summaries if s["arm"] == "injected" and s["model"] == model]
        si = [s for s in summaries if s["arm"] == "self_induced" and s["model"] == model]
        if len(inj) != 3 or len(si) != 1:
            raise SystemExit(f"unexpected summary rows for {model}: {len(inj)} injected, {len(si)} self-induced")
        ccr = sum(inj) / len(inj)
        hcr_si = si[0]["hcr_si"]
        turn1 = si[0]["turn1_false_rate"]
        rows.append(
            {
                "size": label,
                "model": model,
                "injected_ccr": ccr,
                "hcr_si": hcr_si,
                "turn1_false_rate": turn1,
                "end_to_end_selfinduced": hcr_si * turn1,
                "n_turn1_false": si[0].get("n_turn1_false"),
            }
        )

    small, large = rows[0], rows[-1]
    injected_fold = small["injected_ccr"] / large["injected_ccr"]
    end_to_end_fold = small["end_to_end_selfinduced"] / large["end_to_end_selfinduced"]
    hcr_si_fold = small["hcr_si"] / large["hcr_si"]

    report = {
        "source_metrics": to_project_relative(metrics_path),
        "ladder": rows,
        "injected_ccr_fold_reduction": injected_fold,
        "end_to_end_selfinduced_fold_reduction": end_to_end_fold,
        "conditional_hcr_si_fold_reduction": hcr_si_fold,
        "end_to_end_smallest": small["end_to_end_selfinduced"],
        "end_to_end_largest": large["end_to_end_selfinduced"],
        "paper_values": {
            "injected_fold": round(injected_fold, 1),
            "end_to_end_fold": round(end_to_end_fold, 1),
            "end_to_end_range": [round(small["end_to_end_selfinduced"], 2), round(large["end_to_end_selfinduced"], 2)],
        },
        "definition": {
            "end_to_end_selfinduced": "turn1_false_rate x hcr_si; the unconditional rate at which a model states and then repeats its own falsehood",
            "conditional_hcr_si_fold_reduction": "reported for contrast only; its denominator shrinks with scale, so it is not the deployment-facing quantity",
        },
    }
    out_path = resolve_project_path(args.output)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            "wrote": to_project_relative(out_path),
            "injected_fold": round(injected_fold, 2),
            "end_to_end_fold": round(end_to_end_fold, 2),
            "end_to_end": [round(small["end_to_end_selfinduced"], 3), round(large["end_to_end_selfinduced"], 3)],
            "conditional_hcr_si_fold": round(hcr_si_fold, 2),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
