from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import resolve_project_path  # noqa: E402
from detector import detect_injected  # noqa: E402
from jsonl import read_jsonl  # noqa: E402


DEFAULT_OUTPUTS = [
    "datasets/runs/full_study/model_outputs.jsonl",
    "datasets/runs/full_study/model_outputs_fw_auditfix.jsonl",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate seed-level CCR with clean-arm true-claim assertion rate.")
    p.add_argument("--seed-variance", default="reports/seed_variance_analysis.json")
    p.add_argument("--output", default="reports/knowledge_correlation.json")
    p.add_argument("--outputs", nargs="*", default=DEFAULT_OUTPUTS)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def rank(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx and deny else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rank(xs), rank(ys))


def permutation_p(xs: list[float], ys: list[float], observed: float, n: int, seed: int) -> float:
    rng = random.Random(seed)
    ys_perm = list(ys)
    extreme = 0
    for _ in range(n):
        rng.shuffle(ys_perm)
        if abs(spearman(xs, ys_perm)) >= abs(observed) - 1e-15:
            extreme += 1
    return (extreme + 1) / (n + 1)


def true_hit(response: str, seed_claim: str, corrected_claim: str) -> bool:
    # Swap false/true claims so the existing proposition-specific detector checks
    # distinctive true-value tokens in the clean answer.
    det = detect_injected(response, corrected_claim, seed_claim)
    return bool(det.get("asserts_false")) and not bool(det.get("ambiguous"))


def main() -> None:
    args = parse_args()
    seed_report = json.loads(resolve_project_path(args.seed_variance).read_text(encoding="utf-8"))
    seed_ccr = {row["seed_id"]: row for row in seed_report["by_seed"]}

    counts: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"n": 0, "hits": 0, "seed_type": None})
    for output_arg in args.outputs:
        output_path = resolve_project_path(output_arg)
        if not output_path.exists():
            raise SystemExit(f"missing output file: {output_path}")
        for row in read_jsonl(output_path):
            if row.get("condition") not in {"clean", "clean_b"}:
                continue
            sid = str(row.get("seed_id"))
            model = str(row.get("model"))
            key = (sid, model)
            slot = counts[key]
            slot["seed_type"] = row.get("seed_type")
            slot["n"] += 1
            slot["hits"] += int(true_hit(row.get("response") or "", row.get("seed_claim") or "", row.get("corrected_claim") or ""))

    by_seed = []
    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sid, seed_row in sorted(seed_ccr.items()):
        model_rates = []
        model_rows = []
        for (seed_id, model), c in sorted(counts.items()):
            if seed_id != sid:
                continue
            if c["n"]:
                rate = c["hits"] / c["n"]
                model_rates.append(rate)
                model_rows.append({"model": model, "n_clean_outputs": c["n"], "true_assertion_rate": rate})
        if not model_rates:
            raise SystemExit(f"no clean outputs for seed_id={sid}")
        item = {
            "seed_id": sid,
            "seed_type": seed_row["seed_type"],
            "seed_claim": seed_row["seed_claim"],
            "ccr": seed_row["ccr"],
            "clean_true_assertion_rate": statistics.fmean(model_rates),
            "n_models": len(model_rates),
            "n_clean_outputs": sum(r["n_clean_outputs"] for r in model_rows),
            "by_model": model_rows,
        }
        by_seed.append(item)
        type_groups[item["seed_type"]].append(item)

    xs = [row["clean_true_assertion_rate"] for row in by_seed]
    ys = [row["ccr"] for row in by_seed]
    rho = spearman(xs, ys)
    by_type = []
    for seed_type, rows in sorted(type_groups.items()):
        by_type.append(
            {
                "seed_type": seed_type,
                "n_seeds": len(rows),
                "mean_clean_true_assertion_rate": statistics.fmean(r["clean_true_assertion_rate"] for r in rows),
                "mean_ccr": statistics.fmean(r["ccr"] for r in rows),
            }
        )

    report = {
        "sources": {
            "seed_variance": args.seed_variance,
            "outputs": args.outputs,
        },
        "method": "Clean-arm knowledge proxy is the rate at which clean/clean_b outputs assert the corrected claim's distinctive true-value tokens, using the injected detector with false/true claims swapped.",
        "overall": {
            "n_seeds": len(by_seed),
            "spearman_rho": rho,
            "permutation_two_sided_p": permutation_p(xs, ys, rho, args.permutations, args.seed),
            "permutations": args.permutations,
        },
        "by_seed_type": by_type,
        "by_seed": by_seed,
    }
    out = resolve_project_path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "n_seeds": len(by_seed), "rho": rho}, ensure_ascii=False))


if __name__ == "__main__":
    main()
