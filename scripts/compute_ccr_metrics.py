from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl
from manifest import file_sha256, object_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CCR, HCR, and self-induced inheritance metrics.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--injected", default=None, help="Filled claim_annotation_injected.jsonl")
    parser.add_argument("--selfinduced", default=None, help="Filled claim_annotation_selfinduced.jsonl")
    parser.add_argument("--selfinduced-config", default=None, help="Config for a self-induced arm stored under a separate run name")
    parser.add_argument("--selfinduced-tasks", action="append",
                        help="Self-induced stage-2 task file; repeat for merged runs.")
    parser.add_argument("--selfinduced-all-questions", action="store_true",
                        help="Validate the full question bank instead of the frozen primary cohort.")
    parser.add_argument("--placebo", default=None, help="Filled claim_annotation_placebo.jsonl (clean_A vs clean_B)")
    parser.add_argument("--output", default=None, help="Defaults to reports_dir/<run_name>_metrics.json")
    parser.add_argument("--expected-model", action="append",
                        help="Expected model name; repeat for merged analyses (default: config model_pool).")
    parser.add_argument("--recall-audit", default=None,
                        help="Real-output detector audit report for this annotation pool.")
    return parser.parse_args()


def is_ccr_hit(claims: list[dict[str, Any]]) -> bool:
    for c in claims or []:
        if bool(c.get("depends_on_seed")) and c.get("present_in_clean") is False:
            return True
    return False


def _ccr_flags(rows: list[dict[str, Any]]) -> list[int]:
    """Per-pair CCR indicator (0/1) — the raw data a bootstrap resamples."""
    flags = []
    for r in rows:
        a = r.get("annotation", {})
        if a.get("hcr_has_contagious") is None:
            continue
        flags.append(1 if is_ccr_hit(a.get("contaminated_false_claims")) else 0)
    return flags


def _cluster_flags(rows: list[dict[str, Any]], key: str) -> dict[Any, list[int]]:
    """Group per-row CCR flags by a clustering key (seed_claim or model)."""
    groups: dict[Any, list[int]] = {}
    for r in rows:
        a = r.get("annotation") or {}
        # Keep the CI denominator identical to summarize_injected/_ccr_flags.
        # Treating an unannotated row as a negative would bias both the point
        # estimate and the clustered interval downward.
        if a.get("hcr_has_contagious") is None:
            continue
        groups.setdefault(r.get(key), []).append(1 if is_ccr_hit(a.get("contaminated_false_claims")) else 0)
    return groups


def bootstrap_gap_ci_clustered(
    inj_rows: list[dict[str, Any]],
    pl_rows: list[dict[str, Any]],
    key: str,
    n: int,
    seed: int,
    alpha: float = 0.05,
) -> list[float] | None:
    """Matched cluster bootstrap for the injected-minus-placebo CCR gap."""
    import random

    R, P = _cluster_flags(inj_rows, key), _cluster_flags(pl_rows, key)
    keys = sorted(set(R) & set(P), key=str)
    if not keys:
        return None
    rng = random.Random(seed)
    K = len(keys)
    gaps = []
    for _ in range(n):
        drawn = [keys[rng.randrange(K)] for _ in range(K)]
        rv = [x for k in drawn for x in R[k]]
        pv = [x for k in drawn for x in P[k]]
        if not rv or not pv:
            continue
        gaps.append(sum(rv) / len(rv) - sum(pv) / len(pv))
    if not gaps:
        return None
    gaps.sort()
    m = len(gaps)
    return [round(gaps[int((alpha / 2) * m)], 4), round(gaps[min(m - 1, int((1 - alpha / 2) * m))], 4)]


def bootstrap_gap_ci(real: list[int], placebo: list[int], n: int, seed: int, alpha: float = 0.05) -> list[float] | None:
    """Percentile bootstrap CI for the injected-minus-placebo CCR gap."""
    import random

    if not real or not placebo:
        return None
    rng = random.Random(seed)
    nr, npl = len(real), len(placebo)
    gaps = []
    for _ in range(n):
        mr = sum(real[rng.randrange(nr)] for _ in range(nr)) / nr
        mp = sum(placebo[rng.randrange(npl)] for _ in range(npl)) / npl
        gaps.append(mr - mp)
    gaps.sort()
    lo = gaps[int((alpha / 2) * n)]
    hi = gaps[min(n - 1, int((1 - alpha / 2) * n))]
    return [round(lo, 4), round(hi, 4)]


def fmean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def human_reviewed(row: dict, field: str) -> bool:
    return field in ((row.get("annotation") or {}).get("human_fields") or [])


def summarize_injected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("source_dataset", "unknown"), r.get("model", "unknown"))].append(r)
    out = []
    skipped_unannotated = 0
    for (dataset, model), rs in sorted(groups.items()):
        hcr_flags, ccr_flags, clean_seed = [], [], []
        for r in rs:
            a = r.get("annotation", {})
            # Ignore unannotated template rows instead of treating them as negatives.
            if a.get("hcr_has_contagious") is None:
                skipped_unannotated += 1
                continue
            hcr_flags.append(1.0 if a.get("hcr_has_contagious") else 0.0)
            ccr_flags.append(1.0 if is_ccr_hit(a.get("contaminated_false_claims")) else 0.0)
            clean_seed.append(1.0 if a.get("clean_has_seed_claim") else 0.0)
        out.append(
            {
                "arm": "injected",
                "source_dataset": dataset,
                "model": model,
                "n": len(hcr_flags),
                "hcr": fmean(hcr_flags),
                "ccr": fmean(ccr_flags),
                "clean_seed_rate": fmean(clean_seed),
            }
        )
    if skipped_unannotated:
        warnings.warn(f"summarize_injected: skipped {skipped_unannotated} unannotated (template) injected rows.", stacklevel=2)
    return out


def summarize_selfinduced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("model", "unknown")].append(r)
    out = []
    for model, rs in sorted(groups.items()):
        confirmed = [
            r for r in rs
            if human_reviewed(r, "turn1_was_false")
            and r.get("annotation", {}).get("turn1_was_false") is True
        ]
        inheritance_judged = [
            r for r in confirmed
            if human_reviewed(r, "hcr_has_contagious")
            and r.get("annotation", {}).get("hcr_has_contagious") is not None
        ]
        turn1_judged = [
            r for r in rs
            if human_reviewed(r, "turn1_was_false")
            and r.get("annotation", {}).get("turn1_was_false") is not None
        ]
        flags = [
            1.0 if r["annotation"]["hcr_has_contagious"] else 0.0
            for r in inheritance_judged
        ]
        out.append(
            {
                "arm": "self_induced",
                "model": model,
                "n_total": len(rs),
                "n_turn1_judged": len(turn1_judged),
                "n_turn1_false": len(confirmed),
                "n_inheritance_judged": len(inheritance_judged),
                "turn1_false_rate": fmean([
                    1.0 if r["annotation"]["turn1_was_false"] else 0.0
                    for r in turn1_judged
                ]),
                "hcr_si": fmean(flags),
            }
        )
    return out


def pooled_selfinduced(rows: list[dict[str, Any]], cluster: str, n: int, seed: int) -> dict[str, Any]:
    """Pooled inheritance among human-confirmed errors, with a cluster bootstrap."""
    import random

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(cluster, "unknown"))].append(row)

    def counts(sample: list[dict[str, Any]]) -> tuple[int, int]:
        eligible = [
            row for row in sample
            if human_reviewed(row, "turn1_was_false")
            and row.get("annotation", {}).get("turn1_was_false") is True
            and human_reviewed(row, "hcr_has_contagious")
            and isinstance(row.get("annotation", {}).get("hcr_has_contagious"), bool)
        ]
        return sum(row["annotation"]["hcr_has_contagious"] is True for row in eligible), len(eligible)

    inherited, errors = counts(rows)
    rng, names, rates = random.Random(seed), sorted(groups), []
    for _ in range(n):
        drawn = [row for _ in names for row in groups[names[rng.randrange(len(names))]]]
        k, total = counts(drawn)
        if total:
            rates.append(k / total)
    rates.sort()
    ci = None if not rates else [
        round(rates[int(0.025 * len(rates))], 4),
        round(rates[min(len(rates) - 1, int(0.975 * len(rates)))], 4),
    ]
    return {"inherited": inherited, "errors": errors, "rate": inherited / errors if errors else None, "ci95": ci}


def resolve_input_path(arg_value: str | None, default_path: Path, label: str) -> tuple[Path, bool]:
    """Resolve an arm input and fail fast for explicitly requested missing files."""
    if arg_value:
        path = resolve_project_path(arg_value)
        if not path.exists():
            raise SystemExit(f"--{label} path does not exist: {path}")
        return path, True
    return default_path, False


def main() -> None:
    args = parse_args()
    if args.selfinduced_config and not args.selfinduced:
        raise SystemExit("--selfinduced-config requires --selfinduced")
    config = load_config(args.config)
    expected_models = args.expected_model or config["pilot"].get("model_pool", [])
    if len(set(expected_models)) != len(expected_models):
        raise SystemExit("--expected-model contains duplicates")
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    runs_dir = resolve_project_path(config["paths"]["runs_dir"])
    reports_dir = resolve_project_path(config["paths"]["reports_dir"])
    run_dir = runs_dir / run_name
    selfinduced_config = load_config(args.selfinduced_config) if args.selfinduced_config else config
    selfinduced_run_dir = (
        resolve_project_path(selfinduced_config["paths"]["runs_dir"])
        / selfinduced_config["pilot"].get("run_name", "pilot")
    )

    # Explicit arm arguments restrict computation to those arms; otherwise use the run directory.
    any_arm = bool(args.injected or args.selfinduced or args.placebo)
    do_injected = bool(args.injected) or not any_arm
    do_selfinduced = bool(args.selfinduced) or not any_arm
    do_placebo = bool(args.placebo) or not any_arm
    arms_computed = []

    summaries: list[dict[str, Any]] = []
    real_flags: list[int] = []
    inj_rows: list[dict[str, Any]] = []
    pl_rows: list[dict[str, Any]] = []
    si_rows: list[dict[str, Any]] = []
    inj_path, injected_explicit = resolve_input_path(args.injected, run_dir / "claim_annotation_injected.jsonl", "injected")
    if do_injected and inj_path.exists():
        inj_rows = list(read_jsonl(inj_path))
        summaries += summarize_injected(inj_rows)
        real_flags = _ccr_flags(inj_rows)
        arms_computed.append("injected")
    elif do_injected and injected_explicit:
        raise SystemExit(f"--injected path does not exist: {inj_path}")
    si_path, selfinduced_explicit = resolve_input_path(args.selfinduced, run_dir / "claim_annotation_selfinduced.jsonl", "selfinduced")
    if do_selfinduced and si_path.exists():
        si_rows = list(read_jsonl(si_path))
        summaries += summarize_selfinduced(si_rows)
        arms_computed.append("self_induced")
    elif do_selfinduced and selfinduced_explicit:
        raise SystemExit(f"--selfinduced path does not exist: {si_path}")

    # Placebo (clean_A vs clean_B) CCR = decoding-noise false-positive floor.
    placebo_ccr = None
    placebo_flags: list[int] = []
    pl_path, placebo_explicit = resolve_input_path(args.placebo, run_dir / "claim_annotation_placebo.jsonl", "placebo")
    if do_placebo and pl_path.exists():
        pl_rows = list(read_jsonl(pl_path))
        pl = summarize_injected(pl_rows)
        placebo_ccr = fmean([s["ccr"] for s in pl if s["ccr"] is not None])
        placebo_flags = _ccr_flags(pl_rows)
        arms_computed.append("placebo")
    elif do_placebo and placebo_explicit:
        raise SystemExit(f"--placebo path does not exist: {pl_path}")

    # Partial-arm runs use arm-specific output names.
    has_injected = "injected" in arms_computed
    pairs_path = (
        resolve_project_path(config["paths"]["processed_dir"])
        / f"{run_name}_pairs_all.jsonl"
    )
    expected_pair_ids = (
        {r["pair_id"] for r in read_jsonl(pairs_path)}
        if pairs_path.exists() else set()
    )
    expected_pair_models = {
        (pair_id, model)
        for pair_id in expected_pair_ids
        for model in expected_models
    }
    injected_key_list = [(r.get("pair_id"), r.get("model")) for r in inj_rows]
    placebo_key_list = [(r.get("pair_id"), r.get("model")) for r in pl_rows]
    injected_keys = set(injected_key_list)
    placebo_keys = set(placebo_key_list)
    mock_rows = sum(
        1 for row in (inj_rows + pl_rows + si_rows) if row.get("source_mock") is True
    )

    def _injected_annotations_complete(rows: list[dict]) -> bool:
        return all(
            isinstance((row.get("annotation") or {}).get("hcr_has_contagious"), bool)
            and isinstance((row.get("annotation") or {}).get("clean_has_seed_claim"), bool)
            and isinstance((row.get("annotation") or {}).get("contaminated_false_claims"), list)
            for row in rows
        )

    injected_complete = (
        bool(expected_pair_models)
        and len(inj_rows) == len(expected_pair_models)
        and injected_keys == expected_pair_models
        and _injected_annotations_complete(inj_rows)
    )
    placebo_complete = (
        bool(expected_pair_models)
        and len(pl_rows) == len(expected_pair_models)
        and placebo_keys == expected_pair_models
        and _injected_annotations_complete(pl_rows)
    )

    stage2_tasks_paths = (
        [resolve_project_path(path) for path in args.selfinduced_tasks]
        if args.selfinduced_tasks
        else [selfinduced_run_dir / "selfinduced_stage2_tasks.jsonl"]
    )
    stage2_task_rows = [
        row for path in stage2_tasks_paths if path.exists() for row in read_jsonl(path)
    ]
    primary_qids = set() if args.selfinduced_all_questions else set(
        selfinduced_config.get("pilot", {}).get("annotation", {}).get("primary_question_ids") or []
    )
    evaluation_task_rows = [
        row for row in stage2_task_rows
        if not primary_qids or row.get("q_id") in primary_qids
    ]
    expected_si_id_list = [r["task_id"] for r in evaluation_task_rows]
    expected_si_ids = set(expected_si_id_list)
    observed_si_id_list = [r.get("task_id") for r in si_rows]
    observed_si_ids = set(observed_si_id_list)
    self_bank = json.loads(
        resolve_project_path(selfinduced_config["paths"]["selfinduced_bank"]).read_text(
            encoding="utf-8",
        )
    )
    expected_si_q_models = {
        (q["q_id"], model)
        for q in self_bank.get("questions", [])
        if not primary_qids or q["q_id"] in primary_qids
        for model in expected_models
    }
    task_si_q_model_list = [
        (r.get("q_id"), r.get("model")) for r in evaluation_task_rows
    ]
    observed_si_q_model_list = [
        (r.get("q_id"), r.get("model")) for r in si_rows
    ]
    configured_selfinduced_n = len(expected_models) * (
        len(primary_qids)
        if primary_qids
        else int(selfinduced_config["pilot"].get("selfinduced", {}).get("questions_per_model", 0))
    )
    selfinduced_output_complete = (
        bool(expected_si_ids)
        and len(expected_si_id_list) == len(expected_si_ids)
        and len(si_rows) == len(expected_si_ids)
        and observed_si_ids == expected_si_ids
        and (
            configured_selfinduced_n <= 0
            or len(expected_si_ids) == configured_selfinduced_n
        )
        and (
            not expected_si_q_models
            or (
                len(task_si_q_model_list) == len(expected_si_q_models)
                and set(task_si_q_model_list) == expected_si_q_models
                and len(observed_si_q_model_list) == len(expected_si_q_models)
                and set(observed_si_q_model_list) == expected_si_q_models
            )
        )
    )

    selfinduced_human_complete = bool(si_rows) and all(
        human_reviewed(row, "turn1_was_false")
        and (
            row.get("annotation", {}).get("turn1_was_false") is not True
            or human_reviewed(row, "hcr_has_contagious")
        )
        for row in si_rows
    )
    coverage_complete = (
        injected_complete and placebo_complete
        and selfinduced_output_complete and selfinduced_human_complete
        and mock_rows == 0
    )
    # Namespace default reports by run name.
    if args.output:
        out_path = resolve_project_path(args.output)
    elif has_injected:
        out_path = reports_dir / f"{run_name}_metrics.json"
    else:
        out_path = reports_dir / f"{run_name}_metrics_{'_'.join(arms_computed) or 'empty'}.json"

    go = config["pilot"]["go_no_go"]
    min_gap = config["pilot"].get("ccr", {}).get("min_real_minus_placebo_ccr", 0.05)
    inj = [s for s in summaries if s["arm"] == "injected"]
    si = [s for s in summaries if s["arm"] == "self_induced"]

    mean_hcr = fmean([s["hcr"] for s in inj if s["hcr"] is not None])
    mean_ccr = fmean([s["ccr"] for s in inj if s["ccr"] is not None])
    mean_si = fmean([s["hcr_si"] for s in si if s["hcr_si"] is not None])
    real_minus_placebo = (mean_ccr - placebo_ccr) if (mean_ccr is not None and placebo_ccr is not None) else None
    n_boot = int(config["pilot"].get("ccr", {}).get("bootstrap_samples", 2000))
    _seed = int(config.get("random_seed", 0))
    si_seed = int(selfinduced_config.get("random_seed", _seed))
    pooled_si_question = pooled_selfinduced(si_rows, "q_id", n_boot, si_seed) if si_rows else None
    pooled_si_model = pooled_selfinduced(si_rows, "model", n_boot, si_seed) if si_rows else None
    gap_ci = bootstrap_gap_ci(real_flags, placebo_flags, n=n_boot, seed=_seed) if (real_flags and placebo_flags) else None
    # Rows are not exchangeable; keep the row-level interval only as a sensitivity check.
    gap_ci_seed = gap_ci_model = None
    if real_flags and placebo_flags and "injected" in arms_computed and "placebo" in arms_computed:
        gap_ci_seed = bootstrap_gap_ci_clustered(inj_rows, pl_rows, "seed_claim", n=n_boot, seed=_seed)
        gap_ci_model = bootstrap_gap_ci_clustered(inj_rows, pl_rows, "model", n=n_boot, seed=_seed)
    cluster_dispersion = None
    if real_flags and "injected" in arms_computed:
        disp = {}
        for key, label in (("seed_claim", "per_seed"), ("model", "per_model")):
            rates = [sum(v) / len(v) for v in _cluster_flags(inj_rows, key).values() if v]
            if len(rates) > 1:
                disp[label] = {
                    "n": len(rates),
                    "min": round(min(rates), 4),
                    "max": round(max(rates), 4),
                    "stdev": round(statistics.stdev(rates), 4),
                }
        def _width(ci):
            return (ci[1] - ci[0]) if ci else None
        w_row, w_seed, w_model = _width(gap_ci), _width(gap_ci_seed), _width(gap_ci_model)
        disp["ci_width"] = {"row_level": w_row, "seed_clustered": w_seed, "model_clustered": w_model}
        if w_row:
            disp["ci_width_ratio_vs_row_level"] = {
                "seed_clustered": round(w_seed / w_row, 2) if w_seed else None,
                "model_clustered": round(w_model / w_row, 2) if w_model else None,
            }
        cluster_dispersion = disp

    clustered_gate_cis = [ci for ci in (gap_ci_seed, gap_ci_model) if ci is not None]
    gate_cis = clustered_gate_cis or ([gap_ci] if gap_ci is not None else [])
    gap_ci_excludes_zero = bool(gate_cis) and all(ci[0] > 0 for ci in gate_cis)

    decision = {
        "arms_computed": arms_computed,
        "mean_injected_hcr": mean_hcr,
        "mean_ccr": mean_ccr,
        "mean_placebo_ccr": placebo_ccr,
        "real_minus_placebo_ccr": real_minus_placebo,
        "real_minus_placebo_ccr_ci95": gap_ci,
        "real_minus_placebo_ccr_ci95_seed_clustered": gap_ci_seed,
        "real_minus_placebo_ccr_ci95_model_clustered": gap_ci_model,
        "cluster_dispersion": cluster_dispersion,
        "real_minus_placebo_ccr_ci_excludes_zero": gap_ci_excludes_zero,
        "mean_selfinduced_hcr": mean_si,
        "pooled_selfinduced_hcr": pooled_si_question["rate"] if pooled_si_question else None,
        "pooled_selfinduced_inherited_count": pooled_si_question["inherited"] if pooled_si_question else None,
        "pooled_selfinduced_error_count": pooled_si_question["errors"] if pooled_si_question else None,
        "pooled_selfinduced_hcr_ci95_question_clustered": pooled_si_question["ci95"] if pooled_si_question else None,
        "pooled_selfinduced_hcr_ci95_model_clustered": pooled_si_model["ci95"] if pooled_si_model else None,
        "min_ccr_observed": min([s["ccr"] for s in inj if s["ccr"] is not None], default=None),
        "placebo_provided": placebo_ccr is not None,
        "coverage": {
            "expected_injected_pair_model_rows": len(expected_pair_models),
            "observed_injected_rows": len(inj_rows),
            "observed_injected_unique_pair_model_rows": len(injected_keys),
            "duplicate_injected_pair_model_rows": sum(
                count - 1 for count in Counter(injected_key_list).values() if count > 1
            ),
            "observed_placebo_rows": len(pl_rows),
            "observed_placebo_unique_pair_model_rows": len(placebo_keys),
            "duplicate_placebo_pair_model_rows": sum(
                count - 1 for count in Counter(placebo_key_list).values() if count > 1
            ),
            "injected_complete": injected_complete,
            "placebo_complete": placebo_complete,
            "expected_selfinduced_rows": len(expected_si_ids),
            "configured_selfinduced_rows": configured_selfinduced_n or None,
            "expected_selfinduced_task_rows": len(expected_si_id_list),
            "observed_selfinduced_rows": len(si_rows),
            "observed_selfinduced_unique_rows": len(observed_si_ids),
            "expected_selfinduced_question_model_rows": len(expected_si_q_models),
            "observed_selfinduced_unique_question_model_rows": len(
                set(observed_si_q_model_list)
            ),
            "duplicate_selfinduced_rows": sum(
                count - 1 for count in Counter(observed_si_id_list).values() if count > 1
            ),
            "selfinduced_output_complete": selfinduced_output_complete,
            "selfinduced_human_complete": selfinduced_human_complete,
            "complete": coverage_complete,
            "source_mock_rows": mock_rows,
        },
        "thresholds": {
            "min_contaminated_hcr": go.get("min_contaminated_hcr"),
            "min_ccr": go.get("min_ccr"),
            "min_selfinduced_hcr": go.get("min_selfinduced_hcr"),
            "min_real_minus_placebo_ccr": min_gap,
        },
    }
    go_ccr = mean_ccr is not None and mean_ccr >= go.get("min_ccr", 0.10)
    go_hcr = mean_hcr is not None and mean_hcr >= go.get("min_contaminated_hcr", 0.15)
    # The combined decision is defined only when self-induced results are present.
    si_in_run = "self_induced" in arms_computed
    if not si_in_run:
        go_si: bool | None = None
    elif mean_si is None:
        go_si = None
    else:
        go_si = mean_si >= go.get("min_selfinduced_hcr", 0.10)
    # Require separation from the placebo floor and clustered intervals above zero.
    go_placebo = (
        real_minus_placebo is not None
        and real_minus_placebo >= min_gap
        and gap_ci_excludes_zero
    )

    # Bind detector validation to the current detector, seed bank, and construct-gold set.
    det_path = reports_dir / "detector_validation.json"
    det_trustworthy = det_f1 = None
    detector_provenance_current = False
    if det_path.exists():
        try:
            det = json.loads(det_path.read_text(encoding="utf-8"))
            provenance = det.get("provenance", {})
            detector_file = resolve_project_path("auto_labeling/detector.py")
            seed_bank_file = resolve_project_path(config["paths"]["seed_bank"])
            gold_file = resolve_project_path(provenance["construct_gold_path"])
            current_detector_settings = {
                "negation_window": int(
                    config.get("pilot", {}).get("detector", {}).get("negation_window", 4)
                ),
                "min_precision": float(
                    config.get("pilot", {}).get("detector", {}).get("min_precision", 0.90)
                ),
            }
            detector_provenance_current = (
                provenance.get("detector_sha256") == file_sha256(detector_file)
                and provenance.get("seed_bank_sha256") == file_sha256(seed_bank_file)
                and provenance.get("construct_gold_sha256") == file_sha256(gold_file)
                and provenance.get("detector_settings_sha256")
                == object_sha256(current_detector_settings)
            )
            det_trustworthy = (
                bool(det.get("detector_trustworthy"))
                if detector_provenance_current else False
            )
            det_f1 = det.get("f1")
        except Exception:
            det_trustworthy = None
    go_detector = det_trustworthy is True and detector_provenance_current
    real_output_precision_required = go.get("min_real_output_detector_precision")
    recall_audit_path = (
        resolve_project_path(args.recall_audit)
        if args.recall_audit else reports_dir / "injected_recall_human_audit.json"
    )
    recall_audit_current = False
    recall_audit_complete = None
    real_output_precision = None
    go_real_output_detector: bool | None = (
        True if real_output_precision_required is None else None
    )
    if real_output_precision_required is not None and recall_audit_path.exists():
        try:
            recall_report = json.loads(recall_audit_path.read_text(encoding="utf-8"))
            rp = recall_report.get("provenance") or {}
            audit_matches_input = (
                resolve_project_path(rp["annotations_path"]).resolve()
                == inj_path.resolve()
            )

            def _recall_file_current(path_key: str, hash_key: str) -> bool:
                current_path = resolve_project_path(rp[path_key])
                return rp.get(hash_key) == file_sha256(current_path)

            if audit_matches_input:
                recall_audit_current = (
                    float(rp.get("min_precision")) == float(real_output_precision_required)
                    and float(rp.get("min_decidable_fraction"))
                    == float(
                        config.get("pilot", {}).get("annotation", {}).get(
                            "injected_recall_min_decidable_fraction", 0.75
                        )
                    )
                    and _recall_file_current("annotations_path", "annotations_sha256")
                    and _recall_file_current("audit_path", "audit_sha256")
                    and _recall_file_current("compute_script_path", "compute_script_sha256")
                )
                recall_audit_complete = bool(recall_report.get("audit_complete"))
                real_output_precision = recall_report.get("estimated_real_output_precision")
                if recall_audit_current and recall_audit_complete:
                    go_real_output_detector = bool(
                        recall_report.get("detector_real_output_trustworthy")
                    )
        except Exception:
            recall_audit_current = False
            go_real_output_detector = None
    if real_output_precision_required is not None:
        go_detector = bool(go_detector and go_real_output_detector is True)

    kappa_required = si_in_run and go.get("min_kappa") is not None
    kappa_report_path = reports_dir / "selfinduced_second_annotator_chain_check.json"
    kappa_provenance_current = False
    kappa_chain_ok = None
    kappa_turn1 = None
    kappa_inheritance = None
    go_kappa: bool | None = True if not kappa_required else None
    if kappa_required and kappa_report_path.exists():
        try:
            kappa_report = json.loads(kappa_report_path.read_text(encoding="utf-8"))
            kp = kappa_report.get("provenance") or {}

            def _provenance_file_current(path_key: str, hash_key: str) -> bool:
                current_path = resolve_project_path(kp[path_key])
                return kp.get(hash_key) == file_sha256(current_path)

            kappa_provenance_current = (
                float(kp.get("min_kappa")) == float(go["min_kappa"])
                and float(kp.get("min_decidable_fraction"))
                == float(
                    config.get("pilot", {}).get("annotation", {}).get(
                        "min_kappa_decidable_fraction", 0.0
                    )
                )
                and _provenance_file_current("annotations_path", "annotations_sha256")
                and _provenance_file_current("turn1_packet_path", "turn1_packet_sha256")
                and _provenance_file_current("hcr_packet_path", "hcr_packet_sha256")
                and _provenance_file_current("checker_path", "checker_sha256")
            )
            kappa_chain_ok = bool(kappa_report.get("ok"))
            agreement = kappa_report.get("agreement") or {}
            turn1_agreement = agreement.get("turn1_was_false") or {}
            inheritance_agreement = agreement.get("hcr_has_contagious") or {}
            kappa_turn1 = turn1_agreement.get("cohen_kappa")
            kappa_inheritance = inheritance_agreement.get("cohen_kappa")
            if kappa_provenance_current and kappa_chain_ok:
                go_kappa = bool(
                    turn1_agreement.get("go_kappa")
                    and inheritance_agreement.get("go_kappa")
                )
        except Exception:
            kappa_provenance_current = False
            go_kappa = None

    decision["go_ccr"] = go_ccr
    decision["go_contaminated_hcr"] = go_hcr
    decision["go_selfinduced"] = go_si
    decision["go_placebo"] = go_placebo
    decision["detector_f1"] = det_f1
    decision["detector_trustworthy"] = det_trustworthy
    decision["detector_provenance_current"] = detector_provenance_current
    decision["go_detector"] = go_detector
    decision["real_output_detector_precision_required"] = real_output_precision_required
    decision["real_output_detector_precision"] = real_output_precision
    decision["recall_audit_complete"] = recall_audit_complete
    decision["recall_audit_provenance_current"] = recall_audit_current
    decision["go_real_output_detector"] = go_real_output_detector
    decision["kappa_required"] = kappa_required
    decision["kappa_chain_ok"] = kappa_chain_ok
    decision["kappa_provenance_current"] = kappa_provenance_current
    decision["kappa_turn1_was_false"] = kappa_turn1
    decision["kappa_hcr_has_contagious"] = kappa_inheritance
    decision["go_kappa"] = go_kappa
    decision["go_coverage"] = coverage_complete
    # The combined decision is undefined when a required arm or audit is absent.
    if (
        not has_injected or go_si is None or not coverage_complete
        or (kappa_required and go_kappa is None)
    ):
        decision["go"] = None
    else:
        decision["go"] = bool(
            go_ccr and go_hcr and go_placebo and go_si and go_detector and go_kappa
        )
    report = {"summaries": summaries, "decision": decision}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    if summaries:
        fields = sorted({k for s in summaries for k in s})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)

    print(json.dumps({"wrote": str(out_path), "go": decision["go"], "mean_ccr": mean_ccr, "mean_placebo_ccr": placebo_ccr, "real_minus_placebo_ccr": real_minus_placebo, "real_minus_placebo_ccr_ci95": gap_ci, "mean_selfinduced_hcr": mean_si, "detector_trustworthy": det_trustworthy}, ensure_ascii=False))


if __name__ == "__main__":
    main()
