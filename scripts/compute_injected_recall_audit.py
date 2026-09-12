from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl  # noqa: E402
from manifest import file_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute real-output recall estimate from a filled injected recall audit packet.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--annotations", default=None)
    p.add_argument("--audit", default=None)
    p.add_argument("--sampling-manifest", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--model", action="append", help="Restrict a multi-endpoint audit to these models; repeatable.")
    return p.parse_args()


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def recall_from_miss_rate(auto_pos: int, auto_neg: int, miss_rate: float) -> float | None:
    denom = auto_pos + auto_neg * miss_rate
    return auto_pos / denom if denom else None


def corrected_recall(auto_pos: int, auto_neg: int, precision: float, miss_rate: float) -> float | None:
    estimated_tp = auto_pos * precision
    denom = estimated_tp + auto_neg * miss_rate
    return estimated_tp / denom if denom else None


def design_estimates(
    rows: list[dict], detector_labels: dict[tuple, bool], auto_pos: int,
    auto_neg: int, population_by_model: dict | None = None,
) -> dict[str, float] | None:
    if population_by_model:
        estimated_tp = estimated_fn = 0.0
        for model, population in sorted(population_by_model.items()):
            model_rows = [r for r in rows if str(r.get("model", "unknown")) == model]
            positives = [r for r in model_rows if detector_labels[(r.get("pair_id"), r.get("model"))]]
            negatives = [r for r in model_rows if not detector_labels[(r.get("pair_id"), r.get("model"))]]
            if not positives or not negatives:
                return None
            estimated_tp += int(population["positive"]) * sum(
                r["human_asserts_seed_falsehood"] is True for r in positives
            ) / len(positives)
            estimated_fn += int(population["negative"]) * sum(
                r["human_asserts_seed_falsehood"] is True for r in negatives
            ) / len(negatives)
    else:
        positives = [r for r in rows if detector_labels[(r.get("pair_id"), r.get("model"))]]
        negatives = [r for r in rows if not detector_labels[(r.get("pair_id"), r.get("model"))]]
        if not positives or not negatives:
            return None
        estimated_tp = auto_pos * sum(
            r["human_asserts_seed_falsehood"] is True for r in positives
        ) / len(positives)
        estimated_fn = auto_neg * sum(
            r["human_asserts_seed_falsehood"] is True for r in negatives
        ) / len(negatives)
    total = estimated_tp + estimated_fn
    return {
        "precision": estimated_tp / auto_pos if auto_pos else 0.0,
        "miss_rate": estimated_fn / auto_neg if auto_neg else 0.0,
        "recall": estimated_tp / total if total else 0.0,
        "estimated_true_positive_count": estimated_tp,
        "estimated_false_negative_count": estimated_fn,
    }


def clustered_intervals(
    rows: list[dict], detector_labels: dict[tuple, bool], auto_pos: int,
    auto_neg: int, samples: int, seed: int, population_by_model: dict | None = None,
) -> dict[str, list[float] | int | None]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("seed_claim"))].append(row)
    names = sorted(groups)
    rng = random.Random(seed)
    values = {"precision": [], "miss_rate": [], "recall": []}
    for _ in range(samples):
        drawn = [row for _ in names for row in groups[names[rng.randrange(len(names))]]]
        estimates = design_estimates(
            drawn, detector_labels, auto_pos, auto_neg, population_by_model,
        )
        if estimates is None:
            continue
        for key in values:
            values[key].append(estimates[key])

    def interval(xs: list[float]) -> list[float] | None:
        if not xs:
            return None
        xs.sort()
        n = len(xs)
        return [xs[int(0.025 * n)], xs[min(n - 1, int(0.975 * n))]]

    return {
        **{key: interval(value) for key, value in values.items()},
        "valid_replicates": len(values["recall"]),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config["pilot"].get("run_name", "zh_study")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    reports_dir = resolve_project_path(config["paths"]["reports_dir"])
    annotations_path = resolve_project_path(args.annotations) if args.annotations else run_dir / "claim_annotation_injected.jsonl"
    audit_path = resolve_project_path(args.audit) if args.audit else run_dir / "injected_recall_human_audit_sample.jsonl"
    default_sampling_manifest = run_dir / "injected_recall_audit_sampling_manifest.json"
    sampling_manifest_path = (
        resolve_project_path(args.sampling_manifest) if args.sampling_manifest
        else default_sampling_manifest if args.audit is None
        else None
    )
    output_path = resolve_project_path(args.output) if args.output else reports_dir / "injected_recall_human_audit.json"
    min_precision = float(config["pilot"].get("go_no_go", {}).get("min_real_output_detector_precision", 0.90))
    min_decidable_fraction = float(
        config["pilot"].get("annotation", {}).get(
            "injected_recall_min_decidable_fraction", 0.75
        )
    )
    bootstrap_samples = int(config["pilot"].get("ccr", {}).get("bootstrap_samples", 2000))
    bootstrap_seed = int(config.get("random_seed", 0))
    ann = list(read_jsonl(annotations_path))
    selected_models = set(args.model or [])
    if selected_models:
        ann = [row for row in ann if row.get("model") in selected_models]
        if {row.get("model") for row in ann} != selected_models:
            raise ValueError("--model includes a model absent from the source annotations")
    auto_pos = sum(1 for r in ann if r.get("annotation", {}).get("hcr_has_contagious") is True)
    auto_neg = sum(1 for r in ann if r.get("annotation", {}).get("hcr_has_contagious") is not True)
    detector_labels = {
        (r.get("pair_id"), r.get("model")): r.get("annotation", {}).get("hcr_has_contagious") is True
        for r in ann
    }
    audit = list(read_jsonl(audit_path))
    if selected_models:
        audit = [row for row in audit if row.get("model") in selected_models]
    if any((r.get("pair_id"), r.get("model")) not in detector_labels for r in audit):
        raise ValueError("every audit row must match a source annotation by pair_id and model")
    if any(
        isinstance(r.get("detector_label"), bool)
        and r["detector_label"] != detector_labels[(r.get("pair_id"), r.get("model"))]
        for r in audit
    ):
        raise ValueError("audit detector_label disagrees with the source annotation")
    if any(
        r.get("human_uncertain") is True
        and isinstance(r.get("human_asserts_seed_falsehood"), bool)
        for r in audit
    ):
        raise ValueError("an audit row cannot be both uncertain and a boolean judgment")
    filled = [
        r for r in audit
        if isinstance(r.get("human_asserts_seed_falsehood"), bool)
        and r.get("human_uncertain") is not True
    ]
    uncertain = [r for r in audit if r.get("human_uncertain") is True]
    unresolved = len(audit) - len(filled) - len(uncertain)
    sampling_labels = dict(detector_labels)
    sampling_auto_pos, sampling_auto_neg = auto_pos, auto_neg
    sampling_by_model = None
    sampling_manifest = None
    if sampling_manifest_path and sampling_manifest_path.exists():
        sampling_manifest = json.loads(sampling_manifest_path.read_text(encoding="utf-8"))
        if sampling_manifest.get("audit_sha256") != file_sha256(audit_path):
            raise ValueError("sampling manifest does not match the audit packet")
        audit_by_id = {r.get("audit_id"): r for r in audit}
        if len(audit_by_id) != len(audit) or None in audit_by_id:
            raise ValueError("sampling manifest requires unique audit_id values")
        for audit_id in sampling_manifest.get("post_revision_positive_to_negative_audit_ids", []):
            row = audit_by_id.get(audit_id)
            if row is None or detector_labels[(row.get("pair_id"), row.get("model"))]:
                raise ValueError(f"invalid positive-to-negative audit id: {audit_id}")
            sampling_labels[(row.get("pair_id"), row.get("model"))] = True
        for audit_id in sampling_manifest.get("post_revision_negative_to_positive_audit_ids", []):
            row = audit_by_id.get(audit_id)
            if row is None or not detector_labels[(row.get("pair_id"), row.get("model"))]:
                raise ValueError(f"invalid negative-to-positive audit id: {audit_id}")
            sampling_labels[(row.get("pair_id"), row.get("model"))] = False
        sampling_auto_pos = int(sampling_manifest["population_positive"])
        sampling_auto_neg = int(sampling_manifest["population_negative"])
        sampling_by_model = sampling_manifest.get("population_by_model")
        if selected_models and sampling_by_model:
            sampling_by_model = {
                model: population for model, population in sampling_by_model.items()
                if model in selected_models
            }
            if set(sampling_by_model) != selected_models:
                raise ValueError("sampling manifest does not cover every selected --model")
            sampling_auto_pos = sum(int(v["positive"]) for v in sampling_by_model.values())
            sampling_auto_neg = sum(int(v["negative"]) for v in sampling_by_model.values())
        if sampling_by_model and (
            set(sampling_by_model) != {str(r.get("model", "unknown")) for r in ann}
            or sum(int(v["positive"]) for v in sampling_by_model.values()) != sampling_auto_pos
            or sum(int(v["negative"]) for v in sampling_by_model.values()) != sampling_auto_neg
        ):
            raise ValueError("model-stratum populations do not match the sampling population")
    positive_rows = [r for r in filled if sampling_labels[(r.get("pair_id"), r.get("model"))]]
    negative_rows = [r for r in filled if not sampling_labels[(r.get("pair_id"), r.get("model"))]]
    post_positive_rows = [r for r in filled if detector_labels[(r.get("pair_id"), r.get("model"))]]
    post_negative_rows = [r for r in filled if not detector_labels[(r.get("pair_id"), r.get("model"))]]
    if sampling_manifest and not selected_models and (
        len(positive_rows) != int(sampling_manifest["sample_positive"])
        or len(negative_rows) != int(sampling_manifest["sample_negative"])
    ):
        raise ValueError("reconstructed audit strata do not match the sampling manifest")
    true_positives = sum(1 for r in positive_rows if r.get("human_asserts_seed_falsehood") is True)
    missed = sum(1 for r in negative_rows if r.get("human_asserts_seed_falsehood") is True)
    unstratified_precision = true_positives / len(positive_rows) if positive_rows else None
    unstratified_precision_ci = wilson_ci(true_positives, len(positive_rows)) if positive_rows else None
    unstratified_miss_rate = missed / len(negative_rows) if negative_rows else None
    unstratified_miss_rate_ci = wilson_ci(missed, len(negative_rows)) if negative_rows else None
    unstratified_recall = (
        corrected_recall(
            sampling_auto_pos, sampling_auto_neg,
            unstratified_precision, unstratified_miss_rate,
        ) if unstratified_precision is not None and unstratified_miss_rate is not None else None
    )
    primary = design_estimates(
        filled, sampling_labels, sampling_auto_pos, sampling_auto_neg, sampling_by_model,
    )
    precision = primary["precision"] if primary else None
    miss_rate = primary["miss_rate"] if primary else None
    recall_estimate = primary["recall"] if primary else None
    estimated_missed_total = primary["estimated_false_negative_count"] if primary else None
    clustered_ci = clustered_intervals(
        filled, sampling_labels, sampling_auto_pos, sampling_auto_neg,
        bootstrap_samples, bootstrap_seed, sampling_by_model,
    ) if filled else {"precision": None, "miss_rate": None, "recall": None, "valid_replicates": 0}
    unstratified_clustered_ci = clustered_intervals(
        filled, sampling_labels, sampling_auto_pos, sampling_auto_neg,
        bootstrap_samples, bootstrap_seed,
    ) if filled else {"precision": None, "miss_rate": None, "recall": None, "valid_replicates": 0}
    decidable_fraction = len(filled) / len(audit) if audit else 0.0
    per_model = {}
    estimated_tp_stratified = estimated_fn_stratified = 0.0
    for model in sorted({str(r.get("model", "unknown")) for r in ann}):
        model_ann = [r for r in ann if str(r.get("model", "unknown")) == model]
        model_rows = [r for r in filled if str(r.get("model", "unknown")) == model]
        model_pos = [r for r in model_rows if sampling_labels[(r.get("pair_id"), r.get("model"))]]
        model_neg = [r for r in model_rows if not sampling_labels[(r.get("pair_id"), r.get("model"))]]
        model_tp = sum(r["human_asserts_seed_falsehood"] is True for r in model_pos)
        model_fn = sum(r["human_asserts_seed_falsehood"] is True for r in model_neg)
        if sampling_by_model:
            model_auto_pos = int(sampling_by_model[model]["positive"])
            model_auto_neg = int(sampling_by_model[model]["negative"])
        else:
            model_auto_pos = sum(r.get("annotation", {}).get("hcr_has_contagious") is True for r in model_ann)
            model_auto_neg = len(model_ann) - model_auto_pos
        model_precision = model_tp / len(model_pos) if model_pos else None
        model_miss_rate = model_fn / len(model_neg) if model_neg else None
        if model_precision is not None:
            estimated_tp_stratified += model_auto_pos * model_precision
        if model_miss_rate is not None:
            estimated_fn_stratified += model_auto_neg * model_miss_rate
        per_model[model] = {
            "auto_positive_count": model_auto_pos,
            "auto_negative_count": model_auto_neg,
            "audit_positive_count": len(model_pos),
            "audit_negative_count": len(model_neg),
            "true_positive_count": model_tp,
            "false_positive_count": len(model_pos) - model_tp,
            "missed_count": model_fn,
            "precision": model_precision,
            "miss_rate": model_miss_rate,
        }
    stratified_precision = (
        estimated_tp_stratified / sampling_auto_pos if sampling_auto_pos else None
    )
    stratified_recall = (
        estimated_tp_stratified / (estimated_tp_stratified + estimated_fn_stratified)
        if estimated_tp_stratified + estimated_fn_stratified else None
    )
    audit_complete = (
        unresolved == 0
        and bool(positive_rows)
        and bool(negative_rows)
        and decidable_fraction >= min_decidable_fraction
    )
    report = {
        "source_annotations": to_project_relative(annotations_path),
        "source_audit": to_project_relative(audit_path),
        "auto_positive_hcr_count": auto_pos,
        "auto_negative_hcr_count": auto_neg,
        "sampling_population_positive_count": sampling_auto_pos,
        "sampling_population_negative_count": sampling_auto_neg,
        "audit_rows": len(audit),
        "audit_filled_rows": len(filled),
        "audit_uncertain_rows": len(uncertain),
        "audit_unresolved_rows": unresolved,
        "audit_decidable_fraction": decidable_fraction,
        "min_decidable_fraction": min_decidable_fraction,
        "audit_complete": audit_complete,
        "audit_sampling_positive_filled": len(positive_rows),
        "audit_sampling_negative_filled": len(negative_rows),
        "audit_post_revision_positive_filled": len(post_positive_rows),
        "audit_post_revision_negative_filled": len(post_negative_rows),
        "audit_post_revision_true_positive_count": sum(
            r.get("human_asserts_seed_falsehood") is True for r in post_positive_rows
        ),
        "audit_post_revision_missed_count": sum(
            r.get("human_asserts_seed_falsehood") is True for r in post_negative_rows
        ),
        "post_revision_packet_precision": (
            sum(r.get("human_asserts_seed_falsehood") is True for r in post_positive_rows)
            / len(post_positive_rows) if post_positive_rows else None
        ),
        "post_revision_packet_precision_ci95_wilson": wilson_ci(
            sum(r.get("human_asserts_seed_falsehood") is True for r in post_positive_rows),
            len(post_positive_rows),
        ),
        "audit_true_positive_count": true_positives,
        "estimated_real_output_precision": precision,
        "estimated_real_output_precision_ci95_seed_clustered": clustered_ci["precision"],
        "audit_missed_count": missed,
        "audit_missed_rate_among_auto_negatives": miss_rate,
        "audit_missed_rate_ci95_seed_clustered": clustered_ci["miss_rate"],
        "estimated_total_missed": estimated_missed_total,
        "estimated_total_missed_ci95_seed_clustered": (
            [sampling_auto_neg * x for x in clustered_ci["miss_rate"]]
            if clustered_ci["miss_rate"] is not None else None
        ),
        "estimated_real_output_recall": recall_estimate,
        "estimated_real_output_recall_ci95_seed_clustered": clustered_ci["recall"],
        "seed_cluster_bootstrap_valid_replicates": clustered_ci["valid_replicates"],
        "unstratified_sensitivity": {
            "estimated_precision": unstratified_precision,
            "estimated_precision_ci95_wilson": unstratified_precision_ci,
            "estimated_precision_ci95_seed_clustered": unstratified_clustered_ci["precision"],
            "estimated_miss_rate": unstratified_miss_rate,
            "estimated_miss_rate_ci95_wilson": unstratified_miss_rate_ci,
            "estimated_miss_rate_ci95_seed_clustered": unstratified_clustered_ci["miss_rate"],
            "estimated_recall": unstratified_recall,
            "estimated_recall_ci95_seed_clustered": unstratified_clustered_ci["recall"],
            "note": "Sensitivity analysis that pools across the model quotas.",
        },
        "per_model_audit": per_model,
        "model_stratified_reweighting": {
            "estimated_true_positive_count": estimated_tp_stratified,
            "estimated_false_negative_count": estimated_fn_stratified,
            "estimated_precision": stratified_precision,
            "estimated_recall": stratified_recall,
            "note": "Primary design estimate: reweights every model-by-initial-label stratum to its full-run count.",
        },
        "detector_real_output_trustworthy": (
            audit_complete and precision is not None and precision >= min_precision
        ),
        "min_precision_gate": min_precision,
        "ci_method": "Original model-by-detector-label design weights; seed-cluster bootstrap for precision, miss rate, and recall.",
        "reviewer_count": 1,
        "note": "Single-reviewer audit. Primary estimates retain all pre-revision model-by-label sampling strata; post-revision packet precision is diagnostic only.",
        "provenance": {
            "annotations_path": to_project_relative(annotations_path),
            "annotations_sha256": file_sha256(annotations_path),
            "audit_path": to_project_relative(audit_path),
            "audit_sha256": file_sha256(audit_path),
            "sampling_manifest_path": to_project_relative(sampling_manifest_path) if sampling_manifest and sampling_manifest_path else None,
            "sampling_manifest_sha256": file_sha256(sampling_manifest_path) if sampling_manifest and sampling_manifest_path else None,
            "compute_script_path": "scripts/compute_injected_recall_audit.py",
            "compute_script_sha256": file_sha256(Path(__file__)),
            "min_precision": min_precision,
            "min_decidable_fraction": min_decidable_fraction,
        },
    }
    out = output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out), "filled": len(filled), "precision": precision, "recall": recall_estimate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
