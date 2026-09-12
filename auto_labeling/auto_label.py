"""Create injected, placebo, and self-induced annotation files.

Injected and placebo labels come from the deterministic detector. Self-induced
automatic labels are suggestions; human fields remain authoritative.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from config import load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import (  # noqa: E402
    load_output_manifest,
    object_sha256,
    require_model_binding,
    require_task_binding,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import (  # noqa: E402
    configure_nli,
    detect_injected,
    looks_like_abstention,
    nli_asserts_claim,
    nli_entails,
    nli_health,
    nli_relation,
)


def _nli_preflight(config: dict, require: bool, arm: str) -> dict:
    """Probe NLI and report loudly. NLI absence silently disables turn-1 suggestions and the
    injected E2 clean-guard, which an operator would not otherwise notice — so we surface it
    here (and hard-fail with --require-nli, e.g. on the self-induced arm where it matters most).
    """
    nli_cfg = config.get("pilot", {}).get("nli", {})
    configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
    health = nli_health()
    if not health["available"]:
        msg = (
            f"[NLI UNAVAILABLE] model={health['model']!r} error={health['error']!r}. "
            f"DeBERTa-v3 needs `pip install sentencepiece protobuf transformers`. "
            f"Without NLI the {arm} arm gets NO turn-1 suggestions; an enabled injected "
            "E2 clean-guard cannot run."
        )
        if require:
            raise SystemExit(msg + " (--require-nli set: refusing to proceed.)")
        print(msg, file=sys.stderr)
    else:
        print(json.dumps({"nli_preflight": "ok", "model": health["model"]}, ensure_ascii=False), file=sys.stderr)
    return health


def _clean_nli_override(clean_text: str, clean_det: dict, seed_claim: str, guard: bool) -> bool:
    """E2 guard: True if the clean-side rule hit should be DROPPED as a mention-only false positive.

    Only fires when (a) the guard is on, (b) the rule already flagged the clean side, and (c) NLI
    confidently finds the clean text does NOT entail the false proposition. A clean-side
    override can only REMOVE a false suppressor and thus RECOVER a real CCR — it can never fabricate a
    contagion hit (the contaminated side is untouched). Enabling the guard without usable NLI
    is an error; otherwise identical configs would silently produce environment-dependent labels.
    """
    if not guard or not clean_det.get("asserts_false"):
        return False
    entailed = nli_asserts_claim(clean_text, seed_claim)  # bool | None
    if entailed is None:
        raise RuntimeError("nli_clean_guard is enabled but the configured NLI model is unavailable")
    return entailed is False


def _annotation(cont: dict, clean: dict, seed_claim: str, clean_nli_override: bool = False) -> dict[str, Any]:
    # Ambiguous continuations state both the false and true values without an explicit
    # refutation. Count them as needing audit, not as positive contagion. On the clean
    # side, treat ambiguity as present-in-clean so CCR is conservative.
    cont_false = cont["asserts_false"] and not cont.get("ambiguous")
    # E2: a clean-side rule hit that NLI says is mention-only is not a real "clean asserts the seed".
    clean_false = clean["asserts_false"] and not clean_nli_override
    is_ccr = cont_false and not clean_false
    # CCR requires presence in the contaminated response and absence from its clean match.
    return {
        "hcr_has_contagious": cont_false,
        "contaminated_false_claims": (
            [{"text": seed_claim, "depends_on_seed": is_ccr, "present_in_clean": clean_false}] if cont_false else []
        ),
        "clean_has_seed_claim": clean_false,
        "derived_hallucination_count": 1 if is_ccr else 0,
        "chain_depth": 1 if is_ccr else 0,
        "auto_labeled": True,
        "detector": "rule+nli" if clean_nli_override else "rule",
        # Retain refutation, ambiguity, and clean-side rule signals for audit.
        "contaminated_refuted": cont["refuted"],
        "contaminated_ambiguous": cont["ambiguous"],
        "clean_asserts_false": clean_false,
        "clean_rule_asserts_false": clean["asserts_false"],
        "clean_nli_override": clean_nli_override,
        "needs_human_review": bool(cont["ambiguous"] or clean["ambiguous"] or clean_false or clean_nli_override),
    }


def label_injected(config: dict, run_dir: Path, pairs: dict, outputs_path: Path | None = None) -> tuple[list, list]:
    det_cfg = config.get("pilot", {}).get("detector", {})
    neg_window = int(det_cfg.get("negation_window", 4))
    # E2 NLI guard for clean-side mention-only false positives (no-op without a local NLI model).
    nli_guard = bool(det_cfg.get("nli_clean_guard", True))
    required_conditions = {"clean", "contaminated"}
    if config.get("pilot", {}).get("ccr", {}).get("placebo_clean_vs_clean", False):
        required_conditions.add("clean_b")
    if nli_guard:
        nli_cfg = config.get("pilot", {}).get("nli", {})
        configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
    by_key: dict[tuple, dict[str, str]] = defaultdict(dict)
    src: dict[tuple, str] = {}
    source_mock: dict[tuple, bool] = {}
    source_model_artifact: dict[tuple, str | None] = {}
    formal_tasks = {}
    output_file = outputs_path or (run_dir / "model_outputs.jsonl")
    output_manifest = None
    if outputs_path is None:
        formal_tasks_path = run_dir / "generation_tasks.jsonl"
        formal_task_rows = list(read_jsonl(formal_tasks_path))
        if not formal_task_rows:
            raise ValueError("Formal generation task file is empty.")
        formal_tasks = {row["task_id"]: row for row in formal_task_rows}
        if len(formal_tasks) != len(formal_task_rows):
            raise ValueError("Formal generation task file contains duplicate task_id values.")
        output_manifest = load_output_manifest(output_file, formal_tasks_path)
    # outputs_path lets the closed-model arm label model_outputs_api.jsonl with the same detector.
    seen_output_keys: set[tuple[str, str, str]] = set()
    for o in read_jsonl(output_file):
        if outputs_path is None:
            task = formal_tasks.get(o.get("task_id"))
            if task is None:
                raise ValueError(
                    f"Generation output {o.get('task_id')} is unknown."
                )
            require_task_binding(o, task)
            require_model_binding(o, output_manifest)
        output_key = (o["pair_id"], o["model"], o["condition"])
        if output_key in seen_output_keys:
            raise ValueError(
                f"Duplicate generation output for pair/model/condition={output_key}"
            )
        seen_output_keys.add(output_key)
        by_key[(o["pair_id"], o["model"])][o["condition"]] = (o.get("response") or "").strip()
        src[(o["pair_id"], o["model"])] = o.get("source_dataset", "unknown")
        source_mock[(o["pair_id"], o["model"])] = bool(o.get("mock"))
        artifact_key = (o["pair_id"], o["model"])
        artifact_sha = o.get("model_artifact_sha256")
        previous_artifact = source_model_artifact.get(artifact_key)
        if previous_artifact is not None and previous_artifact != artifact_sha:
            raise ValueError(
                f"inconsistent model-weight fingerprints across conditions for {artifact_key}"
            )
        source_model_artifact[artifact_key] = artifact_sha

    real, placebo = [], []
    for (pair_id, model), conds in by_key.items():
        missing = required_conditions - set(conds)
        if missing:
            raise ValueError(
                f"Incomplete generation conditions for pair_id={pair_id}, model={model}: "
                f"missing={sorted(missing)} present={sorted(conds)}"
            )
        p = pairs.get(pair_id)
        if p is None:
            continue
        sc, corr = p["seed_claim"], p.get("corrected_claim", "")
        clean_a = conds.get("clean")
        contaminated = conds.get("contaminated")
        clean_b = conds.get("clean_b")

        if clean_a is not None and contaminated is not None:
            cont_d = detect_injected(contaminated, sc, corr, neg_window)
            cleana_d = detect_injected(clean_a, sc, corr, neg_window)
            cleana_over = _clean_nli_override(clean_a, cleana_d, sc, nli_guard)
            real.append({
                "pair_id": pair_id, "model": model, "source_dataset": src[(pair_id, model)],
                "source_mock": source_mock[(pair_id, model)],
                "source_model_artifact_sha256": source_model_artifact[(pair_id, model)],
                "seed_claim": sc, "annotation": _annotation(cont_d, cleana_d, sc, cleana_over),
                "source_response_sha256": object_sha256((clean_a, contaminated)),
            })
        # Placebo: treat clean_A as if it were the "contaminated" side vs clean_B (the clean side here).
        if clean_a is not None and clean_b is not None:
            a_d = detect_injected(clean_a, sc, corr, neg_window)
            b_d = detect_injected(clean_b, sc, corr, neg_window)
            b_over = _clean_nli_override(clean_b, b_d, sc, nli_guard)
            placebo.append({
                "pair_id": pair_id, "model": model, "source_dataset": src[(pair_id, model)],
                "source_mock": source_mock[(pair_id, model)],
                "source_model_artifact_sha256": source_model_artifact[(pair_id, model)],
                "seed_claim": sc, "annotation": _annotation(a_d, b_d, sc, b_over),
                "source_response_sha256": object_sha256((clean_a, clean_b)),
            })
    return real, placebo


def label_selfinduced(config: dict, run_dir: Path, task_rows: list[dict]) -> list:
    # NLI model id + threshold come from pilot.nli (not hardcoded at the call site), so the
    # exact self-induced labeler is reproducible from the shipped config.
    nli_cfg = config.get("pilot", {}).get("nli", {})
    nli_enabled = bool(nli_cfg.get("enabled", True))
    if nli_enabled:
        configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
    tasks = {r["task_id"]: r for r in task_rows}
    if len(tasks) != len(task_rows):
        raise ValueError("Self-induced stage-2 task file contains duplicate task_id values.")
    tasks_path = run_dir / "selfinduced_stage2_tasks.jsonl"
    outputs_path = run_dir / "selfinduced_stage2_outputs.jsonl"
    output_manifest = load_output_manifest(outputs_path, tasks_path)
    rows = []
    for r in read_jsonl(outputs_path):
        task = tasks.get(r.get("task_id"))
        if task is None:
            raise ValueError(
                f"Self-induced output {r.get('task_id')} is unknown."
            )
        require_task_binding(r, task)
        require_model_binding(r, output_manifest)
        turn1 = (task.get("turn1_response") or "").strip()
        turn2 = (r.get("response") or "").strip()
        # The truth is task-bound at stage 1 and copied through inference/stage 2.
        # Re-reading the mutable bank here can adjudicate old outputs against a new
        # answer key, so missing provenance is a hard error rather than a fallback.
        truth_stmt = (task.get("truth_statement") or "").strip()
        if not truth_stmt:
            raise ValueError(
                f"self-induced output {r.get('task_id')} lacks task-bound truth_statement; "
                "rebuild stage-1/stage-2 tasks with the current pipeline."
            )

        # Automatic checks may suggest a label; human judgment remains authoritative.
        abstained = looks_like_abstention(turn1) if turn1 else False
        rel = (
            nli_relation(turn1, truth_stmt)
            if nli_enabled and turn1 and truth_stmt and not abstained else None
        )
        if abstained:
            turn1_false_suggested = False        # abstention is not a self-hallucination
        elif rel is None:
            turn1_false_suggested = None         # NLI unavailable -> nothing to suggest
        else:
            turn1_false_suggested = (rel == "contradiction")

        # turn-2 inheritance of the turn-1 claim (also a suggestion until confirmed).
        inherit = nli_entails(turn2, turn1) if nli_enabled and turn1 else None
        rows.append({
            "task_id": r.get("task_id"),
            "pair_id": task.get("pair_id"), "model": task.get("model"), "q_id": task.get("q_id"),
            "turn1_response": turn1, "response": turn2,
            "truth_statement": truth_stmt,
            "question": task.get("question"),
            "evidence_hint": task.get("evidence_hint"),
            "source_mock": bool(r.get("mock")),
            "source_model_artifact_sha256": r.get("model_artifact_sha256"),
            "source_response_sha256": object_sha256((turn1, turn2, truth_stmt)),
            "annotation": {
                # Authoritative label: unset until a human confirms. compute_ccr_metrics counts
                # only turn1_was_false is True, so unconfirmed rows are correctly excluded.
                "turn1_was_false": None,
                "hcr_has_contagious": (bool(inherit) if inherit is not None else None),
                "derived_hallucination_count": 1 if inherit else 0,
                "chain_depth": 1 if inherit else 0,
                "auto_labeled": True,
                "detector": "nli",
                "needs_human_review": True,
                "needs_nli": nli_enabled and rel is None and not abstained,
                "nli_disabled": not nli_enabled,
                # Machine SUGGESTIONS to speed the human (not the label):
                "turn1_false_suggested": turn1_false_suggested,
                "turn1_abstained": abstained,
                "nli_turn1_vs_truth": rel,
            },
        })
    return rows


def select_primary_selfinduced_rows(config: dict, rows: list[dict]) -> list[dict]:
    """Restrict human labeling to the question IDs frozen in the config."""
    qids = config.get("pilot", {}).get("annotation", {}).get("primary_question_ids") or []
    if not qids:
        return rows
    if len(qids) != len(set(qids)):
        raise ValueError("annotation.primary_question_ids contains duplicates")
    selected = [row for row in rows if row.get("q_id") in set(qids)]
    models = set(config.get("pilot", {}).get("model_pool", []))
    expected = {(q_id, model) for q_id in qids for model in models}
    observed = {(row.get("q_id"), row.get("model")) for row in selected}
    if observed != expected or len(selected) != len(expected):
        raise ValueError(
            "primary self-induced annotation cohort is incomplete: "
            f"expected={len(expected)} observed={len(selected)} "
            f"missing={len(expected - observed)} unexpected={len(observed - expected)}"
        )
    return selected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Automatic labeling (injected + placebo, or self-induced).")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--arm", choices=["injected", "selfinduced"], default="injected")
    p.add_argument(
        "--require-nli",
        action="store_true",
        help="Exit non-zero if the NLI model cannot load (instead of silently degrading to no suggestions). "
        "Recommended for the self-induced arm and whenever you rely on the injected E2 clean-guard.",
    )
    p.add_argument("--outputs", default=None, help="Override the model_outputs file to label (injected arm), "
                   "e.g. model_outputs_api.jsonl for the closed-model arm.")
    p.add_argument("--suffix", default="", help="Suffix for the written annotation files, e.g. _api -> "
                   "claim_annotation_injected_api.jsonl (keeps the closed arm separate from the open one).")
    p.add_argument("--allow-cohort-change", action="store_true",
                   help="Allow an existing annotation file to be overwritten with a different source_dataset cohort.")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow labeling an incomplete formal output file. Intended only for smoke/subset diagnostics.",
    )
    return p.parse_args()


def _human_label_key(row: dict) -> tuple | None:
    """Stable row key for preserving human labels across regenerated annotation files."""
    tid = row.get("task_id")
    if tid is not None:
        return ("task_id", tid)
    vals = (row.get("pair_id"), row.get("model"), row.get("source_dataset"), row.get("seed_claim"))
    if all(v is not None for v in vals):
        return ("pair_model_dataset_claim", *vals)
    vals = (row.get("pair_id"), row.get("model"), row.get("source_dataset"))
    if all(v is not None for v in vals):
        return ("pair_model_dataset", *vals)
    return None


def _preserve_human_labels(new_rows: list, existing_path: Path, allow_change: bool = False) -> tuple[list, int]:
    """Reject accidental cohort loss, then carry human labels onto rebuilt rows.

    auto_label rebuilds the annotation file from scratch every run; without this, re-running it after the
    server adds models would WIPE the human-confirmed turn1_was_false / inheritance labels (exactly the
    data loss that cost a 300-row annotation pass). We keep the fresh machine suggestions but overlay any
    field the human set (tracked in `human_fields`), so existing labels survive and only NEW rows are blank.
    """
    if not existing_path.exists():
        return new_rows, 0
    old_rows = list(read_jsonl(existing_path))
    if not allow_change:
        old_ds = {str(r.get("source_dataset", "unknown")) for r in old_rows}
        new_ds = {str(r.get("source_dataset", "unknown")) for r in new_rows}
        if old_ds != new_ds:
            raise SystemExit(
                f"Refusing to overwrite {to_project_relative(existing_path)} with a different source_dataset cohort: "
                f"existing={sorted(old_ds)} new={sorted(new_ds)}. Use a distinct --suffix for the new cohort, "
                "or pass --allow-cohort-change only if this replacement is intentional."
            )
        old_keys = {k for r in old_rows if (k := _human_label_key(r)) is not None}
        new_keys = {k for r in new_rows if (k := _human_label_key(r)) is not None}
        if not old_keys.issubset(new_keys):
            raise SystemExit(
                f"Refusing to overwrite {to_project_relative(existing_path)} with a strict subset "
                f"of its prior rows: removed={len(old_keys - new_keys)}. Finish/merge model outputs "
                "first, use a distinct --suffix, or explicitly pass --allow-cohort-change."
            )
    prev = {}
    for r in old_rows:
        key = _human_label_key(r)
        if key is not None:
            prev[key] = r
    kept = 0
    for r in new_rows:
        old_row = prev.get(_human_label_key(r))
        if not old_row:
            continue
        if (
            not old_row.get("source_response_sha256")
            or old_row.get("source_response_sha256") != r.get("source_response_sha256")
        ):
            continue
        pa = old_row.get("annotation", {})
        hf = pa.get("human_fields") or []
        if not hf:
            continue
        a = r["annotation"]
        for k in hf:
            if k in pa:
                a[k] = pa[k]          # overlay the human-decided value(s)
        a["human_fields"] = sorted(hf)
        a["human_confirmed"] = pa.get("human_confirmed", True)
        if "annotator" in pa:
            a["annotator"] = pa["annotator"]
        if "hcr_machine" in pa:
            a["hcr_machine"] = pa["hcr_machine"]
        kept += 1
    return new_rows, kept


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])

    # NLI preflight. For injected it only gates the optional E2 clean-guard, so skip the probe
    # when the guard is off; for self-induced NLI drives the suggestions, so always probe.
    nli_guard_on = bool(config.get("pilot", {}).get("detector", {}).get("nli_clean_guard", True))
    nli_enabled = bool(config.get("pilot", {}).get("nli", {}).get("enabled", True))
    if ((args.arm == "selfinduced" and nli_enabled)
            or (args.arm == "injected" and nli_guard_on)
            or args.require_nli):
        _nli_preflight(
            config,
            require=(args.require_nli or (args.arm == "injected" and nli_guard_on)),
            arm=args.arm,
        )

    if args.arm == "injected":
        outputs_path = resolve_project_path(args.outputs) if args.outputs else None
        if outputs_path is not None and not args.suffix:
            default_outputs = run_dir / "model_outputs.jsonl"
            if outputs_path.resolve() != default_outputs.resolve():
                raise SystemExit(
                    "--outputs points to a non-default model output file; pass --suffix "
                    "(for example --suffix _api) so it cannot overwrite the default annotations."
                )
        sfx = args.suffix
        inj_path = run_dir / f"claim_annotation_injected{sfx}.jsonl"
        pl_path = run_dir / f"claim_annotation_placebo{sfx}.jsonl"
        pairs = {
            p["pair_id"]: p
            for p in read_jsonl(processed_dir / f"{run_name}_pairs_all.jsonl")
        }
        real, placebo = label_injected(config, run_dir, pairs, outputs_path)
        if outputs_path is None and not args.suffix and not args.allow_partial:
            expected = {
                (pair_id, model)
                for pair_id in pairs
                for model in config["pilot"]["model_pool"]
            }
            observed_list = [(r["pair_id"], r["model"]) for r in real]
            observed = set(observed_list)
            if observed != expected or len(observed_list) != len(expected):
                raise SystemExit(
                    "Refusing to label incomplete formal injected outputs: "
                    f"expected={len(expected)} pair/model rows observed={len(observed_list)} "
                    f"unique_observed={len(observed)} "
                    f"missing={len(expected - observed)} unexpected={len(observed - expected)}. "
                    "Finish all models, or use --allow-partial for a non-authoritative diagnostic."
                )
        # Preserve any optional human audit labels across re-runs (same merge as self-induced).
        real, kept_r = _preserve_human_labels(real, inj_path, args.allow_cohort_change)
        write_jsonl(inj_path, real)
        n_e2 = sum(1 for r in real if r["annotation"].get("clean_nli_override"))
        msg = {"arm": "injected", "wrote": to_project_relative(inj_path), "wrote_real": len(real),
               "clean_nli_overrides_e2": n_e2, "human_labels_kept": kept_r}
        if placebo:
            placebo, _ = _preserve_human_labels(placebo, pl_path, args.allow_cohort_change)
            write_jsonl(pl_path, placebo)
            msg["wrote_placebo"] = len(placebo)
            msg["placebo_clean_nli_overrides_e2"] = sum(1 for r in placebo if r["annotation"].get("clean_nli_override"))
        print(json.dumps(msg, ensure_ascii=False))
    else:
        tasks_path = run_dir / "selfinduced_stage2_tasks.jsonl"
        task_rows = list(read_jsonl(tasks_path))
        rows = label_selfinduced(config, run_dir, task_rows)
        if not args.allow_partial:
            expected_id_list = [r["task_id"] for r in task_rows]
            expected_ids = set(expected_id_list)
            observed_id_list = [r.get("task_id") for r in rows]
            observed_ids = set(observed_id_list)
            self_bank = json.loads(
                resolve_project_path(config["paths"]["selfinduced_bank"]).read_text(
                    encoding="utf-8",
                )
            )
            expected_q_models = {
                (q["q_id"], model)
                for q in self_bank.get("questions", [])
                for model in config["pilot"].get("model_pool", [])
            }
            observed_q_model_list = [
                (r.get("q_id"), r.get("model")) for r in rows
            ]
            expected_formal_n = (
                len(config["pilot"].get("model_pool", []))
                * int(
                    config["pilot"].get("selfinduced", {}).get(
                        "questions_per_model", 0,
                    )
                )
            )
            if (
                len(expected_id_list) != len(expected_ids)
                or observed_ids != expected_ids
                or len(observed_id_list) != len(expected_ids)
                or (
                    expected_formal_n > 0
                    and len(expected_ids) != expected_formal_n
                )
                or (
                    expected_q_models
                    and (
                        len(observed_q_model_list) != len(expected_q_models)
                        or set(observed_q_model_list) != expected_q_models
                    )
                )
            ):
                raise SystemExit(
                    "Refusing to label incomplete formal self-induced outputs: "
                    f"expected={len(expected_id_list)} unique_expected={len(expected_ids)} "
                    f"configured_formal_n={expected_formal_n or 'not-set'} "
                    f"observed={len(observed_id_list)} unique_observed={len(observed_ids)} "
                    f"expected_question_models={len(expected_q_models)} "
                    f"observed_unique_question_models={len(set(observed_q_model_list))} "
                    f"missing={len(expected_ids - observed_ids)} unexpected={len(observed_ids - expected_ids)}. "
                    "Finish stage 2, or use --allow-partial for a diagnostic."
                )
        # Generation completeness is checked above on all outputs; human review uses
        # only the question cohort frozen in the config.
        rows = select_primary_selfinduced_rows(config, rows)
        # Preserve human-confirmed labels across recomputation.
        rows, kept = _preserve_human_labels(
            rows, run_dir / "claim_annotation_selfinduced.jsonl", args.allow_cohort_change
        )
        write_jsonl(run_dir / "claim_annotation_selfinduced.jsonl", rows)
        n_needs_nli = sum(1 for r in rows if r["annotation"].get("needs_nli"))
        n_suggest_false = sum(1 for r in rows if r["annotation"].get("turn1_false_suggested") is True)
        n_abstain = sum(1 for r in rows if r["annotation"].get("turn1_abstained"))
        # turn1_was_false is intentionally None for every row: a human must confirm the
        # suggested-false ones before HCR_si has a denominator.
        print(json.dumps({
            "arm": "selfinduced", "wrote": len(rows), "human_labels_kept": kept,
            "turn1_false_suggested": n_suggest_false, "abstained": n_abstain,
            "needs_nli": n_needs_nli,
            "note": f"{kept} existing human labels preserved; remaining rows need human confirmation before HCR_si.",
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
