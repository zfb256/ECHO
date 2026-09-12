from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path
from jsonl import read_jsonl, write_jsonl
from sampling import deterministic_sample


# Build injected, placebo, and self-induced annotation templates.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create claim-level annotation templates (injected pair-centric + self-induced).")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--injected-outputs", default=None, help="Defaults to runs_dir/<run>/model_outputs.jsonl (clean+contaminated)")
    parser.add_argument("--pairs", default=None, help="Defaults to processed_dir/pilot_pairs_all.jsonl")
    parser.add_argument("--selfinduced-outputs", default=None, help="Defaults to runs_dir/<run>/selfinduced_stage2_outputs.jsonl")
    parser.add_argument("--allow-overwrite", action="store_true",
                        help="Overwrite existing claim_annotation_*.jsonl templates. Dangerous after human/auto labels exist.")
    return parser.parse_args()


def injected_annotation() -> dict:
    return {
        "hcr_has_contagious": None,
        "contaminated_false_claims": [
            # one object per false claim the annotator finds in the CONTAMINATED continuation:
            # {"text": "...", "depends_on_seed": true/false, "present_in_clean": true/false}
        ],
        "clean_has_seed_claim": None,
        "derived_hallucination_count": None,
        "chain_depth": None,
        "notes": "",
    }


def selfinduced_annotation() -> dict:
    return {
        "turn1_was_false": None,
        "hcr_has_contagious": None,
        "derived_hallucination_count": None,
        "chain_depth": None,
        "notes": "",
    }


def write_template(path: Path, rows: list[dict], allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing annotation file: {path}. "
            "This script creates blank templates and can erase human/auto labels. "
            "Pass --allow-overwrite only in a fresh sandbox."
        )
    write_jsonl(path, rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    runs_dir = resolve_project_path(config["paths"]["runs_dir"])
    processed_dir = resolve_project_path(config["paths"]["processed_dir"])
    run_dir = runs_dir / run_name

    seed = int(config["random_seed"])
    dbl_frac = float(config["pilot"]["annotation"].get("double_annotation_fraction", 0.0))
    ccr_cfg = config.get("pilot", {}).get("ccr", {})
    placebo_enabled = bool(ccr_cfg.get("placebo_clean_vs_clean", False))

    def select_double(keys: list, offset: int) -> set:
        # Use a ceiling so any positive fraction selects at least one row.
        n = len(keys)
        if n == 0 or dbl_frac <= 0:
            return set()
        k = max(1, min(n, int(math.ceil(n * dbl_frac))))
        selected = deterministic_sample(keys, k, seed + offset)
        # keys may be lists (pair/model) or strings (task_id); normalize to tuples/str.
        return set(tuple(x) if isinstance(x, list) else x for x in selected)

    wrote = {}

    # ---- INJECTED arm (pair-centric) ----
    injected_path = resolve_project_path(args.injected_outputs) if args.injected_outputs else run_dir / "model_outputs.jsonl"
    pairs_path = resolve_project_path(args.pairs) if args.pairs else processed_dir / f"{config['pilot'].get('run_name', 'pilot')}_pairs_all.jsonl"
    if injected_path.exists() and pairs_path.exists():
        pair_meta = {p["pair_id"]: p for p in read_jsonl(pairs_path)}
        # join clean + contaminated (+ clean_b placebo) responses by (pair_id, model)
        joined: dict[tuple, dict] = {}
        for row in read_jsonl(injected_path):
            cond = row.get("condition")
            if cond not in ("clean", "contaminated", "clean_b"):
                continue
            key = (row["pair_id"], row["model"])
            joined.setdefault(key, {})[cond] = row.get("response", "")
        rows = []
        keys = sorted(k for k, v in joined.items() if "clean" in v and "contaminated" in v)
        dbl_keys = select_double([list(k) for k in keys], offset=0)
        for key in keys:
            pair_id, model = key
            meta = pair_meta.get(pair_id, {})
            row = {
                "pair_id": pair_id,
                "model": model,
                "arm": "injected",
                "source_dataset": meta.get("source_dataset"),
                "seed_id": meta.get("seed_id"),
                "seed_claim": meta.get("seed_claim"),
                "corrected_claim": meta.get("corrected_claim"),
                "clean_response": joined[key]["clean"],
                "contaminated_response": joined[key]["contaminated"],
                "double_annotate": key in dbl_keys,
                "annotation": injected_annotation(),
            }
            if key in dbl_keys:
                row["annotation_2"] = injected_annotation()
            rows.append(row)
        out_path = run_dir / "claim_annotation_injected.jsonl"
        write_template(out_path, rows, args.allow_overwrite)
        wrote["injected"] = {"path": str(out_path), "rows": len(rows), "double": len(dbl_keys)}

        # Treat clean_B as the pseudo-exposed side of the placebo comparison.
        if placebo_enabled:
            placebo_rows = []
            placebo_keys = sorted(k for k, v in joined.items() if "clean" in v and "clean_b" in v)
            p_dbl = select_double([list(k) for k in placebo_keys], offset=1)
            for key in placebo_keys:
                pair_id, model = key
                meta = pair_meta.get(pair_id, {})
                row = {
                    "pair_id": pair_id,
                    "model": model,
                    "arm": "placebo",
                    "source_dataset": meta.get("source_dataset"),
                    "seed_id": meta.get("seed_id"),
                    "seed_claim": meta.get("seed_claim"),
                    "corrected_claim": meta.get("corrected_claim"),
                    "clean_response": joined[key]["clean"],        # clean_A
                    "contaminated_response": joined[key]["clean_b"],  # clean_B, relabeled
                    "double_annotate": key in p_dbl,
                    "annotation": injected_annotation(),
                }
                if key in p_dbl:
                    row["annotation_2"] = injected_annotation()
                placebo_rows.append(row)
            pout_path = run_dir / "claim_annotation_placebo.jsonl"
            write_template(pout_path, placebo_rows, args.allow_overwrite)
            wrote["placebo"] = {"path": str(pout_path), "rows": len(placebo_rows), "double": len(p_dbl)}
    else:
        wrote["injected"] = {"skipped": "missing model_outputs.jsonl or pilot_pairs_all.jsonl"}

    # ---- SELF-INDUCED arm ----
    si_path = resolve_project_path(args.selfinduced_outputs) if args.selfinduced_outputs else run_dir / "selfinduced_stage2_outputs.jsonl"
    if si_path.exists():
        si_rows = list(read_jsonl(si_path))
        keys = [r["task_id"] for r in si_rows]
        dbl_keys = select_double(keys, offset=1)
        rows = []
        for r in si_rows:
            row = {
                "task_id": r["task_id"],
                "pair_id": r.get("pair_id"),
                "model": r.get("model"),
                "arm": "self_induced",
                "source_dataset": "selfinduced",
                "q_id": r.get("q_id"),
                "question": r.get("question"),
                "turn1_response": r.get("turn1_response"),
                "response": r.get("response", ""),
                "double_annotate": r["task_id"] in dbl_keys,
                "annotation": selfinduced_annotation(),
            }
            if r["task_id"] in dbl_keys:
                row["annotation_2"] = selfinduced_annotation()
            rows.append(row)
        out_path = run_dir / "claim_annotation_selfinduced.jsonl"
        write_template(out_path, rows, args.allow_overwrite)
        wrote["self_induced"] = {"path": str(out_path), "rows": len(rows), "double": len(dbl_keys)}
    else:
        wrote["self_induced"] = {"skipped": "missing selfinduced_stage2_outputs.jsonl"}

    print(json.dumps(wrote, ensure_ascii=False))


if __name__ == "__main__":
    main()
