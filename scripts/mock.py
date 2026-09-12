from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from jsonl import read_jsonl, write_jsonl
from manifest import file_sha256, object_sha256, write_manifest


# Mock runner for Pilot-V1.5 pipeline smoke tests ONLY. No GPU, no model.
# It fabricates generation outputs and annotations so the whole local -> metrics
# flow can be validated end to end before anything runs on the server.
# Mock outputs MUST NOT be used as research results.
#
# Steps:
#   stage1    : selfinduced_stage1_tasks.jsonl   -> selfinduced_stage1_outputs.jsonl
#   stage2    : selfinduced_stage2_tasks.jsonl   -> selfinduced_stage2_outputs.jsonl
#   injected  : processed/pilot_pairs_all.jsonl  -> model_outputs.jsonl (clean+contaminated)
#   annotate  : claim_annotation_*.jsonl         -> filled in place (*_mockfilled.jsonl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock Pilot-V1.5 generation/annotation for smoke tests only.")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--step", required=True, choices=["stage1", "stage2", "injected", "annotate"])
    parser.add_argument("--allow-clobber", action="store_true",
                        help="Actually overwrite pipeline output files. Intended only for local smoke-test sandboxes.")
    return parser.parse_args()


def run_dir_of(config) -> Path:
    run_name = config["pilot"].get("run_name", "pilot")
    return resolve_project_path(config["paths"]["runs_dir"]) / run_name


def _require_clobber(allow: bool, path: Path) -> None:
    if not allow:
        raise SystemExit(
            f"Mock step would overwrite {path}. Re-run in a smoke-test sandbox with --allow-clobber, "
            "or use the real runner for research data."
        )


def _write_mock_outputs(
    config: dict, config_path: Path, tasks_path: Path, output_path: Path, rows: list[dict]
) -> None:
    write_jsonl(output_path, rows)
    models = sorted({str(row["model"]) for row in rows})
    artifacts = {
        model: {
            "repository": None,
            "model_path": None,
            "directory_sha256": "MOCK-NOT-A-MODEL",
        }
        for model in models
    }
    write_manifest(
        output_path.with_suffix(".manifest.json"),
        config_path,
        config,
        {
            "runner": "mock",
            "tasks_path": to_project_relative(tasks_path),
            "tasks_sha256": file_sha256(tasks_path),
            "output_path": to_project_relative(output_path),
            "output_sha256": file_sha256(output_path),
            "num_outputs": len(rows),
            "models": models,
            "model_artifacts": artifacts,
            "warning": "Synthetic smoke-test output; never valid for research metrics.",
        },
    )


def mock_stage1(config, rng, allow_clobber: bool, config_path: Path) -> None:
    rd = run_dir_of(config)
    _require_clobber(allow_clobber, rd / "selfinduced_stage1_outputs.jsonl")
    tasks = list(read_jsonl(rd / "selfinduced_stage1_tasks.jsonl"))
    rows = []
    for t in tasks:
        # ~50% of the time the mock "hallucinates" a fake specific answer
        if rng.random() < 0.5:
            resp = f"The answer to '{t['question']}' is Jane Falsename (1962), a noted figure."
        else:
            resp = "I am not certain about the precise details of that question."
        rows.append({
            **{k: v for k, v in t.items() if k not in ("prompt", "messages")},
            "response": resp,
            "task_sha256": object_sha256(t),
            "model_artifact_sha256": "MOCK-NOT-A-MODEL",
            "mock": True,
        })
    _write_mock_outputs(
        config, config_path, rd / "selfinduced_stage1_tasks.jsonl",
        rd / "selfinduced_stage1_outputs.jsonl", rows,
    )
    print(json.dumps({"wrote": str(rd / "selfinduced_stage1_outputs.jsonl"), "rows": len(rows), "mock": True}))


def mock_stage2(config, rng, allow_clobber: bool, config_path: Path) -> None:
    rd = run_dir_of(config)
    _require_clobber(allow_clobber, rd / "selfinduced_stage2_outputs.jsonl")
    tasks = list(read_jsonl(rd / "selfinduced_stage2_tasks.jsonl"))
    rows = []
    for t in tasks:
        t1 = t.get("turn1_response", "")
        if "Falsename" in t1 and rng.random() < 0.6:
            resp = "Building on Jane Falsename, she also founded the fictional Falsename Institute in 1970."
        else:
            resp = "I would rather not add unverified details."
        rows.append({
            **{k: v for k, v in t.items() if k not in ("prompt", "messages")},
            "response": resp,
            "task_sha256": object_sha256(t),
            "model_artifact_sha256": "MOCK-NOT-A-MODEL",
            "mock": True,
        })
    _write_mock_outputs(
        config, config_path, rd / "selfinduced_stage2_tasks.jsonl",
        rd / "selfinduced_stage2_outputs.jsonl", rows,
    )
    print(json.dumps({"wrote": str(rd / "selfinduced_stage2_outputs.jsonl"), "rows": len(rows), "mock": True}))


def mock_injected(config, rng, allow_clobber: bool, config_path: Path) -> None:
    rd = run_dir_of(config)
    _require_clobber(allow_clobber, rd / "model_outputs.jsonl")
    tasks = list(read_jsonl(rd / "generation_tasks.jsonl"))
    rows = []
    for task in tasks:
        cond = task["condition"]
        echo_rate = 0.35 if cond == "contaminated" else 0.05
        if rng.random() < echo_rate:
            resp = (
                f"As established earlier, {task.get('seed_claim', '')} "
                "Therefore a related detail follows."
            )
        else:
            resp = "Continuing carefully without relying on unsupported prior claims."
        rows.append({
            **{
                key: value for key, value in task.items()
                if key not in ("prompt", "messages")
            },
            "response": resp,
            "task_sha256": object_sha256(task),
            "model_artifact_sha256": "MOCK-NOT-A-MODEL",
            "mock": True,
        })
    _write_mock_outputs(
        config, config_path, rd / "generation_tasks.jsonl",
        rd / "model_outputs.jsonl", rows,
    )
    print(json.dumps({"wrote": str(rd / "model_outputs.jsonl"), "rows": len(rows), "mock": True}))


def _fill_injected_ann(row, rng):
    contaminated = row.get("contaminated_response", "") or ""
    seed = row.get("seed_claim", "") or ""
    contagious = bool(seed and seed[:20] in contaminated)
    ann = {
        "hcr_has_contagious": contagious,
        "contaminated_false_claims": (
            [{"text": seed, "depends_on_seed": True, "present_in_clean": False}] if contagious else []
        ),
        "clean_has_seed_claim": False,
        "derived_hallucination_count": 1 if contagious else 0,
        "chain_depth": 1 if contagious else 0,
        "notes": "mock",
    }
    return ann


def mock_annotate(config, rng, allow_clobber: bool, _config_path: Path) -> None:
    rd = run_dir_of(config)
    for name, kind in (("claim_annotation_injected.jsonl", "injected"), ("claim_annotation_selfinduced.jsonl", "self")):
        path = rd / name
        if not path.exists():
            continue
        rows_in = list(read_jsonl(path))
        # Footgun guard: the injected arm is labeled by auto_labeling/auto_label.py, not by this mock
        # annotator. If the file is already auto-labeled, filling it here would CLOBBER real labels
        # with fabricated ones (and _fill_injected_ann reads template-era fields that auto_label does
        # not emit). Skip it loudly. The injected smoke path is: mock --step injected -> auto_label.
        if kind == "injected" and any((r.get("annotation") or {}).get("auto_labeled") for r in rows_in):
            print(json.dumps({"skipped": name, "reason": "already auto-labeled by auto_label.py; mock annotate would clobber real labels"}))
            continue
        rows = []
        for row in rows_in:
            if kind == "injected":
                row["annotation"] = _fill_injected_ann(row, rng)
                if "annotation_2" in row:
                    a2 = _fill_injected_ann(row, rng)
                    # inject occasional disagreement so kappa is non-degenerate
                    if rng.random() < 0.15:
                        a2["hcr_has_contagious"] = not a2["hcr_has_contagious"]
                    row["annotation_2"] = a2
            else:
                t1_false = "Falsename" in (row.get("turn1_response") or "")
                contagious = t1_false and "Falsename" in (row.get("response") or "")
                row["annotation"] = {
                    "turn1_was_false": t1_false,
                    "hcr_has_contagious": contagious,
                    "derived_hallucination_count": 1 if contagious else 0,
                    "chain_depth": 1 if contagious else 0,
                    "notes": "mock",
                    "human_fields": [
                        "turn1_was_false", "hcr_has_contagious",
                    ],
                }
                if "annotation_2" in row:
                    a2 = dict(row["annotation"])
                    if rng.random() < 0.15:
                        a2["hcr_has_contagious"] = not a2["hcr_has_contagious"]
                    a2["human_fields"] = [
                        "turn1_was_false", "hcr_has_contagious",
                    ]
                    row["annotation_2"] = a2
            rows.append(row)
        out = path.with_name(path.stem + "_mockfilled.jsonl")
        write_jsonl(out, rows)
        print(json.dumps({"wrote": str(out), "rows": len(rows), "mock": True}))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = resolve_project_path(args.config)
    ensure_dirs(config)
    rng = random.Random(int(config["random_seed"]))
    {"stage1": mock_stage1, "stage2": mock_stage2, "injected": mock_injected, "annotate": mock_annotate}[args.step](
        config, rng, args.allow_clobber, config_path
    )


if __name__ == "__main__":
    main()
