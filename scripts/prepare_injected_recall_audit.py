from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402
from manifest import object_sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a blind human packet for injected-detector real-output recall.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--annotations", default=None)
    p.add_argument("--outputs", nargs="+", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_outputs(paths: list[str]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out = {}
    for path in paths:
        for r in read_jsonl(resolve_project_path(path)):
            key = (r["pair_id"], r["model"], r["condition"])
            if key in out:
                raise ValueError(f"duplicate output key across --outputs files: {key}")
            out[key] = r
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = config["pilot"].get("run_name", "zh_study")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    annotations_path = resolve_project_path(args.annotations) if args.annotations else run_dir / "claim_annotation_injected.jsonl"
    output_paths = args.outputs or [str(run_dir / "model_outputs.jsonl")]
    output_path = resolve_project_path(args.output) if args.output else run_dir / "injected_recall_human_audit_sample.jsonl"
    n = args.n if args.n is not None else int(config["pilot"].get("annotation", {}).get("injected_recall_audit_n", 120))
    seed = args.seed if args.seed is not None else int(config.get("random_seed", 0))
    positive_fraction = float(
        config["pilot"].get("annotation", {}).get(
            "injected_recall_positive_fraction", 0.5
        )
    )
    ann_rows = list(read_jsonl(annotations_path))
    ann_keys = [(r.get("pair_id"), r.get("model")) for r in ann_rows]
    if any(not all(key) for key in ann_keys) or len(set(ann_keys)) != len(ann_keys):
        raise ValueError("annotations contain missing or duplicate pair/model keys")
    outputs = load_outputs(output_paths)
    candidates = []
    for r in ann_rows:
        a = r.get("annotation", {})
        if not isinstance(a.get("hcr_has_contagious"), bool):
            raise ValueError(
                f"annotation row pair_id={r.get('pair_id')} model={r.get('model')} "
                "does not have a final boolean detector label"
            )
        detector_label = a.get("hcr_has_contagious") is True
        key = (r["pair_id"], r["model"], "contaminated")
        o = outputs.get(key)
        if not o:
            continue
        clean = outputs.get((r["pair_id"], r["model"], "clean"))
        if clean is None:
            raise ValueError(f"missing clean output for pair/model={key[:2]}")
        expected_source_hash = object_sha256(
            ((clean.get("response") or "").strip(), (o.get("response") or "").strip())
        )
        if r.get("source_response_sha256") != expected_source_hash:
            raise ValueError(
                f"annotation/output response binding mismatch for pair/model={key[:2]}"
            )
        candidates.append({
            "audit_id": f"injrec_{len(candidates):05d}",
            "pair_id": r["pair_id"],
            "model": r["model"],
            "source_dataset": r.get("source_dataset"),
            "seed_claim": r.get("seed_claim"),
            "corrected_claim": o.get("corrected_claim"),
            "contaminated_response": o.get("response"),
            "detector_label": detector_label,
            "human_asserts_seed_falsehood": None,
            "human_uncertain": False,
            "human_notes": "",
            "instructions": (
                "盲标召回审计：仅当待判断回答断言、预设或直接扩展给定错误命题时标记为真。"
                "纠正或反驳、仅提及、不确定表达和无关错误均标记为假。"
                "不得查看自动标签，不得使用搜索引擎或生成式人工智能。"
            ),
        })
    def stratified(rows: list[dict[str, Any]], target: int, local_seed: int) -> list[dict[str, Any]]:
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(str(row.get("model", "unknown")), []).append(row)
        rng = random.Random(local_seed)
        for pool in by_model.values():
            rng.shuffle(pool)
        chosen: list[dict[str, Any]] = []
        names = sorted(by_model)
        while len(chosen) < min(target, len(rows)):
            progressed = False
            for model in names:
                if by_model[model] and len(chosen) < target:
                    chosen.append(by_model[model].pop())
                    progressed = True
            if not progressed:
                break
        return chosen

    positives = [row for row in candidates if row["detector_label"]]
    negatives = [row for row in candidates if not row["detector_label"]]
    positive_target = min(len(positives), int(round(n * positive_fraction)))
    negative_target = min(len(negatives), n - positive_target)
    sample = stratified(positives, positive_target, seed + 17)
    sample += stratified(negatives, negative_target, seed + 31)
    if len(sample) < min(n, len(candidates)):
        selected_ids = {row["audit_id"] for row in sample}
        remainder = [row for row in candidates if row["audit_id"] not in selected_ids]
        sample += stratified(remainder, n - len(sample), seed + 47)
    random.Random(seed + 59).shuffle(sample)
    detector_positive = sum(1 for row in sample if row["detector_label"])
    detector_negative = len(sample) - detector_positive
    for row in sample:
        row.pop("detector_label")  # Keep the annotator packet genuinely blind.
    out = output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(
            f"refusing to overwrite existing audit packet {out}; move/archive it first"
        )
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in sample) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(out),
        "sample_rows": len(sample),
        "candidate_detector_positive_rows": len(positives),
        "candidate_detector_negative_rows": len(negatives),
        "source_annotations": str(annotations_path),
        "by_model": {
            model: sum(1 for row in sample if str(row.get("model")) == model)
            for model in sorted({str(row.get("model")) for row in sample})
        },
        "sample_detector_positive": detector_positive,
        "sample_detector_negative": detector_negative,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
