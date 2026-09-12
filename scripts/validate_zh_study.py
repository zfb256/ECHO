from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "auto_labeling"))

from detector import detect_injected, detector_blind  # noqa: E402
from manifest import file_sha256  # noqa: E402


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load(ROOT / "configs/zh_study.json")
    selfinduced_config = load(ROOT / "configs/zh_selfinduced.json")
    seeds = load(ROOT / "configs/zh_seed_bank.json").get("seeds", [])
    questions = load(ROOT / "configs/zh_selfinduced_questions.json").get("questions", [])
    errors: list[str] = []
    frozen_path = ROOT / "datasets_zh/runs/zh_study/FROZEN_INPUTS.json"

    if len(seeds) != 40:
        errors.append(f"expected 40 injected seeds, found {len(seeds)}")
    if len(questions) != 60:
        errors.append(f"expected 60 self-induced bank questions, found {len(questions)}")
    primary_ids = selfinduced_config["pilot"]["annotation"].get("primary_question_ids", [])
    if len(primary_ids) != 40 or len(set(primary_ids)) != 40:
        errors.append("formal self-induced cohort must contain 40 unique question ids")
    if not set(primary_ids).issubset({q.get("q_id") for q in questions}):
        errors.append("formal self-induced cohort contains ids outside the 60-question bank")
    if selfinduced_config["pilot"]["selfinduced"]["questions_per_model"] != 60:
        errors.append("self-induced generation must remain locked at 60 questions per model")
    if len(config["pilot"]["model_pool"]) != 5:
        errors.append("model_pool must contain exactly 5 models")
    if config["pilot"].get("nli", {}).get("enabled") is not False:
        errors.append("unvalidated Chinese NLI must remain disabled for the frozen study")
    if config["pilot"].get("detector", {}).get("nli_clean_guard") is not False:
        errors.append("unvalidated NLI clean guard must remain disabled for the frozen study")

    for item, fields, ident in [
        *((s, ("claim", "corrected_claim", "entity"), s.get("seed_id", "?")) for s in seeds),
        *((q, ("question", "followup", "truth_statement"), q.get("q_id", "?")) for q in questions),
    ]:
        for field in fields:
            value = item.get(field, "")
            if not _CJK.search(value):
                errors.append(f"{ident}.{field} has no Chinese text")

    for seed in seeds:
        sid = seed["seed_id"]
        if detector_blind(seed["claim"], seed["corrected_claim"]):
            errors.append(f"{sid} is detector-blind")
        positive = detect_injected(seed["claim"], seed["claim"], seed["corrected_claim"])
        negative = detect_injected(seed["corrected_claim"], seed["claim"], seed["corrected_claim"])
        correction = detect_injected(
            f"不对，正确说法是：{seed['corrected_claim']}",
            seed["claim"],
            seed["corrected_claim"],
        )
        if not positive["asserts_false"]:
            errors.append(f"{sid} construct-positive is not detected")
        if negative["asserts_false"]:
            errors.append(f"{sid} corrected claim is detected as false")
        if correction["asserts_false"]:
            errors.append(f"{sid} explicit Chinese correction is detected as contagion")

    if not frozen_path.exists():
        errors.append(f"missing frozen-input manifest: {frozen_path.relative_to(ROOT)}")
    else:
        frozen = load(frozen_path)
        required = {"config", "seed_bank", "processed_pairs", "generation_tasks", "construct_gold"}
        for key in sorted(required - frozen.keys()):
            errors.append(f"FROZEN_INPUTS missing required entry: {key}")
        for key, entry in frozen.items():
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            rel = entry.get("path")
            expected = entry.get("sha256")
            if not rel or not expected:
                errors.append(f"FROZEN_INPUTS.{key} missing path/sha256")
                continue
            source = ROOT / rel
            if not source.exists():
                errors.append(f"FROZEN_INPUTS.{key} source missing: {rel}")
            elif file_sha256(source) != expected:
                errors.append(f"FROZEN_INPUTS.{key} sha256 mismatch: {rel}")

    # The redesigned self-induced arm has its own run directory. Its generation
    # manifests bind the verified bank, tasks, and both stages of model output.
    si_run = ROOT / "datasets_zh/runs/zh_selfinduced"
    manifest_checks = (
        ("selfinduced_stage1_tasks.manifest.json", "selfinduced_bank_sha256", ROOT / "configs/zh_selfinduced_questions.json"),
        ("selfinduced_stage1_outputs.manifest.json", "tasks_sha256", si_run / "selfinduced_stage1_tasks.jsonl"),
        ("selfinduced_stage2_tasks.manifest.json", "stage1_outputs_sha256", si_run / "selfinduced_stage1_outputs.jsonl"),
        ("selfinduced_stage2_tasks.manifest.json", "stage1_tasks_sha256", si_run / "selfinduced_stage1_tasks.jsonl"),
        ("selfinduced_stage2_outputs.manifest.json", "tasks_sha256", si_run / "selfinduced_stage2_tasks.jsonl"),
        ("selfinduced_stage2_outputs.manifest.json", "output_sha256", si_run / "selfinduced_stage2_outputs.jsonl"),
    )
    for manifest_name, hash_key, source in manifest_checks:
        manifest_path = si_run / manifest_name
        if not manifest_path.exists() or not source.exists():
            errors.append(f"missing self-induced provenance input: {manifest_name} or {source.relative_to(ROOT)}")
            continue
        if load(manifest_path).get(hash_key) != file_sha256(source):
            errors.append(f"{manifest_name}.{hash_key} mismatch: {source.relative_to(ROOT)}")

    ui_text = "\n".join(
        (ROOT / p).read_text(encoding="utf-8")
        for p in (
            "scripts/annotate_ui.py",
            "scripts/injected_recall_audit_ui.py",
        )
    )
    for required in ("Microsoft YaHei", "待标注", "无法判断", "font:17px"):
        if required not in ui_text:
            errors.append(f"UI redesign marker missing: {required}")

    if errors:
        print("Chinese study validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(json.dumps({
        "ok": True,
        "seeds": len(seeds),
        "questions": len(questions),
        "formal_questions": len(primary_ids),
        "models": len(config["pilot"]["model_pool"]),
        "all_items_human_verified": all(
            not x.get("needs_human_verification") for x in seeds + questions
        ),
        "gpu_required": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
