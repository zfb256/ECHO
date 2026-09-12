from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and compare the frozen Chinese-study artifact reports."
    )
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--selfinduced-config", default="configs/zh_selfinduced.json")
    return parser.parse_args()


def run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout}")


def main() -> None:
    args = parse_args()
    config_path = (ROOT / args.config).resolve()
    self_config_path = (ROOT / args.selfinduced_config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    self_config = json.loads(self_config_path.read_text(encoding="utf-8"))
    run_dir = ROOT / config["paths"]["runs_dir"] / config["pilot"]["run_name"]
    self_run_dir = (
        ROOT
        / self_config["paths"]["runs_dir"]
        / self_config["pilot"]["run_name"]
    )
    reports_dir = ROOT / config["paths"]["reports_dir"]
    python = sys.executable

    run([python, "-B", "scripts/validate_zh_study.py"])
    run([python, "-B", "scripts/validate_seed_bank.py", "--config", args.config, "--require-verified"])

    with tempfile.TemporaryDirectory(prefix=".artifact-check.", dir=ROOT) as tmp:
        tmp_dir = Path(tmp)
        checks: list[tuple[str, Path, list[str]]] = [
            (
                "detector validation",
                reports_dir / "detector_validation.json",
                [python, "-B", "auto_labeling/validate_detector.py", "--config", args.config,
                 "--gold", str((run_dir / "construct_gold.jsonl").relative_to(ROOT))],
            ),
            (
                "turn-1 agreement",
                reports_dir / "agreement_selfinduced_turn1.json",
                [python, "-B", "scripts/compute_agreement.py", "--config", args.selfinduced_config,
                 "--annotations", str(self_run_dir / "claim_annotation_selfinduced.jsonl"),
                 "--field", "turn1_was_false"],
            ),
            (
                "inheritance agreement",
                reports_dir / "agreement_selfinduced_hcr.json",
                [python, "-B", "scripts/compute_agreement.py", "--config", args.selfinduced_config,
                 "--annotations", str(self_run_dir / "claim_annotation_selfinduced.jsonl"),
                 "--field", "hcr_has_contagious"],
            ),
            (
                "injected-output audit",
                reports_dir / "injected_recall_human_audit.json",
                [python, "-B", "scripts/compute_injected_recall_audit.py", "--config", args.config,
                 "--annotations", str(run_dir / "claim_annotation_injected.jsonl"),
                 "--audit", str(run_dir / "injected_recall_human_audit_sample.jsonl"),
                 "--sampling-manifest", str(run_dir / "injected_recall_audit_sampling_manifest.json")],
            ),
            (
                "headline metrics",
                reports_dir / "zh_full_metrics.json",
                [python, "-B", "scripts/compute_ccr_metrics.py", "--config", args.config,
                 "--injected", str(run_dir / "claim_annotation_injected.jsonl"),
                 "--placebo", str(run_dir / "claim_annotation_placebo.jsonl"),
                 "--selfinduced", str(self_run_dir / "claim_annotation_selfinduced.jsonl"),
                 "--selfinduced-config", args.selfinduced_config],
            ),
            (
                "open-model estimand robustness",
                reports_dir / "ccr_estimand_robustness.json",
                [python, "-B", "scripts/compute_ccr_estimand_robustness.py", "--config", args.config,
                 "--injected", str(run_dir / "claim_annotation_injected.jsonl"),
                 "--placebo", str(run_dir / "claim_annotation_placebo.jsonl")],
            ),
            (
                "self-induced annotation chain",
                reports_dir / "selfinduced_second_annotator_chain_check.json",
                [python, "-B", "scripts/check_selfinduced_second_annotator_chain.py",
                 "--config", args.selfinduced_config, "--run-dir", str(self_run_dir)],
            ),
        ]

        for provider in ("deepseek", "doubao"):
            checks.extend([
                (
                    f"{provider} metrics",
                    reports_dir / f"zh_api_{provider}_metrics.json",
                    [python, "-B", "scripts/compute_ccr_metrics.py", "--config", args.config,
                     "--injected", str(run_dir / f"claim_annotation_injected_api_{provider}.jsonl"),
                     "--placebo", str(run_dir / f"claim_annotation_placebo_api_{provider}.jsonl")],
                ),
                (
                    f"{provider} estimand robustness",
                    reports_dir / f"ccr_estimand_robustness_{provider}.json",
                    [python, "-B", "scripts/compute_ccr_estimand_robustness.py", "--config", args.config,
                     "--injected", str(run_dir / f"claim_annotation_injected_api_{provider}.jsonl"),
                     "--placebo", str(run_dir / f"claim_annotation_placebo_api_{provider}.jsonl")],
                ),
                (
                    f"{provider} output validation",
                    reports_dir / f"zh_api_{provider}_validation.json",
                    [python, "-B", "scripts/check_api_outputs.py",
                     "--tasks", str(run_dir / "generation_tasks.jsonl"),
                     "--outputs", str(run_dir / f"model_outputs_api_{provider}.jsonl"),
                     "--injected", str(run_dir / f"claim_annotation_injected_api_{provider}.jsonl"),
                     "--placebo", str(run_dir / f"claim_annotation_placebo_api_{provider}.jsonl")],
                ),
            ])

        mismatches: list[str] = []
        for index, (label, saved, command) in enumerate(checks):
            generated = tmp_dir / f"{index:02d}.json"
            run([*command, "--output", str(generated)])
            if not saved.exists():
                mismatches.append(f"{label}: missing saved report {saved.relative_to(ROOT)}")
            elif generated.read_bytes() != saved.read_bytes():
                mismatches.append(f"{label}: saved report differs from recomputation")
            if label == "headline metrics" or label.endswith(" metrics"):
                saved_csv = saved.with_suffix(".csv")
                generated_csv = generated.with_suffix(".csv")
                if not saved_csv.exists() or generated_csv.read_bytes() != saved_csv.read_bytes():
                    mismatches.append(f"{label}: saved CSV differs from recomputation")
            if label.endswith(" metrics") and label != "headline metrics":
                decision = json.loads(generated.read_text(encoding="utf-8"))["decision"]
                if decision.get("real_output_detector_precision") is not None:
                    mismatches.append(f"{label}: inherited an unrelated open-model audit")
                if decision.get("kappa_required") is not False:
                    mismatches.append(f"{label}: inherited a self-induced kappa requirement")

    if mismatches:
        print(json.dumps({"ok": False, "errors": mismatches}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"ok": True, "checks": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
