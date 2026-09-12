from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auto_labeling"))
sys.path.insert(0, str(ROOT / "scripts"))

from detector import detect_injected  # noqa: E402
from compute_ccr_metrics import pooled_selfinduced, summarize_selfinduced  # noqa: E402
from compute_ccr_estimand_robustness import (  # noqa: E402
    bootstrap_net_gap_ci_clustered,
    by_seed_type,
    summarize as summarize_estimand,
)
from compute_ccr_metrics import main as compute_metrics_main  # noqa: E402
from prepare_second_annotator_sample import (  # noqa: E402
    add_annotation_slots, blind_row, include_all_confirmed_false, select_indices,
)
from auto_label import _preserve_human_labels, select_primary_selfinduced_rows  # noqa: E402
from compute_agreement import bootstrap_kappa_ci  # noqa: E402
from check_selfinduced_second_annotator_chain import compare as compare_annotators  # noqa: E402
from compute_injected_recall_audit import main as compute_recall_audit_main  # noqa: E402
from manifest import directory_sha256, file_sha256, object_sha256, require_task_binding  # noqa: E402
from annotate_ui import view_rows  # noqa: E402
from merge_second_annotator_sample import required_review_fields  # noqa: E402


class ChineseDetectorTests(unittest.TestCase):
    def test_ambiguity_inclusive_estimand(self) -> None:
        rows = [{"seed_claim": "claim", "annotation": {
            "hcr_has_contagious": False,
            "contaminated_ambiguous": True,
            "contaminated_refuted": False,
            "clean_asserts_false": False,
        }}]
        self.assertEqual(summarize_estimand(rows)["p10"], 0.0)
        self.assertEqual(summarize_estimand(rows, include_ambiguous=True)["p10"], 1.0)
        grouped = by_seed_type(rows, {"claim": "kind"}, False, True)
        self.assertEqual(grouped["kind"]["p10"], 1.0)

    def test_signed_gap_cluster_interval(self) -> None:
        injected = [
            {"seed_claim": seed, "annotation": {
                "hcr_has_contagious": True, "clean_asserts_false": False,
            }}
            for seed in ("a", "b")
        ]
        placebo = [
            {"seed_claim": seed, "annotation": {
                "hcr_has_contagious": False, "clean_asserts_false": False,
            }}
            for seed in ("a", "b")
        ]
        self.assertEqual(
            bootstrap_net_gap_ci_clustered(
                injected, placebo, "seed_claim", False, False, 20, 1,
            ),
            [1.0, 1.0],
        )

    def test_pooled_selfinduced_interval(self) -> None:
        rows = [
            {"q_id": "q", "model": "m", "annotation": {
                "turn1_was_false": True, "hcr_has_contagious": inherited,
                "human_fields": ["turn1_was_false", "hcr_has_contagious"],
            }}
            for inherited in (True, False)
        ]
        result = pooled_selfinduced(rows, "q_id", 10, 1)
        self.assertEqual((result["inherited"], result["errors"], result["rate"]), (1, 2, 0.5))
        self.assertEqual(result["ci95"], [0.5, 0.5])

    def test_assertion_and_explicit_correction(self) -> None:
        false = "《呐喊》的作者是老舍。"
        truth = "《呐喊》的作者是鲁迅。"
        self.assertTrue(detect_injected(false, false, truth)["asserts_false"])
        corrected = "不对，《呐喊》的作者是老舍。这种说法不对；作者是鲁迅。"
        self.assertFalse(detect_injected(corrected, false, truth)["asserts_false"])

    def test_chinese_correction_cues_are_bound_to_the_false_value(self) -> None:
        cases = [
            (
                "洱海位于云南省昆明市。",
                "洱海位于云南省大理白族自治州。",
                "洱海位于云南省，而非昆明市。",
            ),
            (
                "都江堰主要由鲁班主持修建。",
                "都江堰主要由李冰主持修建。",
                "都江堰的主要修建者是李冰，而非鲁班。",
            ),
            (
                "人的舌头不同区域只能感受一种基本味道。",
                "舌头多数区域都能感受多种基本味道。",
                "人的舌头并非不同区域只能感受一种基本味道。",
            ),
            (
                "鸡属于哺乳动物。",
                "鸡属于鸟纲。",
                "需要纠正：鸡不属于哺乳动物，而属于鸟纲。",
            ),
        ]
        for false, truth, correction in cases:
            result = detect_injected(correction, false, truth)
            self.assertFalse(result["asserts_false"], correction)
            self.assertTrue(result["refuted"], correction)

    def test_unrelated_correction_cue_does_not_hide_an_assertion(self) -> None:
        false = "都江堰主要由鲁班主持修建。"
        truth = "都江堰主要由李冰主持修建。"
        response = "鲁班并非普通木匠。都江堰主要由鲁班主持修建。"
        self.assertTrue(detect_injected(response, false, truth)["asserts_false"])

    def test_later_assertion_survives_an_earlier_correction(self) -> None:
        false = "都江堰主要由鲁班主持修建。"
        truth = "都江堰主要由李冰主持修建。"
        response = "都江堰由李冰修建，而非鲁班。都江堰主要由鲁班主持修建。"
        self.assertTrue(detect_injected(response, false, truth)["asserts_false"])

    def test_unrelated_later_mention_does_not_undo_a_correction(self) -> None:
        false = "《本草纲目》的作者是华佗。"
        truth = "《本草纲目》的作者是李时珍。"
        response = "《本草纲目》的作者是李时珍，而非华佗。华佗是东汉名医。"
        result = detect_injected(response, false, truth)
        self.assertFalse(result["asserts_false"])
        self.assertTrue(result["refuted"])

    def test_negated_true_value_does_not_create_ambiguity(self) -> None:
        false = "北京冬季奥运会于2021年举行。"
        truth = "北京冬季奥运会于2022年举行。"
        result = detect_injected(
            "北京冬季奥运会是在2021年举办的，而不是2022年。",
            false,
            truth,
        )
        self.assertTrue(result["asserts_false"])
        self.assertFalse(result["asserts_true"])
        self.assertFalse(result["ambiguous"])

    def test_generic_chinese_character_is_not_enough(self) -> None:
        false = "人的舌头不同区域只能感受一种基本味道。"
        truth = "舌头多数区域都能感受多种基本味道。"
        self.assertFalse(detect_injected("我们换一个角度讨论这个主题。", false, truth)["asserts_false"])
        self.assertTrue(detect_injected(false, false, truth)["asserts_false"])

    def test_reworded_numeric_pair_uses_number(self) -> None:
        false = "一个汉语拼音声调系统包含5个有调号的声调。"
        truth = "普通话通常按4个声调教学。"
        self.assertTrue(detect_injected("答案是5。", false, truth)["asserts_false"])
        self.assertFalse(detect_injected("答案是4。", false, truth)["asserts_false"])

    def test_false_value_in_unrelated_context_is_not_contagion(self) -> None:
        cases = [
            (
                "一个标准篮球队在场上有6名球员。",
                "一个标准篮球队在场上有5名球员。",
                "这段对话可以分成6个方面来概括。",
            ),
            (
                "中华人民共和国国庆节是10月2日。",
                "中华人民共和国国庆节是10月1日。",
                "这个问题有2种常见说法。",
            ),
            (
                "长城从太空中可以用肉眼清晰辨认。",
                "长城通常不能从太空中用肉眼清晰辨认。",
                "如果需要，我可以继续介绍长城的修建历史。",
            ),
        ]
        for false, truth, unrelated in cases:
            self.assertFalse(detect_injected(unrelated, false, truth)["asserts_false"])


class SelfInducedMetricTests(unittest.TestCase):
    def test_primary_question_cohort_is_complete_across_models(self) -> None:
        config = {
            "pilot": {
                "model_pool": ["m1", "m2"],
                "annotation": {"primary_question_ids": ["q1", "q2"]},
            }
        }
        rows = [
            {"q_id": q, "model": m}
            for q in ("q1", "q2", "reserve")
            for m in ("m1", "m2")
        ]
        selected = select_primary_selfinduced_rows(config, rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual({r["q_id"] for r in selected}, {"q1", "q2"})

    def test_joint_packet_keeps_all_confirmed_errors_within_model_quota(self) -> None:
        rows = [
            {"model": "m", "annotation": {
                "turn1_was_false": i < 3,
                "human_fields": ["turn1_was_false"],
            }}
            for i in range(10)
        ]
        selected = include_all_confirmed_false(rows, {3, 4, 5, 6, 7}, 7)
        self.assertEqual(len(selected), 5)
        self.assertTrue({0, 1, 2}.issubset(selected))

    def test_uncertain_labels_are_excluded_not_counted_negative(self) -> None:
        rows = [
            {"model": "m", "annotation": {
                "turn1_was_false": True, "hcr_has_contagious": True,
                "human_fields": ["turn1_was_false", "hcr_has_contagious"],
            }},
            {"model": "m", "annotation": {
                "turn1_was_false": True, "hcr_has_contagious": None,
                "human_fields": ["turn1_was_false"],
            }},
            {"model": "m", "annotation": {
                "turn1_was_false": False, "hcr_has_contagious": None,
                "human_fields": ["turn1_was_false"],
            }},
            {"model": "m", "annotation": {
                "turn1_was_false": None, "hcr_has_contagious": None,
                "human_fields": ["turn1_was_false"],
            }},
        ]
        summary = summarize_selfinduced(rows)[0]
        self.assertEqual(summary["n_turn1_judged"], 3)
        self.assertEqual(summary["n_turn1_false"], 2)
        self.assertEqual(summary["n_inheritance_judged"], 1)
        self.assertAlmostEqual(summary["turn1_false_rate"], 2 / 3)
        self.assertEqual(summary["hcr_si"], 1.0)

    def test_machine_inheritance_suggestion_is_not_counted_as_human(self) -> None:
        rows = [{
            "model": "m",
            "annotation": {
                "turn1_was_false": True,
                "hcr_has_contagious": True,
                "human_fields": ["turn1_was_false"],
            },
        }]
        summary = summarize_selfinduced(rows)[0]
        self.assertEqual(summary["n_turn1_false"], 1)
        self.assertEqual(summary["n_inheritance_judged"], 0)
        self.assertIsNone(summary["hcr_si"])

    def test_inheritance_slotting_preserves_non_candidates(self) -> None:
        all_rows = [
            {"task_id": "false", "annotation": {"turn1_was_false": True}},
            {"task_id": "true", "annotation": {"turn1_was_false": False}},
        ]
        candidates = [all_rows[0]]
        result = add_annotation_slots(all_rows, candidates, {0}, "inheritance")
        self.assertEqual([r["task_id"] for r in result], ["false", "true"])
        self.assertEqual(result[0]["double_annotate_fields"], ["hcr_has_contagious"])
        self.assertNotIn("double_annotate", result[1])

    def test_blind_packet_is_ui_ready_and_task_bound(self) -> None:
        row = {
            "task_id": "t", "pair_id": "p", "q_id": "q", "model": "m",
            "question": "任务中的问题", "truth_statement": "任务中的答案",
            "evidence_hint": "任务中的依据", "turn1_response": "第一轮",
            "response": "第二轮",
        }
        packet = blind_row(row, 1, "inheritance")
        self.assertEqual(packet["question"], "任务中的问题")
        self.assertEqual(packet["truth_statement"], "任务中的答案")
        self.assertEqual(packet["response"], "第二轮")
        self.assertEqual(packet["double_annotate_fields"], ["hcr_has_contagious"])
        self.assertTrue(packet["question_sha1"])
        with self.assertRaises(ValueError):
            blind_row({**row, "truth_statement": ""}, 2, "turn1")

    def test_blind_packet_can_assign_both_selfinduced_fields(self) -> None:
        row = {
            "task_id": "t", "pair_id": "p", "q_id": "q", "model": "m",
            "question": "问题", "truth_statement": "答案", "evidence_hint": "依据",
            "turn1_response": "第一轮", "response": "第二轮",
        }
        packet = blind_row(row, 1, "both")
        self.assertEqual(
            packet["double_annotate_fields"],
            ["turn1_was_false", "hcr_has_contagious"],
        )

    def test_joint_packet_requires_inheritance_only_after_false_turn1(self) -> None:
        item = {"double_annotate_fields": ["turn1_was_false", "hcr_has_contagious"]}
        self.assertEqual(
            required_review_fields(item, {"turn1_was_false": True}),
            {"turn1_was_false", "hcr_has_contagious"},
        )
        self.assertEqual(
            required_review_fields(item, {"turn1_was_false": False}),
            {"turn1_was_false"},
        )

    def test_human_label_only_survives_identical_response(self) -> None:
        old = {
            "task_id": "t", "source_response_sha256": "same",
            "annotation": {
                "turn1_was_false": True,
                "human_fields": ["turn1_was_false"],
                "human_confirmed": True,
            },
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "ann.jsonl"
            path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
            same = [{"task_id": "t", "source_response_sha256": "same", "annotation": {}}]
            changed = [{"task_id": "t", "source_response_sha256": "changed", "annotation": {}}]
            kept, n_kept = _preserve_human_labels(same, path)
            dropped, n_dropped = _preserve_human_labels(changed, path)
        self.assertEqual(n_kept, 1)
        self.assertIs(kept[0]["annotation"]["turn1_was_false"], True)
        self.assertEqual(n_dropped, 0)
        self.assertNotIn("turn1_was_false", dropped[0]["annotation"])

    def test_partial_relabel_cannot_drop_existing_rows(self) -> None:
        old = [
            {"task_id": "t1", "source_dataset": "selfinduced", "annotation": {}},
            {"task_id": "t2", "source_dataset": "selfinduced", "annotation": {}},
        ]
        new = [
            {"task_id": "t1", "source_dataset": "selfinduced", "annotation": {}},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp) / "ann.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in old),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit):
                _preserve_human_labels(new, path)

    def test_kappa_bootstrap_ci_is_reproducible(self) -> None:
        a = [1, 1, 0, 0, 1, 0, 1, 0]
        b = [1, 0, 0, 0, 1, 0, 1, 1]
        ci1 = bootstrap_kappa_ci(a, b, 200, 7)
        ci2 = bootstrap_kappa_ci(a, b, 200, 7)
        self.assertEqual(ci1, ci2)
        self.assertIsNotNone(ci1)

    def test_kappa_does_not_count_machine_prefill_as_annotator_one(self) -> None:
        rows = [{
            "task_id": "t",
            "annotation": {"hcr_has_contagious": True, "human_fields": []},
            "annotation_2": {
                "hcr_has_contagious": True,
                "human_fields": ["hcr_has_contagious"],
            },
        }]
        report = compare_annotators(rows, {"t"}, "hcr_has_contagious", 0.6)
        self.assertEqual(report["n_compared"], 0)
        self.assertEqual(report["skipped_incomplete"], 1)

    def test_kappa_cannot_pass_on_tiny_decidable_subset(self) -> None:
        rows = []
        ids = set()
        for i in range(80):
            task_id = f"t{i}"
            ids.add(task_id)
            human = i < 2
            rows.append({
                "task_id": task_id,
                "annotation": {
                    "turn1_was_false": True if human else None,
                    "human_fields": ["turn1_was_false"] if human else [],
                },
                "annotation_2": {
                    "turn1_was_false": True if human else None,
                    "human_fields": ["turn1_was_false"] if human else [],
                },
            })
        report = compare_annotators(
            rows, ids, "turn1_was_false", 0.6, 0.75
        )
        self.assertEqual(report["n_compared"], 2)
        self.assertFalse(report["go_kappa"])

    def test_exact_sample_preserves_model_minimum(self) -> None:
        rows = [
            {
                "task_id": f"{model}-{i}",
                "model": model,
                "annotation": {"human_fields": [], "turn1_was_false": None},
            }
            for model in ("a", "b", "c", "d", "e", "f")
            for i in range(40)
        ]
        selected = select_indices(rows, 80 / len(rows), 12, 2025)
        self.assertEqual(len(selected), 80)
        counts = {
            model: sum(1 for i in selected if rows[i]["model"] == model)
            for model in ("a", "b", "c", "d", "e", "f")
        }
        self.assertTrue(all(count >= 12 for count in counts.values()), counts)

    def test_end_to_end_go_requires_complete_human_inheritance(self) -> None:
        def write_jsonl(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            rel = base.relative_to(ROOT).as_posix()
            paths = {
                "datasets_dir": f"{rel}/datasets",
                "models_dir": f"{rel}/models",
                "raw_dir": f"{rel}/datasets/raw",
                "interim_dir": f"{rel}/datasets/interim",
                "processed_dir": f"{rel}/datasets/processed",
                "runs_dir": f"{rel}/datasets/runs",
                "reports_dir": f"{rel}/reports",
                "seed_bank": f"{rel}/seed.json",
                "selfinduced_bank": f"{rel}/si.json",
            }
            config = {
                "random_seed": 7,
                "paths": paths,
                "datasets": [{"key": "d", "sample_size": 2}],
                "pilot": {
                    "run_name": "smoke",
                    "model_pool": ["m"],
                    "annotation": {},
                    "ccr": {"bootstrap_samples": 40, "min_real_minus_placebo_ccr": 0.0},
                    "go_no_go": {
                        "min_ccr": 0.0, "min_contaminated_hcr": 0.0,
                        "min_selfinduced_hcr": 0.0,
                    },
                },
            }
            config_path = base / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            seed_path = base / "seed.json"
            seed_path.write_text(json.dumps({"seeds": []}), encoding="utf-8")
            (base / "si.json").write_text(json.dumps({"questions": []}), encoding="utf-8")
            processed = base / "datasets/processed"
            write_jsonl(processed / "smoke_pairs_all.jsonl", [
                {"pair_id": "d-0000"}, {"pair_id": "d-0001"},
            ])
            run = base / "datasets/runs/smoke"
            def injected_row(pair: str, hit: bool) -> dict:
                return {
                    "pair_id": pair, "model": "m", "source_dataset": "d",
                    "seed_claim": pair,
                    "annotation": {
                        "hcr_has_contagious": hit,
                        "clean_has_seed_claim": False,
                        "contaminated_false_claims": (
                            [{"depends_on_seed": True, "present_in_clean": False}]
                            if hit else []
                        ),
                    },
                }
            injected_rows = [
                injected_row("d-0000", True), injected_row("d-0001", True),
            ]
            write_jsonl(run / "claim_annotation_injected.jsonl", injected_rows)
            write_jsonl(run / "claim_annotation_placebo.jsonl", [
                injected_row("d-0000", False), injected_row("d-0001", False),
            ])
            write_jsonl(run / "selfinduced_stage2_tasks.jsonl", [{"task_id": "si-1"}])
            si_row = {
                "task_id": "si-1", "model": "m",
                "annotation": {
                    "turn1_was_false": True,
                    "hcr_has_contagious": True,
                    "human_fields": ["turn1_was_false", "hcr_has_contagious"],
                },
            }
            write_jsonl(run / "claim_annotation_selfinduced.jsonl", [si_row])
            reports = base / "reports"
            reports.mkdir(parents=True)
            detector_path = ROOT / "auto_labeling/detector.py"
            detector_settings = {"negation_window": 4, "min_precision": 0.9}
            detector_settings_sha = object_sha256(detector_settings)
            gold_path = base / "construct_gold.jsonl"
            gold_path.write_text("{}\n", encoding="utf-8")
            (reports / "detector_validation.json").write_text(json.dumps({
                "detector_trustworthy": True,
                "f1": 1.0,
                "provenance": {
                    "detector_sha256": file_sha256(detector_path),
                    "seed_bank_sha256": file_sha256(seed_path),
                    "construct_gold_path": gold_path.relative_to(ROOT).as_posix(),
                    "construct_gold_sha256": file_sha256(gold_path),
                    "detector_settings_sha256": detector_settings_sha,
                },
            }), encoding="utf-8")

            complete_report = base / "complete.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--output", str(complete_report),
            ]):
                compute_metrics_main()
            complete = json.loads(complete_report.read_text(encoding="utf-8"))
            self.assertTrue(complete["decision"]["coverage"]["complete"])
            self.assertTrue(complete["decision"]["go"])

            # A separately configured self-induced run must still satisfy the
            # authoritative combined coverage gate.
            external_config = json.loads(json.dumps(config))
            external_config["pilot"]["run_name"] = "self"
            external_config["pilot"]["annotation"] = {"primary_question_ids": ["q1"]}
            external_config["pilot"]["selfinduced"] = {"questions_per_model": 1}
            external_bank = base / "external_si.json"
            external_bank.write_text(json.dumps({"questions": [{"q_id": "q1"}]}), encoding="utf-8")
            external_config["paths"]["selfinduced_bank"] = external_bank.relative_to(ROOT).as_posix()
            external_config_path = base / "external_config.json"
            external_config_path.write_text(json.dumps(external_config), encoding="utf-8")
            external_run = base / "datasets/runs/self"
            external_row = dict(si_row, q_id="q1")
            write_jsonl(external_run / "selfinduced_stage2_tasks.jsonl", [{"task_id": "si-1", "q_id": "q1", "model": "m"}])
            write_jsonl(external_run / "claim_annotation_selfinduced.jsonl", [external_row])
            external_report = base / "external.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--injected", str(run / "claim_annotation_injected.jsonl"),
                "--placebo", str(run / "claim_annotation_placebo.jsonl"),
                "--selfinduced", str(external_run / "claim_annotation_selfinduced.jsonl"),
                "--selfinduced-config", str(external_config_path),
                "--output", str(external_report),
            ]):
                compute_metrics_main()
            external = json.loads(external_report.read_text(encoding="utf-8"))
            self.assertTrue(external["decision"]["coverage"]["complete"])
            self.assertTrue(external["decision"]["go"])

            injected_rows[0]["source_mock"] = True
            write_jsonl(run / "claim_annotation_injected.jsonl", injected_rows)
            mock_report = base / "mock.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--output", str(mock_report),
            ]):
                compute_metrics_main()
            mock_result = json.loads(mock_report.read_text(encoding="utf-8"))
            self.assertEqual(mock_result["decision"]["coverage"]["source_mock_rows"], 1)
            self.assertFalse(mock_result["decision"]["coverage"]["complete"])
            self.assertIsNone(mock_result["decision"]["go"])
            injected_rows[0]["source_mock"] = False
            write_jsonl(run / "claim_annotation_injected.jsonl", injected_rows)

            # A configured kappa threshold requires the bound agreement report.
            config["pilot"]["go_no_go"]["min_kappa"] = 0.6
            config_path.write_text(json.dumps(config), encoding="utf-8")
            no_kappa_report = base / "no_kappa.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--output", str(no_kappa_report),
            ]):
                compute_metrics_main()
            no_kappa = json.loads(no_kappa_report.read_text(encoding="utf-8"))
            self.assertIsNone(no_kappa["decision"]["go_kappa"])
            self.assertIsNone(no_kappa["decision"]["go"])
            del config["pilot"]["go_no_go"]["min_kappa"]
            config_path.write_text(json.dumps(config), encoding="utf-8")

            si_row["annotation"]["human_fields"] = ["turn1_was_false"]
            write_jsonl(run / "claim_annotation_selfinduced.jsonl", [si_row])
            incomplete_report = base / "incomplete.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--output", str(incomplete_report),
            ]):
                compute_metrics_main()
            incomplete = json.loads(incomplete_report.read_text(encoding="utf-8"))
            self.assertFalse(incomplete["decision"]["coverage"]["selfinduced_human_complete"])
            self.assertIsNone(incomplete["decision"]["go"])

            # A duplicate cannot be hidden by set equality and allowed to
            # reweight the headline metric.
            si_row["annotation"]["human_fields"] = [
                "turn1_was_false", "hcr_has_contagious",
            ]
            write_jsonl(run / "claim_annotation_selfinduced.jsonl", [si_row])
            write_jsonl(
                run / "claim_annotation_injected.jsonl",
                injected_rows + [injected_rows[-1]],
            )
            duplicate_report = base / "duplicate.json"
            with patch("sys.argv", [
                "compute_ccr_metrics.py", "--config", str(config_path),
                "--output", str(duplicate_report),
            ]):
                compute_metrics_main()
            duplicate = json.loads(duplicate_report.read_text(encoding="utf-8"))
            self.assertEqual(
                duplicate["decision"]["coverage"]["duplicate_injected_pair_model_rows"], 1,
            )
            self.assertFalse(duplicate["decision"]["coverage"]["injected_complete"])
            self.assertIsNone(duplicate["decision"]["go"])

            # Likewise, merely having every pair/model key is insufficient if
            # an annotation row is still blank.
            injected_rows[-1]["annotation"]["hcr_has_contagious"] = None
            write_jsonl(run / "claim_annotation_injected.jsonl", injected_rows)
            blank_report = base / "blank.json"
            with self.assertWarnsRegex(UserWarning, "skipped 1 unannotated"):
                with patch("sys.argv", [
                    "compute_ccr_metrics.py", "--config", str(config_path),
                    "--output", str(blank_report),
                ]):
                    compute_metrics_main()
            blank = json.loads(blank_report.read_text(encoding="utf-8"))
            self.assertFalse(blank["decision"]["coverage"]["injected_complete"])
            self.assertIsNone(blank["decision"]["go"])


class FrozenTaskTests(unittest.TestCase):
    def test_clean_and_contaminated_prompts_differ_only_by_injection(self) -> None:
        path = ROOT / "datasets_zh/processed/zh_study_pairs_all.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 800)
        for row in rows:
            if (ROOT / "EXPORT_MANIFEST.json").exists() and row["source_dataset"] == "lccc":
                self.assertNotIn("clean_prompt", row)
                self.assertNotIn("contaminated_prompt", row)
                continue
            suffix = f"\n用户: {row['probe']}"
            self.assertTrue(row["clean_prompt"].endswith(suffix))
            expected = (
                row["clean_prompt"][:-len(suffix)]
                + f"\n助手: {row['seed_claim']}"
                + suffix
            )
            self.assertEqual(row["contaminated_prompt"], expected)

    def test_stage1_tasks_bind_truth_statement(self) -> None:
        path = ROOT / "datasets_zh/runs/zh_selfinduced/selfinduced_stage1_tasks.jsonl"
        if not path.exists():
            self.skipTest("audit task export not present")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), 300)
        self.assertTrue(all(row.get("truth_statement") for row in rows))

    def test_task_fingerprint_changes_with_prompt(self) -> None:
        task = {
            "task_id": "same-id", "pair_id": "p", "condition": "clean",
            "model": "m", "prompt": "旧提示",
        }
        edited = {**task, "prompt": "新提示"}
        self.assertNotEqual(object_sha256(task), object_sha256(edited))

    def test_task_binding_rejects_tampered_truth(self) -> None:
        task = {
            "task_id": "t", "pair_id": "p", "condition": "self_induced",
            "model": "m", "prompt": "prompt", "truth_statement": "事实",
        }
        output = {
            **{k: v for k, v in task.items() if k != "prompt"},
            "response": "回答",
            "task_sha256": object_sha256(task),
        }
        require_task_binding(output, task)
        output["truth_statement"] = "篡改"
        with self.assertRaises(ValueError):
            require_task_binding(output, task)

    def test_directory_hash_binds_paths_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            (root / "a").write_text("one", encoding="utf-8")
            first = directory_sha256(root)
            (root / "a").write_text("two", encoding="utf-8")
            second = directory_sha256(root)
            (root / "a").write_text("one", encoding="utf-8")
            (root / "b").write_text("", encoding="utf-8")
            third = directory_sha256(root)
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)

    def test_all_annotation_uis_offer_reviewed_filter(self) -> None:
        for relative in (
            "scripts/annotate_ui.py",
            "scripts/injected_recall_audit_ui.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("已标注 / Reviewed", text, relative)

    def test_question_bank_draft_is_unreviewed(self) -> None:
        rows = [{"q_id": "q1", "verification_status": "draft"}]
        view = view_rows("selfinduced_bank", {"kind": "qbank", "label": "", "path": Path("bank.json")}, {}, rows)
        self.assertEqual(view["reviewed"], 0)
        self.assertFalse(view["rows"][0]["done"])

    def test_real_output_audit_estimates_precision_and_misses(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            annotations = base / "annotations.jsonl"
            audit = base / "audit.jsonl"
            report_path = base / "report.json"
            annotations.write_text(
                "".join(
                    json.dumps({
                        "pair_id": f"p{i}",
                        "model": "m",
                        "annotation": {"hcr_has_contagious": label},
                    }) + "\n"
                    for i, label in enumerate((True, True, False, False))
                ),
                encoding="utf-8",
            )
            audit.write_text(
                "".join(
                    json.dumps({
                        "pair_id": f"p{i}",
                        "model": "m",
                        "seed_claim": f"s{i}",
                        "human_asserts_seed_falsehood": human,
                        "human_uncertain": False,
                    }) + "\n"
                    for i, (_detector, human) in enumerate((
                        (True, True), (True, False),
                        (False, True), (False, False),
                    ))
                ),
                encoding="utf-8",
            )
            with patch("sys.argv", [
                "compute_injected_recall_audit.py",
                "--annotations", str(annotations),
                "--audit", str(audit),
                "--output", str(report_path),
            ]):
                compute_recall_audit_main()
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["estimated_real_output_precision"], 0.5)
        self.assertEqual(report["audit_missed_rate_among_auto_negatives"], 0.5)
        self.assertEqual(report["estimated_real_output_recall"], 0.5)
        self.assertEqual(report["per_model_audit"]["m"]["false_positive_count"], 1)
        self.assertEqual(report["model_stratified_reweighting"]["estimated_recall"], 0.5)
        self.assertIsNotNone(report["estimated_real_output_recall_ci95_seed_clustered"])
        self.assertTrue(report["audit_complete"])
        self.assertFalse(report["detector_real_output_trustworthy"])

        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            annotations = base / "annotations.jsonl"
            audit = base / "audit.jsonl"
            manifest = base / "sampling.json"
            report_path = base / "report.json"
            annotations.write_text("".join(json.dumps({
                "pair_id": f"p{i}", "model": "m",
                "annotation": {"hcr_has_contagious": label},
            }) + "\n" for i, label in enumerate((True, True, False, False))), encoding="utf-8")
            audit.write_text("".join(json.dumps({
                "audit_id": f"a{i}", "pair_id": f"p{i}", "model": "m",
                "seed_claim": f"s{i}", "human_asserts_seed_falsehood": human,
                "human_uncertain": False,
            }) + "\n" for i, human in enumerate((True, False, True, False))), encoding="utf-8")
            manifest.write_text(json.dumps({
                "audit_sha256": file_sha256(audit),
                "population_positive": 2, "population_negative": 2,
                "sample_positive": 2, "sample_negative": 2,
                "post_revision_positive_to_negative_audit_ids": ["a2"],
                "post_revision_negative_to_positive_audit_ids": ["a1"],
                "population_by_model": {"m": {"positive": 2, "negative": 2}},
            }), encoding="utf-8")
            with patch("sys.argv", [
                "compute_injected_recall_audit.py", "--annotations", str(annotations),
                "--audit", str(audit), "--sampling-manifest", str(manifest),
                "--output", str(report_path),
            ]):
                compute_recall_audit_main()
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["estimated_real_output_precision"], 1.0)
        self.assertEqual(report["post_revision_packet_precision"], 0.5)
        self.assertEqual(report["audit_sampling_positive_filled"], 2)
        self.assertEqual(report["audit_post_revision_positive_filled"], 2)

    def test_recall_audit_uses_model_by_label_design_weights(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            annotations = base / "annotations.jsonl"
            audit = base / "audit.jsonl"
            manifest = base / "sampling.json"
            report_path = base / "report.json"
            rows = [
                ("a_pos", "a", True, True), ("a_neg", "a", False, True),
                ("b_pos", "b", True, False), ("b_neg", "b", False, False),
            ]
            annotations.write_text("".join(json.dumps({
                "pair_id": pair, "model": model,
                "annotation": {"hcr_has_contagious": label},
            }) + "\n" for pair, model, label, _human in rows), encoding="utf-8")
            audit.write_text("".join(json.dumps({
                "audit_id": pair, "pair_id": pair, "model": model,
                "seed_claim": pair, "human_asserts_seed_falsehood": human,
                "human_uncertain": False,
            }) + "\n" for pair, model, _label, human in rows), encoding="utf-8")
            manifest.write_text(json.dumps({
                "audit_sha256": file_sha256(audit),
                "population_positive": 100, "population_negative": 100,
                "sample_positive": 2, "sample_negative": 2,
                "post_revision_positive_to_negative_audit_ids": [],
                "post_revision_negative_to_positive_audit_ids": [],
                "population_by_model": {
                    "a": {"positive": 90, "negative": 10},
                    "b": {"positive": 10, "negative": 90},
                },
            }), encoding="utf-8")
            with patch("sys.argv", [
                "compute_injected_recall_audit.py", "--annotations", str(annotations),
                "--audit", str(audit), "--sampling-manifest", str(manifest),
                "--output", str(report_path),
            ]):
                compute_recall_audit_main()
            report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["estimated_real_output_precision"], 0.9)
        self.assertEqual(report["estimated_real_output_recall"], 0.9)
        self.assertEqual(report["unstratified_sensitivity"]["estimated_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
