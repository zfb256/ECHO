from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))

from jsonl import read_jsonl, write_jsonl  # noqa: E402
from check_paper_numbers import run_assertions  # noqa: E402
from compute_selfinduced_u4 import require_cohort  # noqa: E402
from merge_ext_annotations import coverage, schema_of  # noqa: E402
from prepare_injected_audit_r2 import blind_row_sha256, deanonymize, prepare  # noqa: E402
from prepare_injected_second_annotator import select_ids  # noqa: E402
from prepare_selfinduced_u4 import immutable_row_sha256, select_u4_rows  # noqa: E402
from run_vllm_inference import model_runtime_profile, rows_in_task_order  # noqa: E402
from run_semantic_judge import (  # noqa: E402
    judge_pair,
    score_rows,
    semantic_sensitivity_interval,
    with_rate_limit_retry,
    wilson_upper,
)


class _Reply:
    def __init__(self, text: str, finish_reason: str = "stop"):
        self.choices = [type("Choice", (), {
            "finish_reason": finish_reason,
            "message": type("Message", (), {"content": text})(),
        })()]


class _Client:
    def __init__(self, text: str, finish_reason: str = "stop"):
        self.chat = type("Chat", (), {
            "completions": type("Completions", (), {
                "create": staticmethod(lambda **_kwargs: _Reply(text, finish_reason)),
            })(),
        })()


class ExtensionWorkflowTests(unittest.TestCase):
    def test_judge_rejects_string_booleans(self) -> None:
        protocol = {"prompt_template": "{seed_claim}{corrected_claim}{response_a}{response_b}",
                    "judge_model": "judge", "temperature": 0, "max_tokens": 8}
        row = {"seed_claim": "f", "corrected_claim": "t",
               "response_contaminated": "a", "response_clean": "b"}
        verdict = judge_pair(_Client('{"a": "false", "b": true}'), protocol, row, False)
        self.assertIsNone(verdict["a"])
        self.assertIsNone(verdict["b"])

    def test_judge_fails_loudly_on_truncation(self) -> None:
        protocol = {"prompt_template": "{seed_claim}{corrected_claim}{response_a}{response_b}",
                    "judge_model": "judge", "temperature": 0, "max_tokens": 64}
        row = {"seed_claim": "f", "corrected_claim": "t",
               "response_contaminated": "a", "response_clean": "b"}
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            judge_pair(_Client("", "length"), protocol, row, False)

    def test_judge_records_content_filter_as_undecidable(self) -> None:
        error = type("Filtered", (Exception,), {"status_code": 400})
        client = _Client("")
        client.chat.completions.create = lambda **_kwargs: (_ for _ in ()).throw(
            error("request considered high risk: content_filter")
        )
        protocol = {"prompt_template": "{seed_claim}{corrected_claim}{response_a}{response_b}",
                    "judge_model": "judge", "temperature": 0, "max_tokens": 64}
        row = {"seed_claim": "f", "corrected_claim": "t",
               "response_contaminated": "a", "response_clean": "b"}
        verdict = judge_pair(client, protocol, row, False)
        self.assertEqual(verdict, {"a": None, "b": None,
                                  "raw": "CONTENT_FILTER", "mock": False})

    def test_judge_retries_only_rate_limits(self) -> None:
        attempts, delays = [], []
        error = type("RateLimited", (Exception,), {"status_code": 429})
        def call():
            attempts.append(1)
            if len(attempts) < 3:
                raise error()
            return "ok"
        self.assertEqual(with_rate_limit_retry(call, delays.append), "ok")
        self.assertEqual(delays, [1, 2])

    def test_judge_checkpoint_is_resumable_and_bound(self) -> None:
        protocol = {"protocol_sha256": "p", "prompt_template": "{seed_claim}{corrected_claim}{response_a}{response_b}"}
        rows = [
            {"pair_id": str(i), "model": "m", "seed_claim": "f", "corrected_claim": "t",
             "response_contaminated": "a", "response_clean": "b"}
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            checkpoint = Path(tmp) / "scores.partial.jsonl"
            first = score_rows(rows, protocol, True, None, checkpoint)
            second = score_rows(rows, protocol, True, None, checkpoint)
            self.assertEqual(first, second)
            self.assertEqual(len(list(read_jsonl(checkpoint))), 2)

            checkpoint.write_bytes(checkpoint.read_bytes() + b"\x00\x00")
            recovered = score_rows(rows, protocol, True, None, checkpoint)
            self.assertEqual(recovered, first)
            self.assertEqual(len(list(read_jsonl(checkpoint))), 2)

    def test_semantic_interval_propagates_error_and_undecidables(self) -> None:
        disagreement = wilson_upper(0, 24)
        self.assertIsNotNone(disagreement)
        low, high = semantic_sensitivity_interval(10, 90, 100, disagreement)
        self.assertAlmostEqual(low, max(0, 0.10 - disagreement))
        self.assertAlmostEqual(high, min(1, 0.20 + disagreement))

    def test_glm_runtime_profile_uses_local_model_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            model = Path(tmp)
            (model / "config.json").write_text('{"model_type":"glm"}', encoding="utf-8")
            (model / "generation_config.json").write_text(
                '{"eos_token_id":[1,2,3]}', encoding="utf-8"
            )
            self.assertEqual(model_runtime_profile(model), {
                "backend": "transformers",
                "tokenizer_mode": "auto",
                "use_fast_tokenizer": True,
                "stop_token_ids": [1, 2, 3],
            })

    def test_per_model_merge_preserves_previous_model(self) -> None:
        tasks = [{"task_id": "a"}, {"task_id": "b"}]
        rows = rows_in_task_order(tasks, {"a": {"task_id": "a"}}, {"b": {"task_id": "b"}})
        self.assertEqual([r["task_id"] for r in rows], ["a", "b"])

    def test_blind_hash_allows_labels_only(self) -> None:
        row = {"audit_id": "a", "contaminated_response": "answer",
               "human_asserts_seed_falsehood": None, "human_uncertain": False, "human_notes": ""}
        edited = {**row, "human_asserts_seed_falsehood": True, "human_notes": "note"}
        self.assertEqual(blind_row_sha256(row), blind_row_sha256(edited))
        self.assertNotEqual(blind_row_sha256(row), blind_row_sha256({**row, "contaminated_response": "changed"}))

    def test_injected_second_annotator_sample_covers_endpoint_strata(self) -> None:
        design = [
            {"audit_id": f"{endpoint}-{label}-{i}", "endpoint_code": endpoint,
             "condition": "contaminated", "detector_label": label}
            for endpoint in ("E01", "E02", "E03")
            for label in (False, True)
            for i in range(4)
        ]
        ids = select_ids(design, 9, 7)
        self.assertEqual(len(ids), 9)
        self.assertEqual({audit_id.split("-")[0] for audit_id in ids}, {"E01", "E02", "E03"})
        self.assertTrue(any("-True-" in audit_id for audit_id in ids))
        self.assertTrue(any("-False-" in audit_id for audit_id in ids))

    def test_model_output_coverage_includes_condition(self) -> None:
        rows = [
            {"model": "m", "pair_id": "p", "condition": condition}
            for condition in ("clean", "contaminated", "clean_b")
        ]
        self.assertEqual(len(coverage(rows, "model_outputs")["m"]), 3)

    def test_selfinduced_schema_allows_second_annotator_fields_only(self) -> None:
        base = {"task_id": "t", "q_id": "q", "model": "m", "annotation": {}}
        self.assertEqual(
            schema_of(base, "selfinduced_annotation"),
            schema_of({**base, "annotation_2": {}, "double_annotate": True},
                      "selfinduced_annotation"),
        )
        self.assertNotEqual(
            schema_of(base, "selfinduced_annotation"),
            schema_of({**base, "unexpected": 1}, "selfinduced_annotation"),
        )

    def test_u4_selects_old_remaining_and_all_new_rows(self) -> None:
        bank = [f"q{i}" for i in range(60)]
        primary = bank[:40]
        old = [
            {"task_id": f"old-{q}", "q_id": q, "model": "old", "annotation": {}}
            for q in bank
        ]
        new = [
            {"task_id": f"new-{q}", "q_id": q, "model": "new", "annotation": {}}
            for q in bank
        ]
        rows, cohorts = select_u4_rows(old, new, bank, primary, ["old"], ["new"])
        self.assertEqual(len(rows), 80)
        self.assertEqual(Counter(row["model"] for row in rows), Counter({"old": 20, "new": 60}))
        self.assertEqual(set(cohorts.values()), {"primary_models_remaining_20", "extension_models_full_60"})
        edited = {**rows[0], "annotation": {
            **rows[0]["annotation"], "turn1_was_false": True,
            "turn1_was_false_uncertain": False, "human_fields": ["turn1_was_false"],
            "human_confirmed": True, "annotator": 1, "needs_human_review": False,
        }}
        self.assertEqual(immutable_row_sha256(rows[0]), immutable_row_sha256(edited))

    def test_u4_metric_cohort_requires_complete_human_labels(self) -> None:
        rows = [{"task_id": "t", "q_id": "q", "model": "m", "annotation": {
            "turn1_was_false": True, "hcr_has_contagious": False,
            "human_fields": ["turn1_was_false", "hcr_has_contagious"],
        }}]
        require_cohort(rows, {"q"}, {"m"}, "test")
        with self.assertRaises(ValueError):
            require_cohort([], {"q"}, {"m"}, "test")

    def test_round2_packet_emits_compatible_sampling_manifest(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            packet, design = base / "packet.jsonl", base / "design.json"
            args = argparse.Namespace(
                config="configs/zh_study.json",
                source=[["datasets_zh/runs/zh_study/claim_annotation_injected.jsonl",
                         "datasets_zh/runs/zh_study/model_outputs.jsonl"]],
                exclude=None, positives_per_endpoint=1, negatives_per_endpoint=1,
                clean_partners=5, seed=7, packet=str(packet), design=str(design),
                deanonymize=False, filled=None, out_contaminated=None, out_pairs=None,
            )
            self.assertEqual(prepare(args), 0)
            filled = list(read_jsonl(packet))
            for row in filled:
                row["human_asserts_seed_falsehood"] = False
            write_jsonl(packet, filled)
            out_cont, out_pairs = base / "contaminated.jsonl", base / "pairs.jsonl"
            args.deanonymize = True
            args.filled = str(packet)
            args.out_contaminated = str(out_cont)
            args.out_pairs = str(out_pairs)
            self.assertEqual(deanonymize(args), 0)
            sampling = json.loads((base / "injected_audit_r2_sampling_manifest.json").read_text())
            self.assertEqual(sampling["sample_positive"], 5)
            self.assertEqual(sampling["sample_negative"], 5)
            self.assertEqual(len(sampling["population_by_model"]), 5)

    def test_assertions_use_half_up_and_check_tex(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            base = Path(tmp)
            (base / "reports").mkdir()
            (base / "paper.tex").write_text("Result: 0.308", encoding="utf-8")
            (base / "reports" / "r.json").write_text('{"value": 0.3075}', encoding="utf-8")
            registry = [{"quantity": "q", "tex": "0.308", "report": "r.json",
                         "path": "/value", "places": 3,
                         "tex_file": str((base / "paper.tex").relative_to(ROOT))}]
            (base / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
            args = argparse.Namespace(
                registry=str((base / "registry.json").relative_to(ROOT)),
                reports=str((base / "reports").relative_to(ROOT)), rounding="half-up",
            )
            self.assertEqual(run_assertions(args), 0)


if __name__ == "__main__":
    unittest.main()
