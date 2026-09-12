from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))
from detector import detect_injected, false_true_spans  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Before/after audit for the detector repair.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--output", default="reports/detector_before_after_audit.json")
    return p.parse_args()


def _tokens(s: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", (s or "").lower().replace(",", ""))


_NEG_CUES = {
    "not", "no", "never", "n't", "isn", "wasn", "aren", "weren", "didn", "doesn", "don",
    "wrong", "incorrect", "false", "untrue", "myth", "mistaken", "actually", "rather",
    "contrary", "misconception", "debunked",
}


def legacy_detect(text: str, seed_claim: str, corrected_claim: str, neg_window: int) -> dict[str, Any]:
    """Approximate the pre-auditfix detector: any distinctive false token can fire.

    This preserves the negation guard but intentionally does not require an ordered
    replacement-span hit. That is the reviewed bug: long false spans could fire on a
    generic/shared-looking fragment. Ambiguity handling is applied in the caller,
    because the old annotation path counted contaminated ambiguous hits as positives.
    """
    false_tokens, true_tokens = false_true_spans(seed_claim, corrected_claim)
    toks = _tokens(text)
    hit_positions = [i for i, tok in enumerate(toks) if tok in set(false_tokens)]
    asserts_false = False
    for i in hit_positions:
        left = max(0, i - neg_window)
        right = min(len(toks), i + 1 + neg_window)
        ctx = toks[left:i] + toks[i + 1:right]
        if not any(w in _NEG_CUES for w in ctx):
            asserts_false = True
            break
    asserts_true = any(tok in set(true_tokens) for tok in toks)
    return {
        "asserts_false": asserts_false,
        "asserts_true": asserts_true,
        "ambiguous": asserts_false and asserts_true,
    }


def prf(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def construct_gold(config: dict, neg_window: int) -> dict[str, Any]:
    seeds = {
        s["seed_id"]: s
        for s in json.loads(resolve_project_path(config["paths"]["seed_bank"]).read_text(encoding="utf-8")).get("seeds", [])
    }
    gold = list(read_jsonl(resolve_project_path("auto_labeling/construct_gold.jsonl")))
    out = {}
    for name, detector in (
        ("legacy_any_token", legacy_detect),
        ("current_ordered_span", lambda text, sc, cc, nw: detect_injected(text, sc, cc, nw)),
    ):
        tp = fp = fn = tn = 0
        for g in gold:
            seed = seeds[g["seed_id"]]
            pred = bool(detector(g["text"], seed["claim"], seed["corrected_claim"], neg_window)["asserts_false"])
            truth = bool(g["label_contains_false"])
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
        out[name] = prf(tp, fp, fn, tn) | {"n_gold": len(gold)}
    return out


def active_metric(path: str) -> dict[str, Any]:
    rows = list(read_jsonl(resolve_project_path(path)))
    h = c = 0
    for r in rows:
        a = r.get("annotation", {})
        if a.get("hcr_has_contagious") is True:
            h += 1
        if any(x.get("depends_on_seed") and x.get("present_in_clean") is False for x in a.get("contaminated_false_claims") or []):
            c += 1
    return {"n": len(rows), "hcr": h / len(rows), "ccr": c / len(rows), "hcr_count": h, "ccr_count": c}


def load_outputs(paths: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for path in paths:
        for r in read_jsonl(resolve_project_path(path)):
            key = (r["pair_id"], r["model"])
            by_key[key][r["condition"]] = r.get("response") or ""
            by_key[key]["source_dataset"] = r.get("source_dataset")
            by_key[key]["seed_claim"] = r.get("seed_claim")
            by_key[key]["corrected_claim"] = r.get("corrected_claim")
    return by_key


def legacy_outputs_metric(output_paths: list[str], neg_window: int) -> dict[str, Any]:
    by_key = load_outputs(output_paths)
    real_h = real_c = real_n = 0
    pl_h = pl_c = pl_n = 0
    by_dataset: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_pair, _model), conds in by_key.items():
        sc, cc = conds["seed_claim"], conds["corrected_claim"]
        ds = conds.get("source_dataset", "unknown")
        if "clean" in conds and "contaminated" in conds:
            cont = legacy_detect(conds["contaminated"], sc, cc, neg_window)
            clean = legacy_detect(conds["clean"], sc, cc, neg_window)
            # Legacy annotation path counted contaminated ambiguous hits as positives.
            cont_false = bool(cont["asserts_false"])
            clean_false = bool(clean["asserts_false"])
            real_h += int(cont_false)
            real_c += int(cont_false and not clean_false)
            real_n += 1
            by_dataset[ds]["real_n"] += 1
            by_dataset[ds]["real_ccr"] += int(cont_false and not clean_false)
        if "clean" in conds and "clean_b" in conds:
            a = legacy_detect(conds["clean"], sc, cc, neg_window)
            b = legacy_detect(conds["clean_b"], sc, cc, neg_window)
            a_false = bool(a["asserts_false"])
            b_false = bool(b["asserts_false"])
            pl_h += int(a_false)
            pl_c += int(a_false and not b_false)
            pl_n += 1
            by_dataset[ds]["pl_n"] += 1
            by_dataset[ds]["pl_ccr"] += int(a_false and not b_false)
    return {
        "injected": {"n": real_n, "hcr": real_h / real_n, "ccr": real_c / real_n, "hcr_count": real_h, "ccr_count": real_c},
        "placebo": {"n": pl_n, "hcr": pl_h / pl_n, "ccr": pl_c / pl_n, "hcr_count": pl_h, "ccr_count": pl_c},
        "by_dataset": {
            ds: {
                "legacy_ccr": v["real_ccr"] / v["real_n"],
                "legacy_placebo_ccr": v["pl_ccr"] / v["pl_n"],
                "n": v["real_n"],
            }
            for ds, v in sorted(by_dataset.items())
        },
    }


def intervention_baseline() -> dict[str, Any]:
    old4 = {"Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct-AWQ", "Llama-3.1-8B-Instruct", "Mistral-7B-Instruct-v0.3"}
    files = sorted(resolve_project_path("reports").glob("echo_r_eval_injected_multi4_*.json"))
    files = [f for f in files if "_suppress_" not in f.name and "forcerw" not in f.name]
    out = {}
    for name, keep in (("current_same_four_models", old4), ("current_all_eight_models", None)):
        c = t = 0
        rows = []
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            if keep is not None and d["model"] not in keep:
                continue
            for r in d["results"]:
                if r["policy"] == "no_op" and r["budget_frac"] == 0.25:
                    c += int(r["contagious_probes"])
                    t += int(r["total_probes"])
                    rows.append({"model": d["model"], "contagious": r["contagious_probes"], "total": r["total_probes"], "rate": r["contagion_rate"]})
        out[name] = {"contagious": c, "total": t, "rate": c / t if t else None, "models": rows}
    return out


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    neg_window = int(config.get("pilot", {}).get("detector", {}).get("negation_window", 4))
    report = {
        "note": (
            "Before/after audit uses the same current inputs. Legacy simulates the pre-auditfix "
            "any-token detector and contaminated-ambiguous-as-positive annotation. It is an audit "
            "of the detector/annotation rule, not a resurrection of deleted stale artifacts."
        ),
        "construct_gold": construct_gold(config, neg_window),
        "headline_same_outputs": {
            "legacy_any_token": legacy_outputs_metric(
                [
                    "datasets/runs/full_study/model_outputs.jsonl",
                    "datasets/runs/full_study/model_outputs_fw_auditfix.jsonl",
                ],
                neg_window,
            ),
            "current_auditfix": {
                "injected": active_metric("datasets/runs/full_study/claim_annotation_injected_all3_auditfix.jsonl"),
                "placebo": active_metric("datasets/runs/full_study/claim_annotation_placebo_all3_auditfix.jsonl"),
            },
        },
        "intervention_no_op_baseline": intervention_baseline(),
    }
    out = resolve_project_path(args.output)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
