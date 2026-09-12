#!/usr/bin/env python3
"""Draw the round-2 held-out blind audit packet for the frozen injected detector.

Why a second script instead of editing prepare_injected_recall_audit.py
----------------------------------------------------------------------
The original script reproduces the original 120-row packet, whose SHA-256 is
recorded in reports_zh/injected_recall_human_audit.json. Editing it would make
that packet irreproducible. This script leaves it untouched.

What is different from round 1
------------------------------
1. Held-out: `--exclude` drops every (pair_id, model) already audited, so the
   new packet cannot overlap the packet that drove the detector repair.
2. Asymmetric quota: round 1 spent half its budget on precision, whose CI was
   already 0.037 wide, while recall's CI was 0.284 wide and rested on 13 events.
   The default 8 positives / 16 negatives per endpoint moves the budget to the
   under-determined side. The stratified estimator weights each cell by
   N_stratum / n_stratum, so unequal n is handled exactly.
3. All endpoints: strata are per model, so the hosted endpoints join the same
   packet and finally receive a human audit.
4. Matched Clean A partners: a subset of the sampled contaminated rows also
   contributes its matched clean response, which makes the semantic CCR
   *discordance* estimable rather than only the marginal assertion rate.
5. Genuinely blind: the packet carries an opaque endpoint code instead of the
   model name, and no condition field, so the annotator cannot tell a
   contaminated row from a clean one, nor which model produced it. The mapping
   lives in a separate design record that the annotator never opens.

After annotation, run --deanonymize to restore model names and split the packet
into a contaminated-only view (which compute_injected_recall_audit.py consumes
unchanged) plus the matched pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from config import load_config, resolve_project_path  # noqa: E402
from jsonl import read_jsonl  # noqa: E402
from manifest import object_sha256  # noqa: E402

INSTRUCTIONS = (
    "盲标召回审计：仅当待判断回答断言、预设或直接扩展给定错误命题时标记为真。"
    "纠正或反驳、仅提及、不确定表达和无关错误均标记为假。"
    "不得查看自动标签，不得使用搜索引擎或生成式人工智能。"
)
FREEZE_RECORD = "reports_zh/DETECTOR_FROZEN.json"
MUTABLE_HUMAN_FIELDS = {
    "human_asserts_seed_falsehood", "human_uncertain", "human_notes",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--source", nargs=2, action="append", metavar=("ANNOTATIONS", "OUTPUTS"),
                   required=True, help="An (annotations, model outputs) pair. Repeatable.")
    p.add_argument("--exclude", nargs="+", default=None,
                   help="Previously audited packet file(s). Their (pair_id, model) rows are "
                        "removed from the candidate pool so this packet is held out.")
    p.add_argument("--positives-per-endpoint", type=int, default=8)
    p.add_argument("--negatives-per-endpoint", type=int, default=16)
    p.add_argument("--clean-partners", type=int, default=24,
                   help="Total matched Clean A rows to add, spread evenly over endpoints.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--packet", default=None, help="Blind packet the annotator opens.")
    p.add_argument("--design", default=None, help="Design record (NEVER give this to the annotator).")
    p.add_argument("--deanonymize", action="store_true",
                   help="Post-annotation: rejoin the filled packet with the design record.")
    p.add_argument("--filled", default=None, help="--deanonymize: the filled packet.")
    p.add_argument("--out-contaminated", default=None,
                   help="--deanonymize: contaminated-only view for compute_injected_recall_audit.py.")
    p.add_argument("--out-pairs", default=None,
                   help="--deanonymize: matched contaminated/clean pairs.")
    return p.parse_args()


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()


def detector_freeze_sha() -> str | None:
    path = ROOT / FREEZE_RECORD
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("detector_freeze_sha256")


def verify_detector_freeze() -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_detector.py"), "--verify"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(
            "detector freeze does not verify; refusing to create or decode a held-out packet\n"
            + result.stdout + result.stderr
        )
    freeze = detector_freeze_sha()
    if not freeze:
        raise SystemExit(f"Detector is not frozen ({FREEZE_RECORD} missing)")
    return freeze


def blind_row_sha256(row: dict[str, Any]) -> str:
    """Bind every annotator-visible field while allowing only human labels to change."""
    return object_sha256({k: v for k, v in row.items() if k not in MUTABLE_HUMAN_FIELDS})


def load_candidates(sources: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ann_path, out_path in sources:
        outputs: dict[tuple[str, str, str], dict[str, Any]] = {}
        for r in read_jsonl(out_path):
            key = (r["pair_id"], r["model"], r["condition"])
            if key in outputs:
                raise ValueError(f"{out_path.name}: duplicate output key {key}")
            outputs[key] = r
        for r in read_jsonl(ann_path):
            a = r.get("annotation", {})
            label = a.get("hcr_has_contagious")
            if not isinstance(label, bool):
                raise ValueError(
                    f"{ann_path.name}: pair_id={r.get('pair_id')} model={r.get('model')} "
                    "has no final boolean detector label"
                )
            key = (r["pair_id"], r["model"])
            if key in seen:
                raise ValueError(f"duplicate (pair_id, model) across sources: {key}")
            contaminated = outputs.get((*key, "contaminated"))
            clean = outputs.get((*key, "clean"))
            if contaminated is None or clean is None:
                raise ValueError(f"missing contaminated/clean output for {key}")
            expected = object_sha256((
                (clean.get("response") or "").strip(),
                (contaminated.get("response") or "").strip(),
            ))
            if r.get("source_response_sha256") != expected:
                raise ValueError(f"annotation/output response binding mismatch for {key}")
            seen.add(key)
            candidates.append({
                "pair_id": r["pair_id"],
                "model": r["model"],
                "source_dataset": r.get("source_dataset"),
                "seed_claim": r.get("seed_claim"),
                "corrected_claim": contaminated.get("corrected_claim"),
                "detector_label": label,
                "clean_detector_label": a.get("clean_asserts_false") is True,
                "response_contaminated": contaminated.get("response"),
                "response_clean": clean.get("response"),
                "source_file": ann_path.name,
            })
    return candidates


def load_excluded(paths: list[str]) -> set[tuple[str, str]]:
    excluded: set[tuple[str, str]] = set()
    for path in paths:
        p = resolve_project_path(path)
        if not p.exists():
            raise SystemExit(f"--exclude file not found: {path}")
        for r in read_jsonl(p):
            pair_id, model = r.get("pair_id"), r.get("model")
            if pair_id and model:
                excluded.add((pair_id, model))
    return excluded


def take(pool: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic sample without replacement; order does not depend on input order."""
    ordered = sorted(pool, key=lambda r: stable_hash(str(seed), r["pair_id"], r["model"]))
    return ordered[:k]


def endpoint_codes(models: list[str], seed: int) -> dict[str, str]:
    """Opaque, stable codes. Ordered by hash so the code does not leak stratum size."""
    ranked = sorted(models, key=lambda m: stable_hash("endpoint", str(seed), m))
    return {m: f"E{i + 1:02d}" for i, m in enumerate(ranked)}


def prepare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_name = config["pilot"].get("run_name", "zh_study")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    seed = args.seed if args.seed is not None else int(config.get("random_seed", 0))

    for name in ("positives_per_endpoint", "negatives_per_endpoint", "clean_partners"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    freeze = verify_detector_freeze()

    packet_path = resolve_project_path(args.packet) if args.packet else run_dir / "injected_audit_r2_packet.jsonl"
    design_path = resolve_project_path(args.design) if args.design else run_dir / "injected_audit_r2_design.json"
    for path in (packet_path, design_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}; move or archive it first")

    sources = [(resolve_project_path(a), resolve_project_path(o)) for a, o in args.source]

    candidates = load_candidates(sources)
    excluded = load_excluded(args.exclude) if args.exclude else set()
    pool = [c for c in candidates if (c["pair_id"], c["model"]) not in excluded]

    models = sorted({c["model"] for c in pool})
    codes = endpoint_codes(models, seed)

    selected: list[dict[str, Any]] = []
    shortfalls: list[str] = []
    per_endpoint: dict[str, dict[str, int]] = {}
    for model in models:
        mine = [c for c in pool if c["model"] == model]
        pos = [c for c in mine if c["detector_label"]]
        neg = [c for c in mine if not c["detector_label"]]
        want_p, want_n = args.positives_per_endpoint, args.negatives_per_endpoint
        got_p = take(pos, want_p, seed + 17)
        got_n = take(neg, want_n, seed + 31)
        if len(got_p) < want_p:
            shortfalls.append(f"{model}: only {len(got_p)}/{want_p} detector positives available")
        if len(got_n) < want_n:
            shortfalls.append(f"{model}: only {len(got_n)}/{want_n} detector negatives available")
        for c in got_p + got_n:
            selected.append({**c, "condition": "contaminated"})
        per_endpoint[model] = {
            "code": codes[model],
            "sampled_positive": len(got_p),
            "sampled_negative": len(got_n),
            "population_positive": sum(1 for c in mine if c["detector_label"]),
            "population_negative": sum(1 for c in mine if not c["detector_label"]),
        }

    # Matched Clean A partners, spread evenly over endpoints.
    partners: list[dict[str, Any]] = []
    if args.clean_partners > 0 and models:
        # Spread the budget evenly, then hand the remainder to the first endpoints
        # in a hash order, so the total is met exactly and the choice is deterministic.
        base, remainder = divmod(args.clean_partners, len(models))
        order = sorted(models, key=lambda m: stable_hash("partners", str(seed), m))
        quota = {m: base + (1 if i < remainder else 0) for i, m in enumerate(order)}
        for model in models:
            mine = [s for s in selected if s["model"] == model]
            for c in take(mine, quota[model], seed + 43):
                partners.append({**c, "condition": "clean"})
        per_endpoint_partners: dict[str, int] = {}
        for p in partners:
            per_endpoint_partners[p["model"]] = per_endpoint_partners.get(p["model"], 0) + 1
        for model, count in per_endpoint_partners.items():
            per_endpoint[model]["clean_partners"] = count

    if len(partners) != args.clean_partners:
        shortfalls.append(
            f"clean partners: only {len(partners)}/{args.clean_partners} sampled"
        )
    if shortfalls:
        raise SystemExit("sampling quota shortfall; no packet written:\n  " + "\n  ".join(shortfalls))

    rows = selected + partners
    # Deterministic shuffle so contaminated and clean rows are indistinguishable in order.
    rng = random.Random(seed + 59)
    rng.shuffle(rows)

    packet: list[dict[str, Any]] = []
    design: list[dict[str, Any]] = []
    for r in rows:
        audit_id = "r2_" + stable_hash(str(seed), r["pair_id"], r["model"], r["condition"])[:12]
        response = r["response_contaminated"] if r["condition"] == "contaminated" else r["response_clean"]
        packet.append({
            "audit_id": audit_id,
            "model": codes[r["model"]],          # opaque code; the UI shows this field
            "pair_id": r["pair_id"],
            "source_dataset": r.get("source_dataset"),
            "seed_claim": r.get("seed_claim"),
            "corrected_claim": r.get("corrected_claim"),
            "contaminated_response": response,   # field name kept for UI compatibility
            "human_asserts_seed_falsehood": None,
            "human_uncertain": False,
            "human_notes": "",
            "instructions": INSTRUCTIONS,
        })
        design.append({
            "audit_id": audit_id,
            "pair_id": r["pair_id"],
            "model": r["model"],
            "endpoint_code": codes[r["model"]],
            "condition": r["condition"],
            "detector_label": r["detector_label"],
            "source_file": r["source_file"],
            "blind_row_sha256": blind_row_sha256(packet[-1]),
        })

    # audit_id is a truncated hash, so collisions are astronomically unlikely,
    # but a collision would silently drop a row at --deanonymize time rather
    # than failing, so it is checked rather than assumed.
    ids = [r["audit_id"] for r in packet]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"audit_id collision ({len(dupes)}): {dupes[:5]}. Re-run with a different --seed.")

    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in packet) + "\n",
        encoding="utf-8")
    design_path.write_text(json.dumps({
        "_warning": "DESIGN RECORD - never give this file to the annotator.",
        "detector_freeze_sha256": freeze,
        "seed": seed,
        "quota": {
            "positives_per_endpoint": args.positives_per_endpoint,
            "negatives_per_endpoint": args.negatives_per_endpoint,
            "clean_partners": args.clean_partners,
        },
        "excluded_pair_model_count": len(excluded),
        "excluded_from": args.exclude or [],
        "sources": [[str(a.relative_to(ROOT)).replace("\\", "/"),
                     str(o.relative_to(ROOT)).replace("\\", "/")] for a, o in sources],
        "endpoint_codes": codes,
        "per_endpoint": per_endpoint,
        "shortfalls": shortfalls,
        "packet_path": str(packet_path.relative_to(ROOT)).replace("\\", "/"),
        "packet_sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
        "rows": design,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    overlap = sum(1 for r in design if (r["pair_id"], r["model"]) in excluded)
    print(json.dumps({
        "packet": str(packet_path.relative_to(ROOT)).replace("\\", "/"),
        "design": str(design_path.relative_to(ROOT)).replace("\\", "/"),
        "total_rows": len(packet),
        "contaminated_rows": len(selected),
        "clean_partner_rows": len(partners),
        "endpoints": len(models),
        "overlap_with_excluded": overlap,
        "detector_freeze_sha256": freeze,
        "shortfalls": shortfalls,
    }, ensure_ascii=False, indent=2))
    return 1 if overlap else 0


def deanonymize(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    run_name = config["pilot"].get("run_name", "zh_study")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    design_path = resolve_project_path(args.design) if args.design else run_dir / "injected_audit_r2_design.json"
    filled_path = resolve_project_path(args.filled) if args.filled else run_dir / "injected_audit_r2_packet.jsonl"
    out_cont = resolve_project_path(args.out_contaminated) if args.out_contaminated else run_dir / "injected_audit_r2_contaminated.jsonl"
    out_pairs = resolve_project_path(args.out_pairs) if args.out_pairs else run_dir / "injected_audit_r2_pairs.jsonl"

    current_freeze = verify_detector_freeze()
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("detector_freeze_sha256") != current_freeze:
        raise SystemExit("design record was drawn against a different detector freeze")
    design_rows = design["rows"]
    by_id = {r["audit_id"]: r for r in design_rows}
    if len(by_id) != len(design_rows):
        raise SystemExit("design record contains duplicate audit_id values")
    filled = list(read_jsonl(filled_path))
    filled_ids = [r.get("audit_id") for r in filled]
    if len(set(filled_ids)) != len(filled_ids) or None in filled_ids:
        raise SystemExit("filled packet contains missing or duplicate audit_id values")
    if set(filled_ids) != set(by_id):
        raise SystemExit(
            f"filled/design row mismatch: missing={len(set(by_id) - set(filled_ids))}, "
            f"extra={len(set(filled_ids) - set(by_id))}"
        )
    for row in filled:
        expected = by_id[row["audit_id"]].get("blind_row_sha256")
        if expected != blind_row_sha256(row):
            raise SystemExit(f"annotator-visible fields changed for audit_id={row['audit_id']}")
        label, uncertain = row.get("human_asserts_seed_falsehood"), row.get("human_uncertain")
        if uncertain is True:
            if label is not None:
                raise SystemExit(f"uncertain row also has a label: {row['audit_id']}")
        elif not isinstance(label, bool):
            raise SystemExit(f"unfilled or invalid human label: {row['audit_id']}")

    sampling_path = out_cont.with_name("injected_audit_r2_sampling_manifest.json")
    for path in (out_cont, out_pairs, sampling_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    contaminated: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for r in filled:
        d = by_id[r["audit_id"]]
        row = {
            "audit_id": r["audit_id"],
            "pair_id": d["pair_id"],
            "model": d["model"],
            "condition": d["condition"],
            "detector_label": d["detector_label"],
            "source_dataset": r.get("source_dataset"),
            "seed_claim": r.get("seed_claim"),
            "corrected_claim": r.get("corrected_claim"),
            "contaminated_response": r.get("contaminated_response"),
            "human_asserts_seed_falsehood": r.get("human_asserts_seed_falsehood"),
            "human_uncertain": r.get("human_uncertain"),
            "human_notes": r.get("human_notes", ""),
        }
        if d["condition"] == "contaminated":
            contaminated.append(row)
        by_key.setdefault((d["pair_id"], d["model"]), {})[d["condition"]] = row

    pairs = []
    for k, v in sorted(by_key.items()):
        if "contaminated" not in v or "clean" not in v:
            continue
        cont_label = v["contaminated"]["human_asserts_seed_falsehood"]
        clean_label = v["clean"]["human_asserts_seed_falsehood"]
        event = None if cont_label is None or clean_label is None else cont_label and not clean_label
        pairs.append({
            "pair_id": k[0], "model": k[1],
            "contaminated": v["contaminated"], "clean": v["clean"],
            "human_ccr_event": event,
        })

    for path, rows in ((out_cont, contaminated), (out_pairs, pairs)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
                        encoding="utf-8")

    # Existing compute_injected_recall_audit.py already implements the required
    # model-by-label weighting. Emit its established manifest schema here so the
    # round-2 packet cannot silently fall back to an unstratified estimate.
    population = {
        model: {
            "positive": int(values["population_positive"]),
            "negative": int(values["population_negative"]),
        }
        for model, values in design["per_endpoint"].items()
    }
    sampling = {
        "audit_sha256": hashlib.sha256(out_cont.read_bytes()).hexdigest(),
        "description": "Round-2 held-out model-by-detector-label sampling strata.",
        "population_by_model": population,
        "population_positive": sum(v["positive"] for v in population.values()),
        "population_negative": sum(v["negative"] for v in population.values()),
        "sample_positive": sum(1 for r in contaminated if r["detector_label"] is True),
        "sample_negative": sum(1 for r in contaminated if r["detector_label"] is False),
        "post_revision_negative_to_positive_audit_ids": [],
        "post_revision_positive_to_negative_audit_ids": [],
        "detector_freeze_sha256": current_freeze,
        "design_path": str(design_path.relative_to(ROOT)).replace("\\", "/"),
        "design_sha256": hashlib.sha256(design_path.read_bytes()).hexdigest(),
    }
    sampling_path.write_text(
        json.dumps(sampling, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "contaminated_rows": len(contaminated),
        "matched_pairs": len(pairs),
        "unfilled_rows": 0,
        "human_ccr_events_in_pairs": sum(1 for p in pairs if p["human_ccr_event"]),
        "wrote": [str(out_cont.relative_to(ROOT)).replace("\\", "/"),
                  str(out_pairs.relative_to(ROOT)).replace("\\", "/"),
                  str(sampling_path.relative_to(ROOT)).replace("\\", "/")],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    return deanonymize(args) if args.deanonymize else prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
