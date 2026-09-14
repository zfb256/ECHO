from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))

from config import load_config, resolve_project_path
from detector import detector_blind


SEED_REQUIRED = {"seed_id", "claim", "corrected_claim", "seed_type", "entity", "evidence_hint"}
SEED_TYPES = {
    "entity_substitution",
    "temporal_shift",
    "relation_swap",
    "numeric_perturbation",
    "unsupported_fabricated_detail",
}
# truth_statement stores the reference fact; evidence_hint stores supporting notes.
SI_REQUIRED = {"q_id", "question", "followup", "hallucination_prone_reason", "evidence_hint", "truth_statement"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the injected seed bank AND the self-induced question bank.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument(
        "--require-verified",
        action="store_true",
    )
    return parser.parse_args()


def unverified_ids(items: list[dict], id_key: str) -> list[str]:
    """q_ids/seed_ids of items still flagged needs_human_verification (missing == verified)."""
    return [it.get(id_key, "?") for it in items if it.get("needs_human_verification")]


def _nonempty(v) -> bool:
    return isinstance(v, str) and v.strip() != ""


def validate_seed_bank(path: Path, seeds: list[dict]) -> list[str]:
    errors: list[str] = []
    if not seeds:
        return [f"{path.name}: no seeds"]
    ids: set[str] = set()
    for i, s in enumerate(seeds):
        miss = SEED_REQUIRED - set(s)
        if miss:
            errors.append(f"{path.name}[{i}] missing fields: {sorted(miss)}")
            continue
        sid = s["seed_id"]
        if sid in ids:
            errors.append(f"{path.name}: duplicate seed_id {sid}")
        ids.add(sid)
        if not _nonempty(s.get("claim")):
            errors.append(f"{path.name}:{sid} empty claim")
        if not _nonempty(s.get("corrected_claim")):
            errors.append(f"{path.name}:{sid} empty corrected_claim")
        # Identical claims cannot define a false-statement intervention.
        if _nonempty(s.get("claim")) and s.get("claim", "").strip() == s.get("corrected_claim", "").strip():
            errors.append(f"{path.name}:{sid} claim == corrected_claim (seed is not false)")
        # The span detector requires tokens that distinguish the false claim.
        if _nonempty(s.get("claim")) and _nonempty(s.get("corrected_claim")) and detector_blind(s["claim"], s["corrected_claim"]):
            errors.append(
                f"{path.name}:{sid} detector-blind: corrected_claim restates the false value, so the "
                f"rule detector can never flag it. Rewrite corrected_claim to NOT repeat the false token(s); "
                f"move the false-vs-true nuance into evidence_hint."
            )
        if s.get("seed_type") not in SEED_TYPES:
            errors.append(f"{path.name}:{sid} invalid seed_type {s.get('seed_type')!r} (allowed: {sorted(SEED_TYPES)})")
    return errors


def validate_selfinduced_bank(path: Path, qs: list[dict]) -> list[str]:
    errors: list[str] = []
    if not qs:
        return [f"{path.name}: no questions"]
    ids: set[str] = set()
    for i, q in enumerate(qs):
        miss = SI_REQUIRED - set(q)
        if miss:
            errors.append(f"{path.name}[{i}] missing fields: {sorted(miss)}")
            continue
        qid = q["q_id"]
        if qid in ids:
            errors.append(f"{path.name}: duplicate q_id {qid}")
        ids.add(qid)
        if not _nonempty(q.get("question")):
            errors.append(f"{path.name}:{qid} empty question")
        if not _nonempty(q.get("followup")):
            errors.append(f"{path.name}:{qid} empty followup")
        if not _nonempty(q.get("truth_statement")):
            errors.append(f"{path.name}:{qid} empty truth_statement (need a clean verifiable proposition for the NLI turn-1 check)")
    return errors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_path = resolve_project_path(config["paths"]["seed_bank"])
    si_path = resolve_project_path(config["paths"]["selfinduced_bank"])

    seeds = json.loads(seed_path.read_text(encoding="utf-8")).get("seeds", [])
    questions = json.loads(si_path.read_text(encoding="utf-8")).get("questions", [])
    errors = (
        validate_seed_bank(seed_path, seeds)
        + validate_selfinduced_bank(si_path, questions)
    )
    if errors:
        print("Bank validation failed:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    unv_seeds = unverified_ids(seeds, "seed_id")
    unv_q = unverified_ids(questions, "q_id")
    if args.require_verified and (unv_seeds or unv_q):
        print("Bank verification gate FAILED (--require-verified): items still need a human pass.")
        if unv_seeds:
            print(f"  seed_bank ({len(unv_seeds)}): {unv_seeds}")
        if unv_q:
            print(f"  selfinduced_bank ({len(unv_q)}): {unv_q}")
        print("  Confirm each in scripts/annotate_ui.py, set needs_human_verification=false, then re-run.")
        raise SystemExit(1)

    print(json.dumps({
        "seed_bank": str(seed_path), "seeds": len(seeds), "seeds_unverified": len(unv_seeds),
        "selfinduced_bank": str(si_path), "questions": len(questions), "questions_unverified": len(unv_q),
        "verified_gate": "passed" if args.require_verified else "not_checked", "ok": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
