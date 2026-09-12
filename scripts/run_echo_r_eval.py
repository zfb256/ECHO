from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "auto_labeling"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "echo_r"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative  # noqa: E402
from jsonl import read_jsonl  # noqa: E402
from detector import configure_nli, detect_injected, false_true_spans  # noqa: E402

from core import Budget, Claim, ProvenanceState, TurnContext  # noqa: E402
from risk import ElaborationRiskScorer, HeuristicRiskScorer  # noqa: E402
from policy import (  # noqa: E402
    EchoRPolicy, NoOpPolicy, OraclePolicy, ReminderPolicy, UniformCheckPolicy,
)
from checker import NLIChecker, OracleChecker  # noqa: E402

# ECHO-R ONLINE multi-turn evaluation. For each confirmed-false seed, roll the dialogue forward N turns
# under each policy at a budget B, applying PASS/DISCOUNT/CHECK between turns and RE-GENERATING the next
# turn from the (edited) context, then measure downstream contagion (turn asserts the seed's false value,
# via the validated detector). Hypothesis under test: ECHO-R's selective allocation beats the uniform
# baselines at matched B. Outcome on real models: it does NOT (see the paper); this script produces that
# negative result and the failure-mode trace. GPU (vLLM) for real; --mock for logic checks only.

DISCOUNT_COST = 0.2
_GENERIC_BAN_TOKENS = {
    "be", "been", "being", "exactly", "human", "larger", "measured", "over", "percent", "than",
}


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def suppression_bad_words(false_tokens: list[str]) -> list[str]:
    """Decode bans should be distinctive false values, not generic fragments.

    The detector can keep broader evidence tokens, but vLLM `bad_words` is a blunt
    instrument: banning "1", "3", or "over" can corrupt unrelated generations.
    """
    out = []
    for tok in false_tokens:
        t = (tok or "").strip().lower()
        if not t or t in _GENERIC_BAN_TOKENS:
            continue
        if t.isdigit() and len(t) < 2:
            continue
        if (not t.isdigit()) and len(t) < 3:
            continue
        out.append(t)
    return sorted(set(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECHO-R online multi-turn intervention evaluation.")
    p.add_argument("--config", default="configs/full_study.json")
    p.add_argument("--model", required=True, help="Model to roll out on; seeds are restricted to this model's own confirmed-false turn-1s.")
    p.add_argument("--selfinduced", default=None, help="Annotated claim_annotation_selfinduced.jsonl (default: run_dir).")
    p.add_argument("--followups", default="echo_r/followups.json")
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--budgets", default="0.25,0.5,1.0", help="Comma list of GLOBAL check budgets as a FRACTION of #seeds (e.g. 0.25 = enough checks for a quarter of the dialogues). The moat shows at tight (<1) budgets.")
    p.add_argument("--risky-ratio", type=float, default=0.5, help="Fraction of turns that are on-topic (risky).")
    p.add_argument("--claims-per-dialogue", type=int, default=1, help="K>1 enables the MULTI-CLAIM rollout: K "
                   "distinct injected claims per dialogue; each on-topic turn probes one of them by entity, "
                   "drawn uniformly at random with replacement. This is the regime "
                   "where ECHO-R's selective allocation can beat uniform (which checks indiscriminately). "
                   "Injected arm only.")
    p.add_argument("--check-cost", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="Cap number of seeds (smoke/subset).")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=8192, help="Cap vLLM context length to fit 40GB GPUs.")
    p.add_argument("--output", default=None, help="Defaults to reports/echo_r_eval_<model>.json")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--mock", action="store_true", help="Fabricate generations (logic check; NOT research data).")
    p.add_argument("--dry-run", action="store_true", help="Validate seed loading / schedules only; no model, no rollout.")
    p.add_argument("--require-nli", action="store_true", help="Deprecated compatibility flag; non-mock runs now require NLI unless --allow-nli-noop is set.")
    p.add_argument("--allow-nli-noop", action="store_true", help="Permit non-mock CHECK to continue when NLI is unavailable. Use only for plumbing tests, never research data.")
    p.add_argument("--suppress", action="store_true", help="法子A: decoding-level cure. When CHECK refutes a claim, BAN its distinctive false tokens in the decoder (vLLM bad_words) for the rest of that dialogue, so the model physically cannot restate the false value. Attacks the re-derivation root that context edits miss.")
    p.add_argument("--force-rewrite", action="store_true", help="法子D: strongest cure. Once CHECK refutes a claim, any later probe targeting THAT claim is answered by serving a grounded/abstain response INSTEAD of the model's free generation (which paraphrases past token-bans). Contribution is the selective ALLOCATION: under a tight global budget, ECHO-R spends checks on about-to-be-probed claims so those probes get served correct answers; uniform wastes checks. Multi-claim injected only.")
    p.add_argument("--rewrite-mode", choices=["grounded", "abstain"], default="grounded", help="法子D rewrite content: grounded = serve the verified truth; abstain = decline to assert. Default grounded.")
    p.add_argument("--trace", action="store_true", help="Diagnostic: dump a per-turn trace (which claim was checked vs probed, risk-score rank of the probed claim, contagion, served/refuted state) to reports/echo_r_trace_<tag>_<model>.jsonl. Feeds the failure-mode diagnostic tables. Additive; does not change any metric. Multi-claim only.")
    p.add_argument("--arm", choices=["selfinduced", "injected"], default="selfinduced",
                   help="selfinduced: C = the model's own confirmed-false turn-1 (model RE-DERIVES it from weights, "
                        "so context interruption is expected to be weak). injected: C = an EXTERNAL false seed the "
                        "model wouldn't generate on its own, so refuting/removing it should let the model revert to "
                        "its true knowledge — this is where ECHO-R's context interruption should actually work.")
    return p.parse_args()


def result_tag(args: argparse.Namespace) -> str:
    multi = args.claims_per_dialogue > 1
    tag = f"{args.arm}_multi{args.claims_per_dialogue}" if multi else args.arm
    if multi and args.suppress:
        tag += "_suppress"
    if multi and args.force_rewrite:
        tag += f"_forcerw-{args.rewrite_mode}"
    return tag


def result_suffix(args: argparse.Namespace) -> str:
    parts = []
    if args.mock:
        parts.append("mock")
    if args.allow_nli_noop:
        parts.append("nli-noop")
    if args.limit is not None:
        parts.append(f"limit{args.limit}")
    return f"_{'_'.join(parts)}" if parts else ""


def resolve_output_path(args: argparse.Namespace) -> Path:
    out_path = (resolve_project_path(args.output) if args.output
                else resolve_project_path("reports") / f"echo_r_eval_{result_tag(args)}_{args.model}{result_suffix(args)}.json")
    out_name = out_path.name.lower()
    if args.mock and args.output and "mock" not in out_name:
        raise SystemExit("--mock output must include 'mock' in the filename; refusing to overwrite research data.")
    if args.allow_nli_noop and args.output:
        safe_name = any(token in out_name for token in ("noop", "diagnostic", "tmp"))
        if not safe_name:
            raise SystemExit(
                "--allow-nli-noop output must include 'noop', 'diagnostic', or 'tmp' in the filename; "
                "refusing to overwrite research data."
            )
    if args.limit is not None and args.output:
        safe_name = any(token in out_name for token in ("limit", "smoke", "tmp", "partial"))
        if not safe_name:
            raise SystemExit(
                "--limit is for smoke/partial ECHO-R runs only. Pass an explicit output filename containing "
                "'limit', 'smoke', 'tmp', or 'partial' so a full research output cannot be overwritten."
            )
    return out_path


def load_seeds_injected(config: dict, model: str) -> list[dict]:
    """Injected-arm seeds straight from the seed bank: C = the false claim (external), truth = corrected.
    No per-model confirmation needed (the falsehood is external by construction). `model` is unused for
    selection (every seed applies to every model); kept for signature symmetry."""
    bank_path = resolve_project_path(config["paths"]["seed_bank"])
    seeds = []
    for s in json.loads(bank_path.read_text(encoding="utf-8")).get("seeds", []):
        C, tr = (s.get("claim") or "").strip(), (s.get("corrected_claim") or "").strip()
        if not C or not tr:
            continue
        false_tokens, _ = false_true_spans(C, tr)
        if not false_tokens:
            continue
        entity = (s.get("entity") or "that topic").strip()
        seeds.append({"q_id": s["seed_id"], "question": f"Briefly, what is the key fact about {entity}?",
                      "C": C, "truth": tr, "entity": entity, "false_tokens": false_tokens,
                      "ban_tokens": suppression_bad_words(false_tokens)})
    return seeds


def load_seeds(config: dict, si_path: Path, model: str) -> list[dict]:
    """Confirmed-false turn-1s for `model`, with the seed claim C and its distinctive false tokens."""
    bank_path = resolve_project_path(config["paths"]["selfinduced_bank"])
    bank = {
        q["q_id"]: {
            "question": q.get("question", ""),
            "truth": q.get("truth_statement", ""),
        }
        for q in json.loads(bank_path.read_text(encoding="utf-8")).get("questions", [])
    }
    seeds = []
    for r in read_jsonl(si_path):
        a = r.get("annotation", {})
        if a.get("turn1_was_false") is not True or r.get("model") != model:
            continue
        C = (r.get("turn1_response") or "").strip()
        meta = bank.get(r.get("q_id"), {})
        tr = meta.get("truth", "")
        question = (r.get("question") or meta.get("question") or "").strip()
        if not C or not tr:
            continue
        false_tokens, _ = false_true_spans(C, tr)
        if not false_tokens:
            continue  # detector-blind on this C -> can't label per-turn contagion; skip
        seeds.append({"q_id": r.get("q_id"), "question": question, "C": C, "truth": tr,
                      "false_tokens": false_tokens, "ban_tokens": suppression_bad_words(false_tokens)})
    return seeds


def make_schedule(n_turns: int, risky_ratio: float, pool: dict, rng: random.Random) -> list[tuple[str, str]]:
    """A per-dialogue interleaving of on-topic (risky) and tangent (safe) follow-ups, randomized.

    The number of risky turns VARIES per dialogue (0..n_turns): some dialogues are all-tangent (C is
    never elaborated -> benign, needs no check), others are elaboration-heavy. This spread is what makes
    GLOBAL budget allocation matter — ECHO-R can skip the benign dialogues and spend its limited checks
    where contagion is actually imminent, while uniform spenders waste checks on benign ones. `risky_ratio`
    only biases the expected count.
    """
    lam = max(0, min(n_turns, int(round(rng.gauss(risky_ratio * n_turns, max(1.0, 0.5 * n_turns))))))
    n_risky = lam
    kinds = ["on_topic"] * n_risky + ["tangent"] * (n_turns - n_risky)
    rng.shuffle(kinds)
    return [(k, rng.choice(pool[k])) for k in kinds]


def build_policies(risk, check_cost: float):
    return {
        "no_op": NoOpPolicy(),
        "echo_r": EchoRPolicy(risk_scorer=risk, check_cost=check_cost),
        "uniform_check": UniformCheckPolicy(check_cost),
        "reminder": ReminderPolicy(),
        "oracle": OraclePolicy(check_cost),  # OracleRiskScorer: C is gold-false -> checks while C is open
    }


class _Roll:
    """Per-seed rollout state. The check budget is SHARED across dialogues (passed to rollout), not here."""
    def __init__(self, seed: dict):
        self.seed = seed
        self.C = Claim(text=seed["C"], turn_index=0, state=ProvenanceState.UNVERIFIED, load_bearing=1.0, is_false_gold=True)
        self.history = [{"role": "user", "content": seed["question"]},
                        {"role": "assistant", "content": seed["C"]}]
        self.contagious = False
        self.spent0 = 0.0

    def context_messages(self, user_text: str, discounted: bool) -> list[dict]:
        msgs = list(self.history)
        if discounted and self.C.state is ProvenanceState.UNVERIFIED:
            # De-emphasize without inserting a mid-dialogue system message: several chat
            # templates only allow system messages at the beginning.
            user_text = "Note: an earlier statement in this dialogue is unverified; do not rely on it.\n\n" + user_text
        msgs.append({"role": "user", "content": user_text})
        return msgs


def _mock_turn(roll: _Roll, kind: str, discounted: bool, rng: random.Random) -> str:
    """Deterministic stand-in for generation: assert C's false value with a prob depending on whether C
    is still presented-as-true and the follow-up pulls on it."""
    C, st = roll.seed["C"], roll.C.state
    if st is ProvenanceState.REFUTED:
        p = 0.05
    elif kind == "tangent":
        p = 0.05
    elif discounted:
        p = 0.40
    else:
        p = 0.85
    return (C + " (restated).") if rng.random() < p else "Here is some unrelated, careful elaboration."


def rollout(policy, seeds, total_budget, checker, schedules, generate, max_tokens) -> dict:
    """Online rollout over ALL dialogues sharing ONE budget pool (turn-interleaved, so early checks on
    benign dialogues exhaust the pool before later risky turns — the realistic online constraint)."""
    rolls = [_Roll(s) for s in seeds]
    budget = Budget(total=total_budget)            # SHARED across dialogues
    n_turns = len(schedules[0])
    for t in range(n_turns):
        pending = []
        for i, r in enumerate(rolls):
            kind, user_text = schedules[i][t]
            ctx = TurnContext("d", t, history_claims=[r.C] if r.C.state is ProvenanceState.UNVERIFIED else [],
                              candidate_claims=[Claim(user_text, t)])
            d = policy.decide(ctx, budget)
            discounted = False
            if d.action.value == "check" and budget.can_spend(checker.cost) and d.target is not None:
                budget.spend(checker.cost)
                r.C.state = checker.check(d.target)
                if r.C.state is ProvenanceState.REFUTED:
                    # (2) INJECT THE TRUTH, not just a disclaimer — replacing C with the corrected fact
                    # suppresses re-assertion far better than telling the model to "disregard".
                    for m in r.history:
                        if m["role"] == "assistant" and m["content"] == r.seed["C"]:
                            m["content"] = f"Correction: my earlier statement was wrong. The correct fact is: {r.seed['truth']}"
            elif d.action.value == "discount" and budget.can_spend(DISCOUNT_COST):
                budget.spend(DISCOUNT_COST)
                discounted = True
            pending.append((r, kind, discounted, user_text))
        prompts = [r.context_messages(ut, disc) for (r, k, disc, ut) in pending]
        texts = generate(prompts, [(r, k, disc) for (r, k, disc, _) in pending], max_tokens, [[] for _ in pending])
        for (r, kind, discounted, user_text), turn_text in zip(pending, texts):
            if detect_injected(turn_text, r.seed["C"], r.seed["truth"])["asserts_false"]:
                r.contagious = True
            r.history.append({"role": "user", "content": user_text})
            r.history.append({"role": "assistant", "content": turn_text})
    n = len(rolls)
    return {"policy": policy.name, "budget_total": total_budget,
            "contagion_rate": round(sum(1 for r in rolls if r.contagious) / n, 3),
            "budget_spent": round(budget.spent, 2), "budget_used_frac": round(budget.spent / total_budget, 2) if total_budget else 0.0, "n": n}


# ---------- MULTI-CLAIM rollout (the regime where selective allocation can beat uniform) ----------

class _MultiRoll:
    """One dialogue carrying K distinct injected claims (different entities). Probes target one entity per
    turn; with budget < K, WHICH claim you check matters — ECHO-R checks the about-to-be-probed one,
    uniform checks indiscriminately."""
    def __init__(self, bundle: list[dict]):
        self.seeds = bundle
        self.dialogue_id = "|".join(str(s.get("q_id")) for s in bundle)
        self.claims = [Claim(text=s["C"], turn_index=0, state=ProvenanceState.UNVERIFIED, load_bearing=1.0, is_false_gold=True)
                       for s in bundle]
        self.history = []
        for s in bundle:
            self.history += [{"role": "user", "content": f"Tell me about {s['entity']}."},
                             {"role": "assistant", "content": s["C"]}]
        self.contagion_turns = 0
        self.probe_turns = 0
        self.banned: list[str] = []   # 法子A: false tokens banned from decoding after a CHECK refutes a claim
        self.check_log: dict[int, dict] = {}  # diagnostic (--trace): claim_idx -> {turn, refuted}

    def open_claims(self):
        return [c for c in self.claims if c.state is ProvenanceState.UNVERIFIED]


def make_schedule_multi(n_turns: int, k: int, risky_ratio: float, pool: dict, rng: random.Random):
    """Per-dialogue schedule of (kind, claim_idx, user_text). on_topic probes a specific claim by entity."""
    sched = []
    for _ in range(n_turns):
        if rng.random() < risky_ratio:
            idx = rng.randrange(k)
            sched.append(("on_topic", idx, None))   # user_text filled at runtime with the entity
        else:
            sched.append(("tangent", None, rng.choice(pool["tangent"])))
    return sched


def _forced_answer(seed: dict, mode: str) -> str:
    """法子D served response for a checked-and-refuted claim (replaces free generation).

    Both modes are contagion-free by construction: 'grounded' emits the truth (which has no
    distinctive false token — detector-blind seeds are filtered at load), 'abstain' asserts
    nothing. The research claim is NOT 'this cures' (trivially it does) but that ECHO-R's
    selective allocation gets MORE probes served this way per unit budget than uniform.
    """
    if mode == "abstain":
        return (f"I can't confirm a specific claim about {seed.get('entity', 'that topic')}; "
                "I won't assert something I haven't verified.")
    return f"The verified fact is: {seed['truth']}"


def _future_probe_turns(sched) -> dict[int, list[int]]:
    """claim_idx -> list of turn indices at which it is probed (from the fixed schedule)."""
    out: dict[int, list[int]] = {}
    for tt, (kind, kidx, _u) in enumerate(sched):
        if kind == "on_topic" and kidx is not None:
            out.setdefault(kidx, []).append(tt)
    return out


def _probe_risk_rank(risk, ctx, probed_claim) -> tuple[int | None, int]:
    """Rank of `probed_claim` among open claims by the ECHO-R risk scorer (1 = highest risk).

    Diagnostic only: measures whether the scorer would prioritise the claim that is about to
    be probed. Returns (rank_or_None, n_open); rank is None when the probed claim is not open.
    """
    openc = ctx.open_unverified() if hasattr(ctx, "open_unverified") else []
    n_open = len(openc)
    if risk is None or not hasattr(risk, "_per_claim") or probed_claim not in openc:
        return None, n_open
    pc = risk._per_claim(ctx)
    # Match the policy's actual tie behavior. `riskiest()` uses max() over
    # open claims, so ties go to the earliest open claim, not to every tied
    # claim. This diagnostic should not count all zero-score ties as rank 1.
    ordered = sorted(enumerate(openc), key=lambda x: (-pc.get(id(x[1]), 0.0), x[0]))
    rank = next((j + 1 for j, (_orig_i, c) in enumerate(ordered) if c is probed_claim), None)
    return rank, n_open


def rollout_multiclaim(policy, bundles, total_budget, checker, schedules, generate, max_tokens,
                       suppress: bool = False, force_rewrite: bool = False, rewrite_mode: str = "grounded",
                       trace: dict | None = None) -> dict:
    rolls = [_MultiRoll(b) for b in bundles]
    budget = Budget(total=total_budget)
    n_turns = len(schedules[0])
    fps = [_future_probe_turns(s) for s in schedules]  # per-dialogue future probe turns (for --trace)
    for t in range(n_turns):
        pending = []
        for i, r in enumerate(rolls):
            kind, k_idx, user_text = schedules[i][t]
            if kind == "on_topic":
                # 法子C: strong re-elicitation probe (raises base contagion so cures are measurable).
                ent = r.seeds[k_idx]["entity"]
                user_text = f"What is the single key fact about {ent}? State it directly in one sentence."
            ctx = TurnContext("d", t, history_claims=r.open_claims(), candidate_claims=[Claim(user_text, t)])
            # diagnostic: rank the about-to-be-probed claim BEFORE the check mutates state
            rank, n_open = (_probe_risk_rank(trace.get("risk"), ctx, r.claims[k_idx])
                            if (trace and kind == "on_topic") else (None, 0))
            d = policy.decide(ctx, budget)
            discounted = False
            checked_idx = None
            if d.action.value == "check" and budget.can_spend(checker.cost) and d.target is not None:
                budget.spend(checker.cost)
                d.target.state = checker.check(d.target)
                try:
                    checked_idx = r.claims.index(d.target)
                except ValueError:
                    checked_idx = None
                refuted = d.target.state is ProvenanceState.REFUTED
                if checked_idx is not None:
                    r.check_log[checked_idx] = {"turn": t, "refuted": bool(refuted)}
                if refuted:
                    si = next(s for s in r.seeds if s["C"] == d.target.text)
                    for m in r.history:
                        if m["role"] == "assistant" and m["content"] == d.target.text:
                            m["content"] = f"Correction: my earlier statement was wrong. The correct fact is: {si['truth']}"
                    if suppress:  # 法子A: ban distinctive false-value tokens from decoding for the rest of the dialogue
                        for tokq in si.get("ban_tokens", si.get("false_tokens", [])):
                            if tokq not in r.banned:
                                r.banned.append(tokq)
            elif d.action.value == "discount" and budget.can_spend(DISCOUNT_COST):
                budget.spend(DISCOUNT_COST)
                discounted = True
            tr = None
            if trace is not None and (kind == "on_topic" or checked_idx is not None):
                tr = {"policy": policy.name, "model": trace.get("model"), "budget_frac": trace.get("budget_frac"),
                      "dialogue": r.dialogue_id, "turn": t, "kind": kind, "probed_idx": k_idx,
                      "risk_rank": rank, "n_open": n_open,
                      "checked_idx": checked_idx,
                      "checked_is_probed": (checked_idx is not None and checked_idx == k_idx),
                      "checked_later_probed": (any(tt > t for tt in fps[i].get(checked_idx, []))
                                               if checked_idx is not None else None)}
            pending.append((r, kind, k_idx, discounted, user_text, tr))
        prompts, bad_words = [], []
        for (r, kind, k_idx, disc, ut, tr) in pending:
            msgs = list(r.history)
            if disc:
                ut = "Note: an earlier statement in this dialogue is unverified; do not rely on it.\n\n" + ut
            prompts.append(msgs + [{"role": "user", "content": ut}])
            bad_words.append(list(r.banned))
        texts = generate(prompts, [(r, kind, k_idx, disc) for (r, kind, k_idx, disc, _u, _tr) in pending], max_tokens, bad_words)
        forced = 0
        for (r, kind, k_idx, disc, ut, tr), turn_text in zip(pending, texts):
            if kind == "on_topic":
                r.probe_turns += 1
                s = r.seeds[k_idx]
                # 法子D: if this probed claim was already CHECKed & refuted, serve a grounded/abstain
                # answer instead of the model's free generation (which can paraphrase the falsehood past
                # a token-ban). The moat is which policy gets more probes into this state per unit budget.
                served = force_rewrite and r.claims[k_idx].state is ProvenanceState.REFUTED
                if served:
                    turn_text = _forced_answer(s, rewrite_mode)
                    forced += 1
                contagious = bool(detect_injected(turn_text, s["C"], s["truth"])["asserts_false"])
                if contagious:
                    r.contagion_turns += 1
                if tr is not None:
                    cl = r.check_log.get(k_idx)
                    tr.update({"probed_contagious": contagious, "probed_served": bool(served),
                               "probed_state": r.claims[k_idx].state.name.lower(),
                               "probed_ever_checked": cl is not None,
                               "probed_checked_before_probe": (cl is not None and cl["turn"] < t),
                               "probed_check_refuted": (cl["refuted"] if cl else None)})
            if tr is not None:
                trace["sink"].append(tr)
            r.history.append({"role": "user", "content": ut})
            r.history.append({"role": "assistant", "content": turn_text})
    probes = sum(r.probe_turns for r in rolls)
    cont = sum(r.contagion_turns for r in rolls)
    return {"policy": policy.name, "budget_total": total_budget,
            "contagion_rate": round(cont / probes, 3) if probes else 0.0,
            "contagious_probes": cont, "total_probes": probes,
            "budget_spent": round(budget.spent, 2),
            "budget_used_frac": round(budget.spent / total_budget, 2) if total_budget else 0.0, "n_dialogues": len(rolls)}


def _mock_turn_multi(roll: "_MultiRoll", kind: str, k_idx, disc: bool, rng: random.Random) -> str:
    if kind != "on_topic" or k_idx is None:
        return "Here is some unrelated, careful elaboration."
    s = roll.seeds[k_idx]
    # 法子A: if the claim's false tokens are banned from decoding, the model physically cannot restate
    # the false value -> no contagion (this is the mechanism the real vLLM bad_words enforces).
    if any(tok in roll.banned for tok in s.get("ban_tokens", s.get("false_tokens", []))):
        return "A careful, unrelated elaboration."
    c = roll.claims[k_idx]
    if c.state is ProvenanceState.REFUTED:
        p = 0.35   # context correction alone only partly helps (the real weakness we observed)
    elif disc:
        p = 0.60
    else:
        p = 0.90
    return (roll.seeds[k_idx]["C"] + " (restated).") if rng.random() < p else "A careful, unrelated elaboration."


def main() -> None:
    os.environ.setdefault("VLLM_USE_V1", "0")
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    run_name = config["pilot"].get("run_name", "pilot")
    run_dir = resolve_project_path(config["paths"]["runs_dir"]) / run_name
    si_path = resolve_project_path(args.selfinduced) if args.selfinduced else run_dir / "claim_annotation_selfinduced.jsonl"
    pool = json.loads(resolve_project_path(args.followups).read_text(encoding="utf-8"))
    seeds = load_seeds_injected(config, args.model) if args.arm == 'injected' else load_seeds(config, si_path, args.model)
    if args.limit:
        seeds = seeds[: args.limit]
    if not seeds:
        raise SystemExit(f"No confirmed-false seeds for model {args.model!r} in {si_path} (need turn1_was_false=True rows).")
    seed0 = int(config.get("random_seed", 0))
    multi = args.claims_per_dialogue > 1
    if args.force_rewrite and not multi:
        raise SystemExit("--force-rewrite (法子D) is multi-claim injected only; use --claims-per-dialogue >1 --arm injected.")
    if multi:
        if args.arm != "injected":
            raise SystemExit("--claims-per-dialogue>1 is injected-arm only (claims must be externally known).")
        K = args.claims_per_dialogue
        sd = list(seeds); random.Random(seed0).shuffle(sd)
        units = [sd[i:i + K] for i in range(0, len(sd) - len(sd) % K, K)]  # bundles of K; drop remainder
        if not units:
            raise SystemExit(f"Not enough seeds ({len(seeds)}) for K={K}.")
        schedules = [make_schedule_multi(args.turns, K, args.risky_ratio, pool, random.Random(stable_seed(seed0, bi)))
                     for bi in range(len(units))]
    else:
        units = seeds
        schedules = [make_schedule(args.turns, args.risky_ratio, pool, random.Random(stable_seed(seed0, s["q_id"], args.model))) for s in seeds]
    budget_fracs = [float(x) for x in args.budgets.split(",")]
    budgets = [(f, max(1.0, round(f * len(units) * args.check_cost, 2))) for f in budget_fracs]  # (frac, total checks)

    ban_ready = sum(1 for s in seeds if s.get("ban_tokens"))
    out_path = resolve_output_path(args)
    print(json.dumps({"model": args.model, "arm": args.arm, "seeds": len(seeds),
                      "claims_per_dialogue": args.claims_per_dialogue, "dialogues": len(units),
                      "turns": args.turns, "budgets": budgets, "risky_ratio": args.risky_ratio,
                      "suppression_ban_ready_seeds": ban_ready}, ensure_ascii=False))
    if args.dry_run:
        print(json.dumps({"dry_run": True, "example_schedule": [s[0] for s in schedules[0]]}, ensure_ascii=False))
        return

    # checker: real run = bounded NLI against the seed's truth_statement (no-oracle path); mock = perfect
    # OracleChecker so the intervention logic is exercised without a local NLI model.
    oracle_checker = OracleChecker(cost=args.check_cost)
    nli_meta = {
        "checker": "oracle" if args.mock else "nli",
        "nli_required": bool(not args.mock and not args.allow_nli_noop),
        "nli_available": None,
        "nli_error": None,
    }
    if args.mock:
        checker = oracle_checker
    else:
        truth_by_text = {s["C"]: s["truth"] for s in seeds}
        nli_cfg = config.get("pilot", {}).get("nli", {})
        checker = NLIChecker(evidence_for=lambda c: truth_by_text.get(c.text, ""), cost=args.check_cost,
                             configure=lambda: configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold")))
        from detector import nli_health
        configure_nli(nli_cfg.get("model"), nli_cfg.get("threshold"))
        h = nli_health()
        nli_meta["nli_available"] = bool(h["available"])
        nli_meta["nli_error"] = h.get("error")
        if not h["available"] and not args.allow_nli_noop:
            raise SystemExit(
                f"[NLI UNAVAILABLE] {h['error']}. The CHECK action needs NLI for non-mock research runs; "
                "install sentencepiece/protobuf/transformers or use --mock/--dry-run for plumbing tests."
            )

    # generation backend. generate(prompts, metas, max_tokens, bad_words) — bad_words[i] is the per-prompt
    # decoding ban-list (法子A); empty for single-claim / unsuppressed.
    if args.mock:
        if multi:
            def generate(prompts, metas, max_tokens, bad_words=None):
                return [_mock_turn_multi(r, kind, k_idx, disc,
                                         random.Random(stable_seed(seed0, r.dialogue_id, len(r.history), kind, k_idx)))
                        for (r, kind, k_idx, disc) in metas]
        else:
            def generate(prompts, metas, max_tokens, bad_words=None):
                return [_mock_turn(r, kind, disc, random.Random(stable_seed(seed0, r.seed["q_id"], len(r.history))))
                        for (r, kind, disc) in metas]
    else:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        model_path = resolve_project_path(config["paths"]["models_dir"]) / args.model
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        llm = LLM(model=str(model_path), dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization,
                  trust_remote_code=True, seed=int(config.get("random_seed", 0)),
                  max_model_len=args.max_model_len)
        base_seed = int(config.get("random_seed", 0))

        def generate(prompts, metas, max_tokens, bad_words=None):
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts]
            # Per-prompt SamplingParams so 法子A's bad_words (banned false tokens) apply per dialogue.
            params = []
            for i in range(len(texts)):
                bw = (bad_words[i] if bad_words else []) or []
                kw = {"temperature": 0.7, "top_p": 1.0, "max_tokens": max_tokens, "seed": base_seed}
                if bw:
                    kw["bad_words"] = bw          # vLLM bans these strings from the generation
                params.append(SamplingParams(**kw))
            outs = llm.generate(texts, params)
            return [o.outputs[0].text.strip() for o in outs]

    # Multi-claim uses per-claim entity overlap (HeuristicRiskScorer) so ECHO-R can target the probed
    # claim; single-claim uses the deictic elaboration scorer.
    risk = HeuristicRiskScorer() if multi else ElaborationRiskScorer()
    trace_rows = [] if (args.trace and multi) else None
    rows = []
    for frac, total in budgets:
        for name, policy in build_policies(risk, args.check_cost).items():
            policy_checker = oracle_checker if name == "oracle" else checker
            if multi:
                tr = ({"sink": trace_rows, "risk": risk, "model": args.model, "budget_frac": frac}
                      if trace_rows is not None else None)
                r = rollout_multiclaim(policy, units, total, policy_checker, schedules, generate, args.max_tokens,
                                       suppress=args.suppress, force_rewrite=args.force_rewrite,
                                       rewrite_mode=args.rewrite_mode, trace=tr)
            else:
                r = rollout(policy, units, total, policy_checker, schedules, generate, args.max_tokens)
            r["budget_frac"] = frac
            rows.append(r)

    tag = result_tag(args)
    suffix = result_suffix(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "arm": args.arm, "claims_per_dialogue": args.claims_per_dialogue,
                                    "n_seeds": len(seeds), "n_dialogues": len(units), "turns": args.turns,
                                    "suppression_ban_ready_seeds": ban_ready,
                                    "mock": bool(args.mock), "limit": args.limit, "nli": nli_meta,
                                    "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": to_project_relative(out_path), "rows": len(rows)}, ensure_ascii=False))
    if trace_rows is not None:
        trace_path = resolve_project_path("reports") / f"echo_r_trace_{tag}_{args.model}{suffix}.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for tr in trace_rows:
                fh.write(json.dumps(tr, ensure_ascii=False) + "\n")
        print(json.dumps({"wrote_trace": to_project_relative(trace_path), "trace_rows": len(trace_rows)}, ensure_ascii=False))
    for frac, total in budgets:
        print(f"\n== global budget frac={frac} ({total} checks for {len(units)} dialogues) ==")
        for r in sorted([x for x in rows if x["budget_frac"] == frac], key=lambda x: x["contagion_rate"]):
            print(f"  {r['policy']:24s} contagion={r['contagion_rate']:.3f}  spent={r['budget_spent']:.1f} ({r['budget_used_frac']:.0%})")


if __name__ == "__main__":
    main()
