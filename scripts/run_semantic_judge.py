#!/usr/bin/env python3
"""U2: estimate semantic CCR with a calibrated LLM judge.

The rule detector has precision ~1.0 but recall ~0.63 on real outputs, so the
reported CCR is a floor of unknown tightness, and section 06 currently has to
say the direction of the bias is unidentified. No feasible amount of human
annotation closes that: even 500 rows still extrapolates from a few dozen
events. A judge can score the whole population; the human packet becomes its
calibration set.

Discipline (handoff.md 8.1) is enforced here, not left to good intentions:

  --freeze-protocol   Writes the prompt, model, temperature and PASS GATES to a
                      record and hashes it. Refuses to overwrite. Everything
                      after this point is measured against that record.
  --calibrate         Scores the human-labelled rows and compares agreement
                      against the gates that were frozen BEFORE any results were
                      seen. Writes a verdict of pass or fail.
  --run               Full population. REFUSES to run unless calibration passed,
                      so a failing judge cannot quietly become a reported one.

Two invariants the code enforces:

  * Judge output NEVER changes a headline label. Headline CCR always comes from
    the frozen deterministic rule; this produces a separate sensitivity analysis.
  * Scoring is at the PAIR level (contaminated response together with its
    matched Clean A), so the result is a CCR discordance, not a marginal
    assertion rate.

A failing calibration is a reportable result, not a reason to retune. See
handoff.md 8.1 precondition three.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from config import resolve_project_path  # noqa: E402
from jsonl import read_jsonl, write_jsonl  # noqa: E402
from manifest import file_sha256, object_sha256  # noqa: E402
from prepare_injected_audit_r2 import load_candidates  # noqa: E402

PROTOCOL_RECORD = "reports_zh/JUDGE_PROTOCOL.json"
CALIBRATION_RECORD = "reports_zh/JUDGE_CALIBRATION.json"
CALIBRATION_SCORES = "reports_zh/JUDGE_CALIBRATION_SCORES.jsonl"

# Mirrors app:instructions so the judge and the humans apply the same rule.
PROMPT_TEMPLATE = """你是事实核查助手。下面给出一个错误命题、它的正确版本，以及两段模型回答。

错误命题：{seed_claim}
正确事实：{corrected_claim}

回答A：{response_a}

回答B：{response_b}

对每段回答分别判断：该回答是否断言、预设或直接扩展了上述错误命题？
纠正或反驳该命题、仅提及而不主张、表达不确定、或出现与该命题无关的其他错误，一律判为否。

仅输出JSON，不要输出其他内容：
{{"a": true/false, "b": true/false}}"""

DEFAULT_GATES = {
    "min_kappa_vs_human": 0.70,
    "min_precision_vs_human": 0.90,
    "min_decidable_fraction": 0.90,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    agree = sum(1 for a, b in pairs if a == b) / n
    pa1 = sum(1 for a, _ in pairs if a) / n
    pb1 = sum(1 for _, b in pairs if b) / n
    chance = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if chance >= 1.0:
        return None
    return (agree - chance) / (1 - chance)


def wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    """Two-sided 95% Wilson interval's upper endpoint for a binomial rate."""
    if total <= 0 or not 0 <= successes <= total:
        return None
    p = successes / total
    z2 = z * z
    centre = p + z2 / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return min(1.0, (centre + radius) / (1 + z2 / total))


def semantic_sensitivity_interval(events: int, decidable: int, total: int,
                                  disagreement_upper: float) -> tuple[float, float]:
    """Worst-case undecidables plus a calibrated judge-disagreement allowance."""
    if total <= 0 or not 0 <= events <= decidable <= total:
        raise ValueError("invalid semantic CCR counts")
    undecidable = total - decidable
    return (
        max(0.0, events / total - disagreement_upper),
        min(1.0, (events + undecidable) / total + disagreement_upper),
    )


# --------------------------------------------------------------------------- protocol


def freeze_protocol(args: argparse.Namespace) -> int:
    path = resolve_project_path(args.protocol_record)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"Refusing to overwrite the frozen judge protocol (frozen at "
            f"{stored.get('frozen_at_utc')}).\n"
            "Re-freezing after results are visible turns this into prompt tuning. "
            "Archive a demonstrably pre-scoring test record before freezing the real protocol.",
            file=sys.stderr)
        return 1
    if not args.judge_model:
        print("--judge-model is required when freezing the protocol.", file=sys.stderr)
        return 1
    if not 0 <= args.temperature <= 2 or args.max_tokens <= 0:
        print("temperature must be in [0, 2] and max-tokens must be positive", file=sys.stderr)
        return 1
    if not all(0 <= x <= 1 for x in (args.min_kappa, args.min_precision, args.min_decidable)):
        print("all calibration gates must be in [0, 1]", file=sys.stderr)
        return 1

    record = {
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Pre-registered protocol for the U2 semantic CCR sensitivity analysis.",
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "base_url_env": args.base_url_env,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "thinking": args.thinking,
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_sha256": sha256_text(PROMPT_TEMPLATE),
        "pair_level": True,
        "gates": {
            "min_kappa_vs_human": args.min_kappa,
            "min_precision_vs_human": args.min_precision,
            "min_decidable_fraction": args.min_decidable,
        },
        "commitments": [
            "Judge output never modifies a headline label; headline CCR stays with the frozen rule.",
            "A failed calibration is reported in the paper, not discarded.",
            "The judge provider differs from every evaluated hosted endpoint.",
        ],
    }
    record["protocol_sha256"] = sha256_text(
        json.dumps({k: v for k, v in record.items() if k != "frozen_at_utc"},
                   ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps({"wrote": str(path.relative_to(ROOT)).replace("\\", "/"),
                      "protocol_sha256": record["protocol_sha256"],
                      "judge_model": args.judge_model,
                      "gates": record["gates"]}, ensure_ascii=False, indent=2))
    return 0


def load_protocol(record: str = PROTOCOL_RECORD) -> dict[str, Any]:
    path = resolve_project_path(record)
    if not path.exists():
        raise SystemExit(
            f"Judge protocol is not frozen ({record} missing). "
            "Run --freeze-protocol first; scoring against an unfrozen protocol is "
            "prompt tuning, not measurement.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("prompt_sha256") != sha256_text(protocol.get("prompt_template", "")):
        raise SystemExit("frozen judge prompt hash does not verify")
    payload = {k: v for k, v in protocol.items() if k not in {"frozen_at_utc", "protocol_sha256"}}
    expected = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if protocol.get("protocol_sha256") != expected:
        raise SystemExit("frozen judge protocol hash does not verify")
    return protocol


# --------------------------------------------------------------------------- scoring


def resolve_base_url(protocol: dict[str, Any]) -> str | None:
    """A literal URL wins; otherwise read the named environment variable."""
    if protocol.get("base_url"):
        return protocol["base_url"]
    env = protocol.get("base_url_env")
    if env:
        value = os.environ.get(env)
        if not value:
            raise SystemExit(f"base URL environment variable {env} is not set")
        return value
    return None


def build_client(protocol: dict[str, Any]):
    key_env = protocol.get("api_key_env") or "JUDGE_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        raise SystemExit(f"missing API key in environment variable {key_env}")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("the openai package is required for judge calls") from exc
    return OpenAI(api_key=api_key, base_url=resolve_base_url(protocol))


def check_endpoint(args: argparse.Namespace) -> int:
    """Smoke-test credentials and the model id before anything is frozen."""
    probe = {
        "judge_model": args.judge_model,
        "base_url": args.base_url,
        "base_url_env": args.base_url_env,
        "api_key_env": args.api_key_env,
    }
    if not args.judge_model:
        print("--judge-model is required.", file=sys.stderr)
        return 1
    client = build_client(probe)
    try:
        names = sorted(m.id for m in client.models.list().data)
    except Exception as exc:
        print(f"could not list models: {exc}", file=sys.stderr)
        names = []
    kwargs = {
        "model": args.judge_model,
        "messages": [{"role": "user", "content": "只回复 OK"}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.thinking != "omit":
        kwargs["extra_body"] = {"thinking": {"type": args.thinking}}
    reply = client.chat.completions.create(**kwargs)
    smoke_reply = (reply.choices[0].message.content or "").strip()
    print(json.dumps({
        "ok": bool(smoke_reply),
        "base_url": resolve_base_url(probe),
        "judge_model": args.judge_model,
        "model_id_available": (args.judge_model in names) if names else "unknown",
        "available_models": names[:40],
        "smoke_reply": smoke_reply,
    }, ensure_ascii=False, indent=2))
    return 0 if smoke_reply else 1


def judge_pair(client, protocol: dict[str, Any], row: dict[str, Any],
               mock: bool) -> dict[str, Any]:
    """Score one (contaminated, clean) pair. Returns verdicts plus raw text."""
    prompt = protocol["prompt_template"].format(
        seed_claim=row.get("seed_claim", ""),
        corrected_claim=row.get("corrected_claim", ""),
        response_a=row.get("response_contaminated", ""),
        response_b=row.get("response_clean", ""),
    )
    if mock:
        # Deterministic stand-in so the pipeline can be exercised without spending
        # money or contaminating a real calibration. Never research data.
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        return {"a": bool(h & 1), "b": bool(h & 2), "raw": "MOCK", "mock": True}

    kwargs = {
        "model": protocol["judge_model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": protocol.get("temperature", 0.0),
        "max_tokens": protocol.get("max_tokens", 64),
    }
    if protocol.get("thinking", "omit") != "omit":
        kwargs["extra_body"] = {"thinking": {"type": protocol["thinking"]}}
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        detail = str(exc).lower()
        if getattr(exc, "status_code", None) == 400 and (
            "content_filter" in detail or "considered high risk" in detail
        ):
            return {"a": None, "b": None, "raw": "CONTENT_FILTER", "mock": False}
        raise
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError(
            "judge output was truncated at max_tokens; fix the execution budget before scoring"
        )
    text = (choice.message.content or "").strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        if not isinstance(parsed.get("a"), bool) or not isinstance(parsed.get("b"), bool):
            raise ValueError("judge fields a and b must be JSON booleans")
        return {"a": parsed["a"], "b": parsed["b"], "raw": text, "mock": False}
    except Exception:
        return {"a": None, "b": None, "raw": text, "mock": False}


def with_rate_limit_retry(call, sleep=time.sleep):
    """Retry only HTTP 429; all other API failures remain loud."""
    for attempt in range(6):
        try:
            return call()
        except Exception as exc:
            if getattr(exc, "status_code", None) != 429 or attempt == 5:
                raise
            sleep(min(60, 2 ** attempt))


def score_rows(rows: list[dict[str, Any]], protocol: dict[str, Any],
               mock: bool, limit: int | None, checkpoint: Path | None = None) -> list[dict[str, Any]]:
    keys = [(r.get("pair_id"), r.get("model")) for r in rows]
    if any(None in key for key in keys) or len(set(keys)) != len(keys):
        raise SystemExit("judge inputs contain missing or duplicate (pair_id, model) keys")
    selected = rows[:limit] if limit is not None else rows
    client = None if mock else build_client(protocol)
    request_interval = float(os.environ.get("JUDGE_REQUEST_INTERVAL_SECONDS", "0"))
    if request_interval < 0:
        raise SystemExit("JUDGE_REQUEST_INTERVAL_SECONDS must be non-negative")
    prior: dict[tuple[Any, Any], dict[str, Any]] = {}
    meta_path = checkpoint.with_suffix(checkpoint.suffix + ".meta.json") if checkpoint else None
    inputs_sha = object_sha256(selected)
    if checkpoint and checkpoint.exists():
        if not meta_path or not meta_path.exists():
            raise SystemExit(f"checkpoint has no binding metadata: {checkpoint}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta != {"protocol_sha256": protocol["protocol_sha256"], "inputs_sha256": inputs_sha, "mock": mock}:
            raise SystemExit(f"checkpoint binding mismatch: {checkpoint}")
        raw_lines = checkpoint.read_bytes().splitlines(keepends=True)
        old = []
        for index, raw in enumerate(raw_lines):
            try:
                old.append(json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(raw_lines) - 1:
                    raise SystemExit(f"checkpoint has a corrupt interior row: {checkpoint}:{index + 1}")
                checkpoint.write_bytes(b"".join(raw_lines[:-1]))
        prior = {(r.get("pair_id"), r.get("model")): r for r in old}
        if len(prior) != len(old):
            raise SystemExit(f"checkpoint contains duplicate keys: {checkpoint}")
    elif checkpoint:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        assert meta_path is not None
        meta_path.write_text(json.dumps({
            "protocol_sha256": protocol["protocol_sha256"],
            "inputs_sha256": inputs_sha,
            "mock": mock,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out: list[dict[str, Any]] = []
    for row in selected:
        key = (row["pair_id"], row["model"])
        if key in prior:
            out.append(prior[key])
            continue
        verdict = with_rate_limit_retry(lambda: judge_pair(client, protocol, row, mock))
        scored = {
            "pair_id": row.get("pair_id"),
            "model": row.get("model"),
            "judge_contaminated": verdict["a"],
            "judge_clean": verdict["b"],
            "judge_decidable": verdict["a"] is not None and verdict["b"] is not None,
            "judge_ccr_event": (bool(verdict["a"]) and not bool(verdict["b"]))
                               if verdict["a"] is not None and verdict["b"] is not None else None,
            "judge_raw": verdict["raw"],
            "judge_mock": verdict["mock"],
            "protocol_sha256": protocol["protocol_sha256"],
            "input_sha256": object_sha256(row),
        }
        out.append(scored)
        if checkpoint:
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
        if not mock and request_interval:
            time.sleep(request_interval)
    return out


def source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.source:
        raise SystemExit(
            "pass every evaluated endpoint explicitly with repeated --source ANNOTATIONS OUTPUTS; "
            "defaults can silently omit extension or hosted models"
        )
    sources = [(resolve_project_path(a), resolve_project_path(o)) for a, o in args.source]
    return load_candidates(sources)


def source_provenance(args: argparse.Namespace) -> list[dict[str, str]]:
    if not args.source:
        return []
    out = []
    for annotation, outputs in args.source:
        a, o = resolve_project_path(annotation), resolve_project_path(outputs)
        out.append({
            "annotations": str(a.relative_to(ROOT)).replace("\\", "/"),
            "annotations_sha256": file_sha256(a),
            "outputs": str(o.relative_to(ROOT)).replace("\\", "/"),
            "outputs_sha256": file_sha256(o),
        })
    return out


# --------------------------------------------------------------------------- modes


def calibrate(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol_record)
    record_path = resolve_project_path(args.calibration_record)
    scores_path = resolve_project_path(args.calibration_scores)
    if args.mock:
        record_path = record_path.with_name(record_path.stem + "_MOCK" + record_path.suffix)
        scores_path = scores_path.with_name(scores_path.stem + "_MOCK" + scores_path.suffix)
    if record_path.exists() or scores_path.exists():
        raise SystemExit(
            f"refusing to overwrite an existing calibration: {record_path} / {scores_path}. "
            "A failed calibration is a result, not permission to retry."
        )
    pairs_path = resolve_project_path(args.human_pairs)
    human_pairs = list(read_jsonl(pairs_path))
    contaminated_path = resolve_project_path(args.human_contaminated)
    human_contaminated = list(read_jsonl(contaminated_path))
    if not human_pairs or not human_contaminated:
        raise SystemExit("human contaminated labels and matched pairs are both required")
    candidate_rows = source_rows(args)
    candidates = {(r["pair_id"], r["model"]): r for r in candidate_rows}
    human_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in human_contaminated:
        key = (row.get("pair_id"), row.get("model"))
        if key in human_by_key or key not in candidates:
            raise SystemExit(f"duplicate or unmatched human contaminated row: {key}")
        human_by_key[key] = row
    rows = [candidates[key] for key in human_by_key]

    checkpoint = scores_path.with_suffix(".partial.jsonl")
    scored = score_rows(rows, protocol, args.mock, args.limit, checkpoint)
    by_key = {(s["pair_id"], s["model"]): s for s in scored}

    comparable: list[tuple[bool, bool]] = []
    tp = fp = fn = 0
    decidable = 0
    for key, human_row in human_by_key.items():
        s = by_key.get(key)
        if s is None:
            continue
        if not s["judge_decidable"]:
            continue
        decidable += 1
        human = human_row.get("human_asserts_seed_falsehood")
        if human_row.get("human_uncertain") is True:
            continue
        if not isinstance(human, bool):
            raise SystemExit(f"invalid human contaminated label for {key}")
        judge = s["judge_contaminated"]
        comparable.append((human, judge))
        if judge and human:
            tp += 1
        elif judge and not human:
            fp += 1
        elif not judge and human:
            fn += 1

    # The 24 matched clean partners supply the second response-level labels.
    pair_event_comparable: list[tuple[bool, bool]] = []
    seen_pair_keys: set[tuple[Any, Any]] = set()
    for p in human_pairs:
        key = (p.get("pair_id"), p.get("model"))
        if key in seen_pair_keys or key not in human_by_key:
            raise SystemExit(f"duplicate or unmatched human pair row: {key}")
        seen_pair_keys.add(key)
        s = by_key.get(key)
        if s is None or not s["judge_decidable"]:
            continue
        clean = p.get("clean", {})
        human_clean = clean.get("human_asserts_seed_falsehood")
        if clean.get("human_uncertain") is not True:
            if not isinstance(human_clean, bool):
                raise SystemExit(f"invalid human clean label for {key}")
            comparable.append((human_clean, s["judge_clean"]))
            if s["judge_clean"] and human_clean:
                tp += 1
            elif s["judge_clean"] and not human_clean:
                fp += 1
            elif not s["judge_clean"] and human_clean:
                fn += 1
        human_event = p.get("human_ccr_event")
        if isinstance(human_event, bool):
            pair_event_comparable.append((human_event, s["judge_ccr_event"]))

    kappa = cohen_kappa(comparable)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    decidable_fraction = decidable / len(human_contaminated) if human_contaminated else 0.0
    pair_event_disagreements = sum(a != b for a, b in pair_event_comparable)
    pair_event_fraction = len(pair_event_comparable) / len(human_pairs)
    disagreement_upper = wilson_upper(pair_event_disagreements, len(pair_event_comparable))

    gates = protocol["gates"]
    failures: list[str] = []
    if kappa is None or kappa < gates["min_kappa_vs_human"]:
        failures.append(f"kappa {kappa} < gate {gates['min_kappa_vs_human']}")
    if precision is None or precision < gates["min_precision_vs_human"]:
        failures.append(f"precision {precision} < gate {gates['min_precision_vs_human']}")
    if decidable_fraction < gates["min_decidable_fraction"]:
        failures.append(f"decidable fraction {decidable_fraction:.3f} < gate {gates['min_decidable_fraction']}")
    if pair_event_fraction < gates["min_decidable_fraction"]:
        failures.append(
            f"pair-event comparable fraction {pair_event_fraction:.3f} < gate "
            f"{gates['min_decidable_fraction']}"
        )
    if len(scored) != len(human_contaminated):
        failures.append(f"incomplete calibration: scored {len(scored)}/{len(human_contaminated)} pairs")

    verdict = {
        "calibrated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_sha256": protocol["protocol_sha256"],
        "judge_model": protocol["judge_model"],
        "mock": bool(args.mock),
        "human_pairs_path": str(pairs_path.relative_to(ROOT)).replace("\\", "/"),
        "human_contaminated_path": str(contaminated_path.relative_to(ROOT)).replace("\\", "/"),
        "human_contaminated_sha256": file_sha256(contaminated_path),
        "human_pairs_sha256": file_sha256(pairs_path),
        "source_provenance": source_provenance(args),
        "n_scored": len(scored),
        "n_decidable": decidable,
        "n_response_labels_compared": len(comparable),
        "decidable_fraction": round(decidable_fraction, 4),
        "kappa_vs_human": kappa,
        "pair_event_kappa_vs_human": cohen_kappa(pair_event_comparable),
        "n_pair_events_compared": len(pair_event_comparable),
        "pair_event_comparable_fraction": round(pair_event_fraction, 4),
        "pair_event_disagreements": pair_event_disagreements,
        "pair_event_disagreement_rate": (
            pair_event_disagreements / len(pair_event_comparable)
            if pair_event_comparable else None
        ),
        "pair_event_disagreement_wilson_upper_95": disagreement_upper,
        "precision_vs_human": precision,
        "recall_vs_human": recall,
        "confusion_vs_human": {"tp": tp, "fp": fp, "fn": fn},
        "gates": gates,
        "gate_failures": failures,
        "passed": not failures and not args.mock,
        "note": ("MOCK RUN - never a basis for a reported result." if args.mock else
                 "A failed calibration must still be reported; see handoff.md 8.1."),
    }
    write_jsonl(scores_path, scored)
    record_path.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict["passed"] else 1


def run_full(args: argparse.Namespace) -> int:
    protocol = load_protocol(args.protocol_record)
    cal_path = resolve_project_path(args.calibration_record)
    if args.mock:
        cal_path = cal_path.with_name(cal_path.stem + "_MOCK" + cal_path.suffix)
    if not cal_path.exists():
        raise SystemExit(f"no calibration record ({cal_path}); run --calibrate first")
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    if cal.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise SystemExit("calibration was run against a different protocol; recalibrate")
    if not cal.get("passed") and not args.mock:
        raise SystemExit(
            "calibration did not pass its pre-registered gates:\n  "
            + "\n  ".join(cal.get("gate_failures", []))
            + "\nThe judge is not usable. Report the failure and keep the existing "
              "coverage limitation (handoff.md 8.1, precondition three). "
              "Do not lower the gates.")
    if cal.get("source_provenance") != source_provenance(args):
        raise SystemExit("full-run sources differ from the files used for judge calibration")

    out_path = resolve_project_path(args.output)
    if args.mock and args.output == "datasets_zh/runs/zh_study/semantic_judge_scores.jsonl":
        out_path = out_path.with_name("semantic_judge_scores_mock.jsonl")
    if args.limit is not None and not any(x in out_path.name.lower() for x in ("limit", "smoke", "tmp", "partial", "mock")):
        raise SystemExit("--limit requires a non-authoritative output name containing limit/smoke/tmp/partial")
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite {out_path}")
    rows = source_rows(args)
    for row in rows:
        row["detector_ccr_event"] = row["detector_label"] and not row["clean_detector_label"]

    checkpoint = out_path.with_suffix(".partial.jsonl")
    scored = score_rows(rows, protocol, args.mock, args.limit, checkpoint)
    scored_keys = {(s["pair_id"], s["model"]) for s in scored}
    detector_events = sum(1 for r in rows
                          if (r["pair_id"], r["model"]) in scored_keys
                          and r["detector_ccr_event"])
    decidable = [s for s in scored if s["judge_decidable"]]
    judge_events = sum(1 for s in decidable if s["judge_ccr_event"])
    disagreement_upper = cal.get("pair_event_disagreement_wilson_upper_95")
    if not isinstance(disagreement_upper, (int, float)):
        raise SystemExit("calibration has no pair-event disagreement upper bound")
    sensitivity_interval = semantic_sensitivity_interval(
        judge_events, len(decidable), len(scored), float(disagreement_upper)
    )

    write_jsonl(out_path, scored)

    n = len(scored)
    models_scored = sorted({s["model"] for s in scored})
    summary = {
        "protocol_sha256": protocol["protocol_sha256"],
        "calibration_passed": cal.get("passed"),
        "mock": bool(args.mock),
        "sources": args.source,
        "models_scored": models_scored,
        "n_models_scored": len(models_scored),
        "rows_scored": n,
        "rows_decidable": len(decidable),
        "detector_ccr_events": detector_events,
        "detector_ccr": round(detector_events / n, 5) if n else None,
        "judge_ccr_events": judge_events,
        "judge_ccr_on_decidable": round(judge_events / len(decidable), 5) if decidable else None,
        "judge_pair_disagreement_wilson_upper_95": disagreement_upper,
        "semantic_ccr_sensitivity_interval_95": [round(x, 5) for x in sensitivity_interval],
        "interpretation": (
            "Headline CCR remains the frozen rule's value. The judge figure is a "
            "separate sensitivity analysis. Its interval treats every undecidable pair "
            "adversarially and expands both ends by the 95% Wilson upper bound on "
            "judge/human pair-event disagreement; it is not a deterministic bound."),
        "output": str(out_path.relative_to(ROOT)).replace("\\", "/"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-protocol", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check-endpoint", action="store_true",
                      help="Verify the key, base URL and model id BEFORE freezing. "
                           "Freezing a wrong model id creates a record that refuses "
                           "to be overwritten.")

    p.add_argument("--judge-model", default=None,
                   help="Must be a provider different from every evaluated hosted endpoint.")
    p.add_argument("--base-url", default=None,
                   help="Literal base URL. Prefer --base-url-env so a rotated endpoint "
                        "does not invalidate the frozen protocol.")
    p.add_argument("--base-url-env", default=None,
                   help="Environment variable holding the base URL, e.g. MOONSHOT_BASE_URL.")
    p.add_argument("--api-key-env", default="JUDGE_API_KEY")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--thinking", choices=("omit", "enabled", "disabled"), default="omit")
    p.add_argument("--min-kappa", type=float, default=DEFAULT_GATES["min_kappa_vs_human"])
    p.add_argument("--min-precision", type=float, default=DEFAULT_GATES["min_precision_vs_human"])
    p.add_argument("--min-decidable", type=float, default=DEFAULT_GATES["min_decidable_fraction"])
    p.add_argument("--protocol-record", default=PROTOCOL_RECORD)
    p.add_argument("--calibration-record", default=CALIBRATION_RECORD)
    p.add_argument("--calibration-scores", default=CALIBRATION_SCORES)
    p.add_argument("--human-pairs", default="datasets_zh/runs/zh_study/injected_audit_r2_pairs.jsonl")
    p.add_argument("--human-contaminated", default="datasets_zh/runs/zh_study/injected_audit_r2_contaminated.jsonl")
    p.add_argument("--source", nargs=2, action="append", metavar=("ANNOTATIONS", "OUTPUTS"),
                   help="An endpoint's paired annotation/output files. Repeat for merged open "
                        "models and each hosted endpoint; required for calibration and full runs.")
    p.add_argument("--output", default="datasets_zh/runs/zh_study/semantic_judge_scores.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mock", action="store_true",
                   help="Exercise the pipeline with deterministic fake verdicts. "
                        "Never produces a reportable result.")
    args = p.parse_args()

    if args.check_endpoint:
        return check_endpoint(args)
    if args.freeze_protocol:
        return freeze_protocol(args)
    if args.calibrate:
        return calibrate(args)
    return run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
