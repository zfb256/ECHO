"""Failure-mode diagnostics for the ECHO-R intervention study (paper Section 6.2 / appendix).

Reads the per-turn traces emitted by run_echo_r_eval.py --trace (reports/echo_r_trace_*.jsonl)
and produces four diagnostic tables that explain WHY selective fact-checking fails to stop
contagion, even with a forced-rewrite guardrail:

  A. Check-probe alignment  -- do the checks land on claims that are actually probed later?
  B. Risk-ranking quality   -- does the risk scorer rank the about-to-be-probed claim on top?
  C. Residual attribution   -- under force-rewrite, where does the surviving contagion come from?
                               (never checked / checked but not refuted / checked too late)
  D. Oracle misalignment    -- at full budget, why does even the oracle leave contagion behind?

The thesis these support: interventions fail not because rewrites are weak but because a scarce
verification budget does not reach the claims that later drive contamination (targeting/timing),
not rewrite strength. Deterministic fields (rank, alignment) are model-independent; the residual
tables use the real contagion labels from the GPU run.

Usage:
  python3 -B scripts/echo_r_diagnostics.py --traces "reports/echo_r_trace_injected_multi4_forcerw-grounded_*.jsonl" \
      --out reports/echo_r_diagnostics_forcerw.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from artifacts import is_research_echo_r_path  # noqa: E402
from config import resolve_project_path, to_project_relative  # noqa: E402


def load(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in glob.glob(str(resolve_project_path(pat))):
            if not is_research_echo_r_path(path):
                continue
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def table_alignment(rows):
    """A. Of all CHECK events, fraction on the currently-probed claim and on any later-probed claim."""
    out = {}
    checks = [r for r in rows if r.get("checked_idx") is not None]
    for pol in sorted({r["policy"] for r in checks}):
        cr = [r for r in checks if r["policy"] == pol]
        out[pol] = {
            "n_checks": len(cr),
            "checked_is_probed_now": _mean([1.0 if r.get("checked_is_probed") else 0.0 for r in cr]),
            "checked_probed_later": _mean([1.0 if r.get("checked_later_probed") else 0.0 for r in cr]),
        }
    return out


def table_ranking(rows):
    """B. Over on_topic probes where the probed claim is open, how well does risk rank it?"""
    out = {}
    probes = [r for r in rows if r.get("kind") == "on_topic" and r.get("risk_rank") is not None]
    for pol in sorted({r["policy"] for r in probes}):
        pr = [r for r in probes if r["policy"] == pol]
        out[pol] = {
            "n_open_probes": len(pr),
            "precision_at_1": _mean([1.0 if r["risk_rank"] == 1 else 0.0 for r in pr]),
            "mean_rank": _mean([r["risk_rank"] for r in pr]),
            "mean_n_open": _mean([r.get("n_open") for r in pr]),
        }
    return out


def _residual_class(r):
    """For a contagious probe under force-rewrite, why did it survive?"""
    if r.get("probed_served"):
        return "served_but_contagious"  # should be ~0 by construction; a red flag if not
    if not r.get("probed_ever_checked"):
        return "never_checked"          # budget/targeting never reached this claim
    if r.get("probed_check_refuted") is False:
        return "checked_not_refuted"    # checker (NLI) failed to refute
    if not r.get("probed_checked_before_probe"):
        return "checked_too_late"       # checked at/after the probe turn
    return "other"


def table_residual(rows, policies=("echo_r", "oracle")):
    """C. Attribution of surviving contagion under the guardrail, per policy x budget."""
    out = {}
    for pol in policies:
        for bf in sorted({r.get("budget_frac") for r in rows if r["policy"] == pol}):
            cont = [r for r in rows if r["policy"] == pol and r.get("budget_frac") == bf
                    and r.get("kind") == "on_topic" and r.get("probed_contagious")]
            if not cont:
                continue
            cls = defaultdict(int)
            for r in cont:
                cls[_residual_class(r)] += 1
            n = len(cont)
            out[f"{pol}@{bf}"] = {"n_contagious": n,
                                  **{k: round(v / n, 4) for k, v in sorted(cls.items())}}
    return out


def table_oracle(rows):
    """D. Oracle at each budget: check coverage of probed claims + why contagion survives."""
    out = {}
    for bf in sorted({r.get("budget_frac") for r in rows if r["policy"] == "oracle"}):
        orows = [r for r in rows if r["policy"] == "oracle" and r.get("budget_frac") == bf]
        probes = [r for r in orows if r.get("kind") == "on_topic"]
        cont = [r for r in probes if r.get("probed_contagious")]
        never = sum(1 for r in cont if not r.get("probed_ever_checked"))
        out[f"oracle@{bf}"] = {
            "n_probes": len(probes),
            "probed_claim_ever_checked": _mean([1.0 if r.get("probed_ever_checked") else 0.0 for r in probes]),
            "n_contagious": len(cont),
            "contagious_never_checked_frac": round(never / len(cont), 4) if cont else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", required=True, help="Glob(s) for echo_r_trace_*.jsonl")
    ap.add_argument("--out", default="reports/echo_r_diagnostics.json")
    args = ap.parse_args()

    rows = load(args.traces)
    if not rows:
        raise SystemExit("No trace rows matched.")
    report = {
        "n_trace_rows": len(rows),
        "models": sorted({r.get("model") for r in rows if r.get("model")}),
        "A_check_probe_alignment": table_alignment(rows),
        "B_risk_ranking_quality": table_ranking(rows),
        "C_residual_attribution_forcerw": table_residual(rows),
        "D_oracle_misalignment": table_oracle(rows),
    }
    out_path = resolve_project_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nwrote {to_project_relative(out_path)}")


if __name__ == "__main__":
    main()
