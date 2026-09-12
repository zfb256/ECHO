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

# Post-hoc analysis of ECHO-R eval reports (no GPU). Two outputs that turn "no single policy wins" into a
# defensible "budget-constrained intervention FRAMEWORK" story:
#   (1) Budget vs contagion curve per policy (and reduction vs no_op).
#   (2) An ADAPTIVE policy SELECTOR evaluated leave-one-model-out: at each budget, pick the policy that is
#       best on AVERAGE over the OTHER models, apply it to the held-out model, and compare to (a) no_op,
#       (b) the best single FIXED policy, (c) the per-model ORACLE-best (the achievable floor). If the
#       selector beats the best fixed policy and approaches the oracle-best, a regime-aware framework —
#       not a fixed policy — is the contribution.

DEPLOYABLE = ["no_op", "reminder", "uniform_check", "echo_r"]  # selectable (excl. diagnostic/oracle upper bound)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc ECHO-R analysis: budget curve + adaptive policy selector.")
    p.add_argument(
        "--reports",
        default="reports/echo_r_eval_injected_multi4_forcerw-grounded_*.json",
        help="Glob of eval reports for one intervention arm, normally one file per model.",
    )
    p.add_argument("--arm", default="injected_multi4_forcerw")
    p.add_argument("--output", default=None, help="Defaults to reports/echo_r_analysis_<arm>.json")
    return p.parse_args()


def load(reports_glob: str) -> dict:
    """-> contagion[model][budget][policy] = exact per-model rate."""
    cont: dict = defaultdict(lambda: defaultdict(dict))
    spent: dict = defaultdict(lambda: defaultdict(dict))
    counts: dict = defaultdict(lambda: defaultdict(dict))
    seen: dict[tuple[str, float, str], str] = {}

    files = [f for f in sorted(glob.glob(str(resolve_project_path(reports_glob)))) if is_research_echo_r_path(f)]
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("mock") is True or d.get("nli", {}).get("nli_required") is False:
            continue
        model = d.get("model") or Path(f).stem
        for r in d.get("results", []):
            b = r.get("budget_frac")
            key = (model, b, r["policy"])
            if key in seen:
                raise SystemExit(
                    "Duplicate ECHO-R result key while loading reports: "
                    f"model={model} budget={b} policy={r['policy']} in both "
                    f"{to_project_relative(seen[key])} and {to_project_relative(f)}. "
                    "Pass a glob for exactly one intervention arm."
                )
            seen[key] = f
            if "contagious_probes" in r and "total_probes" in r and r["total_probes"]:
                rate = r["contagious_probes"] / r["total_probes"]
                counts[model][b][r["policy"]] = {
                    "contagious_probes": r["contagious_probes"],
                    "total_probes": r["total_probes"],
                }
            else:
                rate = r["contagion_rate"]
                counts[model][b][r["policy"]] = None
            cont[model][b][r["policy"]] = rate
            spent[model][b][r["policy"]] = r.get("budget_spent")
    return {"contagion": cont, "spent": spent, "counts": counts, "files": files}


def budget_curve(cont: dict, counts: dict) -> list[dict]:
    """Pooled raw-count contagion per policy per budget + reduction vs no_op."""
    budgets = sorted({b for m in cont for b in cont[m]})
    policies = sorted({p for m in cont for b in cont[m] for p in cont[m][b]})
    rows = []
    for b in budgets:
        noop = _pooled_rate(counts, cont, b, "no_op")
        for p in policies:
            vals = [cont[m][b].get(p) for m in cont if b in cont[m] and p in cont[m][b]]
            pooled = _pooled_rate(counts, cont, b, p)
            rows.append({"budget": b, "policy": p, "mean_contagion": _r(pooled),
                         "reduction_vs_no_op": _r(noop - pooled) if (noop is not None and pooled is not None) else None,
                         "n_models": len(vals),
                         "model_mean_contagion": _r(_mean(vals))})
    return rows


def adaptive_selector(cont: dict) -> list[dict]:
    """Leave-one-model-out selector among DEPLOYABLE policies, per budget."""
    models = list(cont)
    budgets = sorted({b for m in cont for b in cont[m]})
    out = []
    for b in budgets:
        sel_vals, fixed_best_vals, noop_vals, oracle_vals = [], [], [], []
        picks = {}
        # best single FIXED policy at this budget (over all models)
        fixed_means = {p: _mean([cont[m][b].get(p) for m in models if b in cont[m] and p in cont[m][b]])
                       for p in DEPLOYABLE}
        fixed_best_policy = min((p for p in fixed_means if fixed_means[p] is not None), key=lambda p: fixed_means[p])
        for held in models:
            if b not in cont[held]:
                continue
            others = [m for m in models if m != held and b in cont[m]]
            # policy best on average over OTHER models
            avg = {p: _mean([cont[m][b].get(p) for m in others if p in cont[m][b]]) for p in DEPLOYABLE}
            avg = {p: v for p, v in avg.items() if v is not None}
            pick = min(avg, key=lambda p: avg[p]) if avg else "no_op"
            picks[held] = pick
            sel_vals.append(cont[held][b].get(pick))
            fixed_best_vals.append(cont[held][b].get(fixed_best_policy))
            noop_vals.append(cont[held][b].get("no_op"))
            oracle_vals.append(min(cont[held][b].get(p) for p in DEPLOYABLE if p in cont[held][b]))
        out.append({
            "budget": b,
            "no_op": _r(_mean(noop_vals)),
            "best_fixed_policy": fixed_best_policy, "best_fixed_contagion": _r(_mean(fixed_best_vals)),
            "adaptive_selector_contagion": _r(_mean(sel_vals)),
            "per_model_best_deployable_contagion": _r(_mean(oracle_vals)),
            "selector_picks": picks,
        })
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pooled_rate(counts, cont, budget, policy):
    c = t = 0
    missing_counts = False
    vals = []
    for m in cont:
        if budget not in cont[m] or policy not in cont[m][budget]:
            continue
        vals.append(cont[m][budget][policy])
        row = counts.get(m, {}).get(budget, {}).get(policy)
        if row:
            c += row["contagious_probes"]
            t += row["total_probes"]
        else:
            missing_counts = True
    if t and not missing_counts:
        return c / t
    return _mean(vals)


def _r(x):
    return round(x, 4) if isinstance(x, (int, float)) else x


def main() -> None:
    args = parse_args()
    data = load(args.reports)
    cont = data["contagion"]
    if not cont:
        raise SystemExit(f"No reports matched {args.reports}")
    curve = budget_curve(cont, data["counts"])
    sel = adaptive_selector(cont)
    report = {"arm": args.arm, "models": list(cont), "files": [to_project_relative(f) for f in data["files"]],
              "budget_curve": curve, "adaptive_selector": sel}
    out = resolve_project_path(args.output) if args.output else resolve_project_path("reports") / f"echo_r_analysis_{args.arm}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": to_project_relative(out), "models": list(cont)}, ensure_ascii=False))
    print("\n=== ADAPTIVE SELECTOR (leave-one-model-out) vs no_op / best-fixed / oracle-best ===")
    for s in sel:
        print(f"  budget={s['budget']}: no_op={s['no_op']}  best_fixed({s['best_fixed_policy']})={s['best_fixed_contagion']}  "
              f"ADAPTIVE={s['adaptive_selector_contagion']}  per_model_best_deployable={s['per_model_best_deployable_contagion']}")
    print("\n=== BUDGET CURVE (mean contagion; reduction vs no_op) ===")
    for r in curve:
        if r["policy"] == "no_op":
            continue
        print(f"  b={r['budget']} {r['policy']:24s} contagion={r['mean_contagion']}  reduction={r['reduction_vs_no_op']}")


if __name__ == "__main__":
    main()
