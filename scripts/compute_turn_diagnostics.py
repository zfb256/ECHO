from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from artifacts import is_research_echo_r_path  # noqa: E402
from config import resolve_project_path, to_project_relative  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize per-turn contagion from ECHO-R trace files.")
    p.add_argument("--traces", required=True, nargs="+")
    p.add_argument("--output", default="reports/echo_r_turn_diagnostics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files: list[str] = []
    for pat in args.traces:
        files.extend(f for f in glob.glob(str(resolve_project_path(pat))) if is_research_echo_r_path(f))
    groups: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for f in sorted(files):
        for line in Path(f).open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            groups[(r["policy"], r["budget_frac"], int(r["turn"]))].append(r)
    rows = []
    for (policy, budget, turn), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = len(rs)
        rows.append({
            "policy": policy,
            "budget": budget,
            "turn": turn,
            "n": n,
            "contagion_rate": sum(1 for r in rs if r.get("probed_contagious")) / n if n else None,
            "served_rate": sum(1 for r in rs if r.get("probed_served")) / n if n else None,
            "never_checked_rate": sum(1 for r in rs if not r.get("probed_ever_checked")) / n if n else None,
        })
    report = {
        "files": [to_project_relative(f) for f in sorted(files)],
        "rows": rows,
        "note": "Turn is zero-indexed in the trace after the initial injected answer.",
    }
    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"wrote": to_project_relative(out), "n_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
