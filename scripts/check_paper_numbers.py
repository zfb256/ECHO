#!/usr/bin/env python3
"""Scan manuscript numbers against reports; --assert checks registered bindings.

Unregistered numbers require manual verification. Also checks decimal precision.
Requires the separately distributed paper directory.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# LaTeX constructs whose numbers are structural, not empirical.
SKIP_PATTERNS = [
    re.compile(r"\\(?:label|ref|cite[a-z]*|input|includegraphics|usepackage|documentclass)\{[^}]*\}"),
    re.compile(r"\\(?:setlength|tabcolsep|linenumbersep|columnwidth|textwidth)[^\n]*"),
    re.compile(r"(?<!\\)%.*$", re.MULTILINE),         # unescaped LaTeX comments
    re.compile(r"\\begin\{tabular\}\{[^}]*\}"),
]
NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(?![\w])")

# Generic scanning cannot know whether a small integer is structural or a result
# (notably, 5 and 7 are model counts here), so skip nothing silently.
STRUCTURAL: set[str] = set()


def strip_latex(text: str) -> str:
    # LaTeX writes thousands separators as 4{,}000; join them back before the
    # number regex runs, or every such figure is read as a bare "000".
    text = text.replace("{,}", ",")
    for pat in SKIP_PATTERNS:
        text = pat.sub(" ", text)
    return text


def norm(tok: str) -> str:
    return tok.replace(",", "")


def flatten_json(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_json(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_json(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def load_reports(reports_dir: Path) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        try:
            reports[path.name] = flatten_json(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return reports


def half_up(value: float, places: int) -> str:
    """Round the way a person writing a table rounds: 0.3075 -> 0.308.

    Python's format() rounds the binary double, so f"{0.3075:.3f}" is "0.307".
    Both conventions are defensible; a paper must pick one and apply it
    everywhere, so the two are tracked separately.
    """
    return str(Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def numeric_index(reports: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """value-as-string -> list of "report.json:/path" locations, at several roundings."""
    index: dict[str, list[str]] = defaultdict(list)
    for name, flat in reports.items():
        for path, value in flat.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            where = f"{name}:{path}"
            forms = {repr(value), f"{value}"}
            if isinstance(value, float):
                for places in range(0, 6):
                    forms.add(f"{value:.{places}f}")
                    forms.add(f"{value * 100:.{places}f}")   # percentage form
            else:
                forms.add(str(value))
            for form in forms:
                cleaned = form.rstrip("0").rstrip(".") if "." in form else form
                index[form].append(where)
                index[cleaned].append(where)
    return index


def halfup_numeric_index(reports: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for name, flat in reports.items():
        for path, value in flat.items():
            if isinstance(value, bool) or not isinstance(value, float):
                continue
            for places in range(0, 6):
                index[half_up(value, places)].append(f"{name}:{path}")
                index[half_up(value * 100, places)].append(f"{name}:{path}")
    return index


def match(token: str, index: dict[str, list[str]]) -> list[str]:
    value = norm(token)
    hits = index.get(value, [])
    if hits:
        return hits
    cleaned = value.rstrip("0").rstrip(".") if "." in value else value
    return index.get(cleaned, [])


def scan(args: argparse.Namespace) -> int:
    sections = sorted((ROOT / args.sections).glob("*.tex"))
    if not sections:
        raise SystemExit(f"no .tex files under {args.sections}")
    reports = load_reports(ROOT / args.reports)
    if not reports:
        raise SystemExit(f"no .json reports under {args.reports}")
    index = numeric_index(reports)

    halfup_index = halfup_numeric_index(reports)
    matched: list[tuple[str, int, str, str]] = []
    halfup_only: list[tuple[str, int, str, str]] = []
    unverified: list[tuple[str, int, str, str]] = []
    decimals: dict[int, set[str]] = defaultdict(set)

    for path in sections:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = strip_latex(raw)
            for token in NUMBER_RE.findall(line):
                if token in STRUCTURAL:
                    continue
                context = raw.strip()
                context = context[:96] + ("…" if len(context) > 96 else "")
                hits = match(token, index)
                if "." in token:
                    decimals[len(token.split(".")[1])].add(token)
                if hits:
                    matched.append((rel, lineno, token, hits[0]))
                    continue
                hu = halfup_index.get(norm(token), [])
                if hu:
                    # Under the declared half-up convention these ARE matches;
                    # under the python convention they are discrepancies.
                    if args.rounding == "half-up":
                        matched.append((rel, lineno, token, hu[0]))
                    else:
                        halfup_only.append((rel, lineno, token, hu[0]))
                else:
                    unverified.append((rel, lineno, token, context))

    print(f"Sections scanned : {len(sections)}")
    print(f"Reports loaded   : {len(reports)}")
    print(f"Numbers matched  : {len(matched)}")
    print(f"Rounding convention: {args.rounding}")
    print(f"Numbers off-convention: {len(halfup_only)}")
    print(f"Numbers UNVERIFIED: {len(unverified)}")
    print()
    if halfup_only:
        print("OFF-CONVENTION (matches only under round-half-up, not the declared convention)")
        print("Either restate the value, or switch --rounding and say so in the paper.")
        print("-" * 100)
        for rel, lineno, token, where in halfup_only:
            print(f"  {rel}:{lineno}  {token:>12}   <- {where}")
        print()
    if unverified and not args.quiet:
        print("UNVERIFIED (not found in any report - confirm each is derived, structural, or fix it)")
        print("-" * 100)
        for rel, lineno, token, context in unverified:
            print(f"  {rel}:{lineno}  {token:>12}   {context}")
        print()
    if args.show_matched:
        print("MATCHED")
        print("-" * 100)
        for rel, lineno, token, where in matched:
            print(f"  {rel}:{lineno}  {token:>12}   <- {where}")
        print()
    if len(decimals) > 1:
        print("DECIMAL-PLACE SPREAD")
        for places in sorted(decimals):
            sample = sorted(decimals[places])[:8]
            print(f"  {places} dp: {len(decimals[places]):>3} distinct  e.g. {', '.join(sample)}")
        print()
    return 1 if (args.strict and (unverified or halfup_only)) else 0


def run_assertions(args: argparse.Namespace) -> int:
    registry_path = ROOT / args.registry
    if not registry_path.exists():
        raise SystemExit(
            f"registry not found: {args.registry}\n"
            "Create it as a JSON list of bindings, e.g.\n"
            '  [{"quantity": "mean CCR", "tex": "0.264", '
            '"report": "zh_full_metrics.json", "path": "/decision/mean_ccr", '
            '"places": 3, "tex_file": "paper/sections/00_abstract.tex"}]')
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    reports = load_reports(ROOT / args.reports)

    failures: list[str] = []
    checked = 0
    for entry in registry:
        name = entry.get("quantity", "?")
        report = entry["report"]
        flat = reports.get(report)
        if flat is None:
            failures.append(f"{name}: report {report} not found")
            continue
        if entry["path"] not in flat:
            failures.append(f"{name}: path {entry['path']} not in {report}")
            continue
        actual = flat[entry["path"]]
        places = entry.get("places")
        if places is not None and isinstance(actual, float):
            rendered = (
                half_up(actual, places) if args.rounding == "half-up"
                else f"{actual:.{places}f}"
            )
        else:
            rendered = str(actual)
        if rendered != entry["tex"]:
            failures.append(
                f"{name}: paper says {entry['tex']} but {report}{entry['path']} is {rendered}")
        tex_file = entry.get("tex_file")
        if not tex_file:
            failures.append(f"{name}: binding has no tex_file")
        else:
            source = ROOT / tex_file
            if not source.exists():
                failures.append(f"{name}: tex_file {tex_file} not found")
            else:
                tokens = {norm(t) for t in NUMBER_RE.findall(strip_latex(source.read_text(encoding="utf-8")))}
                if norm(entry["tex"]) not in tokens:
                    failures.append(f"{name}: {entry['tex']} does not occur in {tex_file}")
        checked += 1

    print(json.dumps({
        "ok": not failures,
        "checked": checked,
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sections", default="paper/sections")
    p.add_argument("--reports", default="reports_zh")
    p.add_argument("--registry", default="paper/number_registry.json")
    p.add_argument("--assert", dest="do_assert", action="store_true",
                   help="Check the hard bindings in --registry and exit non-zero on mismatch.")
    p.add_argument("--show-matched", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Suppress the unverified listing.")
    p.add_argument("--rounding", choices=("half-up", "python"), default="half-up",
                   help="The paper's declared rounding convention. half-up is what a "
                        "person writing a table does (0.3075 -> 0.308); python is what "
                        "format() does on the binary double (0.3075 -> 0.307). Pick one, "
                        "state it in the paper, and keep it here.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero when any number is unverified.")
    args = p.parse_args()
    return run_assertions(args) if args.do_assert else scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
