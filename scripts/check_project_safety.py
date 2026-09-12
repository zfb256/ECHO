from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]

ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\(?:Users|Data|AAAI|tmp|Temp)\\"),
    re.compile(r"(?<![\w.-])/(home|root|Users|mnt|data|tmp)/"),
]

SCAN_SUFFIXES = {
    ".py", ".json", ".toml", ".md", ".txt", ".example",
    ".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".bbl", ".blg",
}
SKIP_DIRS = {".git", ".cache", ".venv", "venv", "data", "datasets", "models", "runs", "logs", "exclude_use"}
SKIP_FILES = {"check_project_safety.py"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan project files for portability and leakage risks.")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix in SCAN_SUFFIXES or path.name == ".gitignore":
            yield path


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    failures = []

    for path in iter_files(root):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(root)
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"Potential secret in {rel}")
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                failures.append(f"Potential absolute path in {rel}")

    if failures:
        print("Safety check failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("Safety check passed: no obvious secrets or absolute local paths found.")


if __name__ == "__main__":
    main()
