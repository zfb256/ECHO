from __future__ import annotations

from pathlib import Path


NON_RESEARCH_ECHO_R_TOKENS = (
    "_mock",
    "_limit",
    "smoke",
    "partial",
    "nli-noop",
    "diagnostic",
    "_tmp",
)


def is_research_echo_r_path(path: str | Path) -> bool:
    stem = Path(path).stem.lower()
    return not any(token in stem for token in NON_RESEARCH_ECHO_R_TOKENS)
