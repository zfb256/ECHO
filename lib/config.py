from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_project_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def resolve_project_path(path: str | Path) -> Path:
    # Accept absolute and relative paths only when they resolve inside the workspace.
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root() / candidate).resolve()
    workspace = workspace_root().resolve()
    if workspace not in (resolved, *resolved.parents):
        raise ValueError(f"Path escapes workspace root: {candidate}")
    return resolved


def to_project_relative(path: str | Path) -> str:
    """POSIX-relative path string for manifests/outputs.

    Artifacts (manifests, reports) must NOT embed an absolute author path: a Windows
    drive path or a Unix home path leaks the machine layout and the OS username into
    anything shared for reproducibility. We store paths relative to the project (or
    workspace) root instead; falls back to the bare filename if the path is outside the
    workspace.
    """
    p = Path(path).resolve()
    for base in (project_root().resolve(), workspace_root().resolve()):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            continue
    return p.name


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ("datasets_dir", "models_dir", "raw_dir", "interim_dir", "processed_dir", "runs_dir", "reports_dir"):
        path_value = config["paths"][key]
        resolve_project_path(path_value).mkdir(parents=True, exist_ok=True)


def apply_hf_environment(config: dict[str, Any]) -> None:
    # The endpoint can be overridden in the study config.
    endpoint = config.get("hf_endpoint", "https://huggingface.co")
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    model_cache = resolve_project_path(config["paths"]["models_dir"]) / "huggingface_cache"
    dataset_cache = resolve_project_path(config["paths"]["datasets_dir"]) / "huggingface_cache"
    os.environ.setdefault("HF_HOME", str(model_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(dataset_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(model_cache / "transformers"))
