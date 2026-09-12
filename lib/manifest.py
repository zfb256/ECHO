from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import to_project_relative


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash a model directory by relative path and file content."""
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"not a directory: {path}")
    digest = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"directory contains no files: {path}")
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def require_task_binding(output: dict[str, Any], task: dict[str, Any]) -> None:
    """Reject an output whose copied metadata does not exactly match its task."""
    task_id = task.get("task_id")
    expected_hash = object_sha256(task)
    if output.get("task_sha256") != expected_hash:
        raise ValueError(
            f"output task fingerprint mismatch for task_id={task_id}: "
            f"output={output.get('task_sha256')!r} current={expected_hash!r}"
        )
    for field, expected in task.items():
        if field in {"prompt", "messages"}:
            continue
        if output.get(field) != expected:
            raise ValueError(
                f"output metadata mismatch for task_id={task_id}: "
                f"{field} output={output.get(field)!r} task={expected!r}"
            )


def load_output_manifest(output_path: Path, tasks_path: Path) -> dict[str, Any]:
    manifest_path = output_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError(f"missing output manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != file_sha256(output_path):
        raise ValueError(f"output SHA-256 does not match manifest: {output_path}")
    if manifest.get("tasks_sha256") != file_sha256(tasks_path):
        raise ValueError(f"task-file SHA-256 does not match output manifest: {tasks_path}")
    return manifest


def require_model_binding(output: dict[str, Any], manifest: dict[str, Any]) -> None:
    model = output.get("model")
    artifact = (manifest.get("model_artifacts") or {}).get(model, {})
    if output.get("model_artifact_sha256") != artifact.get("directory_sha256"):
        raise ValueError(
            f"output model-weight fingerprint mismatch for task_id={output.get('task_id')}"
        )


def write_manifest(path: Path, config_path: Path, config: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": config.get("random_seed"),
        "hf_endpoint": config.get("hf_endpoint"),
        "config_path": to_project_relative(config_path),
        "config_sha256": file_sha256(config_path),
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
