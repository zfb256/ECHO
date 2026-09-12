from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import apply_hf_environment, ensure_dirs, load_config, resolve_project_path
from jsonl import write_jsonl
from zh_quality import chinese_dialogue_ok, normalize_zh_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download configured datasets from Hugging Face.")
    parser.add_argument("--config", default="configs/zh_study.json")
    parser.add_argument("--dataset", action="append", help="Dataset key to download. Defaults to all configured datasets.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for local smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_hf_environment(config)
    ensure_dirs(config)
    if not os.environ.get("HF_TOKEN"):
        print("Warning: HF_TOKEN is not set. Public datasets should work; private or gated datasets will fail.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install -r requirements.txt") from exc

    selected = set(args.dataset or [])
    raw_dir = resolve_project_path(config["paths"]["raw_dir"])

    for spec in config["datasets"]:
        if selected and spec["key"] not in selected:
            continue

        print(f"Downloading {spec['key']} from {spec['hf_name']} via {config['hf_endpoint']}")
        dataset_kwargs = {}
        if spec.get("hf_config"):
            dataset_kwargs["name"] = spec["hf_config"]
        # Some dataset releases require a loading script on datasets 2.x.
        try:
            ds = load_dataset(spec["hf_name"], **dataset_kwargs, split=spec["split"], trust_remote_code=True)
        except TypeError:
            # very old datasets without the trust_remote_code kwarg
            ds = load_dataset(spec["hf_name"], **dataset_kwargs, split=spec["split"])
        # Smoke tests inspect the first N source rows. Formal preparation instead walks a
        # deterministic shuffle and stops as soon as `sample_size` usable rows survive the
        # filters; this avoids exporting millions of LCCC rows merely to sample 400 later.
        target_retained = None if args.limit is not None else int(spec.get("sample_size", 0) or 0)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        elif target_retained:
            ds = ds.shuffle(seed=int(config.get("random_seed", 0)))

        rows = []
        excluded: dict[str, int] = {}
        scanned = 0
        for idx, row in enumerate(ds):
            scanned += 1
            payload = dict(row)
            if spec.get("language") == "zh":
                payload = normalize_zh_value(payload)
                ok, reason = chinese_dialogue_ok(payload)
                if not ok:
                    excluded[reason] = excluded.get(reason, 0) + 1
                    continue
            rows.append({
                "source_dataset": spec["key"],
                "hf_name": spec["hf_name"],
                "split": spec["split"],
                "row_id": idx,
                "payload": payload,
            })
            if target_retained and len(rows) >= target_retained:
                break
        out_path = raw_dir / f"{spec['key']}_{spec['split']}.jsonl"
        write_jsonl(out_path, rows)
        print(json.dumps({
            "wrote": str(out_path),
            "rows": len(rows),
            "source_rows_scanned": scanned,
            "excluded": excluded,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
