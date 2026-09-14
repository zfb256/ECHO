from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import apply_hf_environment, ensure_dirs, load_config, resolve_project_path, to_project_relative
from jsonl import read_jsonl, write_jsonl
from manifest import (
    directory_sha256,
    file_sha256,
    load_output_manifest,
    object_sha256,
    package_versions,
    require_task_binding,
    write_manifest,
)


# Run local injected and self-induced inference; record decoding settings in manifests.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM inference runner.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--tasks", required=True, help="Path to a *_tasks.jsonl file (project-relative).")
    p.add_argument("--output", default=None, help="Defaults to the matching *_outputs.jsonl next to --tasks.")
    p.add_argument("--model", action="append", help="Restrict to these model names (default: all models present in tasks).")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N tasks (smoke/partial).")
    p.add_argument("--max-tokens", type=int, default=None, help="Override max new tokens (default from config or 256).")
    p.add_argument("--temperature", type=float, default=None, help="Override sampling temperature (default 0.0 = greedy).")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--no-chat-template", action="store_true", help="Feed the raw prompt instead of the model chat template.")
    p.add_argument("--dry-run", action="store_true", help="Validate wiring (paths/grouping) without importing vLLM or generating.")
    return p.parse_args()


def default_output_path(tasks_path: Path) -> Path:
    name = tasks_path.name
    if name == "generation_tasks.jsonl":
        out_name = "model_outputs.jsonl"
    elif name.endswith("_tasks.jsonl"):
        out_name = name[: -len("_tasks.jsonl")] + "_outputs.jsonl"
    else:
        out_name = tasks_path.stem + "_outputs.jsonl"
    return tasks_path.with_name(out_name)


def build_text(task: dict, tokenizer: Any, use_chat_template: bool) -> str:
    """Apply structured chat roles or use the required flat prompt when templates are unavailable."""
    prompt = task["prompt"]
    if not use_chat_template or tokenizer is None or getattr(tokenizer, "chat_template", None) is None:
        return prompt
    messages = task.get("messages") or [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def model_runtime_profile(model_path: Path) -> dict[str, Any]:
    """Read the few engine settings that are part of a model's local contract."""
    model_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    generation_path = model_path / "generation_config.json"
    generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path.exists() else {}
    eos = generation.get("eos_token_id", model_config.get("eos_token_id"))
    stop_token_ids = eos if isinstance(eos, list) else ([eos] if isinstance(eos, int) else None)
    is_glm = model_config.get("model_type") == "glm"
    return {
        "backend": "transformers" if is_glm else "vllm",
        "tokenizer_mode": "auto" if is_glm else "slow",
        "use_fast_tokenizer": is_glm,
        "stop_token_ids": stop_token_ids,
    }


def rows_in_task_order(tasks: list[dict[str, Any]], previous: dict[str, dict[str, Any]],
                       current: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-model invocations without dropping rows from earlier models."""
    merged = {**previous, **current}
    return [merged[t["task_id"]] for t in tasks if t["task_id"] in merged]


def free_gpu_memory() -> None:
    """Release GPU and distributed state; one model per process provides reliable isolation."""
    import gc

    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    for dotted in (
        "vllm.distributed.parallel_state",
        "vllm.distributed",
    ):
        try:
            mod = __import__(dotted, fromlist=["destroy_model_parallel"])
            destroy = getattr(mod, "destroy_model_parallel", None)
            if destroy is not None:
                destroy()
                break
        except Exception:
            continue


def main() -> None:
    # Use V0 to avoid optional V1 dependencies incompatible with this CUDA setup.
    os.environ.setdefault("VLLM_USE_V1", "0")

    args = parse_args()
    config = load_config(args.config)
    apply_hf_environment(config)
    ensure_dirs(config)
    config_path = resolve_project_path(args.config)

    tasks_path = resolve_project_path(args.tasks)
    out_path = resolve_project_path(args.output) if args.output else default_output_path(tasks_path)
    manifest_path = out_path.with_suffix(".manifest.json")
    models_dir = resolve_project_path(config["paths"]["models_dir"])

    seed = int(config.get("random_seed", 0))
    max_tokens = args.max_tokens if args.max_tokens is not None else int(config["pilot"].get("inference", {}).get("max_tokens", 256))

    # Generation arms are sampled; self-induced stage 1 is greedy.
    ccr_cfg = config.get("pilot", {}).get("ccr", {})
    temp_gen = float(ccr_cfg.get("decoding_temperature_generation", 0.0))
    temp_si_stage1 = float(ccr_cfg.get("decoding_temperature_selfinduced_stage1", 0.0))

    def temperature_for(task: dict) -> float:
        if args.temperature is not None:
            return args.temperature  # explicit CLI override wins everywhere
        cond = task.get("condition", "")
        if cond == "self_induced_stage1":
            return temp_si_stage1
        return temp_gen

    all_tasks = list(read_jsonl(tasks_path))
    task_ids = [t.get("task_id") for t in all_tasks]
    if any(not tid for tid in task_ids):
        raise SystemExit(f"Task file contains a missing/empty task_id: {tasks_path}")
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit(f"Task file contains duplicate task_id values: {tasks_path}")
    required = {"task_id", "pair_id", "condition", "model", "prompt"}
    for task in all_tasks:
        missing = required - set(task)
        if missing:
            raise SystemExit(f"Task {task.get('task_id')} missing required fields: {sorted(missing)}")
    all_tasks_by_id = {t["task_id"]: t for t in all_tasks}

    tasks = all_tasks
    wanted = set(args.model) if args.model else None
    if wanted is not None:
        tasks = [task for task in tasks if task.get("model") in wanted]
        if not tasks:
            raise SystemExit(f"No tasks match --model {sorted(wanted)}")
    if args.limit is not None:
        # Prevent limited runs from truncating existing full outputs.
        safe_name = any(token in out_path.name.lower() for token in ("limit", "smoke", "tmp", "partial"))
        if not safe_name:
            raise SystemExit(
                "--limit is for smoke/partial runs only. Pass an explicit output filename containing "
                "'limit', 'smoke', 'tmp', or 'partial' so a full research output cannot be truncated."
            )
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"No tasks in {tasks_path}")

    models_in_order: list[str] = []
    for t in tasks:
        if t["model"] not in models_in_order:
            models_in_order.append(t["model"])
    run_models = [m for m in models_in_order if (wanted is None or m in wanted)]
    if not run_models:
        raise SystemExit(f"No models remain after filtering {sorted(wanted) if wanted else ''}")

    temperatures_by_condition: dict[str, float] = {}
    for t in tasks:
        condition = t.get("condition", "")
        temperatures_by_condition.setdefault(condition, temperature_for(t))

    print(json.dumps({
        "tasks": to_project_relative(tasks_path), "output": to_project_relative(out_path), "num_tasks": len(tasks),
        "models": run_models, "max_tokens": max_tokens, "temperatures_by_condition": temperatures_by_condition,
        "model_paths": {m: to_project_relative(models_dir / m) for m in run_models},
    }, ensure_ascii=False))

    missing_model_paths = [
        str(models_dir / model) for model in run_models
        if not (models_dir / model).exists()
    ]
    if missing_model_paths:
        raise SystemExit(
            "Model path(s) not found before inference: "
            + ", ".join(missing_model_paths)
            + ". Download the configured weights or create verified symlinks first."
        )

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "note": "task schema and selected model paths validated; no generation performed",
        }, ensure_ascii=False))
        return

    prior_manifest: dict[str, Any] = {}
    if out_path.exists():
        if not manifest_path.exists():
            raise SystemExit(
                f"Refusing to resume {out_path}: its manifest is missing. "
                "Use a new output path or archive the unverifiable output."
            )
        try:
            prior_manifest = load_output_manifest(out_path, tasks_path)
        except ValueError as exc:
            raise SystemExit(f"Refusing to resume unverifiable output: {exc}") from exc

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install vllm (and run on a GPU host).") from exc
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        AutoModelForCausalLM = None  # type: ignore
        AutoTokenizer = None  # type: ignore

    use_chat = not args.no_chat_template
    model_artifacts = dict(prior_manifest.get("model_artifacts") or {})
    model_repositories = config.get("pilot", {}).get("model_repositories", {})
    software_versions = package_versions(("vllm", "transformers", "torch"))
    hardware: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        software_versions["cuda"] = str(torch.version.cuda)
        software_versions["cudnn"] = str(torch.backends.cudnn.version())
        if torch.cuda.is_available():
            hardware = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
                "gpu_count_visible": torch.cuda.device_count(),
            }
    except Exception:
        software_versions["cuda"] = None
        software_versions["cudnn"] = None

    def task_seed(task: dict) -> int:
        # Use task-specific seeds for distinct stochastic draws from identical clean prompts.
        h = int(hashlib.sha256(task["task_id"].encode("utf-8")).hexdigest()[:8], 16)
        return (seed + h) % (2**31 - 1)

    def make_sampling(temperature: float, s: int, stop_token_ids: list[int] | None) -> Any:
        return SamplingParams(
            temperature=temperature, top_p=1.0, max_tokens=max_tokens, seed=s,
            **({"stop_token_ids": stop_token_ids} if stop_token_ids else {}),
        )

    results: dict[str, dict[str, Any]] = {}  # task_id -> output row
    started = time.time()
    for model in run_models:
        model_path = models_dir / model
        if not model_path.exists():
            raise SystemExit(f"Model path not found: {model_path} (place weights under models/ or symlink there).")
        model_tasks = [t for t in tasks if t["model"] == model]
        runtime = model_runtime_profile(model_path)
        # Hash local model bytes rather than mutable repository identifiers.
        model_artifacts[model] = {
            "repository": model_repositories.get(model),
            "model_path": to_project_relative(model_path),
            "directory_sha256": directory_sha256(model_path),
            "runtime_profile": runtime,
        }

        checkpoint = out_path.with_name(f"{out_path.stem}.{model}.partial.jsonl")
        checkpoint_meta = checkpoint.with_suffix(checkpoint.suffix + ".meta.json")
        binding = {
            "config_sha256": file_sha256(config_path),
            "tasks_sha256": file_sha256(tasks_path),
            "selected_tasks_sha256": object_sha256(model_tasks),
            "runner_sha256": file_sha256(Path(__file__)),
            "model_artifact_sha256": model_artifacts[model]["directory_sha256"],
            "runtime_profile": runtime,
        }
        if checkpoint.exists():
            if not checkpoint_meta.exists() or json.loads(
                checkpoint_meta.read_text(encoding="utf-8")
            ) != binding:
                raise SystemExit(f"checkpoint binding mismatch: {checkpoint}")
            old = list(read_jsonl(checkpoint))
            if len({r.get("task_id") for r in old}) != len(old):
                raise SystemExit(f"checkpoint contains duplicate task_id values: {checkpoint}")
            for row in old:
                task = all_tasks_by_id.get(row.get("task_id"))
                if task is None or row.get("model") != model:
                    raise SystemExit(f"checkpoint contains an unexpected task: {checkpoint}")
                require_task_binding(row, task)
                if row.get("model_artifact_sha256") != binding["model_artifact_sha256"]:
                    raise SystemExit(f"checkpoint model-weight binding mismatch: {checkpoint}")
                results[row["task_id"]] = row
        else:
            if checkpoint_meta.exists():
                raise SystemExit(f"checkpoint metadata exists without scores: {checkpoint_meta}")
            checkpoint_meta.write_text(
                json.dumps(binding, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        model_tasks = [t for t in model_tasks if t["task_id"] not in results]
        if not model_tasks:
            continue

        def record(task: dict[str, Any], text: str, prompt_tokens: int,
                   completion_tokens: int, finish_reason: str, latency: float) -> None:
            row = {k: v for k, v in task.items() if k not in ("prompt", "messages")}
            row.update({
                "response": text.strip(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason,
                "latency_seconds": round(latency, 4),
                "task_sha256": object_sha256(task),
                "model_artifact_sha256": model_artifacts[model]["directory_sha256"],
                "decoding_temperature": temperature_for(task),
            })
            results[task["task_id"]] = row
            with checkpoint.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        tokenizer = None
        if use_chat and AutoTokenizer is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True, use_fast=runtime["use_fast_tokenizer"]
            )

        prompts = [build_text(t, tokenizer, use_chat) for t in model_tasks]
        if runtime["backend"] == "transformers":
            if tokenizer is None or AutoModelForCausalLM is None:
                raise SystemExit("transformers backend requires transformers and a chat tokenizer")
            import torch  # type: ignore

            hf_model = AutoModelForCausalLM.from_pretrained(
                str(model_path), trust_remote_code=True,
                torch_dtype=getattr(torch, args.dtype), device_map="cuda",
            ).eval()
            for task, prompt in zip(model_tasks, prompts):
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                temperature = temperature_for(task)
                sample_seed = task_seed(task)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                t0 = time.time()
                token_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    **({"temperature": temperature} if temperature > 0 else {}),
                    **({"eos_token_id": runtime["stop_token_ids"]} if runtime["stop_token_ids"] else {}),
                )[0, inputs["input_ids"].shape[1]:]
                record(
                    task, tokenizer.decode(token_ids, skip_special_tokens=True),
                    int(inputs["input_ids"].shape[1]), int(token_ids.numel()),
                    "length" if token_ids.numel() >= max_tokens else "stop", time.time() - t0,
                )
            del hf_model
        else:
            llm = LLM(
                model=str(model_path), tokenizer_mode=runtime["tokenizer_mode"], dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, seed=seed,
                **({"max_model_len": args.max_model_len} if args.max_model_len else {}),
            )
            sampling = [
                make_sampling(temperature_for(t), task_seed(t), runtime["stop_token_ids"])
                for t in model_tasks
            ]
            t0 = time.time()
            outputs = llm.generate(prompts, sampling)
            if len(outputs) != len(model_tasks):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(model_tasks)} tasks for model={model}"
                )
            latency = (time.time() - t0) / max(1, len(model_tasks))
            for task, out in zip(model_tasks, outputs):
                gen = out.outputs[0]
                record(
                    task, gen.text, len(out.prompt_token_ids), len(gen.token_ids),
                    gen.finish_reason, latency,
                )
            del llm

        free_gpu_memory()

    # Merge by task ID: replace duplicates with new results and drop obsolete tasks.
    merged: dict[str, dict[str, Any]] = {}
    if out_path.exists():
        for prev in read_jsonl(out_path):
            tid = prev.get("task_id")
            if tid is not None:
                task = all_tasks_by_id.get(tid)
                if task is None:
                    raise ValueError(f"Existing output contains unknown task_id={tid}")
                require_task_binding(prev, task)
                prior_artifact = (prior_manifest.get("model_artifacts") or {}).get(
                    prev.get("model"), {}
                )
                if prev.get("model_artifact_sha256") != prior_artifact.get(
                    "directory_sha256"
                ):
                    raise ValueError(
                        f"Existing output lacks matching model-weight provenance "
                        f"for task_id={tid}"
                    )
                merged[tid] = prev
    # Preserve original task order in the output (rows for tasks not yet generated are skipped).
    output_order = tasks if args.limit is not None else all_tasks
    rows = rows_in_task_order(output_order, merged, results)

    # Record all models in merged output, including earlier invocations.
    models_in_output = sorted({r.get("model") for r in rows if r.get("model")})

    missing_artifacts = sorted(set(models_in_output) - set(model_artifacts))
    if missing_artifacts:
        raise RuntimeError(
            f"Output contains model(s) without weight provenance: {missing_artifacts}"
        )
    backends = sorted({model_artifacts[m]["runtime_profile"]["backend"] for m in models_in_output})
    write_jsonl(out_path, rows)
    write_manifest(
        manifest_path, config_path, config,
        {
            "runner": backends[0] if len(backends) == 1 else "mixed",
            "backends": backends,
            "runner_path": "scripts/run_vllm_inference.py",
            "runner_sha256": file_sha256(Path(__file__)),
            "tasks_path": to_project_relative(tasks_path),
            "tasks_sha256": file_sha256(tasks_path),
            "output_path": to_project_relative(out_path),
            "output_sha256": file_sha256(out_path),
            "num_outputs": len(rows),
            "models": models_in_output,
            "models_this_invocation": run_models,
            "model_artifacts": {
                model: model_artifacts[model] for model in models_in_output
            },
            "software_versions": software_versions,
            "hardware": hardware,
            "decoding": {
                "temperatures_by_condition": temperatures_by_condition,
                "per_condition_temperature": {
                    "generation_arms": temp_gen,
                    "selfinduced_stage1": temp_si_stage1,
                    "cli_override": args.temperature,
                },
                "max_tokens": max_tokens,
                "top_p": 1.0,
                "seed": seed,
            },
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "chat_template": use_chat,
            "wall_seconds": round(time.time() - started, 2),
        },
    )
    print(json.dumps({"wrote": str(out_path), "rows": len(rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
