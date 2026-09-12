from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from config import ensure_dirs, load_config, resolve_project_path, to_project_relative
from ids import stable_id
from jsonl import read_jsonl, write_jsonl
from manifest import write_manifest


# Hosted inference using provider SDKs or OpenAI-compatible endpoints.


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hosted-model API inference for ECHO.")
    p.add_argument("--config", default="configs/zh_study.json")
    p.add_argument("--tasks", required=True, help="A *_tasks.jsonl file (reused from the open-model export).")
    p.add_argument("--model", required=True, help="Provider model identifier, e.g. deepseek-v4-flash.")
    p.add_argument("--output", default=None, help="Defaults to <tasks dir>/model_outputs_api.jsonl (kept separate from open-model outputs).")
    p.add_argument("--limit", type=int, default=None, help="Only run the first N unique tasks (smoke/subset).")
    p.add_argument("--max-tokens", type=int, default=256, help="Match the open-model cap (256).")
    p.add_argument("--temperature", type=float, default=None, help="Override the condition-specific config value")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1, help="Number of parallel API calls for independent tasks.")
    p.add_argument("--base-url", default=None, help="Generic OpenAI-compatible endpoint (overrides the per-provider default).")
    p.add_argument("--api-key-env", default=None, help="Env var holding the key for a generic --base-url provider.")
    p.add_argument(
        "--api-mode",
        choices=("auto", "chat", "responses"),
        default="auto",
        help="OpenAI-compatible API surface. auto uses Responses API for Doubao/Ark and chat completions otherwise.",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate wiring (routing/dedupe) without importing SDKs or calling APIs.")
    p.add_argument("--mock", action="store_true", help="Fabricate outputs without any API call or key (plumbing only; NOT research data).")
    return p.parse_args()


def safe_limited_output_name(name: str) -> bool:
    return any(token in name.lower() for token in ("limit", "smoke", "tmp", "partial"))


_OPENAI_COMPAT = {
    "gpt":      (None, "OPENAI_API_KEY"),
    "o1":       (None, "OPENAI_API_KEY"),
    "o3":       (None, "OPENAI_API_KEY"),
    "o4":       (None, "OPENAI_API_KEY"),
    "chatgpt":  (None, "OPENAI_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
}


def resolve_provider(model: str, args: argparse.Namespace) -> dict:
    """Return {kind, label, base_url, key_env}. kind is 'anthropic' or 'openai_compat'."""
    m = model.lower()
    if m.startswith("claude"):
        return {"kind": "anthropic", "label": "anthropic", "base_url": None, "key_env": "ANTHROPIC_API_KEY"}
    if args.base_url:  # explicit generic OpenAI-compatible endpoint
        return {"kind": "openai_compat", "label": "openai_compat", "base_url": args.base_url,
                "key_env": args.api_key_env or "OPENAI_API_KEY"}
    for prefix, vals in _OPENAI_COMPAT.items():
        base, key_env = vals[:2]
        base_env = vals[2] if len(vals) > 2 else None
        if m.startswith(prefix):
            label = "openai" if base is None else prefix
            if base_env and os.environ.get(base_env):
                base = os.environ[base_env]
            return {"kind": "openai_compat", "label": label, "base_url": base, "key_env": key_env}
    raise SystemExit(
        f"Cannot route model {model!r}. Use a known prefix (claude*/gpt*/o*/deepseek*/qwen*) "
        f"or pass --base-url + --api-key-env for a generic OpenAI-compatible endpoint."
    )


def resolve_api_mode(model: str, prov: dict, requested: str) -> str:
    if prov["kind"] == "anthropic":
        return "messages"
    if requested != "auto":
        return requested
    base_url = (prov.get("base_url") or "").lower()
    if model.lower().startswith("doubao") or "ark.cn-" in base_url or "volces.com" in base_url:
        return "responses"
    return "chat"


def messages_for(task: dict) -> list[dict]:
    """Chat messages for the API. Self-induced stage-2 carries structured `messages`
    ([user, assistant, user]) so the model's own turn-1 is a real assistant turn; otherwise
    wrap the flat `prompt` as a single user message."""
    msgs = task.get("messages")
    if isinstance(msgs, list) and msgs:
        return [{"role": m["role"], "content": m["content"]} for m in msgs]
    return [{"role": "user", "content": task["prompt"]}]


def dedupe_tasks(tasks: list[dict], model: str) -> list[dict]:
    """Open-model task files repeat each (pair, condition) prompt once PER model; the prompt is
    model-independent, so for ONE closed model we keep a single copy per (pair_id, condition)
    and re-key task_id to this model."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for t in tasks:
        key = (t.get("pair_id"), t.get("condition"))
        if key in seen:
            continue
        seen.add(key)
        nt = dict(t)
        nt["model"] = model
        nt["task_id"] = stable_id(t.get("pair_id"), model, t.get("condition"))
        out.append(nt)
    return out


def call_anthropic(client: Any, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking={"type": "disabled"},
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return {
        "response": text.strip(),
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
        "finish_reason": resp.stop_reason,
    }


def call_openai(client: Any, model: str, provider: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    kwargs = {}
    if provider == "deepseek":
        # DeepSeek V4 enables thinking by default; disable it for replication decoding.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    choice = resp.choices[0]
    return {
        "response": (choice.message.content or "").strip(),
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "finish_reason": choice.finish_reason,
    }


def _responses_input(messages: list[dict]) -> list[dict]:
    return [
        {
            "role": m["role"],
            "content": [{"type": "input_text", "text": str(m["content"])}],
        }
        for m in messages
    ]


def _get_usage_field(usage: Any, *names: str) -> int:
    for name in names:
        value = getattr(usage, name, None) if usage is not None else None
        if value is not None:
            return int(value)
    return 0


def call_openai_responses(client: Any, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    resp = client.responses.create(
        model=model,
        input=_responses_input(messages),
        max_output_tokens=max_tokens,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = getattr(resp, "output_text", "") or ""
    usage = getattr(resp, "usage", None)
    return {
        "response": text.strip(),
        "prompt_tokens": _get_usage_field(usage, "input_tokens", "prompt_tokens"),
        "completion_tokens": _get_usage_field(usage, "output_tokens", "completion_tokens"),
        "finish_reason": getattr(resp, "status", None) or "stop",
    }


def make_client(prov: dict):
    if prov["kind"] == "anthropic":
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit("Missing dependency: pip install anthropic") from exc
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install openai") from exc
    key = os.environ.get(prov["key_env"])
    if not key:
        raise SystemExit(f"Set {prov['key_env']} for provider {prov['label']!r}.")
    kwargs = {"api_key": key, "timeout": 120}
    if prov["base_url"]:
        kwargs["base_url"] = prov["base_url"]  # DeepSeek / Qwen / generic OpenAI-compatible
    return OpenAI(**kwargs)


def call_with_retry(fn, max_retries: int):
    import random

    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — SDKs raise provider-specific errors; back off and retry
            status_code = getattr(exc, "status_code", None)
            if status_code in {400, 401, 403, 404}:
                raise SystemExit(f"API call failed with non-retryable status {status_code}: {exc}") from exc
            last = exc
            time.sleep(min(2 ** attempt + random.uniform(0, 1), 30))
    raise SystemExit(f"API call failed after {max_retries} attempts: {last}")


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    config = load_config(args.config)
    ensure_dirs(config)
    config_path = resolve_project_path(args.config)

    tasks_path = resolve_project_path(args.tasks)
    if args.output:
        out_path = resolve_project_path(args.output)
    else:
        parts = ["model_outputs_api"]
        if args.mock:
            parts.append("mock")
        if args.limit is not None:
            parts.append(f"limit{args.limit}")
        out_name = "_".join(parts) + ".jsonl"
        out_path = tasks_path.with_name(out_name)
    if args.mock and "mock" not in out_path.name:
        raise SystemExit("--mock output must include 'mock' in the filename; refusing to overwrite research data.")
    if args.limit is not None and not safe_limited_output_name(out_path.name):
        raise SystemExit(
            "--limit is for smoke/partial API runs only. Pass an explicit output filename containing "
            "'limit', 'smoke', 'tmp', or 'partial' so a full research output cannot be confused with a subset."
        )
    if args.limit is None and safe_limited_output_name(out_path.name):
        raise SystemExit(
            "Output filename looks like a smoke/partial run, but --limit is not set. "
            "Use a full-run filename, or pass --limit for a subset."
        )
    prov = resolve_provider(args.model, args)
    api_mode = resolve_api_mode(args.model, prov, args.api_mode)

    ccr = config.get("pilot", {}).get("ccr", {})
    temp_gen = float(ccr.get("decoding_temperature_generation", 0.7))
    temp_si1 = float(ccr.get("decoding_temperature_selfinduced_stage1", 0.0))

    def temperature_for(task: dict) -> float:
        if args.temperature is not None:
            return args.temperature
        return temp_si1 if task.get("condition") == "self_induced_stage1" else temp_gen

    tasks = dedupe_tasks(list(read_jsonl(tasks_path)), args.model)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit(f"No tasks in {tasks_path}")

    print(json.dumps({
        "tasks": to_project_relative(tasks_path), "output": to_project_relative(out_path),
        "model": args.model, "provider": prov["label"], "base_url": prov["base_url"] or "(default)",
        "api_mode": api_mode, "unique_tasks": len(tasks), "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "temperatures": {"generation": temp_gen, "selfinduced_stage1": temp_si1},
    }, ensure_ascii=False))

    if args.dry_run:
        print(json.dumps({"dry_run": True, "note": "routing + dedupe validated; no API call"}, ensure_ascii=False))
        return

    thread_local = threading.local()

    def get_client():
        if args.mock:
            return None
        client = getattr(thread_local, "client", None)
        if client is None:
            client = make_client(prov)
            thread_local.client = client
        return client

    def generate_row(task: dict) -> dict:
        client = get_client()
        msgs = messages_for(task)
        temp = temperature_for(task)
        if args.mock:
            gen = {"response": f"[MOCK {args.model}] continuation for {task.get('condition')}",
                   "prompt_tokens": 0, "completion_tokens": 0, "finish_reason": "stop"}
        elif prov["kind"] == "anthropic":
            gen = call_with_retry(lambda: call_anthropic(client, args.model, msgs, args.max_tokens, temp), args.max_retries)
        elif api_mode == "responses":
            gen = call_with_retry(lambda: call_openai_responses(client, args.model, msgs, args.max_tokens, temp), args.max_retries)
        else:
            gen = call_with_retry(lambda: call_openai(client, args.model, prov["label"], msgs, args.max_tokens, temp), args.max_retries)
        row = {k: v for k, v in task.items() if k not in ("prompt", "messages")}
        row.update(gen)
        row["decoding_temperature"] = temp
        row["api_provider"] = prov["label"]
        row["api_mode"] = api_mode
        return row

    run_task_ids = {t["task_id"] for t in tasks}
    expected_by_tid = {t["task_id"]: t for t in tasks}
    merged: dict[str, dict] = {}
    if out_path.exists():
        for prev in read_jsonl(out_path):
            tid = prev.get("task_id")
            if tid is None:
                continue
            if tid in run_task_ids:
                expected = expected_by_tid[tid]
                prev_api_mode = prev.get("api_mode")
                if prev.get("model") != args.model:
                    raise SystemExit(f"Refusing to resume {to_project_relative(out_path)}: task_id {tid} has a different model.")
                if prev.get("condition") != expected.get("condition"):
                    raise SystemExit(f"Refusing to resume {to_project_relative(out_path)}: task_id {tid} has a different condition.")
                if prev.get("api_provider") and prev.get("api_provider") != prov["label"]:
                    raise SystemExit(f"Refusing to resume {to_project_relative(out_path)}: task_id {tid} has a different provider.")
                if prev_api_mode and prev_api_mode != api_mode:
                    raise SystemExit(f"Refusing to resume {to_project_relative(out_path)}: task_id {tid} has api_mode={prev_api_mode!r}, expected {api_mode!r}.")
            merged[tid] = prev

    results: dict[str, dict] = {}
    started = time.time()
    pending = [t for t in tasks if t["task_id"] not in merged]
    completed = len(tasks) - len(pending)
    if completed:
        print(json.dumps({"progress": completed, "of": len(tasks), "resumed": True}, ensure_ascii=False), file=sys.stderr)

    def persist() -> list[dict]:
        rows_now = [merged[t["task_id"]] for t in tasks if t["task_id"] in merged]
        for tid, r in merged.items():
            if tid not in run_task_ids:
                rows_now.append(r)
        write_jsonl(out_path, rows_now)
        return rows_now

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {}
    try:
        futures = {executor.submit(generate_row, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            row = future.result()
            completed += 1
            results[task["task_id"]] = row
            merged[task["task_id"]] = row
            persist()
            if completed % 25 == 0 or completed == len(tasks):
                print(json.dumps({"progress": completed, "of": len(tasks)}, ensure_ascii=False), file=sys.stderr)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    rows = persist()
    completed_for_run = sum(1 for t in tasks if t["task_id"] in merged)

    if not args.mock:
        manifest_path = out_path.with_suffix(".manifest.json")
        write_manifest(manifest_path, config_path, config, {
            "runner": "api", "provider": prov["label"], "api_mode": api_mode,
            "base_url": prov["base_url"], "model": args.model,
            "tasks_path": to_project_relative(tasks_path), "output_path": to_project_relative(out_path),
            "num_outputs": completed_for_run, "rows_written_this_run": len(results), "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "temperatures": {"generation": temp_gen, "selfinduced_stage1": temp_si1},
            "wall_seconds": round(time.time() - started, 2),
        })
    print(json.dumps({
        "wrote": to_project_relative(out_path), "rows_this_model": len(results),
        "rows_total": len(rows), "model": args.model, "api_mode": api_mode,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
