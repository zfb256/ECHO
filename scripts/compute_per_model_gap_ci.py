"""Rebuild Figure 3 data with the bootstrap used by the formal reports."""

import json
from pathlib import Path

from compute_ccr_metrics import _ccr_flags, bootstrap_gap_ci_clustered
from jsonl import read_jsonl
from manifest import file_sha256

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports_zh"


def main():
    config_path = ROOT / "configs/zh_study.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = config["random_seed"]
    replicates = config["pilot"]["ccr"]["bootstrap_samples"]
    hashes = {str(config_path.relative_to(ROOT)): file_sha256(config_path)}

    def read_pair(directory, injected, placebo):
        paths = [ROOT / directory / name for name in (injected, placebo)]
        for path in paths:
            hashes[str(path.relative_to(ROOT))] = file_sha256(path)
        rows = [list(read_jsonl(path)) for path in paths]
        keys = [{(r["model"], r["pair_id"]) for r in arm} for arm in rows]
        assert keys[0] == keys[1] and all(len(k) == len(r) for k, r in zip(keys, rows))
        return rows

    def estimate(injected, placebo, report=None):
        flags = [_ccr_flags(arm) for arm in (injected, placebo)]
        assert all(len(f) == len(r) and f for f, r in zip(flags, (injected, placebo)))
        gap = sum(flags[0]) / len(flags[0]) - sum(flags[1]) / len(flags[1])
        ci = bootstrap_gap_ci_clustered(injected, placebo, "seed_claim", replicates, seed)
        assert ci is not None
        if report:
            path = REPORTS / report
            decision = json.loads(path.read_text(encoding="utf-8"))["decision"]
            assert abs(gap - decision["real_minus_placebo_ccr"]) < 1e-12
            assert ci == decision["real_minus_placebo_ccr_ci95_seed_clustered"], (report, ci)
            hashes[str(path.relative_to(ROOT))] = file_sha256(path)
        return {"gap": gap, "lo": ci[0], "hi": ci[1], "n": len(injected)}

    injected, placebo = read_pair(
        "datasets_zh/runs/zh_study_ext", "merged7_claim_annotation_injected.jsonl",
        "merged7_claim_annotation_placebo.jsonl")
    names = {
        **{f"Qwen2.5-{size}B-Instruct": f"Qwen2.5-{size}B" for size in (1.5, 3, 7, 14)},
        "internlm2_5-7b-chat": "InternLM2.5-7B",
        "glm-4-9b-chat-hf": "GLM-4-9B", "Yi-1.5-9B-Chat": "Yi-1.5-9B",
    }
    assert {r["model"] for r in injected} == set(names)
    models = {label: estimate([r for r in injected if r["model"] == model],
                              [r for r in placebo if r["model"] == model])
              for model, label in names.items()}
    aggregate = estimate(injected, placebo, "zh_full_metrics_7models.json")
    for provider, label in (("deepseek", "DeepSeek-V4-Flash"), ("doubao", "Doubao-Seed-2.1-Pro")):
        pair = read_pair("datasets_zh/runs/zh_study", f"claim_annotation_injected_api_{provider}.jsonl",
                         f"claim_annotation_placebo_api_{provider}.jsonl")
        models[label] = estimate(*pair, report=f"zh_api_{provider}_metrics.json")
    data = {"models": models, "open_weight_aggregate": aggregate,
            "bootstrap": {"replicates": replicates, "random_seed": seed, "cluster": "seed_claim",
                          "function": "compute_ccr_metrics.bootstrap_gap_ci_clustered",
                          "quantiles": "sorted draws at floor(0.025*N), floor(0.975*N); rounded to 4 decimals"},
            "input_sha256": hashes}
    output = REPORTS / "per_model_gap_ci.json"
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}; aggregate and both hosted intervals match formal reports.")


if __name__ == "__main__":
    main()
