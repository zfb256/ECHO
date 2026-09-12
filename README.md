# ECHO

Controlled measurement of hallucination recurrence in Chinese dialogue.

The study evaluates seven open-weight models and two hosted endpoints. Hunyuan
and Kimi are semantic judges, not evaluated subjects. The frozen rule supplies
headline labels; human audits and semantic scoring assess measurement validity.

## Contents

- `configs/`: verified proposition/question banks and experiment settings.
- `scripts/`, `lib/`, `auto_labeling/`: generation, annotation, and analysis code.
- `datasets_zh/runs/`: saved outputs, labels, and provenance records.
- `reports_zh/`: results and frozen judge protocols.
- `docs/ANNOTATION_GUIDELINES_ZH_EN.md`: bilingual annotation instructions.
- `tests/`: offline checks, including synthetic API calls.

## Where to start

| Question | File under `reports_zh/` |
|---|---|
| Main seven-model results | `zh_full_metrics_7models.json` |
| Hosted replications | `zh_api_deepseek_metrics.json` / `zh_api_doubao_metrics.json` |
| Placebo and ambiguity sensitivity | `ccr_estimand_robustness_7models.json` |
| Held-out detector coverage | `injected_recall_human_audit_r2_open7.json` |
| Self-induced primary and expanded cohorts | `selfinduced_u4_results.json` |
| Full semantic-judge comparison | `DUAL_JUDGE_FULL_COMPARISON.json` |
| Semantic placebo / correct-statement control | `HUNYUAN_PLACEBO_SUMMARY.json` / `TRUE_FACT_CONTROL_SUMMARY.json` |

Unsuffixed five-model reports and diagnostic audits are historical records,
not substitutes for the seven-model results. Versioned execution-failure files
document protocol amendments; they are not final judge labels. Additional
analysis scripts are retained for provenance and are not a required run order.

## Recompute the seven-model contrast

```bash
python3 scripts/compute_ccr_estimand_robustness.py \
  --injected datasets_zh/runs/zh_study_ext/merged7_claim_annotation_injected.jsonl \
  --placebo datasets_zh/runs/zh_study_ext/merged7_claim_annotation_placebo.jsonl \
  --output reports_zh/ccr_recheck.json
```

`reports_zh/zh_full_metrics.json` retains the original five-model analysis.
The manuscript is distributed separately, not inside this directory.

## Environment

Run commands from this directory. Install dependencies with
`python3 -m pip install -r requirements.txt` in a virtual
environment. Local runs used an NVIDIA A800-SXM4-40GB GPU. Exact backend versions
and model hashes are in the run manifests. API generation requires credentials
supplied through environment variables; no credentials are distributed.
Saved-result inspection and the command above require no GPU or API calls.

## Offline tests

```bash
python3 -B -m unittest discover -s tests -p 'test_*.py'
```

The API smoke test uses synthetic inputs with `--mock` and verifies resume
without duplicate outputs; it makes no API calls. Tests check retained KdConv
prompts and verify that LCCC prompts are withheld. The stage-1 task-binding test
is skipped because generation-task files are not distributed.

## Anonymous distribution

The anonymous export omits the paper, figures, model weights, credentials,
caches, private working notes, and raw dialogue datasets. LCCC dialogue prompts are withheld because
the upstream dataset is restricted to research use. Source locators identify
the upstream rows; saved output records retain task and model hashes.
Export exclusions and any file transformations are recorded in
`EXPORT_MANIFEST.json`; original run hashes are historical provenance, not hashes
of transformed release files. Generation requires restoring the omitted inputs.

Third-party licenses and citations remain applicable. See the manuscript's
Ethics Statement for data-use restrictions and AI-assistance disclosure.
