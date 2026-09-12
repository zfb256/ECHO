"""Prepare a post-hoc true-assistant-statement control, before new generation."""
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
from ids import stable_id
from jsonl import read_jsonl, write_jsonl
from manifest import file_sha256

MODELS = ('Qwen2.5-1.5B-Instruct', 'internlm2_5-7b-chat', 'Qwen2.5-14B-Instruct')


def transform(row):
    old = '\n助手: ' + row['seed_claim'] + '\n用户: '
    if row['prompt'].count(old) != 1:
        raise ValueError('expected exactly one injected assistant turn')
    return dict(row, condition='true_fact_control',
                task_id=stable_id(row['pair_id'], row['model'], 'true_fact_control'),
                prompt=row['prompt'].replace(old, '\n助手: ' + row['corrected_claim'] + '\n用户: ', 1))


def main():
    source = ROOT / 'datasets_zh/runs/zh_study/generation_tasks.jsonl'
    target = ROOT / 'datasets_zh/runs/zh_true_fact_control/tasks.jsonl'
    design = ROOT / 'reports_zh/TRUE_FACT_CONTROL_DESIGN.json'
    if target.exists() or design.exists():
        raise SystemExit('refusing to overwrite prepared control')
    rows = [transform(r) for r in read_jsonl(source)
            if r['condition'] == 'contaminated' and r['model'] in MODELS]
    assert Counter(r['model'] for r in rows) == Counter({m:800 for m in MODELS})
    assert len({r['task_id'] for r in rows}) == 2400
    write_jsonl(target, rows)
    record = dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        status='Prepared before control generation; post-hoc relative to original study.',
        models=MODELS, selection='Fixed post-hoc selection spanning the original open7 detector CCR range; InternLM is intermediate, not the median.',
        n=2400, source_sha256=file_sha256(source), tasks_sha256=file_sha256(target),
        intervention='Replace only the injected assistant false statement with its verified correction; keep history and probe.',
        decoding=dict(temperature=0.7, top_p=1.0, max_tokens=256, native_chat_template=True),
        primary='Compare frozen-detector false-assertion rates across contaminated, true_fact_control, clean and clean_b, on the same 800 pairs per model.',
        secondary='Exploratory asserts_true and refuted fields reported separately; neither is a validated general semantic truth metric.',
        uncertainty='2000 seed-cluster bootstrap replicates; RNG 20260909; paired differences resampled together.',
        reporting='Report all selected models and all results; separate post-hoc analysis, never merge into original Table 1 aggregate; do not claim truth-independent adoption.')
    with design.open('x') as f:
        json.dump(record, f, indent=2)
    print('Prepared 2400 tasks; no generation performed.')


if __name__ == '__main__':
    main()
