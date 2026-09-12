"""Score the fixed true-fact extension with the unchanged rule detector."""
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
sys.path.insert(0, str(ROOT / 'auto_labeling'))
from detector import detect_injected
from jsonl import read_jsonl, write_jsonl
from manifest import file_sha256, object_sha256


def bootstrap_diff(rows, left, right):
    clusters = defaultdict(list)
    for r in rows:
        clusters[r['seed_id']].append(int(r[left]['false']) - int(r[right]['false']))
    totals = np.array([[sum(v), len(v)] for _, v in sorted(clusters.items())])
    picks = np.random.default_rng(20260909).integers(0, len(totals), (2000, len(totals)))
    samples = totals[picks].sum(axis=1)
    return dict(difference=float(totals[:, 0].sum()/totals[:, 1].sum()),
                ci95_seed_clustered=np.quantile(samples[:, 0]/samples[:, 1], [.025,.975]).tolist())


def main():
    subprocess.run([sys.executable, str(ROOT/'scripts/freeze_detector.py'), '--verify'], check=True)
    base = ROOT/'datasets_zh/runs/zh_true_fact_control'
    design_path = ROOT/'reports_zh/TRUE_FACT_CONTROL_DESIGN.json'
    design = json.loads(design_path.read_text())
    manifest = json.loads((base/'model_outputs.manifest.json').read_text())
    assert file_sha256(base/'tasks.jsonl') == design['tasks_sha256'] == manifest['tasks_sha256']
    assert file_sha256(base/'model_outputs.jsonl') == manifest['output_sha256']
    tasks = list(read_jsonl(base/'tasks.jsonl'))
    outputs = list(read_jsonl(base/'model_outputs.jsonl'))
    task_map = {r['task_id']:r for r in tasks}
    out_map = {r['task_id']:r for r in outputs}
    assert len(task_map) == len(out_map) == len(tasks) == len(outputs) == 2400
    assert set(task_map) == set(out_map)
    assert Counter(r['model'] for r in outputs) == Counter({m:800 for m in design['models']})
    original_path = ROOT/'datasets_zh/runs/zh_study_ext/merged7_model_outputs.jsonl'
    original = list(read_jsonl(original_path))
    old = {(r['pair_id'],r['model'],r['condition']):r for r in original}
    assert len(old) == len(original)
    rows = []
    for task in tasks:
        out = out_map[task['task_id']]
        assert out['task_sha256'] == object_sha256(task)
        assert all(out[k] == task[k] for k in ('model','pair_id','condition','seed_claim','corrected_claim'))
        group = {c:old[(task['pair_id'],task['model'],c)] for c in ('contaminated','clean','clean_b')}
        group['true_fact_control'] = out
        row = {k:task[k] for k in ('pair_id','model','seed_id','source_dataset')}
        for condition, response in group.items():
            assert response['model_artifact_sha256'] == out['model_artifact_sha256']
            assert response['decoding_temperature'] == .7
            assert response.get('response','').strip()
            assert not response.get('mock') and not response.get('source_mock')
            d = detect_injected(response['response'], task['seed_claim'], task['corrected_claim'], 6)
            row[condition] = dict(false=d['asserts_false'] and not d['ambiguous'],
                false_inclusive=d['asserts_false'], true=d['asserts_true'],
                refuted=d['refuted'], ambiguous=d['ambiguous'],
                length_capped=response.get('finish_reason') == 'length')
        rows.append(row)
    groups = {'all_selected':rows}
    groups.update({m:[r for r in rows if r['model']==m] for m in design['models']})
    summaries = {}
    for name, group in groups.items():
        conditions = {}
        for c in ('contaminated','true_fact_control','clean','clean_b'):
            counts = {k:sum(r[c][k] for r in group) for k in group[0][c]}
            conditions[c] = dict(n=len(group), counts=counts,
                                 rates={k:v/len(group) for k,v in counts.items()})
        summaries[name] = dict(conditions=conditions,
            contaminated_minus_true=bootstrap_diff(group,'contaminated','true_fact_control'),
            true_minus_clean=bootstrap_diff(group,'true_fact_control','clean'))
    write_jsonl(base/'scored_pairs.jsonl', rows)
    report = dict(design_sha256=file_sha256(design_path), outputs_sha256=file_sha256(base/'model_outputs.jsonl'),
        original_outputs_sha256=file_sha256(original_path), script_sha256=file_sha256(Path(__file__)),
        definition='False rates exclude mixed unrefuted false/true responses; inclusive rates and unvalidated true/refuted proxies also reported. Marginal rates, not CCR.',
        summaries=summaries)
    (ROOT/'reports_zh/TRUE_FACT_CONTROL_SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
