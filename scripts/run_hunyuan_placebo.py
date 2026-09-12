"""Post-hoc Clean A/B extension of the frozen Hunyuan judge; resumable.

--prepare binds the analysis before any new API calls. Default runs and summarizes.
The shared scorer's judge_contaminated/judge_clean fields mean Clean A/Clean B here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from run_semantic_judge import ROOT, load_protocol, score_rows
from jsonl import read_jsonl, write_jsonl
from manifest import file_sha256, object_sha256

OUTPUTS = ROOT / 'datasets_zh/runs/zh_study_ext/merged7_model_outputs.jsonl'
INJECTED = ROOT / 'datasets_zh/runs/zh_study_ext/tencent_semantic_judge_scores.jsonl'
CALIBRATION = ROOT / 'reports_zh/TENCENT_JUDGE_CALIBRATION.json'
DESIGN = ROOT / 'reports_zh/HUNYUAN_PLACEBO_DESIGN.json'
SCORES = ROOT / 'datasets_zh/runs/zh_study_ext/hunyuan_placebo_scores.jsonl'
SUMMARY = ROOT / 'reports_zh/HUNYUAN_PLACEBO_SUMMARY.json'


def make_pairs(outputs):
    groups = defaultdict(dict)
    for row in outputs:
        key = (row['pair_id'], row['model'])
        condition = row['condition']
        if condition in groups[key]:
            raise ValueError(f'duplicate output: {key}, {condition}')
        if row.get('mock') or row.get('source_mock'):
            raise ValueError('mock source output')
        groups[key][condition] = row
    pairs = []
    for (pair_id, model), group in sorted(groups.items()):
        if set(group) != {'clean', 'clean_b', 'contaminated'}:
            raise ValueError(f'incomplete conditions: {pair_id}, {model}')
        a, b = group['clean'], group['clean_b']
        for field in ('seed_claim', 'corrected_claim', 'seed_id', 'source_dataset'):
            if not a.get(field) or any(r.get(field) != a[field] for r in group.values()):
                raise ValueError(f'mismatched {field}: {pair_id}, {model}')
        if any(not str(r.get('response', '')).strip() for r in (a, b)):
            raise ValueError(f'empty clean response: {pair_id}, {model}')
        pairs.append(dict(pair_id=pair_id, model=model, seed_id=a['seed_id'],
                          source_dataset=a['source_dataset'], seed_claim=a['seed_claim'],
                          corrected_claim=a['corrected_claim'],
                          response_contaminated=a['response'], response_clean=b['response'],
                          pair_type='clean_a_vs_clean_b'))
    return pairs


def summarize(rows, scores, injected, protocol):
    import numpy as np

    def index(items):
        result = {(r['pair_id'], r['model']): r for r in items}
        if len(result) != len(items):
            raise ValueError('duplicate scores')
        return result

    p, i = index(scores), index(injected)
    expected = {(r['pair_id'], r['model']) for r in rows}
    if set(p) != expected or not expected.issubset(i):
        raise ValueError('score coverage mismatch')
    grouped = defaultdict(list)
    for row in rows:
        key = (row['pair_id'], row['model'])
        a, b = i[key], p[key]
        for s in (a, b):
            if s['judge_mock'] or s['protocol_sha256'] != protocol['protocol_sha256']:
                raise ValueError('mock or wrong-protocol score')
        if b['input_sha256'] != object_sha256(row):
            raise ValueError('placebo input binding mismatch')
        grouped['all'].append((row, a, b))
        grouped['model:' + row['model']].append((row, a, b))
        grouped['dataset:' + row['source_dataset']].append((row, a, b))
    result = {}
    for name, group in grouped.items():
        complete = [(r, a, b) for r, a, b in group if a['judge_decidable'] and b['judge_decidable']]
        n = len(group)
        diffs = [int(a['judge_ccr_event']) - int(b['judge_ccr_event']) for _, a, b in complete]
        missing = n - len(complete)
        result[name] = dict(n=n, comparable=len(complete),
            injected_events=sum(a['judge_ccr_event'] for _, a, _ in complete),
            placebo_events=sum(b['judge_ccr_event'] for _, _, b in complete),
            gap_on_comparable=sum(diffs) / len(diffs) if diffs else None,
            undecidable_only_range=[(sum(diffs)-missing)/n, (sum(diffs)+missing)/n])
        # Bootstrap paired differences together, conditional on observed judge labels.
        if not missing:
            clusters = defaultdict(list)
            for (r, _, _), diff in zip(complete, diffs):
                clusters[r['seed_id']].append(diff)
            totals = np.array([[sum(v), len(v)] for _, v in sorted(clusters.items())])
            picks = np.random.default_rng(20260908).integers(0, len(totals), (2000, len(totals)))
            boot = totals[picks].sum(axis=1)
            result[name]['gap_ci95_seed_clustered_conditional_on_judge'] = np.quantile(
                boot[:, 0] / boot[:, 1], [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    protocol = load_protocol('reports_zh/TENCENT_JUDGE_PROTOCOL.json')
    cal = json.loads(CALIBRATION.read_text())
    if not cal.get('passed') or cal.get('mock') or cal['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('original calibration invalid')
    rows = make_pairs(list(read_jsonl(OUTPUTS)))
    if len(rows) != 5600 or sorted(Counter(r['model'] for r in rows).values()) != [800]*7:
        raise ValueError('expected 800 pairs for each of seven open models')
    design = dict(protocol_sha256=protocol['protocol_sha256'], inputs_sha256=object_sha256(rows),
        outputs_sha256=file_sha256(OUTPUTS), injected_scores_sha256=file_sha256(INJECTED),
        calibration_sha256=file_sha256(CALIBRATION), n=5600,
        scope='Post-hoc open7 extension; A=Clean A, B=Clean B; frozen prompt unchanged.',
        inference='Paired injected-minus-placebo; 2000 seed-cluster bootstraps, RNG 20260908; conditional on judge labels.',
        limitation='Clean-clean pair type has no human calibration. Do not transfer the contaminated-clean Wilson error allowance.',
        reporting='Retain all undecidables; report full results regardless of direction; do not alter headline detector labels.')
    if DESIGN.exists():
        prior = json.loads(DESIGN.read_text())
        if {k:v for k,v in prior.items() if k != 'created_at_utc'} != design:
            raise ValueError('extension design binding mismatch')
    else:
        if not args.prepare:
            raise ValueError('run --prepare before scoring')
        with DESIGN.open('x') as f:
            json.dump(dict(design, created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), f, indent=2)
    print('Input and design checks passed: 5600 clean-clean pairs', flush=True)
    if args.prepare:
        return
    scores = list(read_jsonl(SCORES)) if SCORES.exists() else score_rows(
        rows, protocol, False, None, SCORES.with_suffix('.partial.jsonl'))
    result = summarize(rows, scores, list(read_jsonl(INJECTED)), protocol)
    if not SCORES.exists():
        write_jsonl(SCORES, scores)
    report = dict(design_sha256=file_sha256(DESIGN), scores_sha256=file_sha256(SCORES),
                  interpretation=design['limitation'], results=result)
    SUMMARY.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Service retries transport/server/rate-limit failures; hard errors stop.
        from openai import APIConnectionError, APIStatusError
        if isinstance(exc, APIConnectionError) or (isinstance(exc, APIStatusError) and
                (exc.status_code == 429 or exc.status_code >= 500)):
            print(f'Transient API failure ({type(exc).__name__}); checkpoint retained.', flush=True)
            sys.exit(75)
        raise
