import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_true_fact_control import transform
from compute_true_fact_control import bootstrap_diff


def test_bootstrap_keeps_pair_direction():
    rows = [dict(seed_id=str(i), a={'false':True}, b={'false':False}) for i in range(4)]
    assert bootstrap_diff(rows, 'a', 'b') == dict(difference=1., ci95_seed_clustered=[1.,1.])
    assert bootstrap_diff(rows, 'b', 'a') == dict(difference=-1., ci95_seed_clustered=[-1.,-1.])


def test_replace_only_assistant_seed():
    r = dict(pair_id='p', model='m', seed_claim='false', corrected_claim='true',
             prompt='history false\n助手: false\n用户: probe', task_id='old')
    s = transform(r)
    assert s['prompt'] == 'history false\n助手: true\n用户: probe'
    assert s['task_id'] != r['task_id']
    assert r['prompt'].endswith('助手: false\n用户: probe')
    with pytest.raises(ValueError):
        transform(dict(r, prompt='missing assistant turn'))
