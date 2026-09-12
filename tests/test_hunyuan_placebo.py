import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_hunyuan_placebo import make_pairs


def test_clean_pair_mapping_and_missing_condition():
    rows = [dict(pair_id='p', model='m', condition=c, response=c,
                 seed_claim='false', corrected_claim='true', seed_id='s', source_dataset='d')
            for c in ('contaminated', 'clean', 'clean_b')]
    pair, = make_pairs(rows)
    assert pair['response_contaminated'] == 'clean'
    assert pair['response_clean'] == 'clean_b'
    assert pair['pair_type'] == 'clean_a_vs_clean_b'
    with pytest.raises(ValueError, match='incomplete'):
        make_pairs(rows[:2])
    with pytest.raises(ValueError, match='duplicate'):
        make_pairs(rows + rows[:1])
