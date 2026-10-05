import pandas as pd
import pytest
from src.chronological import build_features, chronological_split, validate_matches
from src.collect_history import parse_page
from src.evaluate import train
from app import create_app

def matches():
    return pd.DataFrame([{'match_id': str(i), 'match_date': (pd.Timestamp('2025-01-01')+pd.Timedelta(days=i//2)).date().isoformat(),
        'team1': 'Alpha', 'team2': 'Beta', 'score1': 2 if i%3 else 0,
        'score2': 0 if i%3 else 2} for i in range(100)])

def test_outcome_does_not_change_own_or_earlier_features():
    data = matches()
    original, _ = build_features(data)
    data.loc[50, ['score1', 'score2']] = [0,2]
    changed, _ = build_features(data)
    pd.testing.assert_frame_equal(original.iloc[:50].drop(columns='team1_won'), changed.iloc[:50].drop(columns='team1_won'))
    pd.testing.assert_series_equal(original.iloc[50].drop('team1_won'), changed.iloc[50].drop('team1_won'))

def test_same_day_does_not_share_outcomes():
    features, _ = build_features(matches())
    assert features.iloc[0].team1_matches == features.iloc[1].team1_matches == 0
    assert features.iloc[2].team1_matches == 2

def test_split_dates_do_not_overlap():
    parts = chronological_split(build_features(matches())[0])
    assert parts[0].match_date.max() < parts[1].match_date.min()
    assert parts[1].match_date.max() < parts[2].match_date.min()

def test_conflicting_duplicates_rejected():
    data = matches()
    bad = data.iloc[[0]].copy()
    bad['score1'], bad['score2'] = 2,0
    with pytest.raises(ValueError, match='Conflicting'):
        validate_matches(pd.concat([data,bad]))

def test_html_date_and_source():
    html = '<div class="wf-label mod-large">Sun, October 4, 2026 <span>Yesterday</span></div><div><a class="match-item" href="/123/alpha-v-beta"><div class="match-item-vs-team-name">Alpha</div><div class="match-item-vs-team-name">Beta</div><div class="match-item-vs-team-score">2</div><div class="match-item-vs-team-score">0</div></a></div>'
    row = parse_page(html)[0]
    assert row['match_date'] == '2026-10-04'
    assert row['match_id'] == '123'

def test_missing_model_serves_setup_page(tmp_path):
    client = create_app(tmp_path/'missing.csv',tmp_path).test_client()
    assert client.get('/').status_code == 200
    assert client.get('/health').json['ready'] is False
    assert client.post('/predict',json={'team1':'Alpha','team2':'Beta'}).status_code == 503

def test_train_export_and_predict_roundtrip(tmp_path):
    data = tmp_path/'matches.csv'
    matches().to_csv(data,index=False)
    report = train(data,tmp_path/'artifacts')
    assert report['splits']['test']['model']['n'] == 20
    client = create_app(data,tmp_path/'artifacts').test_client()
    assert client.get('/health').json['ready'] is True
    response = client.post('/predict',json={'team1':'Alpha','team2':'Beta'})
    assert response.status_code == 200
    assert 0 <= response.json['team1_probability'] <= 1
    assert client.post('/predict',json={'team1':'Alpha','team2':'Alpha'}).status_code == 400
    assert client.post('/predict',json=[]).status_code == 400
    assert client.get('/').status_code == 200
