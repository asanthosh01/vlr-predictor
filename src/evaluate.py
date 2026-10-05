"""Chronological train/validation/test evaluation and portable model export."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from src.chronological import FEATURES, build_features, chronological_split

def metrics(y, probability):
    y, probability = np.asarray(y), np.asarray(probability)
    correct = int(((probability >= .5) == y).sum())
    n, z = len(y), 1.96
    p = correct/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'n': n, 'accuracy': float(accuracy_score(y, probability >= .5)),
        'accuracy_95ci_wilson': [float(center-half), float(center+half)],
        'balanced_accuracy': float(balanced_accuracy_score(y, probability >= .5)),
        'brier_score': float(brier_score_loss(y, probability)),
        'roc_auc': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None}

def train(data_path, output):
    data_path, output = Path(data_path), Path(output)
    data = pd.read_csv(data_path)
    features, _ = build_features(data)
    train_set, validation, test = chronological_split(features)
    if train_set.team1_won.nunique() != 2:
        raise ValueError('Training data must contain wins by both team positions')
    model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=.05,
        subsample=.8, colsample_bytree=.8, random_state=42, n_jobs=2, eval_metric='logloss')
    model.fit(train_set[FEATURES], train_set.team1_won)
    prior = float(train_set.team1_won.mean())
    report = {'schema_version': 1, 'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'matches': len(features), 'features': FEATURES,
        'method': '60/20/20 chronological split by date; fixed hyperparameters; no tuning on test',
        'history_policy': 'Earlier dates only. Earlier validation/test outcomes update history for later matches; model stays frozen.',
        'probabilities': 'Raw XGBoost probabilities, not calibrated',
        'splits': {}, 'feature_importance': dict(zip(FEATURES, map(float, model.feature_importances_)))}
    for name, part in [('train', train_set), ('validation', validation), ('test', test)]:
        report['splits'][name] = {'start': str(part.match_date.min().date()),
            'end': str(part.match_date.max().date()),
            'model': metrics(part.team1_won, model.predict_proba(part[FEATURES])[:, 1]),
            'majority_baseline': metrics(part.team1_won, np.full(len(part), prior)),
            'elo_baseline': metrics(part.team1_won, 1/(1+10**(-part.elo_difference/400)))}
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(output/'model.json')
    (output/'evaluation.json').write_text(json.dumps(report, indent=2))
    features.to_csv(output/'features.csv', index=False)
    print(json.dumps(report['splits']['test'], indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/matches.csv')
    parser.add_argument('--out', default='artifacts')
    args = parser.parse_args()
    train(args.data, args.out)
