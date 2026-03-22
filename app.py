"""
Flask Web Interface for VLR Match Predictor V2
Now includes recent form and head-to-head features
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from difflib import get_close_matches
from collections import defaultdict

app = Flask(__name__)

# Load V2 model and data
model = joblib.load('models/match_predictor_v2.joblib')
rankings = pd.read_csv('data/raw/rankings_all_regions_20251016_132604.csv')

# Load all historical matches for recent form and H2H
import glob
match_files = glob.glob('data/raw/scraped_matches_*.csv')
all_matches = pd.concat([pd.read_csv(f) for f in match_files], ignore_index=True)
all_matches = all_matches.drop_duplicates(subset=['team1', 'team2', 'score1', 'score2'])
all_matches = all_matches.reset_index(drop=True)

print(f"✓ Loaded {len(all_matches)} historical matches for feature calculation")

# Prepare rankings data
rankings[['wins', 'losses']] = rankings['record'].str.split('–', expand=True)
rankings['wins'] = pd.to_numeric(rankings['wins'])
rankings['losses'] = pd.to_numeric(rankings['losses'])
rankings['total_games'] = rankings['wins'] + rankings['losses']
rankings['win_rate'] = rankings['wins'] / rankings['total_games']

# Create team lookup
team_lookup = {}
team_names_lower = []
for idx, row in rankings.iterrows():
    team_lookup[row['team'].lower()] = idx
    team_names_lower.append(row['team'].lower())

def find_team(team_name):
    """Find team using fuzzy matching"""
    if team_name.lower() in team_lookup:
        idx = team_lookup[team_name.lower()]
        return rankings.iloc[idx]
    
    matches = get_close_matches(team_name.lower(), team_names_lower, n=1, cutoff=0.75)
    if matches:
        idx = team_lookup[matches[0]]
        return rankings.iloc[idx]
    
    return None

def get_team_features(team_name):
    """Get ranking features for a team"""
    team_row = find_team(team_name)
    
    if team_row is not None:
        return {
            'rank': int(team_row['rank']),
            'win_rate': float(team_row['win_rate']),
            'total_games': int(team_row['total_games']),
            'wins': int(team_row['wins']),
            'found': True,
            'display_name': team_row['team']
        }
    else:
        return {
            'rank': 999,
            'win_rate': 0.5,
            'total_games': 0,
            'wins': 0,
            'found': False,
            'display_name': team_name
        }

def calculate_recent_form(team_name, window=5):
    """Calculate team's recent form (last N matches)"""
    team_matches = []
    
    for idx, row in all_matches.iterrows():
        if row['team1'] == team_name:
            won = row['score1'] > row['score2']
            team_matches.append(won)
        elif row['team2'] == team_name:
            won = row['score2'] > row['score1']
            team_matches.append(won)
    
    # Take last N matches
    recent = team_matches[-window:] if len(team_matches) >= window else team_matches
    
    if not recent:
        return {
            'recent_win_rate': 0.5,
            'recent_wins': 0,
            'recent_matches': 0
        }
    
    wins = sum(recent)
    
    return {
        'recent_win_rate': wins / len(recent),
        'recent_wins': wins,
        'recent_matches': len(recent)
    }

def calculate_head_to_head(team1, team2):
    """Calculate head-to-head history between two teams"""
    h2h_matches = []
    
    for idx, row in all_matches.iterrows():
        if (row['team1'] == team1 and row['team2'] == team2):
            team1_won = row['score1'] > row['score2']
            h2h_matches.append(team1_won)
        elif (row['team1'] == team2 and row['team2'] == team1):
            team1_won = row['score2'] > row['score1']
            h2h_matches.append(team1_won)
    
    if not h2h_matches:
        return {
            'h2h_matches': 0,
            'h2h_team1_wins': 0,
            'h2h_team1_win_rate': 0.5
        }
    
    team1_wins = sum(h2h_matches)
    
    return {
        'h2h_matches': len(h2h_matches),
        'h2h_team1_wins': team1_wins,
        'h2h_team1_win_rate': team1_wins / len(h2h_matches)
    }

@app.route('/')
def home():
    """Main page"""
    teams = sorted(rankings['team'].tolist())
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request with V2 features"""
    data = request.json
    team1_name = data.get('team1')
    team2_name = data.get('team2')
    
    if not team1_name or not team2_name:
        return jsonify({'error': 'Both teams required'}), 400
    
    if team1_name == team2_name:
        return jsonify({'error': 'Teams must be different'}), 400
    
    # Get ranking features
    t1 = get_team_features(team1_name)
    t2 = get_team_features(team2_name)
    
    # Get recent form features
    t1_form = calculate_recent_form(t1['display_name'])
    t2_form = calculate_recent_form(t2['display_name'])
    
    # Get head-to-head features
    h2h = calculate_head_to_head(t1['display_name'], t2['display_name'])
    
    # Create feature vector with ALL 21 features IN CORRECT ORDER
    features = {

    # Ranking features (11)
    'team1_rank': t1['rank'],
    'team1_win_rate': t1['win_rate'],
    'team1_total_games': t1['total_games'],
    'team1_wins': t1['wins'],
    'team2_rank': t2['rank'],
    'team2_win_rate': t2['win_rate'],
    'team2_total_games': t2['total_games'],
    'team2_wins': t2['wins'],
    'rank_difference': t1['rank'] - t2['rank'],
    'win_rate_difference': t1['win_rate'] - t2['win_rate'],
    'experience_difference': t1['total_games'] - t2['total_games'],
    
    # Recent form features (6)
    'team1_recent_win_rate': t1_form['recent_win_rate'],
    'team1_recent_wins': t1_form['recent_wins'],
    'team1_recent_matches': t1_form['recent_matches'],
    'team2_recent_win_rate': t2_form['recent_win_rate'],
    'team2_recent_wins': t2_form['recent_wins'],
    'team2_recent_matches': t2_form['recent_matches'],
    
    # Head-to-head features (3) - MOVED BEFORE recent_form_difference
    'h2h_matches_played': h2h['h2h_matches'],
    'h2h_team1_wins': h2h['h2h_team1_wins'],
    'h2h_team1_win_rate': h2h['h2h_team1_win_rate'],
    
    # Derived feature (1) - MOVED TO END
    'recent_form_difference': t1_form['recent_win_rate'] - t2_form['recent_win_rate']

}
    
    # Make prediction
    X = pd.DataFrame([features])
    prediction_proba = model.predict_proba(X)[0]
    prediction = model.predict(X)[0]
    
    # Prepare response
    result = {
        'team1': {
            'name': t1['display_name'],
            'win_probability': float(prediction_proba[1] * 100),
            'rank': t1['rank'],
            'win_rate': f"{t1['win_rate']*100:.1f}%",
            'total_games': t1['total_games'],
            'recent_form': f"{t1_form['recent_wins']}/{t1_form['recent_matches']}" if t1_form['recent_matches'] > 0 else "N/A",
            'recent_win_rate': f"{t1_form['recent_win_rate']*100:.0f}%" if t1_form['recent_matches'] > 0 else "N/A",
            'found': t1['found']
        },
        'team2': {
            'name': t2['display_name'],
            'win_probability': float(prediction_proba[0] * 100),
            'rank': t2['rank'],
            'win_rate': f"{t2['win_rate']*100:.1f}%",
            'total_games': t2['total_games'],
            'recent_form': f"{t2_form['recent_wins']}/{t2_form['recent_matches']}" if t2_form['recent_matches'] > 0 else "N/A",
            'recent_win_rate': f"{t2_form['recent_win_rate']*100:.0f}%" if t2_form['recent_matches'] > 0 else "N/A",
            'found': t2['found']
        },
        'head_to_head': {
            'matches': h2h['h2h_matches'],
            'team1_wins': h2h['h2h_team1_wins'],
            'team2_wins': h2h['h2h_matches'] - h2h['h2h_team1_wins']
        },
        'prediction': 'team1' if prediction == 1 else 'team2',
        'confidence': float(max(prediction_proba) * 100),
        'model_version': 'V2 (71.7% accuracy)'
    }
    
    return jsonify(result)

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 VLR Match Predictor V2 - Web Interface")
    print("=" * 70)
    print(f"Model loaded: ✓ (V2 - 71.7% CV accuracy)")
    print(f"Teams in database: {len(rankings)}")
    print(f"Historical matches: {len(all_matches)}")
    print(f"\nServer starting at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)