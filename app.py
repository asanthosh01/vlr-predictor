"""
Flask Web Interface for VLR Match Predictor
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from difflib import get_close_matches

app = Flask(__name__)

# Load model and data
model = joblib.load('models/match_predictor.joblib')
rankings = pd.read_csv('data/raw/rankings_all_regions_20251013_195912.csv')

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
    
    # Fuzzy match
    matches = get_close_matches(team_name.lower(), team_names_lower, n=1, cutoff=0.75)
    if matches:
        idx = team_lookup[matches[0]]
        return rankings.iloc[idx]
    
    return None

def get_team_features(team_name):
    """Get features for a team"""
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

@app.route('/')
def home():
    """Main page"""
    # Get list of all teams for dropdown
    teams = sorted(rankings['team'].tolist())
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    data = request.json
    team1_name = data.get('team1')
    team2_name = data.get('team2')
    
    if not team1_name or not team2_name:
        return jsonify({'error': 'Both teams required'}), 400
    
    if team1_name == team2_name:
        return jsonify({'error': 'Teams must be different'}), 400
    
    # Get team features
    t1 = get_team_features(team1_name)
    t2 = get_team_features(team2_name)
    
    # Create feature vector
    features = {
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
        'experience_difference': t1['total_games'] - t2['total_games']
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
            'found': t1['found']
        },
        'team2': {
            'name': t2['display_name'],
            'win_probability': float(prediction_proba[0] * 100),
            'rank': t2['rank'],
            'win_rate': f"{t2['win_rate']*100:.1f}%",
            'total_games': t2['total_games'],
            'found': t2['found']
        },
        'prediction': 'team1' if prediction == 1 else 'team2',
        'confidence': float(max(prediction_proba) * 100)
    }
    
    return jsonify(result)

@app.route('/search_teams', methods=['GET'])
def search_teams():
    """Search teams by name"""
    query = request.args.get('q', '').lower()
    if len(query) < 2:
        return jsonify([])
    
    matches = [team for team in rankings['team'].tolist() if query in team.lower()]
    return jsonify(matches[:10])

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 VLR Match Predictor - Web Interface")
    print("=" * 70)
    print(f"Model loaded: ✓")
    print(f"Teams in database: {len(rankings)}")
    print(f"\nServer starting at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(debug=True, port=5000)