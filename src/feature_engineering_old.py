cat << 'EOF' > src/feature_engineering.py
"""
Feature Engineering for Match Prediction
Transforms raw match data into features for ML model
"""

import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        self.rankings = None
        self.players = None
        
    def load_reference_data(self):
        """Load rankings and player stats for feature creation"""
        self.rankings = pd.read_csv('data/raw/team_rankings_na.csv')
        self.players = pd.read_csv('data/raw/player_stats_na_90d.csv')
        
        # Clean rankings data
        self.rankings[['wins', 'losses']] = self.rankings['record'].str.split('–', expand=True)
        self.rankings['wins'] = pd.to_numeric(self.rankings['wins'])
        self.rankings['losses'] = pd.to_numeric(self.rankings['losses'])
        self.rankings['total_games'] = self.rankings['wins'] + self.rankings['losses']
        self.rankings['win_rate'] = self.rankings['wins'] / self.rankings['total_games']
        
        # Create team lookup dictionary
        self.team_stats = self.rankings.set_index('team').to_dict('index')
        
        print("✓ Reference data loaded")
        
    def get_team_features(self, team_name):
        """Get features for a single team"""
        if team_name in self.team_stats:
            stats = self.team_stats[team_name]
            return {
                'rank': int(stats.get('rank', 999)),
                'win_rate': float(stats.get('win_rate', 0.5)),
                'total_games': int(stats.get('total_games', 0)),
                'wins': int(stats.get('wins', 0)),
            }
        else:
            return {
                'rank': 999,
                'win_rate': 0.5,
                'total_games': 0,
                'wins': 0,
            }
    
    def create_match_features(self, matches_df):
        features = []
        
        for idx, row in matches_df.iterrows():
            team1 = row['team1']
            team2 = row['team2']
            
            t1_features = self.get_team_features(team1)
            t2_features = self.get_team_features(team2)
            
            feature_dict = {
                'team1_rank': t1_features['rank'],
                'team1_win_rate': t1_features['win_rate'],
                'team1_total_games': t1_features['total_games'],
                'team1_wins': t1_features['wins'],
                'team2_rank': t2_features['rank'],
                'team2_win_rate': t2_features['win_rate'],
                'team2_total_games': t2_features['total_games'],
                'team2_wins': t2_features['wins'],
                'rank_difference': t1_features['rank'] - t2_features['rank'],
                'win_rate_difference': t1_features['win_rate'] - t2_features['win_rate'],
                'experience_difference': t1_features['total_games'] - t2_features['total_games'],
                'team1_won': int(row['score1'] > row['score2'])
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def prepare_training_data(self, output_path='data/processed/training_data.csv'):
        print("=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        self.load_reference_data()
        
        matches = pd.read_csv('data/raw/recent_matches.csv')
        print(f"Loaded {len(matches)} matches")
        
        features_df = self.create_match_features(matches)
        print(f"Created {len(features_df.columns)-1} features")

        features_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(features_df)}")
        print(f"Team 1 wins: {features_df['team1_won'].sum()} ({features_df['team1_won'].mean()*100:.1f}%)")
        print(f"Team 2 wins: {(1-features_df['team1_won']).sum()} ({(1-features_df['team1_won'].mean())*100:.1f}%)")
        print("\nFeature columns:", list(features_df.columns))
        print("=" * 60)
        
        return features_df

if __name__ == "__main__":
    engineer = FeatureEngineer()
    features = engineer.prepare_training_data()
    
    print("\nSample features:")
    print(features.head())
EOF