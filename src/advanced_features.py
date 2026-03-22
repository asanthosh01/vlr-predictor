"""
Advanced Feature Engineering - Recent Form & Head-to-Head
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import glob

class AdvancedFeatureEngineer:
    def __init__(self):
        self.matches_df = None
        self.team_match_history = defaultdict(list)
        
    def load_all_matches(self):
        """Load and sort all matches chronologically"""
        all_files = glob.glob('data/raw/scraped_matches_*.csv')
        dfs = [pd.read_csv(f) for f in all_files]
        self.matches_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates
        self.matches_df = self.matches_df.drop_duplicates(
            subset=['team1', 'team2', 'score1', 'score2']
        )
        
        print(f"✓ Loaded {len(self.matches_df)} unique matches")
        return self.matches_df
    
    def calculate_recent_form(self, team_name, current_match_idx, window=5):
        """
        Calculate team's win rate in last N matches before current match
        """
        team_matches = []
        
        # Get all matches involving this team before current match
        for idx in range(current_match_idx):
            row = self.matches_df.iloc[idx]
            
            if row['team1'] == team_name:
                won = row['score1'] > row['score2']
                team_matches.append({
                    'won': won,
                    'idx': idx
                })
            elif row['team2'] == team_name:
                won = row['score2'] > row['score1']
                team_matches.append({
                    'won': won,
                    'idx': idx
                })
        
        # Take last N matches
        recent = team_matches[-window:] if len(team_matches) >= window else team_matches
        
        if not recent:
            return {
                'recent_win_rate': 0.5,
                'recent_wins': 0,
                'recent_matches': 0,
                'form_trend': 0
            }
        
        wins = sum(m['won'] for m in recent)
        
        return {
            'recent_win_rate': wins / len(recent),
            'recent_wins': wins,
            'recent_matches': len(recent),
            'form_trend': wins / len(recent) if len(recent) > 0 else 0
        }
    
    def calculate_head_to_head(self, team1, team2, current_match_idx):
        """
        Calculate historical head-to-head record between two teams
        """
        h2h_matches = []
        
        for idx in range(current_match_idx):
            row = self.matches_df.iloc[idx]
            
            # Check if these two teams played each other
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
    
    def create_advanced_features(self):
        """
        Create dataset with advanced features
        """
        print("\n" + "=" * 70)
        print("CREATING ADVANCED FEATURES")
        print("=" * 70)
        
        self.load_all_matches()
        
        # IMPORTANT: Reset index after deduplication
        self.matches_df = self.matches_df.reset_index(drop=True)
        
        advanced_data = []
        
        for idx, row in self.matches_df.iterrows():
            if idx % 50 == 0:
                print(f"Processing match {idx}/{len(self.matches_df)}...")
            
            team1 = row['team1']
            team2 = row['team2']
            
            # Recent form for both teams
            t1_form = self.calculate_recent_form(team1, idx, window=5)
            t2_form = self.calculate_recent_form(team2, idx, window=5)
            
            # Head-to-head
            h2h = self.calculate_head_to_head(team1, team2, idx)
            
            # Create feature dict
            features = {
                # Team 1 recent form
                'team1_recent_win_rate': t1_form['recent_win_rate'],
                'team1_recent_wins': t1_form['recent_wins'],
                'team1_recent_matches': t1_form['recent_matches'],
                
                # Team 2 recent form
                'team2_recent_win_rate': t2_form['recent_win_rate'],
                'team2_recent_wins': t2_form['recent_wins'],
                'team2_recent_matches': t2_form['recent_matches'],
                
                # Head-to-head
                'h2h_matches_played': h2h['h2h_matches'],
                'h2h_team1_wins': h2h['h2h_team1_wins'],
                'h2h_team1_win_rate': h2h['h2h_team1_win_rate'],
                
                # Derived features
                'recent_form_difference': t1_form['recent_win_rate'] - t2_form['recent_win_rate'],
                
                # Original match data
                'team1': team1,
                'team2': team2,
                'score1': row['score1'],
                'score2': row['score2'],
                'team1_won': int(row['score1'] > row['score2'])
            }
            
            advanced_data.append(features)
        
        df = pd.DataFrame(advanced_data)
        
        # Save
        output_path = 'data/processed/advanced_features.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✓ Created {len(df)} samples with advanced features")
        print(f"✓ Saved to {output_path}")
        
        # Stats
        print("\n" + "=" * 70)
        print("FEATURE STATISTICS")
        print("=" * 70)
        print(f"\nMatches with H2H history: {(df['h2h_matches_played'] > 0).sum()}")
        print(f"Average H2H matches: {df['h2h_matches_played'].mean():.2f}")
        print(f"\nTeam 1 recent form: {df['team1_recent_win_rate'].mean():.3f}")
        print(f"Team 2 recent form: {df['team2_recent_win_rate'].mean():.3f}")
        
        return df

if __name__ == "__main__":
    engineer = AdvancedFeatureEngineer()
    df = engineer.create_advanced_features()
    
    print("\nSample features:")
    print(df[['team1', 'team2', 'team1_recent_win_rate', 'team2_recent_win_rate', 
              'h2h_matches_played', 'team1_won']].head(10))