"""
Enhanced Feature Engineering with Fuzzy Matching and Multi-Region Support
"""

import pandas as pd
import numpy as np
from datetime import datetime
from difflib import get_close_matches
import glob

class EnhancedFeatureEngineer:
    def __init__(self):
        self.rankings = None
        self.players = None
        self.team_lookup = {}
        
    def load_latest_data(self):
        """Load the most recent data files"""
        # Find the latest files
        ranking_files = glob.glob('data/raw/rankings_all_regions_*.csv')
        player_files = glob.glob('data/raw/players_all_regions_*.csv')
        match_files = glob.glob('data/raw/matches_*.csv')
        
        if ranking_files:
            latest_rankings = max(ranking_files)
            self.rankings = pd.read_csv(latest_rankings)
            print(f"Loaded rankings from {latest_rankings}")
        else:
            # Fallback to old data
            self.rankings = pd.read_csv('data/raw/team_rankings_na.csv')
            print("Using old NA-only rankings")
        
        if player_files:
            latest_players = max(player_files)
            self.players = pd.read_csv(latest_players)
            print(f"Loaded players from {latest_players}")
        else:
            self.players = pd.read_csv('data/raw/player_stats_na_90d.csv')
            print("Using old NA-only player stats")
        
        # Clean rankings data
        self.rankings[['wins', 'losses']] = self.rankings['record'].str.split('–', expand=True)
        self.rankings['wins'] = pd.to_numeric(self.rankings['wins'])
        self.rankings['losses'] = pd.to_numeric(self.rankings['losses'])
        self.rankings['total_games'] = self.rankings['wins'] + self.rankings['losses']
        self.rankings['win_rate'] = self.rankings['wins'] / self.rankings['total_games']
        
        # Create fuzzy matching lookup
        self.create_team_lookup()
        
        print(f"✓ Loaded {len(self.rankings)} teams and {len(self.players)} players")
        
    def create_team_lookup(self):
        """Create lookup dictionary for fuzzy team name matching"""
        team_names = self.rankings['team'].tolist()
        team_names_lower = [name.lower() for name in team_names]
        
        # Store both exact and lowercase matches
        for i, name in enumerate(team_names):
            self.team_lookup[name.lower()] = i
            self.team_lookup[name] = i
        
        self.team_names_lower = team_names_lower
        print(f"Created lookup for {len(set(team_names))} unique teams")
        
    def find_team_match(self, team_name):
        """Find matching team using exact or fuzzy matching"""
        # Try exact match
        if team_name.lower() in self.team_lookup:
            idx = self.team_lookup[team_name.lower()]
            return self.rankings.iloc[idx]
        
        # Try fuzzy match
        matches = get_close_matches(team_name.lower(), self.team_names_lower, n=1, cutoff=0.75)
        if matches:
            idx = self.team_lookup[matches[0]]
            return self.rankings.iloc[idx]
        
        return None
    
    def get_team_features(self, team_name):
        """Get features for a single team with fuzzy matching"""
        team_row = self.find_team_match(team_name)
        
        if team_row is not None:
            return {
                'rank': int(team_row['rank']),
                'win_rate': float(team_row['win_rate']),
                'total_games': int(team_row['total_games']),
                'wins': int(team_row['wins']),
                'found': True
            }
        else:
            # Unknown team - return defaults
            return {
                'rank': 999,
                'win_rate': 0.5,
                'total_games': 0,
                'wins': 0,
                'found': False
            }
    
    def filter_vct_matches(self, matches_df):
        """Filter out Game Changers matches to focus on VCT"""
        gc_indicators = ['gc', 'gozen', 'game changers', 'shopify rebellion gold']
        
        def is_gc_match(row):
            team1_lower = row['team1'].lower()
            team2_lower = row['team2'].lower()
            return any(ind in team1_lower or ind in team2_lower for ind in gc_indicators)
        
        before_count = len(matches_df)
        vct_matches = matches_df[~matches_df.apply(is_gc_match, axis=1)].copy()
        after_count = len(vct_matches)
        
        print(f"Filtered matches: {before_count} → {after_count} (removed {before_count - after_count} GC matches)")
        
        return vct_matches
    
    def create_match_features(self, matches_df, filter_gc=True):
        """Create features for each match"""
        if filter_gc:
            matches_df = self.filter_vct_matches(matches_df)
        
        features = []
        teams_found = 0
        teams_not_found = 0
        
        for idx, row in matches_df.iterrows():
            team1 = row['team1']
            team2 = row['team2']
            
            # Get team features
            t1_features = self.get_team_features(team1)
            t2_features = self.get_team_features(team2)
            
            # Track matching success
            if t1_features['found']:
                teams_found += 1
            else:
                teams_not_found += 1
            
            if t2_features['found']:
                teams_found += 1
            else:
                teams_not_found += 1
            
            # Create feature vector
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
        
        print(f"\nTeam matching results:")
        print(f"  Teams found: {teams_found} ({teams_found/(teams_found+teams_not_found)*100:.1f}%)")
        print(f"  Teams not found: {teams_not_found}")
        
        return pd.DataFrame(features)
    
    def prepare_training_data(self, output_path='data/processed/training_data_v2.csv', filter_gc=True):
        """Full pipeline: load data, create features, save"""
        print("=" * 60)
        print("ENHANCED FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        # Load reference data
        self.load_latest_data()
        
        # Load latest matches
        match_files = glob.glob('data/raw/matches_*.csv')
        if match_files:
            latest_matches = max(match_files)
            matches = pd.read_csv(latest_matches)
            print(f"Loaded {len(matches)} matches from {latest_matches}")
        else:
            matches = pd.read_csv('data/raw/recent_matches.csv')
            print("Using old match data")

        # Create features
        features_df = self.create_match_features(matches, filter_gc=filter_gc)
        print(f"Created {len(features_df.columns)-1} features for {len(features_df)} matches")

        # Save processed data
        features_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

        # Summary stats
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(features_df)}")
        print(f"Team 1 wins: {features_df['team1_won'].sum()} ({features_df['team1_won'].mean()*100:.1f}%)")
        print(f"Team 2 wins: {(1-features_df['team1_won']).sum()} ({(1-features_df['team1_won'].mean())*100:.1f}%)")
        
        # Check how many teams have real rankings
        real_rankings = (features_df['team1_rank'] != 999).sum() + (features_df['team2_rank'] != 999).sum()
        total_teams = len(features_df) * 2
        print(f"\nTeams with real rankings: {real_rankings}/{total_teams} ({real_rankings/total_teams*100:.1f}%)")
        
        print("\nFeature columns:", list(features_df.columns))
        print("=" * 60)
        
        return features_df

if __name__ == "__main__":
    engineer = EnhancedFeatureEngineer()
    features = engineer.prepare_training_data(filter_gc=True)
    
    print("\nSample features:")
    print(features.head(10))