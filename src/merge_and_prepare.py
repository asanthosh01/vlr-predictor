"""
Merge all data sources and prepare comprehensive training dataset
"""

import pandas as pd
import glob
from feature_engineering import EnhancedFeatureEngineer

class DataMerger:
    def __init__(self):
        self.engineer = EnhancedFeatureEngineer()
        
    def merge_all_matches(self):
        """Combine API and scraped matches"""
        print("=" * 70)
        print("MERGING ALL DATA SOURCES")
        print("=" * 70)
        
        all_matches = []
        
        # Get scraped matches
        scraped_files = glob.glob('data/raw/scraped_matches_*.csv')
        for file in scraped_files:
            df = pd.read_csv(file)
            print(f"✓ Loaded {len(df)} matches from {file}")
            all_matches.append(df)
        
        # Get API matches
        api_files = glob.glob('data/raw/matches_*.csv')
        for file in api_files:
            df = pd.read_csv(file)
            # Standardize column names to match scraped format
            if 'tournament_name' in df.columns:
                df['tournament'] = df['tournament_name']
            if 'round_info' in df.columns:
                df['stage'] = df['round_info']
            print(f"✓ Loaded {len(df)} matches from {file}")
            all_matches.append(df)
        
        # Combine
        combined = pd.concat(all_matches, ignore_index=True)
        
        # Remove duplicates based on teams and scores
        before = len(combined)
        combined = combined.drop_duplicates(subset=['team1', 'team2', 'score1', 'score2'])
        after = len(combined)
        
        print(f"\n✓ Combined {before} matches")
        print(f"✓ Removed {before - after} duplicates")
        print(f"✓ Final dataset: {after} unique matches")
        
        return combined
    
    def prepare_full_dataset(self, filter_gc=True):
        """Create features for all matches"""
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING ON FULL DATASET")
        print("=" * 70)
        
        # Merge matches
        all_matches = self.merge_all_matches()
        
        # Load reference data
        self.engineer.load_latest_data()
        
        # Create features
        features_df = self.engineer.create_match_features(all_matches, filter_gc=filter_gc)
        
        # Save
        output_path = 'data/processed/training_data_full.csv'
        features_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(features_df)} samples to {output_path}")
        
        # Stats
        print("\n" + "=" * 70)
        print("FINAL DATASET STATISTICS")
        print("=" * 70)
        print(f"Total training samples: {len(features_df)}")
        print(f"Team 1 wins: {features_df['team1_won'].sum()} ({features_df['team1_won'].mean()*100:.1f}%)")
        print(f"Team 2 wins: {(~features_df['team1_won'].astype(bool)).sum()} ({(1-features_df['team1_won'].mean())*100:.1f}%)")
        
        # Real rankings coverage
        teams_with_rank = ((features_df['team1_rank'] != 999).sum() + 
                          (features_df['team2_rank'] != 999).sum())
        total_teams = len(features_df) * 2
        print(f"\nTeams with real rankings: {teams_with_rank}/{total_teams} ({teams_with_rank/total_teams*100:.1f}%)")
        
        print("\n✓ Ready for model training!")
        print("=" * 70)
        
        return features_df

if __name__ == "__main__":
    merger = DataMerger()
    dataset = merger.prepare_full_dataset(filter_gc=True)
    
    print("\nSample data:")
    print(dataset.head(10))