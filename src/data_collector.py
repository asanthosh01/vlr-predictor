"""
VLR.gg Data Collector
Fetches match data using the vlrggapi REST API
"""

import requests
import pandas as pd
import time
from datetime import datetime
import json

class VLRDataCollector:
    def __init__(self):
        self.base_url = "https://vlrggapi.vercel.app"
        
    def get_match_results(self, pages=5):
        """
        Fetch completed match results
        """
        print("Fetching match results...")
        
        try:
            response = requests.get(f"{self.base_url}/match?q=results")
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('data', {}).get('status') == 200:
                matches = data['data']['segments']
                print(f"✓ Fetched {len(matches)} matches")
                return matches
            else:
                print("✗ Failed to fetch data")
                return []
                
        except requests.RequestException as e:
            print(f"✗ Error: {e}")
            return []
    
    def get_team_rankings(self, region='na'):
        """
        Fetch team rankings for a specific region
        """
        print(f"Fetching {region.upper()} rankings...")
        
        try:
            response = requests.get(f"{self.base_url}/rankings?region={region}")
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 200:
                rankings = data['data']
                print(f"✓ Fetched {len(rankings)} teams")
                return rankings
            else:
                print("✗ Failed to fetch rankings")
                return []
                
        except requests.RequestException as e:
            print(f"✗ Error: {e}")
            return []
    
    def get_player_stats(self, region='na', timespan='90'):
        """
        Fetch player statistics
        """
        print(f"Fetching player stats ({region.upper()}, last {timespan} days)...")

        try:
            response = requests.get(
                f"{self.base_url}/stats?region={region}&timespan={timespan}"
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('data', {}).get('status') == 200:
                stats = data['data']['segments']
                print(f"✓ Fetched stats for {len(stats)} players")
                return stats
            else:
                print("✗ Failed to fetch stats")
                return []
                
        except requests.RequestException as e:
            print(f"✗ Error: {e}")
            return []
    
    def save_to_csv(self, data, filename):
        """
        Save data to CSV in the data/raw folder
        """
        if not data:
            print("No data to save")
            return
        
        df = pd.DataFrame(data)
        filepath = f"data/raw/{filename}"
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath}")
        
        return df

# Test the collector
if __name__ == "__main__":
    print("=" * 60)
    print("VLR Data Collector Test")
    print("=" * 60)
    
    collector = VLRDataCollector()
    
    # Test 1: Get match results
    print("\n[1] Testing match results...")
    matches = collector.get_match_results()
    if matches:
        print(f"    Sample match: {matches[0]['team1']} vs {matches[0]['team2']}")
        collector.save_to_csv(matches, 'recent_matches.csv')
    
    # Test 2: Get team rankings
    print("\n[2] Testing team rankings...")
    rankings = collector.get_team_rankings(region='na')
    if rankings:
        print(f"    #1 Team: {rankings[0]['team']}")
        collector.save_to_csv(rankings, 'team_rankings_na.csv')
    
    # Test 3: Get player stats
    print("\n[3] Testing player stats...")
    stats = collector.get_player_stats(region='na', timespan='90')
    if stats:
        print(f"    Top player: {stats[0]['player']} ({stats[0]['org']})")
        collector.save_to_csv(stats, 'player_stats_na_90d.csv')
    
    print("\n" + "=" * 60)
    print("✓ All tests complete! Check data/raw/ folder for CSVs")
    print("=" * 60)