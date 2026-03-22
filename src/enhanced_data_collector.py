"""
Enhanced Data Collector - Handles VCT and Game Changers separately
"""

import requests
import pandas as pd
import time
from datetime import datetime
from difflib import get_close_matches

class EnhancedVLRCollector:
    def __init__(self):
        self.base_url = "https://vlrggapi.vercel.app"
        # VCT regions
        self.vct_regions = ['na', 'eu', 'ap', 'br', 'la', 'oce', 'kr', 'jp', 'cn']
        
    def get_match_results(self):
        """Fetch completed match results"""
        print("Fetching match results...")
        try:
            response = requests.get(f"{self.base_url}/match?q=results")
            response.raise_for_status()
            data = response.json()
            
            if data.get('data', {}).get('status') == 200:
                matches = data['data']['segments']
                print(f"Fetched {len(matches)} matches")
                return matches
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def get_rankings_all_regions(self):
        """Fetch team rankings from all VCT regions"""
        all_rankings = []
        
        for region in self.vct_regions:
            print(f"  Fetching {region.upper()} rankings...")
            try:
                response = requests.get(f"{self.base_url}/rankings?region={region}")
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 200:
                    rankings = data['data']
                    for team in rankings:
                        team['source_region'] = region
                        team['is_gc'] = False
                    all_rankings.extend(rankings)
                    print(f"    {len(rankings)} teams")
                    time.sleep(0.5)  # Be nice to API
            except Exception as e:
                print(f"    Error for {region}: {e}")
                continue
        
        return all_rankings
    
    def get_player_stats_all_regions(self, timespan='90'):
        """Fetch player stats from all regions"""
        all_players = []
        
        for region in self.vct_regions:
            print(f"  Fetching {region.upper()} player stats...")
            try:
                response = requests.get(
                    f"{self.base_url}/stats?region={region}&timespan={timespan}"
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get('data', {}).get('status') == 200:
                    players = data['data']['segments']
                    for player in players:
                        player['source_region'] = region
                    all_players.extend(players)
                    print(f"    {len(players)} players")
                    time.sleep(0.5)
            except Exception as e:
                print(f"    Error for {region}: {e}")
                continue
        
        return all_players
    
    def categorize_teams(self, matches_df):
        """
        Analyze which teams are VCT vs Game Changers vs unknown
        """
        print("\n" + "=" * 70)
        print("TEAM CATEGORIZATION ANALYSIS")
        print("=" * 70)
        
        all_teams = set(matches_df['team1'].tolist() + matches_df['team2'].tolist())
        
        # Identify GC teams (have "GC", "Gozen", "X", or other GC indicators)
        gc_indicators = ['gc', 'gozen', 'game changers', 'shopify rebellion gold']
        
        gc_teams = set()
        vct_teams = set()
        unknown_teams = set()
        
        for team in all_teams:
            team_lower = team.lower()
            if any(indicator in team_lower for indicator in gc_indicators):
                gc_teams.add(team)
            else:
                # Could be VCT or unknown - we'll check against rankings
                unknown_teams.add(team)
        
        print(f"\nTotal unique teams in matches: {len(all_teams)}")
        print(f"  • Identified Game Changers teams: {len(gc_teams)}")
        print(f"  • Potential VCT teams: {len(unknown_teams)}")
        
        if gc_teams:
            print(f"\nSample GC teams:")
            for team in list(gc_teams)[:5]:
                print(f"  • {team}")
        
        return {
            'all_teams': all_teams,
            'gc_teams': gc_teams,
            'vct_teams': vct_teams,
            'unknown_teams': unknown_teams
        }
    
    def analyze_coverage(self, matches_df, rankings_df):
        """
        Analyze how much of our match data can be covered by rankings
        """
        print("\n" + "=" * 70)
        print("DATA COVERAGE ANALYSIS")
        print("=" * 70)
        
        # Get team categorization
        team_cats = self.categorize_teams(matches_df)
        
        # Create ranking lookup
        ranking_teams = set(rankings_df['team'].str.lower().tolist())
        
        # Check coverage
        matched = 0
        unmatched = []
        
        for team in team_cats['all_teams']:
            if team.lower() in ranking_teams:
                matched += 1
            else:
                # Try fuzzy matching
                close_matches = get_close_matches(team.lower(), ranking_teams, n=1, cutoff=0.8)
                if close_matches:
                    matched += 1
                else:
                    unmatched.append(team)
        
        print(f"\nRanking Coverage:")
        print(f"  • Teams in rankings: {len(ranking_teams)}")
        print(f"  • Teams in matches: {len(team_cats['all_teams'])}")
        print(f"  • Successfully matched: {matched} ({matched/len(team_cats['all_teams'])*100:.1f}%)")
        print(f"  • Unmatched: {len(unmatched)} ({len(unmatched)/len(team_cats['all_teams'])*100:.1f}%)")
        
        if unmatched:
            print(f"\nUnmatched teams (first 10):")
            for team in list(unmatched)[:10]:
                gc_indicator = " [GC]" if team in team_cats['gc_teams'] else ""
                print(f"  • {team}{gc_indicator}")
        
        print("\n💡 Insight:")
        gc_unmatched = len([t for t in unmatched if t in team_cats['gc_teams']])
        if gc_unmatched > len(unmatched) * 0.5:
            print(f"  Most unmatched teams ({gc_unmatched}/{len(unmatched)}) are Game Changers teams.")
            print(f"  This is expected - GC teams have separate rankings.")
            print(f"  Consider: Collect GC-specific rankings or exclude GC matches from VCT model.")
        
        print("=" * 70)
    
    def collect_all_data(self):
        """Comprehensive data collection"""
        print("=" * 70)
        print("ENHANCED DATA COLLECTION - ALL REGIONS")
        print("=" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Collect matches
        print("\n[1/3] Collecting match results...")
        matches = self.get_match_results()
        df_matches = None
        if matches:
            df_matches = pd.DataFrame(matches)
            filepath = f"data/raw/matches_{timestamp}.csv"
            df_matches.to_csv(filepath, index=False)
            print(f"Saved {len(matches)} matches to {filepath}")
        
        # 2. Collect rankings from all regions
        print("\n[2/3] Collecting VCT team rankings from all regions...")
        rankings = self.get_rankings_all_regions()
        df_rankings = None
        if rankings:
            df_rankings = pd.DataFrame(rankings)
            filepath = f"data/raw/rankings_all_regions_{timestamp}.csv"
            df_rankings.to_csv(filepath, index=False)
            print(f"Saved {len(rankings)} teams to {filepath}")
        
        # 3. Collect player stats from all regions
        print("\n[3/3] Collecting player stats from all regions...")
        players = self.get_player_stats_all_regions()
        if players:
            df_players = pd.DataFrame(players)
            filepath = f"data/raw/players_all_regions_{timestamp}.csv"
            df_players.to_csv(filepath, index=False)
            print(f"Saved {len(players)} players to {filepath}")
        
        # 4. Analyze coverage
        if df_matches is not None and df_rankings is not None:
            self.analyze_coverage(df_matches, df_rankings)
        
        print("\n" + "=" * 70)
        print("COLLECTION COMPLETE!")
        print("=" * 70)
        print(f"\nTotal collected:")
        print(f"  • {len(matches) if matches else 0} matches")
        print(f"  • {len(rankings) if rankings else 0} VCT teams across {len(self.vct_regions)} regions")
        print(f"  • {len(players) if players else 0} players")
        print("=" * 70)

if __name__ == "__main__":
    collector = EnhancedVLRCollector()
    collector.collect_all_data()