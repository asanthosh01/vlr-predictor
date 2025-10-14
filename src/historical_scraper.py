"""
Historical Match Scraper for VLR.gg
Scrapes past match results to build comprehensive training dataset
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re

class VLRHistoricalScraper:
    def __init__(self):
        self.base_url = "https://www.vlr.gg"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
        self.matches = []
        
    def get_results_page(self, page=1):
        """
        Fetch a single page of match results
        vlr.gg uses ?page=X for pagination
        """
        url = f"{self.base_url}/matches/results"
        if page > 1:
            url += f"?page={page}"
        
        print(f"Fetching page {page}...")
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"✗ Error fetching page {page}: {e}")
            return None
    
    def parse_match_block(self, match_div):
        """
        Parse a single match block from the HTML
        """
        try:
            # Get team names
            teams = match_div.find_all('div', class_='match-item-vs-team-name')
            if len(teams) < 2:
                return None
            
            team1 = teams[0].get_text(strip=True)
            team2 = teams[1].get_text(strip=True)
            
            # Get scores
            scores = match_div.find_all('div', class_='match-item-vs-team-score')
            score1 = scores[0].get_text(strip=True) if len(scores) > 0 else '0'
            score2 = scores[1].get_text(strip=True) if len(scores) > 1 else '0'
            
            # Convert scores to integers
            try:
                score1 = int(score1)
                score2 = int(score2)
            except:
                return None  # Skip matches without valid scores
            
            # Get tournament info
            tournament_div = match_div.find('div', class_='match-item-event')
            tournament = tournament_div.get_text(strip=True) if tournament_div else "Unknown"
            
            # Get match date/time
            time_div = match_div.find('div', class_='match-item-time')
            time_completed = time_div.get_text(strip=True) if time_div else "Unknown"
            
            # Get match page URL
            match_link = match_div.find('a', class_='match-item')
            match_url = match_link['href'] if match_link and 'href' in match_link.attrs else ""
            
            # Get event stage/round
            stage_div = match_div.find('div', class_='match-item-event-series')
            stage = stage_div.get_text(strip=True) if stage_div else "Unknown"
            
            return {
                'team1': team1,
                'team2': team2,
                'score1': score1,
                'score2': score2,
                'tournament': tournament,
                'stage': stage,
                'time_completed': time_completed,
                'match_url': match_url
            }
            
        except Exception as e:
            print(f"  ⚠ Error parsing match: {e}")
            return None
    
    def scrape_matches(self, num_pages=10):
        """
        Scrape multiple pages of match results
        """
        print("=" * 70)
        print("VLR.GG HISTORICAL MATCH SCRAPER")
        print("=" * 70)
        
        all_matches = []
        
        for page in range(1, num_pages + 1):
            html = self.get_results_page(page)
            
            if not html:
                print(f"✗ Failed to fetch page {page}, stopping")
                break
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find all match containers
            match_divs = soup.find_all('a', class_='wf-module-item')
            
            if not match_divs:
                print(f"✗ No matches found on page {page}, stopping")
                break
            
            print(f"  Found {len(match_divs)} matches on page {page}")
            
            page_matches = 0
            for match_div in match_divs:
                match_data = self.parse_match_block(match_div)
                if match_data:
                    all_matches.append(match_data)
                    page_matches += 1
            
            print(f"  ✓ Parsed {page_matches} valid matches")
            
            # Be respectful to the server
            time.sleep(2)  # 2 second delay between pages
        
        self.matches = all_matches
        
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        print(f"Total matches collected: {len(all_matches)}")
        
        return all_matches
    
    def save_matches(self, filename=None):
        """Save scraped matches to CSV"""
        if not self.matches:
            print("No matches to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/raw/scraped_matches_{timestamp}.csv"
        
        df = pd.DataFrame(self.matches)
        df.to_csv(filename, index=False)
        print(f"✓ Saved {len(self.matches)} matches to {filename}")
        
        return df
    
    def analyze_data(self):
        """Quick analysis of scraped data"""
        if not self.matches:
            return
        
        df = pd.DataFrame(self.matches)
        
        print("\n" + "=" * 70)
        print("DATA ANALYSIS")
        print("=" * 70)
        
        print(f"\nTotal matches: {len(df)}")
        print(f"\nUnique teams: {len(set(df['team1'].tolist() + df['team2'].tolist()))}")
        print(f"\nUnique tournaments: {df['tournament'].nunique()}")
        
        print("\nTop 10 tournaments by match count:")
        print(df['tournament'].value_counts().head(10))
        
        print("\nSample matches:")
        print(df[['team1', 'team2', 'score1', 'score2', 'tournament']].head())
        
        print("=" * 70)

# Test the scraper
if __name__ == "__main__":
    scraper = VLRHistoricalScraper()
    
    print("Starting scraper...")
    print("This will scrape 10 pages (~500 matches)")
    print("Estimated time: 2-3 minutes")
    print()
    
    # Scrape matches
    matches = scraper.scrape_matches(num_pages=10)
    
    # Analyze
    scraper.analyze_data()
    
    # Save
    df = scraper.save_matches()
    
    print("\n✓ Done! Check data/raw/ folder for the CSV file")