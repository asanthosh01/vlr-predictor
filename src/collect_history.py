"""Collect dated, completed VLR results with stable IDs and source URLs."""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from src.chronological import validate_matches

def parse_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    records = []
    for match in soup.select('a.match-item'):
        header = match.find_previous('div', class_='wf-label mod-large')
        if header is None:
            continue
        label = next(header.stripped_strings, '')
        try:
            day = datetime.strptime(label, '%a, %B %d, %Y').date().isoformat()
        except ValueError:
            continue
        teams = [x.get_text(' ', strip=True) for x in match.select('.match-item-vs-team-name')]
        scores = [x.get_text(strip=True) for x in match.select('.match-item-vs-team-score')]
        link = match.get('href', '')
        identifier = re.match(r'^/(\d+)/', link)
        if len(teams) != 2 or len(scores) != 2 or not identifier:
            continue
        if not all(x.isdigit() for x in scores) or scores[0] == scores[1]:
            continue
        records.append({'match_id': identifier.group(1), 'match_date': day,
            'team1': teams[0], 'team2': teams[1], 'score1': int(scores[0]),
            'score2': int(scores[1]), 'source_url': 'https://www.vlr.gg' + link})
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=int, default=20)
    parser.add_argument('--out', type=Path, default=Path('data/raw/matches.csv'))
    args = parser.parse_args()
    if not 1 <= args.pages <= 100:
        parser.error('--pages must be between 1 and 100')
    session, rows = requests.Session(), []
    session.headers['User-Agent'] = 'VLR-Predictor educational research'
    for page in range(1, args.pages+1):
        response = session.get('https://www.vlr.gg/matches/results', params={'page': page}, timeout=30)
        response.raise_for_status()
        found = parse_page(response.text)
        if not found:
            raise ValueError(f'No dated completed matches found on page {page}; inspect site markup')
        rows.extend(found)
        print(f'Page {page}: {len(found)} matches', flush=True)
        if page < args.pages:
            time.sleep(2)
    data = validate_matches(pd.DataFrame(rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    args.out.with_suffix('.provenance.json').write_text(json.dumps({
        'source': 'https://www.vlr.gg/matches/results', 'pages': args.pages,
        'retrieved_at': datetime.now(timezone.utc).isoformat(), 'unique_matches': len(data),
        'date_precision': 'day; same-day outcomes excluded from pre-match features'}, indent=2))
    print(f'Saved {len(data)} unique matches to {args.out}')

if __name__ == '__main__':
    main()
