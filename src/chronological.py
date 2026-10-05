"""Reproducible pre-match features; never use present-day ranking snapshots."""
from collections import defaultdict, deque
from dataclasses import dataclass, field
import pandas as pd

FEATURES = ['elo_difference', 'win_rate_difference', 'experience_difference',
            'recent_form_difference', 'h2h_win_rate', 'h2h_matches',
            'team1_matches', 'team2_matches']

def validate_matches(frame):
    required = {'match_id', 'match_date', 'team1', 'team2', 'score1', 'score2'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    frame = frame.copy()
    frame['match_date'] = pd.to_datetime(frame['match_date'], errors='raise').dt.normalize()
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing match values are not allowed')
    for col in ['team1', 'team2']:
        frame[col] = frame[col].astype(str).str.strip()
        if (frame[col] == '').any():
            raise ValueError('Team names must be nonempty')
    for col in ['score1', 'score2']:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if (frame[col] < 0).any() or (frame[col] % 1 != 0).any():
            raise ValueError('Scores must be nonnegative integers')
    if (frame.team1 == frame.team2).any():
        raise ValueError('A team cannot play itself')
    if (frame.score1 == frame.score2).any():
        raise ValueError('Only completed, decisive matches are supported')
    if frame.match_id.duplicated().any():
        duplicated = frame[frame.match_id.duplicated(keep=False)]
        if (duplicated.groupby('match_id')[list(required - {'match_id'})].nunique() > 1).any().any():
            raise ValueError('Conflicting rows for the same match ID')
        frame = frame.drop_duplicates('match_id')
    return frame.sort_values(['match_date', 'match_id']).reset_index(drop=True)

@dataclass
class History:
    elo: dict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    wins: dict = field(default_factory=lambda: defaultdict(int))
    games: dict = field(default_factory=lambda: defaultdict(int))
    recent: dict = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    h2h: dict = field(default_factory=lambda: defaultdict(lambda: [0, 0]))

    def features(self, a, b):
        winrate = lambda t: (self.wins[t] + 1) / (self.games[t] + 2)
        recent = lambda t: sum(self.recent[t]) / len(self.recent[t]) if self.recent[t] else .5
        key = tuple(sorted([a, b]))
        first_wins, count = self.h2h[key]
        a_wins = first_wins if a == key[0] else count - first_wins
        return dict(zip(FEATURES, [self.elo[a] - self.elo[b], winrate(a) - winrate(b),
            self.games[a] - self.games[b], recent(a) - recent(b),
            (a_wins + 1) / (count + 2), count, self.games[a], self.games[b]]))

    def update_day(self, rows):
        # Matches on the same day have uncertain completion order: all use the
        # previous day's ratings. Their outcomes become available the next day.
        deltas = defaultdict(float)
        for row in rows:
            a, b = row.team1, row.team2
            y = int(row.score1 > row.score2)
            expected = 1 / (1 + 10 ** ((self.elo[b] - self.elo[a]) / 400))
            change = 24 * (y - expected)
            deltas[a] += change
            deltas[b] -= change
            for team, won in [(a, y), (b, 1-y)]:
                self.games[team] += 1
                self.wins[team] += won
                self.recent[team].append(won)
            key = tuple(sorted([a, b]))
            self.h2h[key][0] += y if a == key[0] else 1-y
            self.h2h[key][1] += 1
        for team, delta in deltas.items():
            self.elo[team] += delta

def build_features(matches):
    matches = validate_matches(matches)
    state, records = History(), []
    for day, group in matches.groupby('match_date', sort=True):
        rows = list(group.itertuples(index=False))
        for row in rows:
            records.append({'match_id': row.match_id, 'match_date': day,
                'team1_won': int(row.score1 > row.score2),
                **state.features(row.team1, row.team2)})
        state.update_day(rows)
    return pd.DataFrame(records), state

def chronological_split(features):
    days = sorted(features.match_date.unique())
    if len(days) < 10 or len(features) < 50:
        raise ValueError('Need at least 50 matches across 10 dates for evaluation')
    first, second = max(1, int(len(days)*.6)), int(len(days)*.8)
    return [features[features.match_date.isin(d)] for d in
            [days[:first], days[first:second], days[second:]]]
