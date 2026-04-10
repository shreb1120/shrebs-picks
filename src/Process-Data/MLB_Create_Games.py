"""Merge MLB team stats + pitcher stats + odds into a training dataset.

Produces Data/MLB_dataset.sqlite with ~74 features per game.
For seasons without odds data (pre-2023), fetches scores from the MLB API
and builds training rows without moneyline/OU odds.
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

BASE_DIR = Path(__file__).resolve().parents[2]
TEAM_DB = BASE_DIR / "Data" / "MLB_TeamData.sqlite"
PITCHER_DB = BASE_DIR / "Data" / "MLB_PitcherData.sqlite"
ODDS_DB = BASE_DIR / "Data" / "MLB_OddsData.sqlite"
DATASET_DB = BASE_DIR / "Data" / "MLB_dataset.sqlite"

MLB_SCHEDULE_API = "https://statsapi.mlb.com/api/v1/schedule"

# Features we extract from team stats (already prefixed with bat_/pit_)
TEAM_BAT_FEATURES = [
    'bat_AVG', 'bat_OBP', 'bat_SLG', 'bat_OPS', 'bat_wRC+', 'bat_ISO',
    'bat_BABIP', 'bat_wOBA', 'bat_K%', 'bat_BB%', 'bat_HR_per_G',
    'bat_R_per_G', 'bat_SB_per_G', 'bat_WAR', 'bat_RBI_per_G',
]

TEAM_PIT_FEATURES = [
    'pit_ERA', 'pit_WHIP', 'pit_FIP', 'pit_xFIP', 'pit_K/9', 'pit_BB/9',
    'pit_HR/9', 'pit_K%', 'pit_BB%', 'pit_LOB%', 'pit_BABIP', 'pit_WAR',
]

SP_FEATURES = [
    'sp_ERA', 'sp_WHIP', 'sp_FIP', 'sp_xFIP', 'sp_K/9', 'sp_BB/9',
    'sp_HR/9', 'sp_K%', 'sp_BB%', 'sp_WAR', 'sp_GS', 'sp_IP_per_GS',
]


def _load_team_stats(season):
    """Load team stats for a season."""
    table = f"team_stats_{season}"
    with sqlite3.connect(TEAM_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    return df


def _load_pitcher_stats(season):
    """Load individual pitcher stats for a season."""
    table = f"pitcher_stats_{season}"
    with sqlite3.connect(PITCHER_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    return df


def _load_game_starters(season):
    """Load game-level starter assignments."""
    table = f"game_starters_{season}"
    with sqlite3.connect(PITCHER_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
    return df


def _load_odds(season_key):
    """Load odds/scores for a season. Returns None if not available."""
    if not ODDS_DB.exists():
        return None
    try:
        with sqlite3.connect(ODDS_DB) as con:
            tables = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (season_key,),
            ).fetchone()
            if not tables:
                return None
            df = pd.read_sql_query(f'SELECT * FROM "{season_key}"', con)
            return df if not df.empty else None
    except Exception:
        return None


def _fetch_season_scores(season):
    """Fetch completed game scores from MLB Schedule API for a season."""
    print(f"  Fetching scores from MLB API for {season}...")
    params = {
        'sportId': 1,
        'season': season,
        'gameType': 'R',
        'hydrate': 'linescore',
    }
    resp = requests.get(MLB_SCHEDULE_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get('dates', []):
        game_date = date_entry['date']
        for game in date_entry.get('games', []):
            status = game.get('status', {}).get('detailedState', '')
            if status not in ('Final', 'Completed Early', 'Game Over'):
                continue

            home_info = game.get('teams', {}).get('home', {})
            away_info = game.get('teams', {}).get('away', {})

            home_score = home_info.get('score', 0) or 0
            away_score = away_info.get('score', 0) or 0

            rows.append({
                'Date': game_date,
                'Home': home_info.get('team', {}).get('name', ''),
                'Away': away_info.get('team', {}).get('name', ''),
                'Points': home_score + away_score,
                'Win_Margin': home_score - away_score,
                'OU': 0,  # No line available
                'ML_Home': 0,
                'ML_Away': 0,
                'Days_Rest_Home': 4,
                'Days_Rest_Away': 4,
            })

    # Compute days rest
    df = pd.DataFrame(rows)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        last_played = {}
        rest_home = []
        rest_away = []
        for _, row in df.iterrows():
            d = row['Date']
            home, away = row['Home'], row['Away']
            rest_home.append((d - last_played[home]).days if home in last_played else 4)
            rest_away.append((d - last_played[away]).days if away in last_played else 4)
            last_played[home] = d
            last_played[away] = d
        df['Days_Rest_Home'] = rest_home
        df['Days_Rest_Away'] = rest_away

    time.sleep(1)
    return df


def _get_league_avg_sp(pitcher_df):
    """Compute league-average starter stats for imputation."""
    sp_cols = [c for c in pitcher_df.columns if c.startswith('sp_')]
    avg = {}
    for col in sp_cols:
        avg[col] = pitcher_df[col].median()
    return avg


def _lookup_pitcher_stats(pitcher_df, pitcher_name, league_avg):
    """Look up a pitcher's stats by name, falling back to league average."""
    if not pitcher_name:
        return league_avg

    matches = pitcher_df[pitcher_df['Name'] == pitcher_name]
    if matches.empty:
        return league_avg

    row = matches.iloc[0]
    stats = {}
    for col in SP_FEATURES:
        if col in row.index:
            val = row[col]
            stats[col] = val if pd.notna(val) else league_avg.get(col, 0)
        else:
            stats[col] = league_avg.get(col, 0)
    return stats


def _get_team_features(team_stats_df, team_name, prefix):
    """Extract team features with a home_/away_ prefix."""
    from src.Utils.MLB_Dictionaries import normalize_team_name, FULL_TO_ABBREV

    canonical = normalize_team_name(team_name)

    # Try matching by full name or abbreviation
    matches = team_stats_df[team_stats_df['Team'] == canonical]
    if matches.empty:
        abbrev = FULL_TO_ABBREV.get(canonical)
        if abbrev:
            matches = team_stats_df[team_stats_df['Team'] == abbrev]
    if matches.empty:
        # Fuzzy: try partial match
        for _, row in team_stats_df.iterrows():
            if canonical in str(row['Team']) or str(row['Team']) in canonical:
                matches = team_stats_df[team_stats_df.index == row.name]
                break

    if matches.empty:
        return {f"{prefix}{col}": np.nan for col in TEAM_BAT_FEATURES + TEAM_PIT_FEATURES}

    row = matches.iloc[0]
    features = {}
    for col in TEAM_BAT_FEATURES + TEAM_PIT_FEATURES:
        val = row.get(col, np.nan)
        features[f"{prefix}{col}"] = val
    return features


def build_dataset(seasons, dataset_name="mlb_dataset"):
    """Build the full training dataset across multiple seasons."""
    all_rows = []

    for season in seasons:
        print(f"Processing season {season}...")

        try:
            team_stats = _load_team_stats(season)
        except Exception as e:
            print(f"  No team stats for {season}: {e}")
            continue

        try:
            pitcher_stats = _load_pitcher_stats(season)
        except Exception as e:
            print(f"  No pitcher stats for {season}: {e}")
            pitcher_stats = pd.DataFrame()

        try:
            game_starters = _load_game_starters(season)
        except Exception as e:
            print(f"  No game starters for {season}: {e}")
            game_starters = pd.DataFrame()

        # Try loading odds; fall back to MLB API scores
        season_key = str(season)
        odds = _load_odds(season_key)
        if odds is None:
            print(f"  No odds data for {season}, fetching scores from MLB API...")
            odds = _fetch_season_scores(season)
            if odds is None or odds.empty:
                print(f"  No game data available for {season}, skipping.")
                continue

        league_avg = _get_league_avg_sp(pitcher_stats) if not pitcher_stats.empty else {}

        from src.Utils.MLB_Dictionaries import normalize_team_name

        for _, game in odds.iterrows():
            home_team = normalize_team_name(game['Home'], season)
            away_team = normalize_team_name(game['Away'], season)
            game_date = str(game['Date'])

            # Get team features
            home_features = _get_team_features(team_stats, home_team, 'home_')
            away_features = _get_team_features(team_stats, away_team, 'away_')

            # Get starter names from game_starters table
            home_sp_name = ''
            away_sp_name = ''
            if not game_starters.empty:
                # Normalize game_date for matching (handle both date formats)
                date_str = game_date[:10]  # Take YYYY-MM-DD portion
                starter_match = game_starters[
                    (game_starters['game_date'] == date_str) &
                    (game_starters['home_team'].apply(
                        lambda x: normalize_team_name(x, season)
                    ) == home_team)
                ]
                if not starter_match.empty:
                    home_sp_name = starter_match.iloc[0].get('home_starter', '')
                    away_sp_name = starter_match.iloc[0].get('away_starter', '')

            # Get pitcher features
            home_sp_stats = _lookup_pitcher_stats(pitcher_stats, home_sp_name, league_avg)
            away_sp_stats = _lookup_pitcher_stats(pitcher_stats, away_sp_name, league_avg)

            home_sp_features = {f"home_{k}": v for k, v in home_sp_stats.items()}
            away_sp_features = {f"away_{k}": v for k, v in away_sp_stats.items()}

            # Targets
            total_runs = game.get('Points', 0)
            ou_line = game.get('OU', 0)
            win_margin = game.get('Win_Margin', 0)
            home_win = 1 if win_margin > 0 else 0

            if ou_line and total_runs:
                if total_runs > ou_line:
                    ou_cover = 1  # Over
                elif total_runs < ou_line:
                    ou_cover = 0  # Under
                else:
                    ou_cover = 2  # Push
            else:
                ou_cover = np.nan

            row = {
                'Date': game_date,
                'Home': home_team,
                'Away': away_team,
                'Home_Starter': home_sp_name,
                'Away_Starter': away_sp_name,
                'season': season,
                **home_features,
                **away_features,
                **home_sp_features,
                **away_sp_features,
                'Days_Rest_Home': game.get('Days_Rest_Home', 4),
                'Days_Rest_Away': game.get('Days_Rest_Away', 4),
                'OU_line': ou_line,
                'Home_Team_Win': home_win,
                'OU_Cover': ou_cover,
                'Total_Runs': total_runs,
                'ML_Home': game.get('ML_Home', 0),
                'ML_Away': game.get('ML_Away', 0),
            }
            all_rows.append(row)

    if not all_rows:
        print("No game data found.")
        return

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=['Home_Team_Win'])

    # Sort by date for time-series training
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    with sqlite3.connect(DATASET_DB) as con:
        df.to_sql(dataset_name, con, if_exists='replace', index=False)

    has_ou = df['OU_Cover'].notna().sum()
    print(f"Saved {len(df)} games to {dataset_name}")
    print(f"  Games with OU data: {has_ou}")
    print(f"  Games without OU data: {len(df) - has_ou}")
    print(f"Features: {len(df.columns)} columns")
    print(f"Home win rate: {df['Home_Team_Win'].mean():.3f}")
    if has_ou > 0:
        ou_valid = df[df['OU_Cover'].isin([0, 1])]
        print(f"Over rate: {(ou_valid['OU_Cover'] == 1).mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Build MLB training dataset.")
    parser.add_argument(
        '--seasons', nargs='+', type=int,
        help='Season years to include (e.g., 2019 2020 2021).',
    )
    parser.add_argument(
        '--dataset', default='mlb_dataset',
        help='Output table name in MLB_dataset.sqlite.',
    )
    parser.add_argument(
        '--backfill', action='store_true',
        help='Use all seasons 2019-2025.',
    )
    args = parser.parse_args()

    if args.backfill:
        seasons = list(range(2019, 2027))
    elif args.seasons:
        seasons = args.seasons
    else:
        from datetime import datetime
        seasons = [datetime.today().year]

    build_dataset(seasons, args.dataset)


if __name__ == "__main__":
    main()
