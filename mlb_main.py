"""MLB prediction CLI — fetches today's games, builds features, runs models."""

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from src.DataProviders.MLBOddsApiProvider import MLBOddsApiProvider
from src.Predict import MLB_XGBoost_Runner
from src.Utils.MLB_Dictionaries import normalize_team_name, mlb_team_index

BASE_DIR = Path(__file__).resolve().parent
TEAM_DB = BASE_DIR / "Data" / "MLB_TeamData.sqlite"
PITCHER_DB = BASE_DIR / "Data" / "MLB_PitcherData.sqlite"

MLB_SCHEDULE_API = "https://statsapi.mlb.com/api/v1/schedule"

# Training feature columns — must match MLB_Create_Games.py output order
# (excluding DROP_COLUMNS from the training scripts)
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


def fetch_todays_games_and_starters(date=None):
    """Fetch today's MLB games and probable starters from MLB Stats API."""
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")

    params = {
        'sportId': 1,
        'date': date,
        'hydrate': 'probablePitcher',
    }
    resp = requests.get(MLB_SCHEDULE_API, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            home_info = game.get('teams', {}).get('home', {})
            away_info = game.get('teams', {}).get('away', {})

            home_team = normalize_team_name(
                home_info.get('team', {}).get('name', '')
            )
            away_team = normalize_team_name(
                away_info.get('team', {}).get('name', '')
            )

            home_sp = home_info.get('probablePitcher', {}).get('fullName', '')
            away_sp = away_info.get('probablePitcher', {}).get('fullName', '')

            if home_team in mlb_team_index and away_team in mlb_team_index:
                games.append({
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_starter': home_sp,
                    'away_starter': away_sp,
                })

    return games


def load_team_stats(season=None):
    """Load current season team stats from SQLite."""
    if season is None:
        season = datetime.today().year
    table = f"team_stats_{season}"
    with sqlite3.connect(TEAM_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def load_pitcher_stats(season=None):
    """Load current season pitcher stats from SQLite."""
    if season is None:
        season = datetime.today().year
    table = f"pitcher_stats_{season}"
    with sqlite3.connect(PITCHER_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def _get_team_features(team_stats, team_name, prefix):
    """Get team features for a specific team."""
    from src.Utils.MLB_Dictionaries import FULL_TO_ABBREV

    matches = team_stats[team_stats['Team'] == team_name]
    if matches.empty:
        abbrev = FULL_TO_ABBREV.get(team_name)
        if abbrev:
            matches = team_stats[team_stats['Team'] == abbrev]
    if matches.empty:
        return {f"{prefix}{c}": 0.0 for c in TEAM_BAT_FEATURES + TEAM_PIT_FEATURES}

    row = matches.iloc[0]
    features = {}
    for col in TEAM_BAT_FEATURES + TEAM_PIT_FEATURES:
        val = row.get(col)
        features[f"{prefix}{col}"] = float(val) if pd.notna(val) else 0.0
    return features


def _get_pitcher_features(pitcher_stats, pitcher_name, prefix, league_avg):
    """Get starting pitcher features."""
    if not pitcher_name:
        return {f"{prefix}{c}": league_avg.get(c, 0.0) for c in SP_FEATURES}

    matches = pitcher_stats[pitcher_stats['Name'] == pitcher_name]
    if matches.empty:
        return {f"{prefix}{c}": league_avg.get(c, 0.0) for c in SP_FEATURES}

    row = matches.iloc[0]
    features = {}
    for col in SP_FEATURES:
        val = row.get(col)
        features[f"{prefix}{col}"] = float(val) if pd.notna(val) else league_avg.get(col, 0.0)
    return features


def build_features(game_list, team_stats, pitcher_stats, odds=None):
    """Build feature matrices for ML and UO models."""
    sp_cols = [c for c in pitcher_stats.columns if c.startswith('sp_')]
    league_avg = {c: pitcher_stats[c].median() for c in sp_cols}

    rows_ml = []
    rows_uo = []
    valid_games = []
    todays_uo = []
    home_odds = []
    away_odds = []
    starters = []

    for game in game_list:
        home = game['home_team']
        away = game['away_team']

        home_bat = _get_team_features(team_stats, home, 'home_')
        away_bat = _get_team_features(team_stats, away, 'away_')
        home_sp = _get_pitcher_features(pitcher_stats, game['home_starter'], 'home_', league_avg)
        away_sp = _get_pitcher_features(pitcher_stats, game['away_starter'], 'away_', league_avg)

        # Days rest: default to 1 for MLB daily games
        features = {
            **home_bat,
            **away_bat,
            **home_sp,
            **away_sp,
            'Days_Rest_Home': 1,
            'Days_Rest_Away': 1,
        }

        # ML features (no OU_line)
        rows_ml.append(features)

        # UO features (with OU_line)
        ou_line = 0
        game_key = f"{home}:{away}"
        if odds and game_key in odds:
            ou_line = odds[game_key].get('under_over_odds') or 0
            home_ml = odds[game_key].get(home, {}).get('money_line_odds')
            away_ml = odds[game_key].get(away, {}).get('money_line_odds')
        else:
            home_ml = away_ml = None

        uo_features = {**features, 'OU_line': ou_line}
        rows_uo.append(uo_features)

        valid_games.append((home, away))
        todays_uo.append(ou_line)
        home_odds.append(home_ml)
        away_odds.append(away_ml)
        starters.append((game['home_starter'], game['away_starter']))

    df_ml = pd.DataFrame(rows_ml).astype(float)
    df_uo = pd.DataFrame(rows_uo).astype(float)

    return df_ml.values, df_uo.values, valid_games, todays_uo, home_odds, away_odds, starters


def main(args):
    odds = None
    if args.odds:
        try:
            odds = MLBOddsApiProvider(sportsbook=args.odds).get_odds()
        except Exception as e:
            print(f"MLB Odds API failed: {e}")

    today = datetime.today()
    date_str = today.strftime("%Y-%m-%d")
    print(f"Fetching MLB games for {date_str}...")

    game_list = fetch_todays_games_and_starters(date_str)
    if not game_list:
        print("No MLB games found for today.")
        return

    print(f"Found {len(game_list)} games.")

    season = today.year
    team_stats = load_team_stats(season)
    pitcher_stats = load_pitcher_stats(season)

    data_ml, data_uo, games, todays_uo, home_odds, away_odds, starters = build_features(
        game_list, team_stats, pitcher_stats, odds
    )

    if args.odds and odds:
        print(f"------------------{args.odds} odds data------------------")
        for game_key in odds.keys():
            home, away = game_key.split(":")
            print(
                f"{away} ({odds[game_key][away]['money_line_odds']}) @ "
                f"{home} ({odds[game_key][home]['money_line_odds']})"
            )

    print("------------- MLB XGBoost Model Predictions ------------")
    MLB_XGBoost_Runner.mlb_xgb_runner(
        data_ml, data_uo, games, home_odds, away_odds,
        kelly_criterion=args.kc, starters=starters,
    )
    print("--------------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLB Model Predictions')
    parser.add_argument('-odds', help='Sportsbook (fanduel, draftkings, betmgm, betonline)')
    parser.add_argument('-kc', action='store_true', help='Show Kelly Criterion')
    args = parser.parse_args()
    main(args)
