"""Fetch MLB individual pitcher stats and game-level starter assignments.

Uses MLB Stats API (statsapi.mlb.com) — free, reliable, no scraping.
Computes FIP, xFIP from raw counting stats. Aggregates bWAR from pybaseball.
Stores pitcher season stats in Data/MLB_PitcherData.sqlite.
"""

import argparse
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "Data" / "MLB_PitcherData.sqlite"

MLB_STATS_API_PITCHERS = "https://statsapi.mlb.com/api/v1/stats"
MLB_SCHEDULE_API = "https://statsapi.mlb.com/api/v1/schedule"

FIP_CONSTANT = 3.10


def _compute_fip(hr, bb, hbp, k, ip):
    if ip <= 0:
        return 0.0
    return ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + FIP_CONSTANT


def _compute_xfip(hr, bb, hbp, k, ip, air_outs):
    if ip <= 0:
        return 0.0
    fb_estimate = air_outs + hr
    if fb_estimate <= 0:
        return _compute_fip(hr, bb, hbp, k, ip)
    expected_hr = fb_estimate * 0.105
    return ((13 * expected_hr + 3 * (bb + hbp) - 2 * k) / ip) + FIP_CONSTANT


def _get_pitcher_war(season):
    """Get individual pitcher WAR from pybaseball bWAR data."""
    try:
        from pybaseball import bwar_pitch
        bp = bwar_pitch()
        season_data = bp[bp['year_ID'] == season]
        # Map mlb_ID -> WAR (use mlb_ID for matching with MLB Stats API player IDs)
        war_map = {}
        for _, row in season_data.iterrows():
            mlb_id = row.get('mlb_ID')
            if pd.notna(mlb_id):
                mlb_id = int(mlb_id)
                war_map[mlb_id] = war_map.get(mlb_id, 0.0) + (row['WAR'] if pd.notna(row['WAR']) else 0.0)
        return war_map
    except Exception as e:
        print(f"  Warning: Could not fetch bWAR pitcher data: {e}")
        return {}


def fetch_pitcher_stats(season, min_gs=1):
    """Pull individual pitcher stats from MLB Stats API, filtered to starters."""
    print(f"Fetching pitcher stats for {season}...")

    war_map = _get_pitcher_war(season)

    all_pitchers = []
    params = {
        'season': season, 'group': 'pitching', 'stats': 'season',
        'sportId': 1, 'playerPool': 'all', 'limit': 500,
        'offset': 0,
    }
    while True:
        resp = requests.get(MLB_STATS_API_PITCHERS, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data['stats'][0]['splits']
        if not splits:
            break
        all_pitchers.extend(splits)
        if len(splits) < 500:
            break
        params['offset'] += 500
        time.sleep(0.5)

    rows = []
    for split in all_pitchers:
        stat = split['stat']
        gs = stat.get('gamesStarted', 0)
        if gs < min_gs:
            continue

        ip = float(stat.get('inningsPitched', 0) or 0)
        ip_per_gs = ip / max(gs, 1)
        player_id = split['player']['id']

        hr = stat.get('homeRuns', 0)
        bb = stat.get('baseOnBalls', 0)
        hbp = stat.get('hitBatsmen', 0) or stat.get('hitByPitch', 0) or 0
        k = stat.get('strikeOuts', 0)
        air_outs = stat.get('airOuts', 0)

        fip = _compute_fip(hr, bb, hbp, k, ip)
        xfip = _compute_xfip(hr, bb, hbp, k, ip, air_outs)
        war = war_map.get(player_id, 0.0)

        rows.append({
            'Name': split['player']['fullName'],
            'Team': split['team']['name'],
            'player_id': player_id,
            'sp_ERA': float(stat.get('era', 0) or 0),
            'sp_WHIP': float(stat.get('whip', 0) or 0),
            'sp_K/9': float(stat.get('strikeoutsPer9Inn', 0) or 0),
            'sp_BB/9': float(stat.get('walksPer9Inn', 0) or 0),
            'sp_HR/9': float(stat.get('homeRunsPer9', 0) or 0),
            'sp_K%': k / max(stat.get('battersFaced', 1), 1),
            'sp_BB%': bb / max(stat.get('battersFaced', 1), 1),
            'sp_GS': gs,
            'sp_WAR': round(war, 1),
            'sp_IP_per_GS': round(ip_per_gs, 2),
            'sp_FIP': round(fip, 2),
            'sp_xFIP': round(xfip, 2),
        })

    return pd.DataFrame(rows)


def fetch_game_starters(season, db_path=DB_PATH):
    """Fetch game-level starter assignments from MLB Stats API."""
    print(f"Fetching game starters for {season}...")

    params = {
        'sportId': 1,
        'season': season,
        'gameType': 'R',
        'hydrate': 'probablePitcher,decisions',
    }
    resp = requests.get(MLB_SCHEDULE_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for date_entry in data.get('dates', []):
        game_date = date_entry['date']
        for game in date_entry.get('games', []):
            status = game.get('status', {}).get('detailedState', '')
            home_info = game.get('teams', {}).get('home', {})
            away_info = game.get('teams', {}).get('away', {})

            home_team = home_info.get('team', {}).get('name', '')
            away_team = away_info.get('team', {}).get('name', '')
            home_sp = home_info.get('probablePitcher', {}).get('fullName', '')
            away_sp = away_info.get('probablePitcher', {}).get('fullName', '')

            rows.append({
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_starter': home_sp,
                'away_starter': away_sp,
                'game_pk': game.get('gamePk'),
                'status': status,
            })

    df = pd.DataFrame(rows)
    table_name = f"game_starters_{season}"

    with sqlite3.connect(db_path) as con:
        df.to_sql(table_name, con, if_exists='replace', index=False)

    completed = df[df['status'].isin(['Final', 'Completed Early', 'Game Over'])]
    print(f"  Saved {len(df)} game records ({len(completed)} completed) to {table_name}")
    return df


def fetch_and_store(season, db_path=DB_PATH):
    """Fetch pitcher stats and game starters for a season."""
    pitcher_df = fetch_pitcher_stats(season)
    pitcher_df['season'] = season

    table_name = f"pitcher_stats_{season}"
    with sqlite3.connect(db_path) as con:
        pitcher_df.to_sql(table_name, con, if_exists='replace', index=False)
    print(f"  Saved {len(pitcher_df)} pitcher records to {table_name}")

    time.sleep(1)
    fetch_game_starters(season, db_path)

    return pitcher_df


def get_league_avg_pitcher_stats(season, db_path=DB_PATH):
    """Return league-average pitcher stats for imputation of missing starters."""
    table_name = f"pitcher_stats_{season}"
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
    sp_cols = [c for c in df.columns if c.startswith('sp_')]
    return {col: df[col].median() for col in sp_cols}


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB pitcher stats via MLB Stats API.")
    parser.add_argument('--seasons', nargs='+', type=int, help='Season years to fetch.')
    parser.add_argument('--backfill', action='store_true', help='Fetch all seasons 2019-2025.')
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.backfill:
        seasons = list(range(2019, 2026))
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = [datetime.today().year]

    for season in seasons:
        try:
            fetch_and_store(season)
            time.sleep(2)
        except Exception as e:
            print(f"  Failed for {season}: {e}")


if __name__ == "__main__":
    main()
