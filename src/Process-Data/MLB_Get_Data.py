"""Fetch MLB team batting and pitching stats via MLB Stats API.

Computes advanced sabermetric stats (FIP, xFIP, wOBA, wRC+, etc.)
from raw counting stats. Aggregates bWAR from pybaseball.

Stores season-level team stats in Data/MLB_TeamData.sqlite,
one table per season (e.g., team_stats_2024).
"""

import argparse
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "Data" / "MLB_TeamData.sqlite"
MLB_STATS_API = "https://statsapi.mlb.com/api/v1/teams/stats"

# ── Standard linear weights for wOBA (approximate, stable across years) ──────
# These are close to the 2019-2024 average FanGraphs weights.
WOBA_WEIGHTS = {
    'bb': 0.690, 'hbp': 0.720, '1b': 0.880, '2b': 1.240,
    '3b': 1.560, 'hr': 2.010,
}
WOBA_SCALE = 1.15  # wOBA/OBP scaling factor (approximate)
LG_WOBA = 0.315    # League average wOBA (approximate)
LG_R_PA = 0.110    # League runs per PA (approximate)
FIP_CONSTANT = 3.10  # ~stable across years


def _compute_fip(hr, bb, hbp, k, ip):
    """Fielding Independent Pitching."""
    if ip <= 0:
        return 0.0
    return ((13 * hr + 3 * (bb + hbp) - 2 * k) / ip) + FIP_CONSTANT


def _compute_xfip(hr, bb, hbp, k, ip, fb):
    """Expected FIP — normalizes HR to league-average HR/FB rate (~10.5%)."""
    if ip <= 0 or fb <= 0:
        return 0.0
    expected_hr = fb * 0.105
    return ((13 * expected_hr + 3 * (bb + hbp) - 2 * k) / ip) + FIP_CONSTANT


def _compute_babip_pitching(h, hr, bf, k, bb, hbp, sf):
    """BABIP for pitchers = (H - HR) / (BF - K - HR - BB - HBP + SF)."""
    denom = bf - k - hr - bb - hbp + sf
    if denom <= 0:
        return 0.300  # league average fallback
    return (h - hr) / denom


def _compute_lob_pct(h, bb, hbp, hr, r):
    """LOB% = (H + BB + HBP - R) / (H + BB + HBP - 1.4 * HR)."""
    denom = h + bb + hbp - 1.4 * hr
    if denom <= 0:
        return 0.700  # league average fallback
    return (h + bb + hbp - r) / denom


def _compute_woba(bb, hbp, singles, doubles, triples, hr, pa):
    """Weighted On-Base Average."""
    if pa <= 0:
        return 0.0
    w = WOBA_WEIGHTS
    return (w['bb'] * bb + w['hbp'] * hbp + w['1b'] * singles +
            w['2b'] * doubles + w['3b'] * triples + w['hr'] * hr) / pa


def _compute_wrc_plus(woba, pa, lg_woba=LG_WOBA, lg_r_pa=LG_R_PA):
    """wRC+ (park-neutral approximation — no park factors applied)."""
    if pa <= 0 or lg_r_pa <= 0:
        return 100.0
    wrc_per_pa = ((woba - lg_woba) / WOBA_SCALE) + lg_r_pa
    return (wrc_per_pa / lg_r_pa) * 100


def _get_pitcher_war_from_db(season):
    """Aggregate team pitching WAR from the pitcher stats DB (fallback)."""
    pitcher_db = BASE_DIR / "Data" / "MLB_PitcherData.sqlite"
    if not pitcher_db.exists():
        return {}
    try:
        import sqlite3 as _sql
        with _sql.connect(pitcher_db) as con:
            df = pd.read_sql_query(
                f'SELECT Team, sp_WAR FROM "pitcher_stats_{season}"', con
            )
        return df.groupby('Team')['sp_WAR'].sum().to_dict()
    except Exception:
        return {}


def _estimate_batting_war(runs, lg_runs_per_game=4.5, games=162):
    """Rough batting WAR estimate: (team runs - league avg runs) / 10.
    WAR ≈ runs above replacement / ~10 runs per win."""
    lg_total = lg_runs_per_game * games
    return (runs - lg_total) / 10.0


def _get_team_war(season):
    """Aggregate team-level WAR from pybaseball bWAR data, with fallbacks."""
    bat_war, pit_war = {}, {}
    try:
        from pybaseball import bwar_bat, bwar_pitch
        bat = bwar_bat()
        bat_season = bat[(bat['year_ID'] == season) & (bat['pitcher'] == 'N')]
        bat_war = bat_season.groupby('team_ID')['WAR'].sum().to_dict()

        pit = bwar_pitch()
        pit_season = pit[pit['year_ID'] == season]
        pit_war = pit_season.groupby('team_ID')['WAR'].sum().to_dict()
    except Exception as e:
        print(f"  Warning: bWAR download failed, using fallbacks: {e}")

    # Fallback: aggregate pitching WAR from individual pitcher DB
    if not pit_war:
        pit_war_by_name = _get_pitcher_war_from_db(season)
        # Convert full team names to bref IDs
        for name, bref_id in TEAM_TO_BREF.items():
            if name in pit_war_by_name:
                pit_war[bref_id] = pit_war_by_name[name]

    return bat_war, pit_war


# Map MLB Stats API team names to baseball-reference team IDs
TEAM_TO_BREF = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL', 'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU', 'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
    'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP', 'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR', 'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSN',
}


def _fetch_team_hitting(season, bat_war):
    """Pull team-level batting stats from MLB Stats API + compute advanced metrics."""
    params = {'season': season, 'group': 'hitting', 'stats': 'season', 'sportIds': 1}
    resp = requests.get(MLB_STATS_API, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for split in data['stats'][0]['splits']:
        team_name = split['team']['name']
        stat = split['stat']
        g = stat.get('gamesPlayed', 1) or 1
        pa = stat.get('plateAppearances', 1) or 1

        h = stat.get('hits', 0)
        doubles = stat.get('doubles', 0)
        triples = stat.get('triples', 0)
        hr = stat.get('homeRuns', 0)
        singles = h - doubles - triples - hr
        bb = stat.get('baseOnBalls', 0)
        hbp = stat.get('hitByPitch', 0)

        woba = _compute_woba(bb, hbp, singles, doubles, triples, hr, pa)
        wrc_plus = _compute_wrc_plus(woba, pa)

        bref_id = TEAM_TO_BREF.get(team_name, '')
        war = bat_war.get(bref_id, 0.0)
        if war == 0.0 and not bat_war:
            # Fallback: estimate batting WAR from runs scored
            war = round(_estimate_batting_war(stat.get('runs', 0), games=g), 1)

        rows.append({
            'Team': team_name,
            'bat_AVG': float(stat.get('avg', 0) or 0),
            'bat_OBP': float(stat.get('obp', 0) or 0),
            'bat_SLG': float(stat.get('slg', 0) or 0),
            'bat_OPS': float(stat.get('ops', 0) or 0),
            'bat_BABIP': float(stat.get('babip', 0) or 0),
            'bat_K%': stat.get('strikeOuts', 0) / max(pa, 1),
            'bat_BB%': bb / max(pa, 1),
            'bat_ISO': float(stat.get('slg', 0) or 0) - float(stat.get('avg', 0) or 0),
            'bat_HR_per_G': hr / g,
            'bat_R_per_G': stat.get('runs', 0) / g,
            'bat_SB_per_G': stat.get('stolenBases', 0) / g,
            'bat_RBI_per_G': stat.get('rbi', 0) / g,
            'bat_wOBA': round(woba, 3),
            'bat_wRC+': round(wrc_plus, 1),
            'bat_WAR': round(war, 1),
        })
    return pd.DataFrame(rows)


def _fetch_team_pitching(season, pit_war):
    """Pull team-level pitching stats from MLB Stats API + compute advanced metrics."""
    params = {'season': season, 'group': 'pitching', 'stats': 'season', 'sportIds': 1}
    resp = requests.get(MLB_STATS_API, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for split in data['stats'][0]['splits']:
        team_name = split['team']['name']
        stat = split['stat']
        bf = max(stat.get('battersFaced', 1), 1)

        ip_str = stat.get('inningsPitched', '0')
        ip = float(ip_str) if ip_str else 0.0

        hr = stat.get('homeRuns', 0)
        bb = stat.get('baseOnBalls', 0)
        hbp = stat.get('hitBatsmen', 0) or stat.get('hitByPitch', 0) or 0
        k = stat.get('strikeOuts', 0)
        h = stat.get('hits', 0)
        r = stat.get('runs', 0)
        sf = stat.get('sacFlies', 0)

        # Estimate fly balls from air outs (rough proxy)
        air_outs = stat.get('airOuts', 0)
        # Fly balls ≈ air outs + HR (air outs don't include HR)
        fb_estimate = air_outs + hr

        fip = _compute_fip(hr, bb, hbp, k, ip)
        xfip = _compute_xfip(hr, bb, hbp, k, ip, fb_estimate)
        babip = _compute_babip_pitching(h, hr, bf, k, bb, hbp, sf)
        lob_pct = _compute_lob_pct(h, bb, hbp, hr, r)

        bref_id = TEAM_TO_BREF.get(team_name, '')
        war = pit_war.get(bref_id, 0.0)

        rows.append({
            'Team': team_name,
            'pit_ERA': float(stat.get('era', 0) or 0),
            'pit_WHIP': float(stat.get('whip', 0) or 0),
            'pit_K/9': float(stat.get('strikeoutsPer9Inn', 0) or 0),
            'pit_BB/9': float(stat.get('walksPer9Inn', 0) or 0),
            'pit_HR/9': float(stat.get('homeRunsPer9', 0) or 0),
            'pit_K%': k / bf,
            'pit_BB%': bb / bf,
            'pit_FIP': round(fip, 2),
            'pit_xFIP': round(xfip, 2),
            'pit_LOB%': round(lob_pct, 3),
            'pit_BABIP': round(babip, 3),
            'pit_WAR': round(war, 1),
        })
    return pd.DataFrame(rows)


def fetch_and_store(season, db_path=DB_PATH):
    """Fetch both batting and pitching stats, merge, and write to SQLite."""
    print(f"Fetching MLB team stats for {season}...")

    bat_war, pit_war = _get_team_war(season)

    batting = _fetch_team_hitting(season, bat_war)
    time.sleep(1)
    pitching = _fetch_team_pitching(season, pit_war)

    merged = batting.merge(pitching, on='Team', how='outer')
    merged['season'] = season

    table_name = f"team_stats_{season}"
    with sqlite3.connect(db_path) as con:
        merged.to_sql(table_name, con, if_exists='replace', index=False)

    print(f"  Saved {len(merged)} teams to {table_name}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Fetch MLB team stats via MLB Stats API.")
    parser.add_argument('--seasons', nargs='+', type=int, help='Season years to fetch.')
    parser.add_argument('--backfill', action='store_true', help='Fetch all seasons 2019-2025.')
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.backfill:
        seasons = list(range(2019, 2026))
    elif args.seasons:
        seasons = args.seasons
    else:
        from datetime import datetime
        seasons = [datetime.today().year]

    for season in seasons:
        try:
            fetch_and_store(season)
            time.sleep(1)
        except Exception as e:
            print(f"  Failed for {season}: {e}")


if __name__ == "__main__":
    main()
