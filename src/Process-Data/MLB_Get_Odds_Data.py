"""Fetch MLB odds and scores via sbrscrape.

Stores in Data/MLB_OddsData.sqlite, following the same pattern as
the NBA Get_Odds_Data.py script.
"""

import argparse
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import toml
from sbrscrape import Scoreboard

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "MLB_OddsData.sqlite"
MIN_DELAY = 1
MAX_DELAY = 3


def load_config():
    return toml.load(CONFIG_PATH)


def iter_dates(start_date, end_date):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)


def fetch_scoreboard(date_pointer):
    try:
        return Scoreboard(sport="MLB", date=date_pointer)
    except Exception as exc:
        print(f"Failed to fetch MLB odds for {date_pointer}: {exc}")
        return None


def parse_date_value(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def table_exists(con, table_name):
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cur.fetchone() is not None


def select_current_season(config, today):
    for key, value in config.get("mlb-odds-data", {}).items():
        start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
        if start <= today <= end:
            return key, value, start, end
    return None, None, None, None


def get_existing_dates(con, table_name):
    if not table_exists(con, table_name):
        return set()
    cur = con.execute(f'SELECT DISTINCT Date FROM "{table_name}"')
    dates = set()
    for (v,) in cur.fetchall():
        d = parse_date_value(v)
        if d:
            dates.add(d)
    return dates


def get_teams_last_played(con, table_name, before_date):
    if not table_exists(con, table_name):
        return {}
    df = pd.read_sql_query(
        f'SELECT Date, Home, Away FROM "{table_name}" WHERE Date < ?',
        con, params=[before_date],
    )
    last = {}
    for row in df.itertuples(index=False):
        d = parse_date_value(row.Date)
        if not d:
            continue
        last[row.Home] = max(d, last.get(row.Home, d))
        last[row.Away] = max(d, last.get(row.Away, d))
    return last


def append_game_rows(rows, date_pointer, game, sportsbook, teams_last_played):
    def days_rest(team):
        lp = teams_last_played.get(team)
        if lp is None:
            return 4  # MLB default rest
        return (date_pointer - lp).days

    home = game["home_team"]
    away = game["away_team"]

    teams_last_played[home] = date_pointer
    teams_last_played[away] = date_pointer

    try:
        ou = game["total"].get(sportsbook) or game["total"].get("fanduel") or next(
            (v for v in game["total"].values() if v is not None), None
        )
        ml_home = game["home_ml"].get(sportsbook) or game["home_ml"].get("fanduel") or next(
            (v for v in game["home_ml"].values() if v is not None), None
        )
        ml_away = game["away_ml"].get(sportsbook) or game["away_ml"].get("fanduel") or next(
            (v for v in game["away_ml"].values() if v is not None), None
        )
        spread = game.get("away_spread", {}).get(sportsbook)
    except (StopIteration, AttributeError):
        return

    if ou is None or ml_home is None:
        return

    rows.append({
        "Date": date_pointer,
        "Home": home,
        "Away": away,
        "OU": ou,
        "Spread": spread,
        "ML_Home": ml_home,
        "ML_Away": ml_away,
        "Points": (game.get("away_score") or 0) + (game.get("home_score") or 0),
        "Win_Margin": (game.get("home_score") or 0) - (game.get("away_score") or 0),
        "Days_Rest_Home": days_rest(home),
        "Days_Rest_Away": days_rest(away),
    })


def collect_odds_for_dates(dates, sportsbook, teams_last_played):
    all_rows = []
    for date_pointer in dates:
        print("Getting MLB odds data:", date_pointer)
        sb = fetch_scoreboard(date_pointer)
        if not sb or not hasattr(sb, "games") or not sb.games:
            time.sleep(random.randint(MIN_DELAY, MAX_DELAY))
            continue

        for game in sb.games:
            try:
                append_game_rows(all_rows, date_pointer, game, sportsbook, teams_last_played)
            except KeyError:
                print(f"No {sportsbook} odds for game: {game}")

        time.sleep(random.randint(MIN_DELAY, MAX_DELAY))

    return all_rows


def main(sportsbook="fanduel", backfill=False, season=None, today=None, db_path=DB_PATH):
    config = load_config()
    if today is None:
        today = datetime.today().date()

    with sqlite3.connect(db_path) as con:
        if backfill:
            items = config.get("mlb-odds-data", {}).items()
            if season:
                items = [(k, v) for k, v in items if k == season]
                if not items:
                    print("Season not found:", season)
                    return
            for season_key, value in items:
                start = datetime.strptime(value["start_date"], "%Y-%m-%d").date()
                end = datetime.strptime(value["end_date"], "%Y-%m-%d").date()
                fetch_end = min(today - timedelta(days=1), end)
                existing = get_existing_dates(con, season_key)
                teams_lp = get_teams_last_played(con, season_key, start)
                new_dates = [d for d in iter_dates(start, fetch_end) if d not in existing]
                if not new_dates:
                    print(f"No new dates for {season_key}")
                    continue
                rows = collect_odds_for_dates(new_dates, sportsbook, teams_lp)
                if rows:
                    pd.DataFrame(rows).to_sql(season_key, con, if_exists="append", index=False)
            return

        season_key, value, start, end = select_current_season(config, today)
        if not season_key:
            print("No current MLB season for today:", today)
            return

        fetch_end = min(today, end)
        existing = get_existing_dates(con, season_key)
        latest = max(existing) if existing else None
        fetch_start = start if latest is None else latest + timedelta(days=1)
        if fetch_start > fetch_end:
            print("No new MLB odds dates. Latest:", latest)
            return

        teams_lp = get_teams_last_played(con, season_key, fetch_start)
        rows = collect_odds_for_dates(iter_dates(fetch_start, fetch_end), sportsbook, teams_lp)
        if rows:
            pd.DataFrame(rows).to_sql(season_key, con, if_exists="append", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch MLB odds data from SBR.")
    parser.add_argument("--sportsbook", default="fanduel")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--season", help="Limit to a single season key (e.g. 2024).")
    args = parser.parse_args()
    main(sportsbook=args.sportsbook, backfill=args.backfill, season=args.season)
