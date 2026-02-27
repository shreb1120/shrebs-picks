#!/usr/bin/env python3
"""
NBA ML Picks Dashboard
Runs XGBoost Moneyline + Over/Under models directly, fetches full odds
(moneyline, spreads, totals), caches results, auto-refreshes.
"""

import os
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, render_template
from sbrscrape import Scoreboard

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env file from project root
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from src.DataProviders.OddsApiProvider import OddsApiProvider
from src.Utils.Dictionaries import team_index_current
from src.Utils.tools import get_json_data, to_data_frame

app = Flask(__name__)

# ── Database ─────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "picks_history.db"


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            ml_pick TEXT,
            ml_pick_side TEXT,
            ml_confidence REAL,
            ml_home_prob REAL,
            ml_away_prob REAL,
            ou_line REAL,
            ou_pick TEXT,
            ou_confidence REAL,
            ml_best_odds INTEGER,
            ml_best_book TEXT,
            ou_best_odds INTEGER,
            ou_best_book TEXT,
            home_score INTEGER,
            away_score INTEGER,
            total_points INTEGER,
            winner TEXT,
            ml_result TEXT,
            ou_result TEXT,
            ml_profit REAL,
            ou_profit REAL,
            settled_at TEXT,
            UNIQUE(game_date, home_team, away_team)
        )
    """)
    conn.commit()
    conn.close()


def save_picks_to_db(picks_data):
    """Save generated picks to SQLite. Idempotent via INSERT OR IGNORE."""
    if not picks_data or not picks_data.get("games"):
        return
    game_date = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    for game in picks_data["games"]:
        # Find best ML odds across books for the picked side
        ml_best_odds = None
        ml_best_book = None
        ou_best_odds = None
        ou_best_book = None
        pick_side = game.get("pick_side", "home")
        ou_pick = game.get("ou_pick")

        for book_name, book in game.get("books", {}).items():
            ml = book.get("moneyline", {})
            odds_key = f"ml_{pick_side}"
            odds_val = ml.get(odds_key)
            if odds_val is not None:
                if ml_best_odds is None or odds_val > ml_best_odds:
                    ml_best_odds = odds_val
                    ml_best_book = book_name

            tt = book.get("totals", {})
            if ou_pick and tt:
                if ou_pick == "OVER":
                    o = tt.get("over_odds")
                elif ou_pick == "UNDER":
                    o = tt.get("under_odds")
                else:
                    o = None
                if o is not None and (ou_best_odds is None or o > ou_best_odds):
                    ou_best_odds = o
                    ou_best_book = book_name

        conn.execute("""
            INSERT OR IGNORE INTO picks
            (game_date, home_team, away_team, ml_pick, ml_pick_side, ml_confidence,
             ml_home_prob, ml_away_prob, ou_line, ou_pick, ou_confidence,
             ml_best_odds, ml_best_book, ou_best_odds, ou_best_book)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_date, game["home_team"], game["away_team"],
            game.get("pick"), pick_side, game.get("confidence"),
            game.get("model_home_prob"), game.get("model_away_prob"),
            game.get("ou_line"), ou_pick, game.get("ou_confidence"),
            ml_best_odds, ml_best_book, ou_best_odds, ou_best_book,
        ))
    conn.commit()
    conn.close()


def normalize_team(name):
    """Normalize team names for matching sbrscrape results."""
    return (name or "").replace("Los Angeles Clippers", "LA Clippers").strip()


_last_settle_time = 0
SETTLE_COOLDOWN = 1800  # 30 minutes


def settle_unsettled_picks(force=False):
    """Settle past picks using sbrscrape scores. Returns count of settled games."""
    global _last_settle_time
    now = time.time()
    if not force and (now - _last_settle_time) < SETTLE_COOLDOWN:
        return 0
    _last_settle_time = now

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    unsettled_dates = conn.execute(
        "SELECT DISTINCT game_date FROM picks WHERE ml_result IS NULL AND game_date < ?",
        (today,)
    ).fetchall()

    settled_count = 0
    for row in unsettled_dates:
        game_date = row["game_date"]
        try:
            dt = datetime.strptime(game_date, "%Y-%m-%d")
            sb = Scoreboard(sport="NBA", date=dt)
            games = sb.games if hasattr(sb, "games") else []
        except Exception as e:
            app.logger.warning(f"Failed to fetch scores for {game_date}: {e}")
            continue

        # Build lookup: normalized home team -> game scores
        score_map = {}
        for g in games:
            status = (g.get("status") or "").lower()
            if "final" not in status:
                continue
            home = normalize_team(g.get("home_team", ""))
            away = normalize_team(g.get("away_team", ""))
            home_score = safe_int(g.get("home_score"))
            away_score = safe_int(g.get("away_score"))
            if home and away and home_score is not None and away_score is not None:
                score_map[home] = {
                    "home": home, "away": away,
                    "home_score": home_score, "away_score": away_score,
                }

        # Match picks to scores
        picks = conn.execute(
            "SELECT * FROM picks WHERE game_date = ? AND ml_result IS NULL",
            (game_date,)
        ).fetchall()

        for pick in picks:
            match = score_map.get(normalize_team(pick["home_team"]))
            if not match:
                continue

            home_score = match["home_score"]
            away_score = match["away_score"]
            total_points = home_score + away_score
            winner = pick["home_team"] if home_score > away_score else pick["away_team"]

            # ML result
            ml_result = "win" if pick["ml_pick"] == winner else "loss"
            ml_profit = 0.0
            if pick["ml_best_odds"]:
                if ml_result == "win":
                    ml_profit = round(american_to_decimal(pick["ml_best_odds"]) * 100, 2)
                else:
                    ml_profit = -100.0

            # O/U result
            ou_result = None
            ou_profit = 0.0
            if pick["ou_line"] and pick["ou_pick"]:
                if total_points > pick["ou_line"]:
                    actual_ou = "OVER"
                elif total_points < pick["ou_line"]:
                    actual_ou = "UNDER"
                else:
                    actual_ou = "PUSH"

                if actual_ou == "PUSH":
                    ou_result = "push"
                    ou_profit = 0.0
                elif pick["ou_pick"] == actual_ou:
                    ou_result = "win"
                    ou_profit = round(american_to_decimal(pick["ou_best_odds"] or -110) * 100, 2) if pick["ou_best_odds"] else 90.91
                else:
                    ou_result = "loss"
                    ou_profit = -100.0

            conn.execute("""
                UPDATE picks SET home_score=?, away_score=?, total_points=?,
                    winner=?, ml_result=?, ou_result=?, ml_profit=?, ou_profit=?,
                    settled_at=?
                WHERE id=?
            """, (
                home_score, away_score, total_points, winner,
                ml_result, ou_result, ml_profit, ou_profit,
                datetime.now().isoformat(), pick["id"],
            ))
            settled_count += 1

    conn.commit()
    conn.close()
    return settled_count


# Initialize DB on import
init_db()

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 3600  # 1 hour
MODEL_DIR = PROJECT_ROOT / "Models" / "XGBoost_Models"
SCHEDULE_PATH = PROJECT_ROOT / "Data" / "nba-2025-UTC.csv"
DATA_URL = (
    "https://stats.nba.com/stats/leaguedashteamstats?"
    "Conference=&DateFrom=&DateTo=&Division=&GameScope=&GameSegment=&Height=&"
    "ISTRound=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&"
    "OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0&"
    "PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=2025-26&"
    "SeasonSegment=&SeasonType=Regular%20Season&ShotClockRange=&StarterBench=&"
    "TeamID=0&TwoWay=0&VsConference=&VsDivision="
)

SPORTSBOOKS = ["fanduel", "draftkings", "betmgm", "betonline"]

# ── Cache ───────────────────────────────────────────────────────────────────
_cache = {"data": None, "timestamp": 0, "error": None}

# ── Model Loading ───────────────────────────────────────────────────────────
_ml_model = None
_uo_model = None
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")


def _select_best_model(kind):
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")
    return max(candidates, key=lambda p: float(m.group(1)) if (m := ACCURACY_PATTERN.search(p.name)) else 0)


def load_models():
    global _ml_model, _uo_model
    if _ml_model is None:
        ml_path = _select_best_model("ML")
        _ml_model = xgb.Booster()
        _ml_model.load_model(str(ml_path))
        app.logger.info(f"Loaded ML model: {ml_path.name}")
    if _uo_model is None:
        uo_path = _select_best_model("UO")
        _uo_model = xgb.Booster()
        _uo_model.load_model(str(uo_path))
        app.logger.info(f"Loaded UO model: {uo_path.name}")
    return _ml_model, _uo_model


# ── Math helpers ────────────────────────────────────────────────────────────
def american_to_decimal(odds):
    if odds >= 100:
        return odds / 100.0
    return 100.0 / abs(odds)


def implied_prob(odds):
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def calc_ev(model_prob, odds):
    payout = american_to_decimal(odds) * 100
    return round(model_prob * payout - (1 - model_prob) * 100, 2)


def calc_kelly(model_prob, odds):
    dec = american_to_decimal(odds)
    f = (dec * model_prob - (1 - model_prob)) / dec
    return round(max(0, f) * 100, 2)


def safe_int(val, default=None):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=None):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ── Data Pipeline ───────────────────────────────────────────────────────────
def _fetch_odds_from_api():
    """Try The Odds API for all sportsbooks. Returns dict or None on failure."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        app.logger.warning("ODDS_API_KEY not set — falling back to sbrscrape (no betonline)")
        return None
    try:
        provider = OddsApiProvider(api_key=api_key)
        result = provider.get_full_odds(SPORTSBOOKS)
        # Log what we got back
        if result:
            sample_key = next(iter(result))
            books_found = list(result[sample_key].get("books", {}).keys())
            app.logger.info(f"Odds API returned {len(result)} games, books: {books_found}")
        else:
            app.logger.warning("Odds API returned no games")
        return result
    except Exception as e:
        app.logger.warning(f"Odds API failed: {e}")
        return None


def _fetch_odds_from_sbrscrape():
    """Fallback: fetch odds from sbrscrape. Returns dict or empty."""
    app.logger.info("Using sbrscrape fallback (betonline not available via sbrscrape)")
    sbr_books = [b for b in SPORTSBOOKS if b in ("fanduel", "draftkings", "betmgm")]
    try:
        sb = Scoreboard(sport="NBA")
        games = sb.games if hasattr(sb, "games") else []
    except Exception as e:
        app.logger.warning(f"sbrscrape fallback also failed: {e}")
        return {}

    all_games = {}
    for game in games:
        home = game.get("home_team", "").replace("Los Angeles Clippers", "LA Clippers")
        away = game.get("away_team", "").replace("Los Angeles Clippers", "LA Clippers")
        if not home or not away:
            continue

        game_key = f"{home}:{away}"
        game_data = {"home_team": home, "away_team": away, "books": {}}

        for book in sbr_books:
            book_data = {}

            # Moneyline
            ml_home = safe_int(game.get("home_ml", {}).get(book))
            ml_away = safe_int(game.get("away_ml", {}).get(book))
            if ml_home and ml_away:
                book_data["ml_home"] = ml_home
                book_data["ml_away"] = ml_away

            # Spread
            spread_home = safe_float(game.get("home_spread", {}).get(book))
            spread_away = safe_float(game.get("away_spread", {}).get(book))
            spread_home_odds = safe_int(game.get("home_spread_odds", {}).get(book))
            spread_away_odds = safe_int(game.get("away_spread_odds", {}).get(book))
            if spread_home is not None:
                book_data["spread_home"] = spread_home
                book_data["spread_away"] = spread_away
                book_data["spread_home_odds"] = spread_home_odds
                book_data["spread_away_odds"] = spread_away_odds

            # Totals
            total = safe_float(game.get("total", {}).get(book))
            over_odds = safe_int(game.get("over_odds", {}).get(book))
            under_odds = safe_int(game.get("under_odds", {}).get(book))
            if total:
                book_data["total"] = total
                book_data["over_odds"] = over_odds
                book_data["under_odds"] = under_odds

            if book_data:
                game_data["books"][book] = book_data

        all_games[game_key] = game_data
    return all_games


def fetch_full_odds():
    """Fetch full odds — Odds API primary, sbrscrape fallback."""
    result = _fetch_odds_from_api()
    if result is not None:
        return result
    return _fetch_odds_from_sbrscrape()


def fetch_team_stats():
    stats_json = get_json_data(DATA_URL)
    return to_data_frame(stats_json)


def load_schedule():
    return pd.read_csv(SCHEDULE_PATH, parse_dates=["Date"], date_format="%d/%m/%Y %H:%M")


def calc_days_rest(team, schedule_df, today):
    team_games = schedule_df[
        (schedule_df["Home Team"] == team) | (schedule_df["Away Team"] == team)
    ]
    prev = team_games[team_games["Date"] <= today].sort_values("Date", ascending=False)
    if len(prev) > 0:
        last_date = prev.iloc[0]["Date"]
        return min((timedelta(days=1) + today - last_date).days, 9)
    return 7


def build_game_features(home_team, away_team, df, schedule_df, today):
    home_idx = team_index_current.get(home_team)
    away_idx = team_index_current.get(away_team)
    if home_idx is None or away_idx is None:
        return None
    home_series = df.iloc[home_idx]
    away_series = df.iloc[away_idx]
    stats = pd.concat([home_series, away_series])
    stats["Days-Rest-Home"] = calc_days_rest(home_team, schedule_df, today)
    stats["Days-Rest-Away"] = calc_days_rest(away_team, schedule_df, today)
    return stats


def generate_picks():
    """Main pipeline: fetch data → run ML + UO models → return structured picks."""
    ml_model, uo_model = load_models()

    all_games_odds = fetch_full_odds()
    df = fetch_team_stats()
    if df.empty:
        return {"error": "Could not fetch NBA team stats", "games": [], "updated": datetime.now().isoformat()}

    schedule_df = load_schedule()
    today = datetime.today()

    # Build game list from odds
    games_list = []
    for key, gdata in all_games_odds.items():
        home = gdata["home_team"]
        away = gdata["away_team"]
        if home in team_index_current and away in team_index_current:
            games_list.append((home, away))

    if not games_list:
        return {"error": "No games scheduled today", "games": [], "updated": datetime.now().isoformat()}

    results = []
    for home_team, away_team in games_list:
        features = build_game_features(home_team, away_team, df, schedule_df, today)
        if features is None:
            continue

        # ML features: drop TEAM_ID, TEAM_NAME (same as training)
        ml_feature_vals = features.drop(labels=["TEAM_ID", "TEAM_NAME"], errors="ignore")
        X_ml = ml_feature_vals.values.astype(float).reshape(1, -1)

        # ML prediction: [P(away_win), P(home_win)]
        ml_probs = ml_model.predict(xgb.DMatrix(X_ml))
        prob_away = float(ml_probs[0][0])
        prob_home = float(ml_probs[0][1])

        game_key = f"{home_team}:{away_team}"
        game_odds = all_games_odds.get(game_key, {})

        # Get the O/U line from the first available book for the UO model
        ou_line = None
        for book in SPORTSBOOKS:
            bdata = game_odds.get("books", {}).get(book, {})
            if bdata.get("total"):
                ou_line = bdata["total"]
                break

        # UO prediction: [P(under), P(over), P(push)]
        ou_pick = None
        ou_confidence = None
        ou_probs_dict = None
        if ou_line is not None:
            uo_feature_vals = ml_feature_vals.copy()
            uo_feature_vals["OU"] = ou_line
            X_uo = uo_feature_vals.values.astype(float).reshape(1, -1)
            uo_probs = uo_model.predict(xgb.DMatrix(X_uo))
            p_under = float(uo_probs[0][0])
            p_over = float(uo_probs[0][1])
            p_push = float(uo_probs[0][2]) if uo_probs.shape[1] > 2 else 0.0
            ou_pred = int(np.argmax(uo_probs[0]))
            ou_labels = {0: "UNDER", 1: "OVER", 2: "PUSH"}
            ou_pick = ou_labels.get(ou_pred, "PUSH")
            ou_confidence = round(float(uo_probs[0][ou_pred]) * 100, 1)
            ou_probs_dict = {
                "under": round(p_under * 100, 1),
                "over": round(p_over * 100, 1),
                "push": round(p_push * 100, 1),
            }

        game_data = {
            "home_team": home_team,
            "away_team": away_team,
            "model_home_prob": round(prob_home * 100, 1),
            "model_away_prob": round(prob_away * 100, 1),
            "pick": home_team if prob_home > prob_away else away_team,
            "pick_side": "home" if prob_home > prob_away else "away",
            "confidence": round(max(prob_home, prob_away) * 100, 1),
            "ou_line": ou_line,
            "ou_pick": ou_pick,
            "ou_confidence": ou_confidence,
            "ou_probs": ou_probs_dict,
            "books": {},
        }

        # Per-book data
        for book in SPORTSBOOKS:
            bdata = game_odds.get("books", {}).get(book, {})
            if not bdata:
                continue

            book_entry = {}

            # Moneyline
            ml_home = bdata.get("ml_home")
            ml_away = bdata.get("ml_away")
            if ml_home and ml_away:
                ev_home = calc_ev(prob_home, ml_home)
                ev_away = calc_ev(prob_away, ml_away)
                edge_home = round((prob_home - implied_prob(ml_home)) * 100, 1)
                edge_away = round((prob_away - implied_prob(ml_away)) * 100, 1)
                kelly_home = calc_kelly(prob_home, ml_home)
                kelly_away = calc_kelly(prob_away, ml_away)
                best_ml = None
                if ev_home > 0 and ev_home >= ev_away:
                    best_ml = {"side": "home", "team": home_team, "ev": ev_home, "edge": edge_home}
                elif ev_away > 0:
                    best_ml = {"side": "away", "team": away_team, "ev": ev_away, "edge": edge_away}
                book_entry["moneyline"] = {
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "ev_home": ev_home,
                    "ev_away": ev_away,
                    "edge_home": edge_home,
                    "edge_away": edge_away,
                    "kelly_home": kelly_home,
                    "kelly_away": kelly_away,
                    "implied_home": round(implied_prob(ml_home) * 100, 1),
                    "implied_away": round(implied_prob(ml_away) * 100, 1),
                    "best_bet": best_ml,
                }

            # Spread
            spread_home = bdata.get("spread_home")
            spread_away = bdata.get("spread_away")
            spread_home_odds = bdata.get("spread_home_odds")
            spread_away_odds = bdata.get("spread_away_odds")
            if spread_home is not None:
                book_entry["spread"] = {
                    "home": spread_home,
                    "away": spread_away,
                    "home_odds": spread_home_odds,
                    "away_odds": spread_away_odds,
                }

            # Totals
            total = bdata.get("total")
            over_odds = bdata.get("over_odds")
            under_odds = bdata.get("under_odds")
            if total:
                book_entry["totals"] = {
                    "line": total,
                    "over_odds": over_odds,
                    "under_odds": under_odds,
                }

            if book_entry:
                game_data["books"][book] = book_entry

        results.append(game_data)

    results.sort(key=lambda g: g["confidence"], reverse=True)

    return {
        "games": results,
        "game_count": len(results),
        "updated": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "date": today.strftime("%A, %B %d, %Y"),
        "error": None,
    }


def get_cached_picks():
    now = time.time()
    if _cache["data"] and (now - _cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _cache["data"]
    try:
        app.logger.info("Regenerating picks...")
        data = generate_picks()
        _cache["data"] = data
        _cache["timestamp"] = now
        _cache["error"] = None
        save_picks_to_db(data)
        return data
    except Exception as e:
        app.logger.error(f"Failed to generate picks: {traceback.format_exc()}")
        error_data = {
            "error": str(e), "games": [],
            "updated": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "date": datetime.now().strftime("%A, %B %d, %Y"),
        }
        if _cache["data"]:
            _cache["data"]["error"] = f"Refresh failed: {e} (showing stale data)"
            return _cache["data"]
        _cache["data"] = error_data
        _cache["timestamp"] = now
        return error_data


def get_record():
    """Compute season record and yesterday's record from picks DB."""
    conn = get_db()
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    overall = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE ml_result='win') as wins,
            COUNT(*) FILTER (WHERE ml_result='loss') as losses
        FROM picks WHERE ml_result IS NOT NULL
    """).fetchone()

    yest = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE ml_result='win') as wins,
            COUNT(*) FILTER (WHERE ml_result='loss') as losses
        FROM picks WHERE ml_result IS NOT NULL AND game_date = ?
    """, (yesterday,)).fetchone()
    conn.close()

    overall_wins = overall["wins"] or 0
    overall_losses = overall["losses"] or 0
    total = overall_wins + overall_losses
    yesterday_wins = yest["wins"] or 0
    yesterday_losses = yest["losses"] or 0

    return {
        "overall_wins": overall_wins,
        "overall_losses": overall_losses,
        "overall_pct": round(overall_wins / total * 100, 1) if total else 0,
        "yesterday_wins": yesterday_wins,
        "yesterday_losses": yesterday_losses,
        "yesterday_total": yesterday_wins + yesterday_losses,
    }


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html", data=get_cached_picks(), record=get_record())


@app.route("/api/picks")
def api_picks():
    return jsonify(get_cached_picks())


@app.route("/api/refresh")
def api_refresh():
    _cache["timestamp"] = 0
    return jsonify(get_cached_picks())


@app.route("/history")
def history():
    settle_unsettled_picks()
    conn = get_db()

    # Stats
    ml_stats = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE ml_result='win') as ml_wins,
            COUNT(*) FILTER (WHERE ml_result='loss') as ml_losses,
            COALESCE(SUM(ml_profit) FILTER (WHERE ml_result IS NOT NULL), 0) as ml_pl,
            COUNT(*) FILTER (WHERE ou_result='win') as ou_wins,
            COUNT(*) FILTER (WHERE ou_result='loss') as ou_losses,
            COUNT(*) FILTER (WHERE ou_result='push') as ou_pushes,
            COALESCE(SUM(ou_profit) FILTER (WHERE ou_result IS NOT NULL), 0) as ou_pl,
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE ml_result IS NULL) as pending
        FROM picks
    """).fetchone()

    stats = dict(ml_stats)
    ml_decided = stats["ml_wins"] + stats["ml_losses"]
    ou_decided = stats["ou_wins"] + stats["ou_losses"]
    stats["ml_accuracy"] = round(stats["ml_wins"] / ml_decided * 100, 1) if ml_decided else 0
    stats["ou_accuracy"] = round(stats["ou_wins"] / ou_decided * 100, 1) if ou_decided else 0

    # All picks grouped by date
    picks = conn.execute(
        "SELECT * FROM picks ORDER BY game_date DESC, ml_confidence DESC"
    ).fetchall()
    conn.close()

    # Group by date
    from collections import OrderedDict
    grouped = OrderedDict()
    for p in picks:
        d = p["game_date"]
        if d not in grouped:
            grouped[d] = []
        grouped[d].append(dict(p))

    return render_template("history.html", stats=stats, grouped=grouped)


@app.route("/api/settle")
def api_settle():
    count = settle_unsettled_picks(force=True)
    return jsonify({"settled": count})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=8002, debug=False)
