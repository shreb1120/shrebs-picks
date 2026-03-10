#!/usr/bin/env python3
"""
NBA ML Picks Dashboard
Runs XGBoost Moneyline + Over/Under models directly, fetches full odds
(moneyline, spreads, totals), caches results, auto-refreshes.
"""

import atexit
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import bcrypt
import numpy as np
import requests
import pandas as pd
import xgboost as xgb
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
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

# Kalshi provider (lazy-loaded, only if credentials are configured)
_kalshi_provider = None


def get_kalshi_provider():
    global _kalshi_provider
    if _kalshi_provider is not None:
        return _kalshi_provider
    api_key = os.environ.get("KALSHI_API_KEY")
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not api_key or not key_path:
        return None
    try:
        from src.DataProviders.KalshiProvider import KalshiProvider
        _kalshi_provider = KalshiProvider(api_key=api_key, private_key_path=key_path)
        return _kalshi_provider
    except Exception as e:
        app.logger.warning(f"Kalshi provider init failed: {e}")
        return None


# Telegram bot (lazy-loaded, only if credentials are configured)
_telegram_bot = None


def get_telegram_bot():
    global _telegram_bot
    if _telegram_bot is not None:
        return _telegram_bot
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    try:
        from src.Notifications.TelegramBot import TelegramBot
        _telegram_bot = TelegramBot(token=token, chat_id=chat_id)
        _telegram_bot.start_polling()
        return _telegram_bot
    except Exception as e:
        app.logger.warning(f"Telegram bot init failed: {e}")
        return None

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "1fb2b4cc128a1681d5d68aabdb2107f88adc31963674c5ae785a79c25aa4f7c4")

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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bankroll (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            starting_amount REAL NOT NULL DEFAULT 1000.0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS kalshi_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            pick_id INTEGER,
            kalshi_ticker TEXT NOT NULL,
            bet_side TEXT NOT NULL,
            bet_type TEXT NOT NULL DEFAULT 'ml',
            model_prob REAL,
            kalshi_price REAL,
            edge REAL,
            kelly_fraction REAL,
            stake_cents INTEGER,
            contracts INTEGER,
            status TEXT NOT NULL DEFAULT 'pending',
            kalshi_order_id TEXT,
            fill_price REAL,
            result TEXT,
            payout_cents INTEGER,
            profit_cents INTEGER,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            placed_at TEXT,
            settled_at TEXT,
            error_message TEXT,
            FOREIGN KEY (pick_id) REFERENCES picks(id)
        )
    """)
    conn.commit()
    conn.close()


def _game_date_et(game):
    """Derive the game date in Eastern Time from commence_time, or fall back to now() in ET."""
    from zoneinfo import ZoneInfo
    eastern = ZoneInfo("America/New_York")
    ct = game.get("start_time_raw")
    if ct:
        try:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            return dt.astimezone(eastern).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.now(tz=eastern).strftime("%Y-%m-%d")


def save_picks_to_db(picks_data):
    """Save generated picks to SQLite. Idempotent via INSERT OR IGNORE."""
    if not picks_data or not picks_data.get("games"):
        return
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

        game_date = _game_date_et(game)
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
SETTLE_COOLDOWN = 300  # 5 minutes


def _settle_pick(conn, pick, home_score, away_score):
    """Settle a single pick row given final scores. Returns True if settled."""
    total_points = home_score + away_score
    winner = pick["home_team"] if home_score > away_score else pick["away_team"]

    ml_result = "win" if pick["ml_pick"] == winner else "loss"
    ml_profit = 0.0
    if pick["ml_best_odds"]:
        if ml_result == "win":
            ml_profit = round(american_to_decimal(pick["ml_best_odds"]) * 100, 2)
        else:
            ml_profit = -100.0

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
    return True


def settle_from_live_scores():
    """Settle picks using Odds API score cache. Called on every score poll."""
    scores = fetch_live_scores()
    if not scores:
        return 0

    conn = get_db()
    unsettled = conn.execute(
        "SELECT * FROM picks WHERE ml_result IS NULL"
    ).fetchall()

    settled_count = 0
    for pick in unsettled:
        game_key = f"{pick['home_team']}:{pick['away_team']}"
        info = scores.get(game_key)
        if not info or info["status"] != "final":
            continue
        if info["home_score"] is None or info["away_score"] is None:
            continue

        if _settle_pick(conn, pick, info["home_score"], info["away_score"]):
            settled_count += 1

    if settled_count:
        conn.commit()
        app.logger.info(f"Settled {settled_count} picks from live scores API")
    conn.close()
    return settled_count


def settle_unsettled_picks(force=False):
    """Settle past picks using sbrscrape scores (fallback). Returns count of settled games."""
    global _last_settle_time
    now = time.time()
    if not force and (now - _last_settle_time) < SETTLE_COOLDOWN:
        return 0
    _last_settle_time = now

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    unsettled_dates = conn.execute(
        "SELECT DISTINCT game_date FROM picks WHERE ml_result IS NULL AND game_date <= ?",
        (today,)
    ).fetchall()

    settled_count = 0
    for row in unsettled_dates:
        game_date = row["game_date"]
        dt = datetime.strptime(game_date, "%Y-%m-%d")

        # Check the saved date AND the next day, since picks may be saved
        # with the server date while games are scheduled in Eastern Time
        score_map = {}
        for offset in (0, 1):
            check_dt = dt + timedelta(days=offset)
            try:
                sb = Scoreboard(sport="NBA", date=check_dt)
                games = sb.games if hasattr(sb, "games") else []
            except Exception as e:
                app.logger.warning(f"Failed to fetch scores for {check_dt.date()}: {e}")
                continue

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
            if _settle_pick(conn, pick, match["home_score"], match["away_score"]):
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

# ── Live Score Cache ────────────────────────────────────────────────────────
SCORE_CACHE_TTL = 30  # seconds
_score_cache = {"data": {}, "timestamp": 0}


def fetch_live_scores():
    """Fetch live NBA scores from The Odds API. Returns dict keyed by 'Home:Away'."""
    now = time.time()
    if _score_cache["data"] and (now - _score_cache["timestamp"]) < SCORE_CACHE_TTL:
        return _score_cache["data"]

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return _score_cache["data"]

    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/scores/",
            params={"apiKey": api_key, "daysFrom": 1},
            timeout=10,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        app.logger.warning(f"Score fetch failed: {e}")
        return _score_cache["data"]

    scores = {}
    for event in events:
        home = OddsApiProvider._normalize_team_name(event.get("home_team", ""))
        away = OddsApiProvider._normalize_team_name(event.get("away_team", ""))
        if not home or not away:
            continue

        completed = event.get("completed", False)
        event_scores = event.get("scores")

        if event_scores is None:
            status = "pregame"
            home_score = None
            away_score = None
        else:
            status = "final" if completed else "live"
            home_score = None
            away_score = None
            for s in event_scores:
                name = OddsApiProvider._normalize_team_name(s.get("name", ""))
                if name == home:
                    home_score = safe_int(s.get("score"))
                elif name == away:
                    away_score = safe_int(s.get("score"))

        game_key = f"{home}:{away}"
        scores[game_key] = {
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "status": status,
            "commence_time": event.get("commence_time"),
        }

    _score_cache["data"] = scores
    _score_cache["timestamp"] = now
    return scores

# ── Model Loading ───────────────────────────────────────────────────────────
_ml_model = None
_uo_model = None
_ml_calibrator = None
_uo_calibrator = None
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")


def _select_best_model(kind):
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")
    return max(candidates, key=lambda p: float(m.group(1)) if (m := ACCURACY_PATTERN.search(p.name)) else 0)


class BoosterWrapper:
    """Needed for unpickling calibrator saved by training script."""
    _estimator_type = "classifier"

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        raw = self.booster.predict(xgb.DMatrix(X))
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw


def _load_calibrator(model_path):
    """Load calibration model (.pkl) that sits alongside the XGBoost .json."""
    import joblib
    import __main__
    # The pickle was saved from __main__ context during training,
    # so it looks for __main__.BoosterWrapper when unpickling
    if not hasattr(__main__, "BoosterWrapper"):
        __main__.BoosterWrapper = BoosterWrapper
    cal_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not cal_path.exists():
        return None
    try:
        return joblib.load(cal_path)
    except Exception as e:
        app.logger.warning(f"Failed to load calibrator {cal_path.name}: {e}")
        return None


def load_models():
    global _ml_model, _uo_model, _ml_calibrator, _uo_calibrator
    if _ml_model is None:
        ml_path = _select_best_model("ML")
        _ml_model = xgb.Booster()
        _ml_model.load_model(str(ml_path))
        _ml_calibrator = _load_calibrator(ml_path)
        app.logger.info(f"Loaded ML model: {ml_path.name} (calibrated={_ml_calibrator is not None})")
    if _uo_model is None:
        uo_path = _select_best_model("UO")
        _uo_model = xgb.Booster()
        _uo_model.load_model(str(uo_path))
        _uo_calibrator = _load_calibrator(uo_path)
        app.logger.info(f"Loaded UO model: {uo_path.name} (calibrated={_uo_calibrator is not None})")
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
        if game.get("date"):
            game_data["commence_time"] = game["date"]

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


def _format_start_time_et(commence_time_str):
    """Convert an ISO-8601 UTC timestamp to 'H:MM PM ET' display string."""
    if not commence_time_str:
        return None
    try:
        from zoneinfo import ZoneInfo
        ct = datetime.fromisoformat(commence_time_str.replace("Z", "+00:00"))
        ct_et = ct.astimezone(ZoneInfo("America/New_York"))
        return ct_et.strftime("%-I:%M %p ET")
    except Exception:
        return None


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
        if _ml_calibrator is not None:
            ml_probs = _ml_calibrator.predict_proba(X_ml)
        else:
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

        # UO prediction: binary model returns P(over) as a single float
        ou_pick = None
        ou_confidence = None
        ou_probs_dict = None
        if ou_line is not None:
            uo_feature_vals = ml_feature_vals.copy()
            rest_home = uo_feature_vals.pop("Days-Rest-Home")
            rest_away = uo_feature_vals.pop("Days-Rest-Away")
            uo_feature_vals["OU"] = ou_line
            uo_feature_vals["Days-Rest-Home"] = rest_home
            uo_feature_vals["Days-Rest-Away"] = rest_away
            X_uo = uo_feature_vals.values.astype(float).reshape(1, -1)
            if _uo_calibrator is not None:
                uo_probs = _uo_calibrator.predict_proba(X_uo)
                p_under = float(uo_probs[0][0])
                p_over = float(uo_probs[0][1])
            else:
                uo_probs = uo_model.predict(xgb.DMatrix(X_uo))
                if uo_probs.ndim > 1:
                    p_under = float(uo_probs[0][0])
                    p_over = float(uo_probs[0][1])
                else:
                    p_over = float(uo_probs[0])
                    p_under = 1.0 - p_over
            ou_pick = "OVER" if p_over > 0.5 else "UNDER"
            ou_confidence = round(max(p_over, p_under) * 100, 1)
            ou_probs_dict = {
                "under": round(p_under * 100, 1),
                "over": round(p_over * 100, 1),
                "push": 0.0,
            }

        start_time_et = _format_start_time_et(game_odds.get("commence_time"))

        game_data = {
            "home_team": home_team,
            "away_team": away_team,
            "start_time_et": start_time_et,
            "start_time_raw": game_odds.get("commence_time"),
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


# ── Auth ────────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        if user and bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("index"))
        error = "Invalid username or password"
    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("index"))
    # Lock registration after first user
    conn = get_db()
    user_count = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()["cnt"]
    conn.close()
    if user_count > 0:
        return redirect(url_for("login"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if not username or not password:
            error = "Username and password are required"
        elif len(password) < 6:
            error = "Password must be at least 6 characters"
        elif password != confirm:
            error = "Passwords do not match"
        else:
            pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            conn = get_db()
            try:
                conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pw_hash))
                user_id = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()["id"]
                conn.execute("INSERT INTO bankroll (user_id, starting_amount) VALUES (?, 1000.0)", (user_id,))
                conn.commit()
                session["user_id"] = user_id
                session["username"] = username
                return redirect(url_for("index"))
            except sqlite3.IntegrityError:
                error = "Username already exists"
            finally:
                conn.close()
    return render_template("register.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("dashboard.html", data=get_cached_picks(), record=get_record())


@app.route("/api/picks")
@login_required
def api_picks():
    return jsonify(get_cached_picks())


@app.route("/api/refresh")
@login_required
def api_refresh():
    _cache["timestamp"] = 0
    return jsonify(get_cached_picks())


@app.route("/api/scores")
@login_required
def api_scores():
    scores = fetch_live_scores()
    # Auto-settle any finished games
    try:
        settle_from_live_scores()
    except Exception as e:
        app.logger.warning(f"Auto-settle on score poll failed: {e}")
    # Determine smart poll interval
    statuses = [s["status"] for s in scores.values()]
    if "live" in statuses:
        poll_interval = 30
    elif all(s == "final" for s in statuses) and statuses:
        poll_interval = 300
    else:
        poll_interval = 60
    return jsonify({"scores": scores, "poll_interval": poll_interval})


@app.route("/history")
@login_required
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
@login_required
def api_settle():
    count = settle_unsettled_picks(force=True)
    return jsonify({"settled": count})


@app.route("/api/bankroll")
@login_required
def api_bankroll():
    user_id = session["user_id"]
    conn = get_db()
    bankroll_row = conn.execute(
        "SELECT starting_amount FROM bankroll WHERE user_id = ?", (user_id,)
    ).fetchone()
    starting = bankroll_row["starting_amount"] if bankroll_row else 1000.0

    # Daily P/L from settled picks
    daily_pl = conn.execute("""
        SELECT game_date,
               COALESCE(SUM(ml_profit), 0) + COALESCE(SUM(ou_profit), 0) as daily_pl
        FROM picks
        WHERE ml_result IS NOT NULL OR ou_result IS NOT NULL
        GROUP BY game_date
        ORDER BY game_date ASC
    """).fetchall()
    conn.close()

    dates = []
    values = []
    running = starting
    total_pl = 0.0
    for row in daily_pl:
        total_pl += row["daily_pl"]
        running = starting + total_pl
        dates.append(row["game_date"])
        values.append(round(running, 2))

    current = round(running, 2) if dates else starting
    roi = round(total_pl / starting * 100, 1) if starting else 0

    return jsonify({
        "starting": starting,
        "current": current,
        "total_pl": round(total_pl, 2),
        "roi": roi,
        "dates": dates,
        "values": values,
    })



# ── Kalshi Integration ────────────────────────────────────────────────────

KALSHI_BET_SIZE_CENTS = 1000  # $10 flat bet per pick
KALSHI_OU_MIN_CONFIDENCE = 53.0  # O/U picks with 53%+ confidence (profitable threshold)
KALSHI_ML_MIN_CONFIDENCE = 999  # ML disabled — favorite juice kills ROI
KALSHI_MIN_EDGE = 0.0  # Only bet when model has positive edge over price
KALSHI_STARTING_BALANCE_CENTS = 50000  # $500 paper starting balance
KALSHI_PAPER_MODE = True  # Paper trading — no real Kalshi API needed


def _odds_to_implied_price(american_odds):
    """Convert American odds to implied probability (0-100 cents)."""
    if not american_odds:
        return 50
    if american_odds > 0:
        return round(100 / (american_odds / 100 + 1))
    else:
        return round(abs(american_odds) / (abs(american_odds) / 100 + 1))


def get_paper_balance():
    """Calculate current paper trading balance from starting balance + settled P/L."""
    conn = get_db()
    total_pl = conn.execute(
        "SELECT COALESCE(SUM(profit_cents), 0) as pl FROM kalshi_bets WHERE status = 'settled'"
    ).fetchone()["pl"]
    pending_stakes = conn.execute(
        "SELECT COALESCE(SUM(stake_cents), 0) as s FROM kalshi_bets WHERE status IN ('placed', 'filled')"
    ).fetchone()["s"]
    conn.close()
    available = KALSHI_STARTING_BALANCE_CENTS + total_pl - pending_stakes
    return {
        "balance_cents": KALSHI_STARTING_BALANCE_CENTS + total_pl,
        "available_cents": available,
        "pending_cents": pending_stakes,
        "total_pl_cents": total_pl,
    }


def find_kalshi_opportunities(picks_data, bankroll_override_cents=None):
    """Find betting opportunities from model picks.

    Paper mode: uses sportsbook odds as simulated prices (no Kalshi API needed).
    O/U only at 53%+ confidence with positive edge. ML disabled (bad ROI on favorites).
    Sorted by confidence descending.
    """
    bet_size = KALSHI_BET_SIZE_CENTS
    opportunities = []
    games = picks_data.get("games", [])

    for game in games:
        home = game["home_team"]
        away = game["away_team"]
        game_key = f"{home}:{away}"

        # --- O/U opportunity ---
        if game.get("ou_pick") and game.get("ou_probs") and game.get("ou_line"):
            ou_pick = game["ou_pick"]
            ou_confidence = game.get("ou_confidence", 0)

            if ou_confidence >= KALSHI_OU_MIN_CONFIDENCE:
                ou_line = game["ou_line"]
                model_ou_prob = game["ou_probs"]["over" if ou_pick == "OVER" else "under"] / 100.0

                # Simulate price from sportsbook odds
                best_ou_odds = None
                for book_data in game.get("books", {}).values():
                    totals = book_data.get("totals", {})
                    if totals and totals.get("line") == ou_line:
                        odds_key = "over_odds" if ou_pick == "OVER" else "under_odds"
                        odds = totals.get(odds_key)
                        if odds:
                            best_ou_odds = odds
                            break

                simulated_price = _odds_to_implied_price(best_ou_odds)
                side = "yes" if ou_pick == "OVER" else "no"

                edge = round((model_ou_prob - simulated_price / 100.0) * 100, 1)
                contracts = bet_size // simulated_price if simulated_price > 0 else 0
                if contracts > 0 and edge >= KALSHI_MIN_EDGE:
                    stake_cents = contracts * simulated_price
                    potential_profit = contracts * (100 - simulated_price)
                    opportunities.append({
                        "home_team": home,
                        "away_team": away,
                        "game_key": game_key,
                        "ticker": f"PAPER-OU-{game_key}",
                        "title": f"{away} @ {home} O/U {ou_line}",
                        "bet_type": "ou",
                        "pick": f"{ou_pick} {ou_line}",
                        "side": side,
                        "model_prob": round(model_ou_prob * 100, 1),
                        "kalshi_price": simulated_price,
                        "edge": edge,
                        "stake_cents": stake_cents,
                        "contracts": contracts,
                        "price_cents": simulated_price,
                        "potential_profit_cents": potential_profit,
                        "confidence": ou_confidence,
                        "volume": 0,
                        "paper": True,
                        "ou_line": ou_line,
                        "best_odds": best_ou_odds,
                    })

        # --- ML opportunity ---
        ml_confidence = game.get("confidence", 0)
        ml_pick = game.get("pick")
        if ml_pick and ml_confidence >= KALSHI_ML_MIN_CONFIDENCE:
            pick_side = game.get("pick_side", "home")
            model_ml_prob = ml_confidence / 100.0

            # Simulate price from sportsbook ML odds
            best_ml_odds = None
            for book_data in game.get("books", {}).values():
                ml_data = book_data.get("moneyline", {})
                odds_key = f"ml_{pick_side}"
                odds = ml_data.get(odds_key)
                if odds:
                    best_ml_odds = odds
                    break

            simulated_price = _odds_to_implied_price(best_ml_odds)
            side = "yes"

            contracts = bet_size // simulated_price if simulated_price > 0 else 0
            if contracts > 0:
                stake_cents = contracts * simulated_price
                potential_profit = contracts * (100 - simulated_price)
                opportunities.append({
                    "home_team": home,
                    "away_team": away,
                    "game_key": game_key,
                    "ticker": f"PAPER-ML-{game_key}",
                    "title": f"{ml_pick} to win",
                    "bet_type": "ml",
                    "pick": ml_pick,
                    "side": side,
                    "model_prob": round(model_ml_prob * 100, 1),
                    "kalshi_price": simulated_price,
                    "edge": round((model_ml_prob - simulated_price / 100.0) * 100, 1),
                    "stake_cents": stake_cents,
                    "contracts": contracts,
                    "price_cents": simulated_price,
                    "potential_profit_cents": potential_profit,
                    "confidence": ml_confidence,
                    "volume": 0,
                    "paper": True,
                    "ml_pick_team": ml_pick,
                    "best_odds": best_ml_odds,
                })

    opportunities.sort(key=lambda x: x["confidence"], reverse=True)
    return opportunities


def auto_place_paper_bets():
    """Auto-place all qualifying paper bets for today. Called daily."""
    picks_data = get_cached_picks()
    if not picks_data or not picks_data.get("games"):
        return 0

    opportunities = find_kalshi_opportunities(picks_data)
    if not opportunities:
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    placed = 0

    for opp in opportunities:
        # Skip if already have an open or today's bet for this game+type
        existing = conn.execute(
            "SELECT id FROM kalshi_bets WHERE home_team = ? "
            "AND away_team = ? AND bet_type = ? AND status NOT IN ('error', 'settled', 'cancelled')",
            (opp["home_team"], opp["away_team"], opp["bet_type"])
        ).fetchone()
        if existing:
            continue

        # Check balance
        balance = get_paper_balance()
        if balance["available_cents"] < opp["stake_cents"]:
            app.logger.info(f"Paper: insufficient balance for {opp['pick']}")
            continue

        pick_row = conn.execute(
            "SELECT id FROM picks WHERE game_date = ? AND home_team = ? AND away_team = ?",
            (today, opp["home_team"], opp["away_team"])
        ).fetchone()
        pick_id = pick_row["id"] if pick_row else None

        conn.execute("""
            INSERT INTO kalshi_bets
            (game_date, home_team, away_team, pick_id, kalshi_ticker, bet_side,
             bet_type, model_prob, kalshi_price, edge, kelly_fraction,
             stake_cents, contracts, status, placed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'placed', ?)
        """, (today, opp["home_team"], opp["away_team"], pick_id, opp["ticker"],
              opp["side"], opp["bet_type"], opp["model_prob"], opp["kalshi_price"],
              opp["edge"], 0, opp["stake_cents"], opp["contracts"],
              datetime.now().isoformat()))
        placed += 1
        app.logger.info(f"Paper bet placed: {opp['bet_type'].upper()} {opp['pick']} "
                        f"({opp['confidence']}%) ${opp['stake_cents']/100:.2f}")

    if placed:
        conn.commit()
    conn.close()
    return placed


def settle_paper_bets():
    """Settle paper bets using actual game scores."""
    scores = fetch_live_scores()
    conn = get_db()
    open_bets = conn.execute(
        "SELECT * FROM kalshi_bets WHERE status IN ('placed', 'filled')"
    ).fetchall()

    settled = 0
    for bet in open_bets:
        game_key = f"{bet['home_team']}:{bet['away_team']}"

        # Try live scores first
        info = scores.get(game_key) if scores else None

        # Fallback: check picks table for already-settled scores
        if not info or info.get("status") != "final":
            pick_row = conn.execute(
                "SELECT home_score, away_score FROM picks "
                "WHERE game_date = ? AND home_team = ? AND away_team = ? AND home_score IS NOT NULL",
                (bet["game_date"], bet["home_team"], bet["away_team"])
            ).fetchone()
            if pick_row:
                info = {"status": "final", "home_score": pick_row["home_score"],
                        "away_score": pick_row["away_score"]}

        if not info or info.get("status") != "final":
            continue
        if info["home_score"] is None or info["away_score"] is None:
            continue

        home_score = info["home_score"]
        away_score = info["away_score"]
        total_points = home_score + away_score
        winner = bet["home_team"] if home_score > away_score else bet["away_team"]

        won = False
        if bet["bet_type"] == "ou":
            ou_parts = bet["kalshi_ticker"].replace("PAPER-OU-", "")
            pick_text = conn.execute(
                "SELECT ou_pick, ou_line FROM picks WHERE game_date = ? AND home_team = ? AND away_team = ?",
                (bet["game_date"], bet["home_team"], bet["away_team"])
            ).fetchone()
            if pick_text:
                ou_pick = pick_text["ou_pick"]
                ou_line = pick_text["ou_line"]
                if total_points > ou_line and ou_pick == "OVER":
                    won = True
                elif total_points < ou_line and ou_pick == "UNDER":
                    won = True
                elif total_points == ou_line:
                    # Push — refund
                    conn.execute("""
                        UPDATE kalshi_bets SET status = 'settled', result = 'push',
                        payout_cents = ?, profit_cents = 0, settled_at = ? WHERE id = ?
                    """, (bet["stake_cents"], datetime.now().isoformat(), bet["id"]))
                    settled += 1
                    continue
        elif bet["bet_type"] == "ml":
            ml_pick_team = bet["kalshi_ticker"].replace("PAPER-ML-", "").split(":")[0]
            # The pick team is stored in the ticker as PAPER-ML-Home:Away
            # We need to check who was picked — look at the picks table
            pick_row = conn.execute(
                "SELECT ml_pick FROM picks WHERE game_date = ? AND home_team = ? AND away_team = ?",
                (bet["game_date"], bet["home_team"], bet["away_team"])
            ).fetchone()
            if pick_row:
                won = (pick_row["ml_pick"] == winner)

        if won:
            payout_cents = int(bet["contracts"] * 100)
            profit_cents = payout_cents - bet["stake_cents"]
            result = "win"
        else:
            payout_cents = 0
            profit_cents = -bet["stake_cents"]
            result = "loss"

        conn.execute("""
            UPDATE kalshi_bets SET status = 'settled', result = ?,
            payout_cents = ?, profit_cents = ?, settled_at = ? WHERE id = ?
        """, (result, payout_cents, profit_cents, datetime.now().isoformat(), bet["id"]))
        settled += 1

    if settled:
        conn.commit()
        app.logger.info(f"Settled {settled} paper bets")
    conn.close()
    return settled


@app.route("/kalshi")
@login_required
def kalshi_page():
    picks_data = get_cached_picks()

    # Auto-place paper bets if any qualify
    if KALSHI_PAPER_MODE:
        auto_place_paper_bets()

    balance = get_paper_balance()
    opportunities = find_kalshi_opportunities(picks_data)

    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")

    active_bets = conn.execute(
        "SELECT * FROM kalshi_bets WHERE status NOT IN ('settled', 'cancelled', 'error') "
        "ORDER BY created_at DESC"
    ).fetchall()
    active_bets = [dict(b) for b in active_bets]

    # Today's settled P/L
    today_pl = conn.execute(
        "SELECT COALESCE(SUM(profit_cents), 0) as pl FROM kalshi_bets "
        "WHERE game_date = ? AND status = 'settled'", (today,)
    ).fetchone()

    # All-time stats
    all_stats = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
               COALESCE(SUM(profit_cents), 0) as total_pl
        FROM kalshi_bets WHERE status = 'settled'
    """).fetchone()

    conn.close()
    today_pl_cents = today_pl["pl"] if today_pl else 0

    return render_template("kalshi.html",
                           kalshi_configured=True,
                           paper_mode=KALSHI_PAPER_MODE,
                           balance=balance,
                           opportunities=opportunities,
                           active_bets=active_bets,
                           today_pl_cents=today_pl_cents,
                           all_stats=dict(all_stats) if all_stats else {})


@app.route("/api/kalshi/confirm", methods=["POST"])
@login_required
def kalshi_confirm():
    data = request.get_json()
    ticker = data.get("ticker")
    side = data.get("side")
    contracts = data.get("contracts")
    price_cents = data.get("price_cents")
    home_team = data.get("home_team")
    away_team = data.get("away_team")
    bet_type = data.get("bet_type", "ml")
    model_prob = data.get("model_prob")
    kalshi_price = data.get("kalshi_price")
    edge = data.get("edge")
    stake_cents = data.get("stake_cents")

    if not all([ticker, side, contracts, price_cents]):
        return jsonify({"error": "Missing required fields"}), 400

    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()

    pick_row = conn.execute(
        "SELECT id FROM picks WHERE game_date = ? AND home_team = ? AND away_team = ?",
        (today, home_team, away_team)
    ).fetchone()
    pick_id = pick_row["id"] if pick_row else None

    if KALSHI_PAPER_MODE:
        conn.execute("""
            INSERT INTO kalshi_bets
            (game_date, home_team, away_team, pick_id, kalshi_ticker, bet_side,
             bet_type, model_prob, kalshi_price, edge, kelly_fraction,
             stake_cents, contracts, status, placed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'placed', ?)
        """, (today, home_team, away_team, pick_id, ticker, side,
              bet_type, model_prob, kalshi_price, edge, 0,
              stake_cents, contracts, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "message": f"Paper bet placed: {side.upper()} x{contracts} @ {price_cents}c",
        })

    # Real Kalshi mode
    provider = get_kalshi_provider()
    if not provider:
        conn.close()
        return jsonify({"error": "Kalshi not configured"}), 400

    try:
        market = provider.get_market(ticker)
        current_price = market.get("yes_ask" if side == "yes" else "no_ask", 0)
        if current_price and abs(current_price - price_cents) > 3:
            conn.close()
            return jsonify({
                "error": f"Price moved: was {price_cents}c, now {current_price}c. Refresh and retry.",
                "price_moved": True, "new_price": current_price,
            }), 409
    except Exception as e:
        app.logger.warning(f"Price re-check failed: {e}")

    conn.execute("""
        INSERT INTO kalshi_bets
        (game_date, home_team, away_team, pick_id, kalshi_ticker, bet_side,
         bet_type, model_prob, kalshi_price, edge, kelly_fraction,
         stake_cents, contracts, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'confirmed')
    """, (today, home_team, away_team, pick_id, ticker, side,
          bet_type, model_prob, kalshi_price, edge, 0,
          stake_cents, contracts))
    bet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()

    order = provider.place_order(ticker, side, contracts, price_cents)
    if order and order.get("order_id"):
        conn.execute("""
            UPDATE kalshi_bets SET status = 'placed', kalshi_order_id = ?, placed_at = ?
            WHERE id = ?
        """, (order["order_id"], datetime.now().isoformat(), bet_id))
        conn.commit()
        conn.close()
        return jsonify({
            "success": True, "bet_id": bet_id, "order_id": order["order_id"],
            "message": f"Order placed: {side.upper()} x{contracts} @ {price_cents}c",
        })
    else:
        conn.execute("UPDATE kalshi_bets SET status = 'error', error_message = 'Order failed' WHERE id = ?", (bet_id,))
        conn.commit()
        conn.close()
        return jsonify({"error": "Order placement failed"}), 500


@app.route("/api/kalshi/skip", methods=["POST"])
@login_required
def kalshi_skip():
    data = request.get_json()
    ticker = data.get("ticker")
    if ticker:
        app.logger.info(f"Kalshi opportunity skipped: {ticker}")
    return jsonify({"success": True})


@app.route("/api/kalshi/refresh")
@login_required
def kalshi_refresh():
    picks_data = get_cached_picks()
    opportunities = find_kalshi_opportunities(picks_data)
    balance = get_paper_balance() if KALSHI_PAPER_MODE else None
    if not KALSHI_PAPER_MODE:
        provider = get_kalshi_provider()
        if provider:
            provider.clear_cache()
            balance = provider.get_portfolio_balance()
    return jsonify({"opportunities": opportunities, "balance": balance})


@app.route("/api/kalshi/history")
@login_required
def kalshi_history():
    conn = get_db()
    bets = conn.execute(
        "SELECT * FROM kalshi_bets WHERE status = 'settled' ORDER BY settled_at DESC LIMIT 100"
    ).fetchall()
    conn.close()
    return jsonify({"bets": [dict(b) for b in bets]})


@app.route("/api/kalshi/positions")
@login_required
def kalshi_positions():
    provider = get_kalshi_provider()
    if not provider:
        return jsonify({"error": "Kalshi not configured"}), 400
    positions = provider.get_positions()
    return jsonify({"positions": positions})


@app.route("/api/kalshi/notify")
@login_required
def kalshi_notify_telegram():
    """Trigger Telegram notifications for current Kalshi opportunities."""
    bot = get_telegram_bot()
    if not bot:
        return jsonify({"error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env"}), 400

    provider = get_kalshi_provider()
    if not provider:
        return jsonify({"error": "Kalshi not configured"}), 400

    picks_data = get_cached_picks()
    opportunities = find_kalshi_opportunities(picks_data)

    balance = provider.get_portfolio_balance()
    balance_cents = balance["available_cents"] if balance else None

    # Send summary
    bot.send_daily_summary(opportunities, balance_cents)

    # Send each opportunity with Place/Skip buttons
    sent = 0
    for opp in opportunities:
        import uuid
        callback_id = uuid.uuid4().hex[:12]

        def make_handler(opportunity, cb_id):
            def handler(action):
                with app.app_context():
                    if action == "place":
                        _telegram_place_bet(opportunity, bot)
                    else:
                        bot.send_message(f"Skipped: {opportunity['pick']}")
            return handler

        bot.register_callback(callback_id, make_handler(opp, callback_id))
        bot.send_opportunity(opp, callback_id)
        sent += 1

    return jsonify({"success": True, "sent": sent, "opportunities": len(opportunities)})


def _telegram_place_bet(opp, bot):
    """Place a Kalshi bet triggered from Telegram approval."""
    provider = get_kalshi_provider()
    if not provider:
        bot.send_bet_result(opp, False, "Kalshi provider not available")
        return

    # Re-check current price
    try:
        market = provider.get_market(opp["ticker"])
        current_ask = market.get("yes_ask", 0) if opp["side"] == "yes" else market.get("no_ask", 0)
        if current_ask and abs(current_ask - opp["price_cents"]) > 3:
            bot.send_bet_result(
                opp, False,
                f"Price moved: was {opp['price_cents']}c, now {current_ask}c. Skipping."
            )
            return
    except Exception as e:
        app.logger.warning(f"Price recheck failed: {e}")

    # Place the order
    order = provider.place_order(opp["ticker"], opp["side"], opp["contracts"], opp["price_cents"])

    # Save to DB
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    pick_row = conn.execute(
        "SELECT id FROM picks WHERE game_date = ? AND home_team = ? AND away_team = ?",
        (today, opp["home_team"], opp["away_team"])
    ).fetchone()
    pick_id = pick_row["id"] if pick_row else None

    if order and order.get("order_id"):
        conn.execute("""
            INSERT INTO kalshi_bets
            (game_date, home_team, away_team, pick_id, kalshi_ticker, bet_side,
             bet_type, model_prob, kalshi_price, edge, kelly_fraction,
             stake_cents, contracts, status, kalshi_order_id, placed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'placed', ?, ?)
        """, (today, opp["home_team"], opp["away_team"], pick_id, opp["ticker"],
              opp["side"], opp["bet_type"], opp["model_prob"], opp["kalshi_price"],
              opp["edge"], opp["kelly_fraction"], opp["stake_cents"], opp["contracts"],
              order["order_id"], datetime.now().isoformat()))
        conn.commit()
        conn.close()
        bot.send_bet_result(opp, True, f"Order ID: {order['order_id']}")
    else:
        conn.execute("""
            INSERT INTO kalshi_bets
            (game_date, home_team, away_team, pick_id, kalshi_ticker, bet_side,
             bet_type, model_prob, kalshi_price, edge, kelly_fraction,
             stake_cents, contracts, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'error', 'Order placement failed')
        """, (today, opp["home_team"], opp["away_team"], pick_id, opp["ticker"],
              opp["side"], opp["bet_type"], opp["model_prob"], opp["kalshi_price"],
              opp["edge"], opp["kelly_fraction"], opp["stake_cents"], opp["contracts"]))
        conn.commit()
        conn.close()
        bot.send_bet_result(opp, False, "Order placement failed")


def settle_kalshi_bets():
    """Check and settle placed Kalshi bets."""
    provider = get_kalshi_provider()
    if not provider:
        return 0

    conn = get_db()
    open_bets = conn.execute(
        "SELECT * FROM kalshi_bets WHERE status IN ('placed', 'filled')"
    ).fetchall()

    settled = 0
    for bet in open_bets:
        try:
            # Check order status
            if bet["kalshi_order_id"]:
                order = provider.get_order(bet["kalshi_order_id"])
                if order:
                    order_status = order.get("status", "")
                    if order_status == "canceled":
                        conn.execute(
                            "UPDATE kalshi_bets SET status = 'cancelled', settled_at = ? WHERE id = ?",
                            (datetime.now().isoformat(), bet["id"])
                        )
                        settled += 1
                        continue
                    if order_status == "executed" and bet["status"] != "filled":
                        fill_price = order.get("avg_price", bet["kalshi_price"])
                        conn.execute(
                            "UPDATE kalshi_bets SET status = 'filled', fill_price = ? WHERE id = ?",
                            (fill_price, bet["id"])
                        )

            # Check market settlement
            market = provider.get_market(bet["kalshi_ticker"])
            if not market:
                continue

            market_result = market.get("result")
            if market_result is None:
                continue

            # Market settled: result is "yes" or "no"
            won = (market_result == bet["bet_side"])
            if won:
                payout_cents = int(bet["contracts"] * 100)  # $1 per contract on win
                profit_cents = payout_cents - bet["stake_cents"]
                result = "win"
            else:
                payout_cents = 0
                profit_cents = -bet["stake_cents"]
                result = "loss"

            conn.execute("""
                UPDATE kalshi_bets SET status = 'settled', result = ?,
                payout_cents = ?, profit_cents = ?, settled_at = ?
                WHERE id = ?
            """, (result, payout_cents, profit_cents, datetime.now().isoformat(), bet["id"]))
            settled += 1

        except Exception as e:
            app.logger.warning(f"Kalshi bet settlement error for bet {bet['id']}: {e}")

    if settled:
        conn.commit()
        app.logger.info(f"Settled {settled} Kalshi bets")
    conn.close()
    return settled


_settle_thread_started = False
_settle_stop_event = threading.Event()
SETTLE_INTERVAL = 300  # 5 minutes


def _background_settle_loop():
    """Background thread that auto-settles picks and paper bets."""
    while not _settle_stop_event.is_set():
        try:
            with app.app_context():
                # Primary: settle from Odds API scores (fast, reliable)
                count = settle_from_live_scores()
                # Fallback: sbrscrape catches anything the API missed
                count += settle_unsettled_picks(force=True)
                # Settle paper/Kalshi bets
                if KALSHI_PAPER_MODE:
                    count += settle_paper_bets()
                    # Auto-place paper bets if picks are available
                    placed = auto_place_paper_bets()
                    if placed:
                        app.logger.info(f"Auto-placed {placed} paper bets")
                else:
                    count += settle_kalshi_bets()
                if count:
                    app.logger.info(f"Background settlement: settled {count} picks/bets")
        except Exception as e:
            app.logger.warning(f"Background settlement error: {e}")
        _settle_stop_event.wait(SETTLE_INTERVAL)


def start_settle_thread():
    global _settle_thread_started
    if _settle_thread_started:
        return
    _settle_thread_started = True
    t = threading.Thread(target=_background_settle_loop, daemon=True)
    t.start()
    atexit.register(lambda: _settle_stop_event.set())


start_settle_thread()


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=8002, debug=False)
