"""
Kalshi prediction market API provider.
Handles RSA-PSS authentication, market fetching, order placement, and position tracking.
"""

import base64
import logging
import os
import time
from datetime import datetime, timezone

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Kalshi ticker abbreviation -> project team name
ABBREV_TO_TEAM = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "MIA": "Miami Heat", "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves", "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Reverse lookup: project team name -> Kalshi abbreviation
TEAM_TO_ABBREV = {v: k for k, v in ABBREV_TO_TEAM.items()}

# Series tickers for different NBA market types
SERIES_GAME = "KXNBAGAME"       # game winner
SERIES_TOTAL = "KXNBATOTAL"     # total points
SERIES_SPREAD = "KXNBASPREAD"   # spread


class KalshiProvider:
    def __init__(self, api_key=None, private_key_path=None):
        self.api_key = api_key or os.environ.get("KALSHI_API_KEY")
        self.private_key_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH")
        if not self.api_key or not self.private_key_path:
            raise ValueError(
                "KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH must be set. "
                "Generate API keys at https://kalshi.com/account/api"
            )
        self._private_key = self._load_private_key()
        self._cached_markets = None
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 5 minutes
        self._last_request_time = 0
        self._min_request_interval = 0.1  # 100ms rate limit guard

    def _load_private_key(self):
        with open(self.private_key_path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)

    def _sign_request(self, method, path, timestamp):
        """RSA-PSS signature: sign(timestamp + method + path)."""
        message = f"{timestamp}{method}{path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, method, path, params=None, json_body=None):
        """Make authenticated request to Kalshi API."""
        self._rate_limit()
        timestamp = str(int(time.time() * 1000))
        url = f"{KALSHI_API_BASE}{path}"

        signature = self._sign_request(method.upper(), path, timestamp)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            resp = requests.request(
                method, url, headers=headers,
                params=params, json=json_body, timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError:
            logger.error(f"Kalshi API error {resp.status_code}: {resp.text}")
            raise
        except Exception as e:
            logger.error(f"Kalshi API request failed: {e}")
            raise

    @staticmethod
    def _parse_game_ticker(ticker):
        """Parse a Kalshi NBA ticker to extract game info.

        Tickers follow patterns like:
          KXNBAGAME-26MAR06DALBOS-BOS  (game winner for BOS in DAL@BOS on Mar 6 2026)
          KXNBATOTAL-26MAR06DALBOS-224 (total points > 224 for DAL@BOS)
        """
        parts = ticker.split("-")
        if len(parts) < 3:
            return None

        series = parts[0]
        game_code = parts[1]  # e.g., "26MAR06DALBOS"
        outcome = parts[2]    # e.g., "BOS" or "224"

        # Extract away/home abbreviations (last 6 chars of game_code = 2x 3-letter abbrevs)
        if len(game_code) < 6:
            return None
        away_abbrev = game_code[-6:-3]
        home_abbrev = game_code[-3:]

        away_team = ABBREV_TO_TEAM.get(away_abbrev)
        home_team = ABBREV_TO_TEAM.get(home_abbrev)

        if not away_team or not home_team:
            return None

        # Determine market type
        if series == SERIES_GAME:
            market_type = "ml"
            team_abbrev = outcome  # which team this contract is for
        elif series == SERIES_TOTAL:
            market_type = "ou"
            team_abbrev = None
        elif series == SERIES_SPREAD:
            market_type = "spread"
            team_abbrev = outcome
        else:
            return None

        return {
            "series": series,
            "away_team": away_team,
            "home_team": home_team,
            "away_abbrev": away_abbrev,
            "home_abbrev": home_abbrev,
            "market_type": market_type,
            "outcome": outcome,
            "team_abbrev": team_abbrev,
        }

    def _fetch_markets_for_series(self, series_ticker):
        """Fetch all open markets for a given series."""
        all_markets = []
        cursor = None
        while True:
            params = {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                data = self._request("GET", "/markets", params=params)
            except Exception:
                break
            batch = data.get("markets", [])
            all_markets.extend(batch)
            cursor = data.get("cursor")
            if not cursor or len(batch) < 200:
                break
        return all_markets

    def get_nba_markets(self):
        """Fetch today's open NBA game winner and total points markets.

        Returns:
            dict keyed by "HomeTeam:AwayTeam", values are dicts with:
                "game_markets": {team_name: {ticker, yes_ask, ...}, ...}
                "total_markets": [{ticker, line, yes_ask, ...}, ...]
        """
        now = time.time()
        if self._cached_markets and (now - self._cache_timestamp) < self._cache_ttl:
            return self._cached_markets

        result = {}

        # Fetch game winner markets
        game_markets = self._fetch_markets_for_series(SERIES_GAME)
        for m in game_markets:
            parsed = self._parse_game_ticker(m.get("ticker", ""))
            if not parsed or parsed["market_type"] != "ml":
                continue

            game_key = f"{parsed['home_team']}:{parsed['away_team']}"
            if game_key not in result:
                result[game_key] = {"game_markets": {}, "total_markets": []}

            team_for = ABBREV_TO_TEAM.get(parsed["team_abbrev"], parsed["team_abbrev"])
            result[game_key]["game_markets"][team_for] = {
                "ticker": m["ticker"],
                "title": m.get("title", ""),
                "yes_bid": m.get("yes_bid", 0) or 0,
                "yes_ask": m.get("yes_ask", 100) or 100,
                "no_bid": m.get("no_bid", 0) or 0,
                "no_ask": m.get("no_ask", 100) or 100,
                "volume": m.get("volume", 0),
                "close_time": m.get("close_time"),
            }

        # Fetch total points markets
        total_markets = self._fetch_markets_for_series(SERIES_TOTAL)
        for m in total_markets:
            parsed = self._parse_game_ticker(m.get("ticker", ""))
            if not parsed or parsed["market_type"] != "ou":
                continue

            game_key = f"{parsed['home_team']}:{parsed['away_team']}"
            if game_key not in result:
                result[game_key] = {"game_markets": {}, "total_markets": []}

            try:
                line = float(parsed["outcome"])
            except (ValueError, TypeError):
                continue

            result[game_key]["total_markets"].append({
                "ticker": m["ticker"],
                "title": m.get("title", ""),
                "line": line,
                "yes_bid": m.get("yes_bid", 0) or 0,
                "yes_ask": m.get("yes_ask", 100) or 100,
                "no_bid": m.get("no_bid", 0) or 0,
                "no_ask": m.get("no_ask", 100) or 100,
                "volume": m.get("volume", 0),
            })

        # Sort total markets by line
        for gk in result:
            result[gk]["total_markets"].sort(key=lambda x: x["line"])

        self._cached_markets = result
        self._cache_timestamp = now
        total_count = sum(
            len(v["game_markets"]) + len(v["total_markets"]) for v in result.values()
        )
        logger.info(f"Fetched {total_count} Kalshi NBA markets across {len(result)} games")
        return result

    def get_portfolio_balance(self):
        """Get current portfolio balance in cents."""
        try:
            data = self._request("GET", "/portfolio/balance")
            return {
                "balance_cents": data.get("balance", 0),
                "available_cents": data.get("payout", 0),
            }
        except Exception:
            return None

    def place_order(self, ticker, side, contracts, price_cents):
        """Place a limit order on Kalshi.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            contracts: Number of contracts
            price_cents: Limit price in cents (1-99)

        Returns:
            dict with order details or None on failure
        """
        if contracts <= 0:
            logger.warning("Cannot place order with 0 contracts")
            return None

        price_key = "yes_price" if side == "yes" else "no_price"
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": "limit",
            "count": contracts,
            price_key: price_cents,
        }

        try:
            data = self._request("POST", "/portfolio/orders", json_body=body)
            order = data.get("order", {})
            logger.info(
                f"Order placed: {ticker} {side} x{contracts} @ {price_cents}c "
                f"(order_id: {order.get('order_id')})"
            )
            return order
        except Exception:
            return None

    def get_positions(self):
        """Get current open positions."""
        try:
            data = self._request("GET", "/portfolio/positions")
            return data.get("market_positions", [])
        except Exception:
            return []

    def get_order(self, order_id):
        """Get order status by ID."""
        try:
            data = self._request("GET", f"/portfolio/orders/{order_id}")
            return data.get("order", {})
        except Exception:
            return None

    def get_market(self, ticker):
        """Get single market details (for settlement checking)."""
        try:
            data = self._request("GET", f"/markets/{ticker}")
            return data.get("market", {})
        except Exception:
            return {}

    def clear_cache(self):
        self._cached_markets = None
        self._cache_timestamp = 0
