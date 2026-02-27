import logging
import os

import requests

logger = logging.getLogger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"

BOOKMAKER_KEYS = {
    "betonline": "betonlineag",
    "fanduel": "fanduel",
    "draftkings": "draftkings",
    "betmgm": "betmgm",
    "bovada": "bovada",
    "pointsbet": "pointsbetus",
    "caesars": "williamhill_us",
    "wynn": "wynnbet",
    "bet_rivers_ny": "betrivers",
}

# Known team name divergences between The Odds API and this project
TEAM_NAME_MAP = {
    "Los Angeles Clippers": "LA Clippers",
}


class OddsApiProvider:
    def __init__(self, sportsbook="betonline", api_key=None):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ODDS_API_KEY environment variable is not set. "
                "Get a free key at https://the-odds-api.com"
            )
        self.sportsbook = sportsbook
        self._cached_response = None

    @staticmethod
    def _normalize_team_name(name):
        return TEAM_NAME_MAP.get(name, name)

    def _fetch_odds(self, bookmakers=None):
        if self._cached_response is not None:
            return self._cached_response

        if bookmakers is None:
            key = BOOKMAKER_KEYS.get(self.sportsbook)
            if not key:
                raise ValueError(f"Unknown sportsbook: {self.sportsbook}")
            bookmakers = key

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "bookmakers": bookmakers,
        }

        resp = requests.get(ODDS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()

        remaining = resp.headers.get("x-requests-remaining")
        used = resp.headers.get("x-requests-used")
        logger.info(f"Odds API credits used: {used}, remaining: {remaining}")

        self._cached_response = resp.json()
        return self._cached_response

    def get_odds(self):
        """Return SbrOddsProvider-compatible dict.

        Format:
            {"Home:Away": {"under_over_odds": 215.5,
                           "Home": {"money_line_odds": -110},
                           "Away": {"money_line_odds": 130}}}
        """
        events = self._fetch_odds()
        book_key = BOOKMAKER_KEYS.get(self.sportsbook, self.sportsbook)
        result = {}

        for event in events:
            home = self._normalize_team_name(event.get("home_team", ""))
            away = self._normalize_team_name(event.get("away_team", ""))
            if not home or not away:
                continue

            game_key = f"{home}:{away}"
            ml_home = None
            ml_away = None
            total_line = None

            for bookmaker in event.get("bookmakers", []):
                if bookmaker["key"] != book_key:
                    continue
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            normalized = self._normalize_team_name(outcome["name"])
                            if normalized == home:
                                ml_home = outcome["price"]
                            elif normalized == away:
                                ml_away = outcome["price"]
                    elif market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                total_line = outcome["point"]

            result[game_key] = {
                "under_over_odds": total_line,
                home: {"money_line_odds": ml_home},
                away: {"money_line_odds": ml_away},
            }

        return result

    def get_full_odds(self, bookmakers_list=None):
        """Return web/app.py-compatible dict with per-book odds.

        Format:
            {"Home:Away": {"home_team": "...", "away_team": "...",
                           "books": {"betonline": {"ml_home": -110, "ml_away": 130,
                                                    "total": 215.5, "over_odds": -110,
                                                    "under_odds": -105}}}}
        """
        if bookmakers_list is None:
            bookmakers_list = [self.sportsbook]

        api_keys = [BOOKMAKER_KEYS.get(b, b) for b in bookmakers_list]
        events = self._fetch_odds(bookmakers=",".join(api_keys))

        result = {}
        for event in events:
            home = self._normalize_team_name(event.get("home_team", ""))
            away = self._normalize_team_name(event.get("away_team", ""))
            if not home or not away:
                continue

            game_key = f"{home}:{away}"
            game_data = {"home_team": home, "away_team": away, "books": {}}

            for bookmaker in event.get("bookmakers", []):
                # Reverse-lookup the user-facing name for this bookmaker key
                book_name = None
                for name, key in BOOKMAKER_KEYS.items():
                    if key == bookmaker["key"] and name in bookmakers_list:
                        book_name = name
                        break
                if book_name is None:
                    continue

                book_data = {}
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for outcome in market.get("outcomes", []):
                            normalized = self._normalize_team_name(outcome["name"])
                            if normalized == home:
                                book_data["ml_home"] = outcome["price"]
                            elif normalized == away:
                                book_data["ml_away"] = outcome["price"]
                    elif market["key"] == "spreads":
                        for outcome in market.get("outcomes", []):
                            normalized = self._normalize_team_name(outcome["name"])
                            if normalized == home:
                                book_data["spread_home"] = outcome.get("point")
                                book_data["spread_home_odds"] = outcome["price"]
                            elif normalized == away:
                                book_data["spread_away"] = outcome.get("point")
                                book_data["spread_away_odds"] = outcome["price"]
                    elif market["key"] == "totals":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == "Over":
                                book_data["total"] = outcome["point"]
                                book_data["over_odds"] = outcome["price"]
                            elif outcome["name"] == "Under":
                                book_data["under_odds"] = outcome["price"]

                if book_data:
                    game_data["books"][book_name] = book_data

            result[game_key] = game_data

        return result
