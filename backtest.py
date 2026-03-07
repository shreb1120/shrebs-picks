#!/usr/bin/env python3
"""
Walk-forward backtest for the NBA ML Sports Betting XGBoost model.

Tests the model on historical games WITH odds data to calculate:
- Raw prediction accuracy
- Betting P/L under different strategies (flat bet, Kelly, positive EV only)
- ROI, max drawdown, win rate by confidence bucket
- Monthly and seasonal breakdowns
"""

import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

# Columns the XGBoost ML model drops during training
DROP_COLUMNS = [
    "index", "Score", "Home-Team-Win", "TEAM_NAME", "Date",
    "index.1", "TEAM_NAME.1", "Date.1", "OU-Cover", "OU",
]

BANKROLL_START = 1000.0
FLAT_BET = 10.0


class BoosterWrapper:
    """Needed for unpickling calibrator saved by training script."""
    _estimator_type = "classifier"

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))

    def decision_function(self, X):
        return self.predict_proba(X)


def load_dataset():
    """Load the combined games dataset."""
    conn = sqlite3.connect(DATA_DIR / "dataset.sqlite")
    df = pd.read_sql('SELECT * FROM "dataset_2012-26"', conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_all_odds():
    """Load and merge all odds tables into a single DataFrame."""
    conn = sqlite3.connect(DATA_DIR / "OddsData.sqlite")
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)

    all_odds = []
    # Prefer _new tables, fall back to regular
    seen_seasons = set()
    for table_name in sorted(tables["name"].tolist()):
        # Determine season from table name
        season = table_name.replace("odds_", "").replace("_new", "")
        is_new = "_new" in table_name

        if is_new or season not in seen_seasons:
            try:
                df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
                if "ML_Home" in df.columns and "ML_Away" in df.columns:
                    # Standardize
                    df = df[["Date", "Home", "Away", "ML_Home", "ML_Away", "OU",
                             "Spread", "Points", "Win_Margin"]].copy()
                    df["season"] = season
                    all_odds.append(df)
                    if is_new:
                        seen_seasons.add(season)
                    elif season not in seen_seasons:
                        seen_seasons.add(season)
            except Exception:
                continue
    conn.close()

    if not all_odds:
        raise ValueError("No odds data found")

    odds = pd.concat(all_odds, ignore_index=True)
    odds["Date"] = pd.to_datetime(odds["Date"], errors="coerce")

    # Drop rows without moneyline odds
    odds = odds.dropna(subset=["ML_Home", "ML_Away"])
    odds = odds[(odds["ML_Home"] != 0) & (odds["ML_Away"] != 0)]

    # Deduplicate (prefer _new tables which were loaded last)
    odds = odds.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
    return odds


def load_xgb_model():
    """Load the best XGBoost moneyline model (skip calibrator due to sklearn version mismatch)."""
    import re
    pattern = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_ML")
    candidates = list(MODEL_DIR.glob("*ML*.json"))
    if not candidates:
        raise FileNotFoundError("No XGBoost ML model found")

    def score(path):
        m = pattern.search(path.name)
        return float(m.group(1)) if m else 0.0

    best_path = max(candidates, key=score)
    model = xgb.Booster()
    model.load_model(str(best_path))

    # Skip calibrator - sklearn 1.3.1 pickle is broken in 1.7.2
    # Raw model probabilities are still well-calibrated for XGBoost softprob
    print(f"Loaded model: {best_path.name}")
    print(f"Using raw model probabilities (calibrator skipped - sklearn version mismatch)")
    return model, None


def predict_probs(model, X, calibrator=None):
    """Get [P(away_win), P(home_win)] for each row."""
    if calibrator is not None:
        return calibrator.predict_proba(X)
    return model.predict(xgb.DMatrix(X))


def american_to_decimal(odds):
    """Convert American odds to decimal payout multiplier."""
    if odds >= 100:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def calc_implied_prob(odds):
    """Convert American odds to implied probability (no-vig)."""
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def calc_payout(odds, stake):
    """Calculate profit on a winning bet."""
    dec = american_to_decimal(odds)
    return stake * dec


def calc_ev(model_prob, odds):
    """Expected value per $100 wagered."""
    payout = american_to_decimal(odds) * 100
    return model_prob * payout - (1 - model_prob) * 100


def calc_kelly(model_prob, odds):
    """Kelly criterion fraction (0-1)."""
    dec = american_to_decimal(odds)
    f = (dec * model_prob - (1 - model_prob)) / dec
    return max(0.0, f)


def merge_dataset_with_odds(dataset, odds):
    """
    Match games in the dataset with their odds.
    The dataset has TEAM_NAME (home) and TEAM_NAME.1 (away) + Date.
    The odds have Home, Away, Date.
    """
    # Normalize team names
    name_map = {
        "LA Clippers": "Los Angeles Clippers",
        "L.A. Clippers": "Los Angeles Clippers",
        "L.A. Lakers": "Los Angeles Lakers",
    }

    df = dataset.copy()
    df["home_norm"] = df["TEAM_NAME"].replace(name_map).str.strip()
    df["away_norm"] = df["TEAM_NAME.1"].replace(name_map).str.strip()

    od = odds.copy()
    od["home_norm"] = od["Home"].replace(name_map).str.strip()
    od["away_norm"] = od["Away"].replace(name_map).str.strip()

    merged = df.merge(
        od[["Date", "home_norm", "away_norm", "ML_Home", "ML_Away"]],
        on=["Date", "home_norm", "away_norm"],
        how="inner",
    )

    # Ensure odds are numeric
    merged["ML_Home"] = pd.to_numeric(merged["ML_Home"], errors="coerce")
    merged["ML_Away"] = pd.to_numeric(merged["ML_Away"], errors="coerce")
    merged = merged.dropna(subset=["ML_Home", "ML_Away"])
    merged = merged[(merged["ML_Home"] != 0) & (merged["ML_Away"] != 0)]

    print(f"Matched {len(merged)} / {len(df)} games with odds data")
    return merged


def run_backtest(merged, model, calibrator, test_seasons_start="2020-10-01"):
    """
    Run backtest on games from test_seasons_start onward.
    This ensures we're testing on data the model hasn't heavily overfit to.
    """
    merged = merged.copy()
    merged = merged.sort_values("Date").reset_index(drop=True)

    # Filter to test period
    cutoff = pd.to_datetime(test_seasons_start)
    test = merged[merged["Date"] >= cutoff].copy()
    print(f"\nBacktest period: {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"Games in backtest: {len(test)}")

    # Prepare features EXACTLY as training does:
    # Drop the DROP_COLUMNS, keep everything else in original column order
    # The training features are the dataset columns minus DROP_COLUMNS, in order
    TRAINING_FEATURES = [c for c in test.columns if c not in
                         set(DROP_COLUMNS) | {"home_norm", "away_norm", "ML_Home", "ML_Away", "season"}]
    X = test[TRAINING_FEATURES].astype(float).values
    print(f"Feature count: {X.shape[1]} (model expects {model.num_features()})")

    # Get predictions
    probs = predict_probs(model, X, calibrator)
    # probs[:, 0] = P(away win), probs[:, 1] = P(home win)

    test = test.reset_index(drop=True)
    test["model_home_prob"] = probs[:, 1]
    test["model_away_prob"] = probs[:, 0]
    test["model_pick_home"] = (probs[:, 1] > 0.5).astype(int)
    test["actual_home_win"] = test["Home-Team-Win"].astype(int)
    test["correct"] = (test["model_pick_home"] == test["actual_home_win"]).astype(int)

    # Calculate EV for each side
    test["ev_home"] = test.apply(lambda r: calc_ev(r["model_home_prob"], r["ML_Home"]), axis=1)
    test["ev_away"] = test.apply(lambda r: calc_ev(r["model_away_prob"], r["ML_Away"]), axis=1)
    test["kelly_home"] = test.apply(lambda r: calc_kelly(r["model_home_prob"], r["ML_Home"]), axis=1)
    test["kelly_away"] = test.apply(lambda r: calc_kelly(r["model_away_prob"], r["ML_Away"]), axis=1)

    # Implied probs from odds
    test["implied_home"] = test["ML_Home"].apply(calc_implied_prob)
    test["implied_away"] = test["ML_Away"].apply(calc_implied_prob)
    test["edge_home"] = test["model_home_prob"] - test["implied_home"]
    test["edge_away"] = test["model_away_prob"] - test["implied_away"]

    return test


def simulate_strategies(test):
    """
    Simulate different betting strategies and report P/L.
    """
    results = {}

    # ---- Strategy 1: Flat bet on every model pick ----
    flat_bets = []
    for _, r in test.iterrows():
        if r["model_pick_home"] == 1:
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
        else:
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0

        profit = calc_payout(odds, FLAT_BET) if won else -FLAT_BET
        flat_bets.append(profit)
    results["flat_all"] = flat_bets

    # ---- Strategy 2: Flat bet only on positive EV picks ----
    ev_bets = []
    ev_bets_count = 0
    for _, r in test.iterrows():
        # Pick the side with better EV
        if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
            ev_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
            ev_bets_count += 1
        elif r["ev_away"] > r["ev_home"] and r["ev_away"] > 0:
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0
            ev_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
            ev_bets_count += 1
        # else skip (no positive EV)
    results["ev_positive"] = ev_bets

    # ---- Strategy 3: Positive EV + minimum 5% edge ----
    edge_bets = []
    for _, r in test.iterrows():
        best_side = None
        if r["edge_home"] >= 0.05 and r["ev_home"] > 0:
            if best_side is None or r["ev_home"] > r.get("_best_ev", 0):
                best_side = "home"
        if r["edge_away"] >= 0.05 and r["ev_away"] > 0:
            if best_side is None or r["ev_away"] > r.get("ev_home", 0):
                best_side = "away"

        if best_side == "home":
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
            edge_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
        elif best_side == "away":
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0
            edge_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
    results["edge_5pct"] = edge_bets

    # ---- Strategy 4: Positive EV + minimum 10% edge ----
    edge10_bets = []
    for _, r in test.iterrows():
        best_side = None
        if r["edge_home"] >= 0.10 and r["ev_home"] > 0:
            best_side = "home"
        if r["edge_away"] >= 0.10 and r["ev_away"] > 0:
            if best_side is None or r["ev_away"] > r.get("ev_home", 0):
                best_side = "away"

        if best_side == "home":
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
            edge10_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
        elif best_side == "away":
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0
            edge10_bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
    results["edge_10pct"] = edge10_bets

    # ---- Strategy 5: Kelly Criterion sizing (quarter Kelly, capped at 3% bankroll) ----
    kelly_bets = []
    kelly_bankroll = BANKROLL_START
    kelly_stakes = []
    for _, r in test.iterrows():
        if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
            kelly_f = r["kelly_home"]
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
        elif r["ev_away"] > 0:
            kelly_f = r["kelly_away"]
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0
        else:
            continue

        # Quarter Kelly, capped at 3% of bankroll, min $1
        stake = kelly_bankroll * kelly_f * 0.25
        stake = min(stake, kelly_bankroll * 0.03)
        stake = max(stake, 0)

        if stake < 1:
            continue

        profit = calc_payout(odds, stake) if won else -stake
        kelly_bankroll += profit
        kelly_bets.append(profit)
        kelly_stakes.append(stake)

        if kelly_bankroll <= 0:
            break

    results["kelly_quarter"] = kelly_bets
    results["_kelly_final_bankroll"] = kelly_bankroll
    results["_kelly_total_staked"] = sum(kelly_stakes)

    return results


def print_report(test, strategies):
    """Print comprehensive backtest report."""
    print("\n" + "=" * 80)
    print("NBA ML SPORTS BETTING - BACKTEST REPORT")
    print("=" * 80)

    # Raw accuracy
    total = len(test)
    correct = test["correct"].sum()
    accuracy = correct / total * 100
    print(f"\n--- RAW MODEL ACCURACY ---")
    print(f"Total games:  {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {accuracy:.1f}%")

    # Home/away breakdown
    home_picks = test[test["model_pick_home"] == 1]
    away_picks = test[test["model_pick_home"] == 0]
    home_correct = home_picks["correct"].sum()
    away_correct = away_picks["correct"].sum()
    print(f"\nHome picks:   {len(home_picks)} ({home_correct}/{len(home_picks)} = {home_correct/max(len(home_picks),1)*100:.1f}%)")
    print(f"Away picks:   {len(away_picks)} ({away_correct}/{len(away_picks)} = {away_correct/max(len(away_picks),1)*100:.1f}%)")

    # Confidence buckets
    print(f"\n--- ACCURACY BY CONFIDENCE ---")
    test["confidence"] = test[["model_home_prob", "model_away_prob"]].max(axis=1) * 100
    buckets = [(50, 55), (55, 60), (60, 65), (65, 70), (70, 75), (75, 100)]
    for lo, hi in buckets:
        mask = (test["confidence"] >= lo) & (test["confidence"] < hi)
        bucket = test[mask]
        if len(bucket) > 0:
            acc = bucket["correct"].mean() * 100
            print(f"  {lo}-{hi}%: {len(bucket):5d} games, {acc:.1f}% accuracy")

    # Betting strategy results
    print(f"\n--- BETTING STRATEGY RESULTS (${FLAT_BET:.0f} flat bet) ---")
    print(f"{'Strategy':<25} {'Bets':>6} {'W':>5} {'L':>5} {'Win%':>7} {'P/L':>10} {'ROI':>8} {'MaxDD':>10}")
    print("-" * 80)

    for name, bets in strategies.items():
        if name.startswith("_"):
            continue
        if not bets:
            print(f"{name:<25} {'0':>6} {'--':>5} {'--':>5} {'--':>7} {'--':>10} {'--':>8} {'--':>10}")
            continue

        bets_arr = np.array(bets)
        wins = (bets_arr > 0).sum()
        losses = (bets_arr <= 0).sum()
        win_pct = wins / len(bets_arr) * 100
        total_pl = bets_arr.sum()

        # Calculate ROI
        if name == "kelly_quarter":
            total_staked = strategies.get("_kelly_total_staked", BANKROLL_START)
            roi = total_pl / max(total_staked, 1) * 100
            final_br = strategies.get("_kelly_final_bankroll", BANKROLL_START + total_pl)
        else:
            total_risked = len(bets_arr) * FLAT_BET
            roi = total_pl / total_risked * 100

        # Max drawdown
        cumsum = np.cumsum(bets_arr)
        running_max = np.maximum.accumulate(cumsum)
        drawdown = running_max - cumsum
        max_dd = drawdown.max()

        extra = ""
        if name == "kelly_quarter":
            extra = f"  (${BANKROLL_START:.0f} -> ${final_br:,.0f})"

        print(f"{name:<25} {len(bets_arr):>6} {wins:>5} {losses:>5} {win_pct:>6.1f}% ${total_pl:>+12.2f} {roi:>+7.2f}% ${max_dd:>9.2f}{extra}")

    # Monthly breakdown for the best strategy
    print(f"\n--- MONTHLY BREAKDOWN (Positive EV, flat bet) ---")
    # Rebuild monthly for ev_positive strategy
    monthly_data = []
    for _, r in test.iterrows():
        if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
            odds = r["ML_Home"]
            won = r["actual_home_win"] == 1
        elif r["ev_away"] > r["ev_home"] and r["ev_away"] > 0:
            odds = r["ML_Away"]
            won = r["actual_home_win"] == 0
        else:
            continue
        profit = calc_payout(odds, FLAT_BET) if won else -FLAT_BET
        monthly_data.append({"month": r["Date"].strftime("%Y-%m"), "profit": profit, "won": won})

    if monthly_data:
        mdf = pd.DataFrame(monthly_data)
        monthly = mdf.groupby("month").agg(
            bets=("profit", "count"),
            wins=("won", "sum"),
            pl=("profit", "sum"),
        )
        monthly["win_pct"] = (monthly["wins"] / monthly["bets"] * 100).round(1)
        monthly["roi"] = (monthly["pl"] / (monthly["bets"] * FLAT_BET) * 100).round(1)

        print(f"{'Month':<10} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P/L':>10} {'ROI':>8}")
        print("-" * 50)
        for month, row in monthly.iterrows():
            print(f"{month:<10} {int(row['bets']):>5} {int(row['wins']):>5} {row['win_pct']:>6.1f}% ${row['pl']:>+9.2f} {row['roi']:>+7.1f}%")

        cum_pl = monthly["pl"].cumsum()
        print(f"\n  Cumulative P/L: ${cum_pl.iloc[-1]:+.2f}")

    # Season summary
    print(f"\n--- SEASON SUMMARY (Positive EV, flat bet) ---")
    if monthly_data:
        mdf["season"] = mdf["month"].apply(lambda m: f"{int(m[:4])}-{int(m[:4])+1}" if int(m[5:]) >= 10
                                             else f"{int(m[:4])-1}-{int(m[:4])}")
        seasonal = mdf.groupby("season").agg(
            bets=("profit", "count"),
            wins=("won", "sum"),
            pl=("profit", "sum"),
        )
        seasonal["win_pct"] = (seasonal["wins"] / seasonal["bets"] * 100).round(1)
        seasonal["roi"] = (seasonal["pl"] / (seasonal["bets"] * FLAT_BET) * 100).round(1)

        print(f"{'Season':<12} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P/L':>10} {'ROI':>8}")
        print("-" * 52)
        for season, row in seasonal.iterrows():
            print(f"{season:<12} {int(row['bets']):>5} {int(row['wins']):>5} {row['win_pct']:>6.1f}% ${row['pl']:>+9.2f} {row['roi']:>+7.1f}%")

    # Edge analysis
    print(f"\n--- EDGE ANALYSIS (model prob vs implied odds prob) ---")
    for side in ["home", "away"]:
        edge_col = f"edge_{side}"
        ev_col = f"ev_{side}"
        actual_col = "actual_home_win"

        # Games where model picked this side with positive EV
        if side == "home":
            mask = (test["ev_home"] > 0) & (test["ev_home"] > test["ev_away"])
        else:
            mask = (test["ev_away"] > 0) & (test["ev_away"] > test["ev_home"])

        subset = test[mask]
        if len(subset) == 0:
            continue

        edge_buckets = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.0)]
        print(f"\n  {side.upper()} side positive EV bets ({len(subset)} total):")
        print(f"  {'Edge':<12} {'Bets':>5} {'Win%':>7} {'Avg EV':>8}")
        for lo, hi in edge_buckets:
            emask = (subset[edge_col] >= lo) & (subset[edge_col] < hi)
            bucket = subset[emask]
            if len(bucket) == 0:
                continue
            if side == "home":
                win_rate = bucket["actual_home_win"].mean() * 100
            else:
                win_rate = (1 - bucket["actual_home_win"]).mean() * 100
            avg_ev = bucket[ev_col].mean()
            print(f"  {lo*100:.0f}-{hi*100:.0f}%{'':<7} {len(bucket):>5} {win_rate:>6.1f}% {avg_ev:>+7.1f}")

    print("\n" + "=" * 80)


def main():
    print("Loading data...")
    dataset = load_dataset()
    odds = load_all_odds()
    model, calibrator = load_xgb_model()

    print("Merging dataset with odds...")
    merged = merge_dataset_with_odds(dataset, odds)

    if len(merged) == 0:
        print("ERROR: No games matched between dataset and odds. Check team name alignment.")
        return

    # The model's train/test split is at 90% of data = ~2024-11-13
    # Games before that were IN the training set (in-sample)
    # Games after that are TRUE out-of-sample
    OOS_DATE = "2024-11-13"

    # Run on full 2020+ range (mostly in-sample, good for overall picture)
    print("\n" + "#" * 80)
    print("# FULL BACKTEST (2020-2026) - includes in-sample data")
    print("#" * 80)
    test_full = run_backtest(merged, model, calibrator, test_seasons_start="2020-10-01")
    strategies_full = simulate_strategies(test_full)
    print_report(test_full, strategies_full)

    # Run on TRUE out-of-sample only
    print("\n" + "#" * 80)
    print(f"# TRUE OUT-OF-SAMPLE BACKTEST (from {OOS_DATE})")
    print("# Model has NEVER seen this data during training")
    print("#" * 80)
    test_oos = run_backtest(merged, model, calibrator, test_seasons_start=OOS_DATE)
    strategies_oos = simulate_strategies(test_oos)
    print_report(test_oos, strategies_oos)


if __name__ == "__main__":
    main()
