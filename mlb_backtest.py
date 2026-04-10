#!/usr/bin/env python3
"""
Walk-forward backtest for MLB XGBoost models (moneyline + over/under).

Tests on historical games WITH odds data to calculate:
- Raw prediction accuracy
- Betting P/L under different strategies (flat, +EV, edge, Kelly)
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
MODEL_DIR = BASE_DIR / "Models" / "MLB_XGBoost_Models"

# Columns dropped during ML training
ML_DROP_COLUMNS = [
    "Date", "Home", "Away", "Home_Starter", "Away_Starter", "season",
    "Home_Team_Win", "OU_Cover", "Total_Runs", "OU_line",
    "ML_Home", "ML_Away",
]

# Columns dropped during UO training (keeps OU_line as feature)
UO_DROP_COLUMNS = [
    "Date", "Home", "Away", "Home_Starter", "Away_Starter", "season",
    "Home_Team_Win", "OU_Cover", "Total_Runs",
    "ML_Home", "ML_Away",
]

BANKROLL_START = 1000.0
FLAT_BET = 10.0


class BoosterWrapper:
    _estimator_type = "classifier"

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        preds = self.booster.predict(xgb.DMatrix(X))
        # binary:logistic returns 1D, multi:softprob returns 2D — normalize to 2D
        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds


def load_dataset():
    conn = sqlite3.connect(DATA_DIR / "MLB_dataset.sqlite")
    df = pd.read_sql('SELECT * FROM "mlb_dataset"', conn)
    conn.close()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_model(pattern_key):
    import re
    pat = re.compile(rf"MLB_XGBoost_(\d+(?:\.\d+)?)%_{pattern_key}")
    candidates = list(MODEL_DIR.glob(f"*{pattern_key}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No MLB XGBoost {pattern_key} model found in {MODEL_DIR}")

    def score(path):
        m = pat.search(path.name)
        return float(m.group(1)) if m else 0.0

    best_path = max(candidates, key=score)
    model = xgb.Booster()
    model.load_model(str(best_path))

    cal_path = best_path.with_name(best_path.stem + "_calibration.pkl")
    calibrator = None
    if cal_path.exists():
        try:
            calibrator = joblib.load(cal_path)
            if hasattr(calibrator, "estimators_"):
                for cal_est in calibrator.estimators_:
                    cal_est.estimator = BoosterWrapper(model, 2)
        except Exception:
            calibrator = None

    cal_status = "with calibrator" if calibrator else "raw probabilities"
    print(f"Loaded {pattern_key} model: {best_path.name} ({cal_status})")
    return model, calibrator, score(best_path)


def predict_probs(model, X, calibrator=None):
    if calibrator is not None:
        return calibrator.predict_proba(X)
    return model.predict(xgb.DMatrix(X))


def american_to_decimal(odds):
    if odds >= 100:
        return odds / 100.0
    else:
        return 100.0 / abs(odds)


def calc_implied_prob(odds):
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def calc_payout(odds, stake):
    return stake * american_to_decimal(odds)


def calc_ev(model_prob, odds):
    payout = american_to_decimal(odds) * 100
    return model_prob * payout - (1 - model_prob) * 100


def calc_kelly(model_prob, odds):
    dec = american_to_decimal(odds)
    f = (dec * model_prob - (1 - model_prob)) / dec
    return max(0.0, f)


# ---------- MONEYLINE BACKTEST ----------

def run_ml_backtest(df, model, calibrator, cutoff_date):
    test = df[df["Date"] >= pd.to_datetime(cutoff_date)].copy()
    test = test.dropna(subset=["ML_Home", "ML_Away"])
    test = test[(test["ML_Home"] != 0) & (test["ML_Away"] != 0)]

    print(f"\nML Backtest: {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"Games: {len(test)}")

    feature_cols = [c for c in test.columns if c not in set(ML_DROP_COLUMNS)]
    X = test[feature_cols].astype(float).values
    print(f"Features: {X.shape[1]} (model expects {model.num_features()})")

    probs = predict_probs(model, X, calibrator)

    test = test.reset_index(drop=True)
    test["model_home_prob"] = probs[:, 1]
    test["model_away_prob"] = probs[:, 0]
    test["model_pick_home"] = (probs[:, 1] > 0.5).astype(int)
    test["actual_home_win"] = test["Home_Team_Win"].astype(int)
    test["correct"] = (test["model_pick_home"] == test["actual_home_win"]).astype(int)

    test["ev_home"] = test.apply(lambda r: calc_ev(r["model_home_prob"], r["ML_Home"]), axis=1)
    test["ev_away"] = test.apply(lambda r: calc_ev(r["model_away_prob"], r["ML_Away"]), axis=1)
    test["kelly_home"] = test.apply(lambda r: calc_kelly(r["model_home_prob"], r["ML_Home"]), axis=1)
    test["kelly_away"] = test.apply(lambda r: calc_kelly(r["model_away_prob"], r["ML_Away"]), axis=1)
    test["implied_home"] = test["ML_Home"].apply(calc_implied_prob)
    test["implied_away"] = test["ML_Away"].apply(calc_implied_prob)
    test["edge_home"] = test["model_home_prob"] - test["implied_home"]
    test["edge_away"] = test["model_away_prob"] - test["implied_away"]
    test["confidence"] = test[["model_home_prob", "model_away_prob"]].max(axis=1) * 100

    return test


# ---------- OVER/UNDER BACKTEST ----------

def run_uo_backtest(df, model, calibrator, cutoff_date):
    test = df[df["Date"] >= pd.to_datetime(cutoff_date)].copy()
    test = test.dropna(subset=["OU_line", "OU_Cover"])
    test = test[test["OU_line"] > 0]

    print(f"\nO/U Backtest: {test['Date'].min().date()} to {test['Date'].max().date()}")
    print(f"Games: {len(test)}")

    feature_cols = [c for c in test.columns if c not in set(UO_DROP_COLUMNS)]
    X = test[feature_cols].astype(float).values
    print(f"Features: {X.shape[1]} (model expects {model.num_features()})")

    probs = predict_probs(model, X, calibrator)

    test = test.reset_index(drop=True)
    test["model_over_prob"] = probs[:, 1]
    test["model_under_prob"] = probs[:, 0]
    test["model_pick_over"] = (probs[:, 1] > 0.5).astype(int)
    test["actual_over"] = test["OU_Cover"].astype(int)
    test["correct"] = (test["model_pick_over"] == test["actual_over"]).astype(int)
    test["confidence"] = test[["model_over_prob", "model_under_prob"]].max(axis=1) * 100

    return test


# ---------- STRATEGY SIMULATION ----------

def simulate_ml_strategies(test):
    results = {}

    bets = []
    for _, r in test.iterrows():
        if r["model_pick_home"] == 1:
            odds, won = r["ML_Home"], r["actual_home_win"] == 1
        else:
            odds, won = r["ML_Away"], r["actual_home_win"] == 0
        bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
    results["flat_all"] = bets

    bets = []
    for _, r in test.iterrows():
        if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
            odds, won = r["ML_Home"], r["actual_home_win"] == 1
        elif r["ev_away"] > r["ev_home"] and r["ev_away"] > 0:
            odds, won = r["ML_Away"], r["actual_home_win"] == 0
        else:
            continue
        bets.append(calc_payout(odds, FLAT_BET) if won else -FLAT_BET)
    results["ev_positive"] = bets

    bets = []
    for _, r in test.iterrows():
        side = None
        if r["edge_home"] >= 0.05 and r["ev_home"] > 0:
            side = "home"
        if r["edge_away"] >= 0.05 and r["ev_away"] > 0:
            if side is None or r["ev_away"] > r["ev_home"]:
                side = "away"
        if side == "home":
            bets.append(calc_payout(r["ML_Home"], FLAT_BET) if r["actual_home_win"] == 1 else -FLAT_BET)
        elif side == "away":
            bets.append(calc_payout(r["ML_Away"], FLAT_BET) if r["actual_home_win"] == 0 else -FLAT_BET)
    results["edge_5pct"] = bets

    bets = []
    for _, r in test.iterrows():
        side = None
        if r["edge_home"] >= 0.10 and r["ev_home"] > 0:
            side = "home"
        if r["edge_away"] >= 0.10 and r["ev_away"] > 0:
            if side is None or r["ev_away"] > r["ev_home"]:
                side = "away"
        if side == "home":
            bets.append(calc_payout(r["ML_Home"], FLAT_BET) if r["actual_home_win"] == 1 else -FLAT_BET)
        elif side == "away":
            bets.append(calc_payout(r["ML_Away"], FLAT_BET) if r["actual_home_win"] == 0 else -FLAT_BET)
    results["edge_10pct"] = bets

    bets, bankroll, stakes = [], BANKROLL_START, []
    for _, r in test.iterrows():
        if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
            kelly_f, odds, won = r["kelly_home"], r["ML_Home"], r["actual_home_win"] == 1
        elif r["ev_away"] > 0:
            kelly_f, odds, won = r["kelly_away"], r["ML_Away"], r["actual_home_win"] == 0
        else:
            continue
        stake = min(bankroll * kelly_f * 0.25, bankroll * 0.03, 500.0)
        if stake < 1:
            continue
        profit = calc_payout(odds, stake) if won else -stake
        bankroll += profit
        bets.append(profit)
        stakes.append(stake)
        if bankroll <= 0:
            break
    results["kelly_quarter"] = bets
    results["_kelly_final_bankroll"] = bankroll
    results["_kelly_total_staked"] = sum(stakes)

    return results


def simulate_uo_strategies(test):
    results = {}
    STANDARD_ODDS = -110

    bets = []
    for _, r in test.iterrows():
        won = r["model_pick_over"] == r["actual_over"]
        bets.append(calc_payout(STANDARD_ODDS, FLAT_BET) if won else -FLAT_BET)
    results["flat_all"] = bets

    for thresh, name in [(53, "conf_53pct"), (55, "conf_55pct"), (58, "conf_58pct"), (60, "conf_60pct")]:
        bets = []
        for _, r in test.iterrows():
            if r["confidence"] < thresh:
                continue
            won = r["model_pick_over"] == r["actual_over"]
            bets.append(calc_payout(STANDARD_ODDS, FLAT_BET) if won else -FLAT_BET)
        results[name] = bets

    return results


# ---------- REPORTING ----------

def print_strategy_table(strategies, label):
    print(f"\n--- {label} BETTING STRATEGIES (${FLAT_BET:.0f} flat bet) ---")
    print(f"{'Strategy':<25} {'Bets':>6} {'W':>5} {'L':>5} {'Win%':>7} {'P/L':>12} {'ROI':>8} {'MaxDD':>10}")
    print("-" * 82)

    for name, bets in strategies.items():
        if name.startswith("_"):
            continue
        if not bets:
            print(f"{name:<25} {'0':>6}")
            continue

        arr = np.array(bets)
        wins = (arr > 0).sum()
        losses = (arr <= 0).sum()
        win_pct = wins / len(arr) * 100
        total_pl = arr.sum()

        if name == "kelly_quarter":
            total_staked = strategies.get("_kelly_total_staked", BANKROLL_START)
            roi = total_pl / max(total_staked, 1) * 100
            final_br = strategies.get("_kelly_final_bankroll", BANKROLL_START + total_pl)
        else:
            roi = total_pl / (len(arr) * FLAT_BET) * 100

        cumsum = np.cumsum(arr)
        max_dd = (np.maximum.accumulate(cumsum) - cumsum).max()

        extra = ""
        if name == "kelly_quarter":
            extra = f"  (${BANKROLL_START:.0f} -> ${final_br:,.0f})"

        print(f"{name:<25} {len(arr):>6} {wins:>5} {losses:>5} {win_pct:>6.1f}% ${total_pl:>+11.2f} {roi:>+7.2f}% ${max_dd:>9.2f}{extra}")


def print_accuracy_report(test, model_type="ML"):
    total = len(test)
    correct = test["correct"].sum()
    accuracy = correct / total * 100

    print(f"\n--- RAW MODEL ACCURACY ({model_type}) ---")
    print(f"Total games:  {total}")
    print(f"Correct:      {correct}")
    print(f"Accuracy:     {accuracy:.1f}%")

    if model_type == "ML":
        home_picks = test[test["model_pick_home"] == 1]
        away_picks = test[test["model_pick_home"] == 0]
        print(f"Home picks:   {len(home_picks)} ({home_picks['correct'].sum()}/{len(home_picks)} = {home_picks['correct'].mean()*100:.1f}%)")
        print(f"Away picks:   {len(away_picks)} ({away_picks['correct'].sum()}/{len(away_picks)} = {away_picks['correct'].mean()*100:.1f}%)")
    else:
        over_picks = test[test["model_pick_over"] == 1]
        under_picks = test[test["model_pick_over"] == 0]
        print(f"Over picks:   {len(over_picks)} ({over_picks['correct'].sum()}/{len(over_picks)} = {over_picks['correct'].mean()*100:.1f}%)")
        print(f"Under picks:  {len(under_picks)} ({under_picks['correct'].sum()}/{len(under_picks)} = {under_picks['correct'].mean()*100:.1f}%)")

    print(f"\n--- ACCURACY BY CONFIDENCE ({model_type}) ---")
    buckets = [(50, 53), (53, 55), (55, 58), (58, 60), (60, 65), (65, 70), (70, 75), (75, 100)]
    for lo, hi in buckets:
        mask = (test["confidence"] >= lo) & (test["confidence"] < hi)
        bucket = test[mask]
        if len(bucket) > 0:
            acc = bucket["correct"].mean() * 100
            print(f"  {lo:>2}-{hi:<3}%: {len(bucket):>5} games, {acc:.1f}% accuracy")


def print_monthly_breakdown(test, model_type="ML"):
    print(f"\n--- MONTHLY BREAKDOWN ({model_type}) ---")

    monthly_data = []
    for _, r in test.iterrows():
        if model_type == "ML":
            if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
                odds, won = r["ML_Home"], r["actual_home_win"] == 1
            elif r["ev_away"] > r["ev_home"] and r["ev_away"] > 0:
                odds, won = r["ML_Away"], r["actual_home_win"] == 0
            else:
                continue
            profit = calc_payout(odds, FLAT_BET) if won else -FLAT_BET
        else:
            if r["confidence"] < 55:
                continue
            won = r["model_pick_over"] == r["actual_over"]
            profit = calc_payout(-110, FLAT_BET) if won else -FLAT_BET

        monthly_data.append({"month": r["Date"].strftime("%Y-%m"), "profit": profit, "won": won})

    if not monthly_data:
        print("  No qualifying bets")
        return

    mdf = pd.DataFrame(monthly_data)
    monthly = mdf.groupby("month").agg(bets=("profit", "count"), wins=("won", "sum"), pl=("profit", "sum"))
    monthly["win_pct"] = (monthly["wins"] / monthly["bets"] * 100).round(1)
    monthly["roi"] = (monthly["pl"] / (monthly["bets"] * FLAT_BET) * 100).round(1)

    print(f"{'Month':<10} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P/L':>10} {'ROI':>8}")
    print("-" * 50)
    for month, row in monthly.iterrows():
        print(f"{month:<10} {int(row['bets']):>5} {int(row['wins']):>5} {row['win_pct']:>6.1f}% ${row['pl']:>+9.2f} {row['roi']:>+7.1f}%")

    cum_pl = monthly["pl"].cumsum()
    print(f"\n  Cumulative P/L: ${cum_pl.iloc[-1]:+.2f}")


def print_season_breakdown(test, model_type="ML"):
    print(f"\n--- SEASON SUMMARY ({model_type}) ---")

    season_data = []
    for _, r in test.iterrows():
        if model_type == "ML":
            if r["ev_home"] > r["ev_away"] and r["ev_home"] > 0:
                odds, won = r["ML_Home"], r["actual_home_win"] == 1
            elif r["ev_away"] > r["ev_home"] and r["ev_away"] > 0:
                odds, won = r["ML_Away"], r["actual_home_win"] == 0
            else:
                continue
            profit = calc_payout(odds, FLAT_BET) if won else -FLAT_BET
        else:
            if r["confidence"] < 55:
                continue
            won = r["model_pick_over"] == r["actual_over"]
            profit = calc_payout(-110, FLAT_BET) if won else -FLAT_BET

        season_data.append({"season": str(r["Date"].year), "profit": profit, "won": won})

    if not season_data:
        return

    sdf = pd.DataFrame(season_data)
    seasonal = sdf.groupby("season").agg(bets=("profit", "count"), wins=("won", "sum"), pl=("profit", "sum"))
    seasonal["win_pct"] = (seasonal["wins"] / seasonal["bets"] * 100).round(1)
    seasonal["roi"] = (seasonal["pl"] / (seasonal["bets"] * FLAT_BET) * 100).round(1)

    print(f"{'Season':<10} {'Bets':>5} {'Wins':>5} {'Win%':>7} {'P/L':>10} {'ROI':>8}")
    print("-" * 50)
    for season, row in seasonal.iterrows():
        print(f"{season:<10} {int(row['bets']):>5} {int(row['wins']):>5} {row['win_pct']:>6.1f}% ${row['pl']:>+9.2f} {row['roi']:>+7.1f}%")


def main():
    print("Loading MLB dataset...")
    df = load_dataset()
    print(f"Total games: {len(df)}")

    has_odds = df[(df["ML_Home"] != 0) & (df["ML_Away"] != 0)].copy()
    print(f"Games with ML odds: {len(has_odds)}")
    has_ou = df[df["OU_line"] > 0].copy()
    print(f"Games with O/U line: {len(has_ou)}")

    n = len(df)
    oos_idx = int(n * 0.9)
    oos_date = df.iloc[oos_idx]["Date"].strftime("%Y-%m-%d")
    print(f"\nOut-of-sample cutoff (90/10 split): {oos_date}")

    # ===== MONEYLINE =====
    try:
        ml_model, ml_cal, ml_acc = load_model("ML")
    except FileNotFoundError as e:
        print(f"Skipping ML: {e}")
        ml_model = None

    if ml_model:
        print("\n" + "#" * 80)
        print("# MLB MONEYLINE - FULL BACKTEST (2021+)")
        print("#" * 80)
        ml_test_full = run_ml_backtest(has_odds, ml_model, ml_cal, "2021-01-01")
        ml_strats_full = simulate_ml_strategies(ml_test_full)
        print_accuracy_report(ml_test_full, "ML")
        print_strategy_table(ml_strats_full, "ML FULL")
        print_monthly_breakdown(ml_test_full, "ML")
        print_season_breakdown(ml_test_full, "ML")

        print("\n" + "#" * 80)
        print(f"# MLB MONEYLINE - TRUE OUT-OF-SAMPLE (from {oos_date})")
        print("#" * 80)
        ml_test_oos = run_ml_backtest(has_odds, ml_model, ml_cal, oos_date)
        ml_strats_oos = simulate_ml_strategies(ml_test_oos)
        print_accuracy_report(ml_test_oos, "ML")
        print_strategy_table(ml_strats_oos, "ML OOS")
        print_monthly_breakdown(ml_test_oos, "ML")

    # ===== OVER/UNDER =====
    try:
        uo_model, uo_cal, uo_acc = load_model("UO")
    except FileNotFoundError as e:
        print(f"Skipping UO: {e}")
        uo_model = None

    if uo_model:
        print("\n" + "#" * 80)
        print("# MLB OVER/UNDER - FULL BACKTEST (2021+)")
        print("#" * 80)
        uo_test_full = run_uo_backtest(has_ou, uo_model, uo_cal, "2021-01-01")
        uo_strats_full = simulate_uo_strategies(uo_test_full)
        print_accuracy_report(uo_test_full, "O/U")
        print_strategy_table(uo_strats_full, "O/U FULL")
        print_monthly_breakdown(uo_test_full, "O/U")
        print_season_breakdown(uo_test_full, "O/U")

        print("\n" + "#" * 80)
        print(f"# MLB OVER/UNDER - TRUE OUT-OF-SAMPLE (from {oos_date})")
        print("#" * 80)
        uo_test_oos = run_uo_backtest(has_ou, uo_model, uo_cal, oos_date)
        uo_strats_oos = simulate_uo_strategies(uo_test_oos)
        print_accuracy_report(uo_test_oos, "O/U")
        print_strategy_table(uo_strats_oos, "O/U OOS")
        print_monthly_breakdown(uo_test_oos, "O/U")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
