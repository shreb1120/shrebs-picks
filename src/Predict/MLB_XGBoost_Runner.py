"""MLB XGBoost prediction runner — loads models and runs inference."""

import re
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "MLB_XGBoost_Models"
ACCURACY_PATTERN = re.compile(r"MLB_XGBoost_(\d+(?:\.\d+)?)%_")

mlb_xgb_ml = None
mlb_xgb_uo = None
mlb_xgb_ml_calibrator = None
mlb_xgb_uo_calibrator = None


def _select_model_path(kind):
    candidates = [p for p in MODEL_DIR.glob("*.json") if f"_{kind}_" in p.name]
    if not candidates:
        raise FileNotFoundError(f"No MLB XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


class BoosterWrapper:
    """Wrapper to make XGBoost Booster compatible with sklearn calibration."""
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
    import sys
    # Inject BoosterWrapper into __main__ so joblib can unpickle it
    import __main__
    if not hasattr(__main__, 'BoosterWrapper'):
        __main__.BoosterWrapper = BoosterWrapper
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    try:
        return joblib.load(calibration_path)
    except Exception:
        return None


def _load_models():
    global mlb_xgb_ml, mlb_xgb_uo, mlb_xgb_ml_calibrator, mlb_xgb_uo_calibrator
    if mlb_xgb_ml is None:
        ml_path = _select_model_path("ML")
        mlb_xgb_ml = xgb.Booster()
        mlb_xgb_ml.load_model(str(ml_path))
        mlb_xgb_ml_calibrator = _load_calibrator(ml_path)
    if mlb_xgb_uo is None:
        uo_path = _select_model_path("UO")
        mlb_xgb_uo = xgb.Booster()
        mlb_xgb_uo.load_model(str(uo_path))
        mlb_xgb_uo_calibrator = _load_calibrator(uo_path)


def _predict_probs(model, data, calibrator=None):
    if calibrator is not None:
        return calibrator.predict_proba(data)
    raw = model.predict(xgb.DMatrix(data))
    # Ensure 2D output: (n_samples, n_classes)
    if raw.ndim == 1:
        return np.column_stack([1 - raw, raw])
    return raw


def mlb_xgb_runner(data_ml, data_uo, games, home_team_odds, away_team_odds,
                    kelly_criterion=False, starters=None):
    """Run MLB predictions and print results.

    Args:
        data_ml: Feature array for ML model (no OU_line column).
        data_uo: Feature array for UO model (includes OU_line column).
        games: List of (home_team, away_team) tuples.
        home_team_odds: List of home moneyline odds.
        away_team_odds: List of away moneyline odds.
        kelly_criterion: Whether to print Kelly fractions.
        starters: Optional list of (home_sp, away_sp) tuples for display.
    """
    _load_models()

    try:
        ml_preds = _predict_probs(mlb_xgb_ml, data_ml, mlb_xgb_ml_calibrator)
        uo_preds = _predict_probs(mlb_xgb_uo, data_uo, mlb_xgb_uo_calibrator)

        results = []
        for idx, game in enumerate(games):
            home_team, away_team = game
            winner = int(np.argmax(ml_preds[idx]))
            winner_conf = round(ml_preds[idx][winner] * 100, 1)

            ou_pred = uo_preds[idx]
            if np.ndim(ou_pred) > 0:
                p_over = float(ou_pred[1])
            else:
                p_over = float(ou_pred)
            p_under = 1.0 - p_over
            under_over = 1 if p_over > 0.5 else 0
            ou_conf = round(max(p_over, p_under) * 100, 1)

            winner_team = home_team if winner == 1 else away_team
            loser_team = away_team if winner == 1 else home_team
            winner_color = Fore.GREEN if winner == 1 else Fore.RED
            loser_color = Fore.RED if winner == 1 else Fore.GREEN
            ou_label = "UNDER" if under_over == 0 else "OVER"
            ou_color = Fore.MAGENTA if under_over == 0 else Fore.BLUE

            starter_line = ""
            if starters and idx < len(starters):
                home_sp, away_sp = starters[idx]
                if home_sp or away_sp:
                    starter_line = f" [{away_sp or '?'} vs {home_sp or '?'}]"

            print(
                f"{winner_color}{winner_team}{Style.RESET_ALL}"
                f"{Fore.CYAN} ({winner_conf}%){Style.RESET_ALL}"
                f" vs {loser_color}{loser_team}{Style.RESET_ALL}: "
                f"{ou_color}{ou_label}{Style.RESET_ALL}"
                f"{Fore.CYAN} ({ou_conf}%){Style.RESET_ALL}"
                f"{starter_line}"
            )

            results.append({
                "home_team": home_team,
                "away_team": away_team,
                "ml_pick": winner_team,
                "ml_pick_side": "home" if winner == 1 else "away",
                "ml_confidence": winner_conf,
                "ml_home_prob": round(float(ml_preds[idx][1]) * 100, 1),
                "ml_away_prob": round(float(ml_preds[idx][0]) * 100, 1),
                "ou_pick": ou_label,
                "ou_confidence": ou_conf,
                "home_starter": starters[idx][0] if starters and idx < len(starters) else "",
                "away_starter": starters[idx][1] if starters and idx < len(starters) else "",
            })

        if kelly_criterion:
            print("--------- Expected Value & Kelly Criterion ----------")
            for idx, game in enumerate(games):
                home_team, away_team = game
                ev_home = ev_away = 0
                if home_team_odds[idx] and away_team_odds[idx]:
                    ev_home = float(Expected_Value.expected_value(
                        ml_preds[idx][1], int(home_team_odds[idx]),
                    ))
                    ev_away = float(Expected_Value.expected_value(
                        ml_preds[idx][0], int(away_team_odds[idx]),
                    ))
                hc = Fore.GREEN if ev_home > 0 else Fore.RED
                ac = Fore.GREEN if ev_away > 0 else Fore.RED
                kc_home = kc.calculate_kelly_criterion(home_team_odds[idx], ml_preds[idx][1]) if home_team_odds[idx] else 0
                kc_away = kc.calculate_kelly_criterion(away_team_odds[idx], ml_preds[idx][0]) if away_team_odds[idx] else 0
                print(f"{home_team} EV: {hc}{ev_home}{Style.RESET_ALL} Kelly: {kc_home}%")
                print(f"{away_team} EV: {ac}{ev_away}{Style.RESET_ALL} Kelly: {kc_away}%")

        return results
    finally:
        deinit()
