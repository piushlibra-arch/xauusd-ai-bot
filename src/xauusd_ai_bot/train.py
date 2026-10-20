from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data/processed/xauusd_features.csv"
MODEL_DIR = ROOT / "models"

FEATURES = [
    "return_1",
    "return_4",
    "return_16",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "atr_14_ratio",
    "volatility_20",
    "candle_range_ratio",
    "candle_body_ratio",
    "tick_volume_zscore",
]

TARGET = "target_up"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    required = FEATURES + [TARGET]
    if df[required].isna().any().any():
        raise ValueError("Missing values found in model data.")

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Strict chronological split: 60% train, 20% validation, 20% test.
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    print("\n--- Chronological Model Evaluation ---")
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows:       {len(X_test):,}")
    print(f"Train period:    {df['time'].iloc[0]} to "
          f"{df['time'].iloc[train_end - 1]}")
    print(f"Validation:      {df['time'].iloc[train_end]} to "
          f"{df['time'].iloc[val_end - 1]}")
    print(f"Test period:     {df['time'].iloc[val_end]} to "
          f"{df['time'].iloc[-1]}")

    fitted = {}
    validation_scores = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        auc = roc_auc_score(y_val, probabilities)
        validation_scores[name] = auc
        fitted[name] = model

        print(f"\n{name}")
        print(f"Validation accuracy: "
              f"{accuracy_score(y_val, predictions):.4f}")
        print(f"Validation balanced accuracy: "
              f"{balanced_accuracy_score(y_val, predictions):.4f}")
        print(f"Validation ROC AUC: {auc:.4f}")

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = candidates[best_name]

    # Refit on training + validation data after model selection.
    X_train_val = X.iloc[:val_end]
    y_train_val = y.iloc[:val_end]
    best_model.fit(X_train_val, y_train_val)

    # Final evaluation on the untouched newest 20%.
    test_probabilities = best_model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)

    print("\n--- Final Untouched Test Results ---")
    print(f"Selected model: {best_name}")
    print(f"Test accuracy: "
          f"{accuracy_score(y_test, test_predictions):.4f}")
    print(f"Test balanced accuracy: "
          f"{balanced_accuracy_score(y_test, test_predictions):.4f}")
    print(f"Test ROC AUC: "
          f"{roc_auc_score(y_test, test_probabilities):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Down or flat", "Up"],
        zero_division=0,
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "xauusd_direction_model.joblib"

    joblib.dump(
        {
            "model": best_model,
            "features": FEATURES,
            "target": TARGET,
            "horizon_candles": 4,
            "selected_model": best_name,
        },
        model_path,
    )

    print(f"Saved research model: {model_path}")
    print("\nThis is a research model, not a live trading signal.")


if __name__ == "__main__":
    main()
