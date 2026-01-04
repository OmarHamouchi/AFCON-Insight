import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_PROCESSED / "pre_match_train_2019_2021.csv"
TEST_PATH  = DATA_PROCESSED / "pre_match_test_2023.csv"

PRED_OUT = DATA_PROCESSED / "predictions_prematch_2023.csv"


# =========================================================
# Helpers
# =========================================================
ID_COLS = [
    "season", "timestamp", "match_datetime",
    "home_team_name", "away_team_name",
    "stadium_name", "referee", "attendance"
]
TARGET_COL = "target_outcome"


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    On garde uniquement les features numériques.
    On exclut IDs et target.
    """
    drop_cols = set([c for c in ID_COLS if c in df.columns] + [TARGET_COL])
    numeric_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols


def evaluate_model(name: str, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 : {f1m:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))
    return acc, f1m


# =========================================================
# Main
# =========================================================
def main():
    # 1) Load
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing: {TEST_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # 2) Basic checks
    if TARGET_COL not in train_df.columns:
        raise KeyError(f"Missing target column: {TARGET_COL}")
    if TARGET_COL not in test_df.columns:
        raise KeyError(f"Missing target column in test: {TARGET_COL}")

    # 3) Feature selection
    feature_cols = pick_feature_columns(train_df)

    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found. Check your CSV columns/dtypes.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].astype(str).copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].astype(str).copy()

    print("✅ Data loaded")
    print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    # 4) Preprocess for numeric features
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop"
    )

    # =====================================================
    # 5) Model 1: Logistic Regression (baseline multiclass)
    # =====================================================
    lr = LogisticRegression(
        max_iter=3000,
        multi_class="auto",
        class_weight="balanced",
        n_jobs=-1
    )

    lr_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", lr)
    ])

    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    lr_acc, lr_f1m = evaluate_model("LogisticRegression (baseline)", y_test, lr_pred)

    # =====================================================
    # 6) Model 2: RandomForest (final)
    # =====================================================
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    rf_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", rf)
    ])

    # Petite grille (simple et rapide)
    param_grid = {
        "model__max_depth": [None, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        rf_pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    print("\n✅ Best RF params:", grid.best_params_)

    rf_pred = best_rf.predict(X_test)
    rf_acc, rf_f1m = evaluate_model("RandomForest (tuned)", y_test, rf_pred)

    # =====================================================
    # 7) Choose best model (by macro F1)
    # =====================================================
    best_model = best_rf if rf_f1m >= lr_f1m else lr_pipe
    best_name = "RandomForest" if rf_f1m >= lr_f1m else "LogisticRegression"
    print(f"\n🏆 Selected model: {best_name}")

    model_path = MODELS_DIR / f"prematch_{best_name.lower()}_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"✅ Model saved: {model_path}")

    # =====================================================
    # 8) Export predictions for 2023
    # =====================================================
    probs = None
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_test)
        classes = best_model.named_steps["model"].classes_
    else:
        # fallback: no proba (rare)
        classes = np.unique(y_train)

    out = test_df[[c for c in ID_COLS if c in test_df.columns]].copy()
    out["true_outcome"] = y_test.values
    out["pred_outcome"] = best_model.predict(X_test)

    if probs is not None:
        for i, cls in enumerate(classes):
            out[f"proba_{cls}"] = probs[:, i]

        # simple risk score (1 - max probability)
        out["risk_score"] = 1.0 - out[[f"proba_{c}" for c in classes]].max(axis=1)

    out.to_csv(PRED_OUT, index=False)
    print(f"✅ Predictions exported: {PRED_OUT}")
    print("\nDone.")

if __name__ == "__main__":
    main()
