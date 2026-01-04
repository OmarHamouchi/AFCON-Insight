import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = DATA_PROCESSED / "pre_match_train_2019_2021.csv"
TEST_PATH  = DATA_PROCESSED / "pre_match_test_2023.csv"
MODEL_PATH = MODELS_DIR / "prematch_logisticregression_model.joblib"

GLOBAL_OUT = DATA_PROCESSED / "shap_global_prematch.csv"
LOCAL_OUT  = DATA_PROCESSED / "shap_local_examples_prematch.csv"
CHATBOT_OUT = DATA_PROCESSED / "shap_summary_per_match_prematch.csv"

TARGET_COL = "target_outcome"
ID_COLS = [
    "season", "timestamp", "match_datetime",
    "home_team_name", "away_team_name",
    "stadium_name", "referee", "attendance"
]

def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = set([c for c in ID_COLS if c in df.columns] + [TARGET_COL])
    numeric_cols = [
        c for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric_cols

def choose_example_indices(proba_df: pd.DataFrame, n_each: int = 2, seed: int = 42) -> list[int]:
    df = proba_df.copy().reset_index(drop=True)
    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    df["risk_score"] = 1.0 - df[proba_cols].max(axis=1)

    confident_idx = df.sort_values("risk_score", ascending=True).head(n_each).index.to_list()
    uncertain_idx = df.sort_values("risk_score", ascending=False).head(n_each).index.to_list()

    rng = np.random.default_rng(seed)
    random_idx = rng.choice(df.index.to_numpy(), size=min(n_each, len(df)), replace=False).tolist()

    idx = list(dict.fromkeys(confident_idx + uncertain_idx + random_idx))
    return idx

def ensure_ns_nf(mat: np.ndarray, n_samples: int, n_features: int) -> np.ndarray:
    """
    Force mat to shape (n_samples, n_features).
    Handles common SHAP orientation differences: (n_features, n_samples).
    """
    if mat.shape == (n_samples, n_features):
        return mat
    if mat.shape == (n_features, n_samples):
        return mat.T
    raise ValueError(f"Unexpected SHAP shape {mat.shape}, expected {(n_samples, n_features)} or {(n_features, n_samples)}")

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    pipe = joblib.load(MODEL_PATH)
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    classes = list(model.classes_)

    feature_cols = pick_feature_columns(train_df)
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_test  = test_df[TARGET_COL].astype(str).copy()

    print("✅ Loaded data & model")
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"Features: {len(feature_cols)}")
    print("Classes:", classes)

    X_train_t = preprocessor.transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    background = X_train_t[: min(50, X_train_t.shape[0])]
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer.shap_values(X_test_t)

    n_samples = X_test_t.shape[0]
    n_features = X_test_t.shape[1]

    # Normalize shap_values format to list[class] -> (n_samples, n_features)
    if isinstance(shap_values, list):
        shap_values = [ensure_ns_nf(np.asarray(sv), n_samples, n_features) for sv in shap_values]
    else:
        # Sometimes SHAP returns ndarray; handle (n_classes, n_samples, n_features) or (n_samples, n_features, n_classes)
        sv = np.asarray(shap_values)
        if sv.ndim == 3 and sv.shape[0] == len(classes) and sv.shape[1:] in [(n_samples, n_features), (n_features, n_samples)]:
            mats = []
            for k in range(len(classes)):
                mats.append(ensure_ns_nf(sv[k], n_samples, n_features))
            shap_values = mats
        elif sv.ndim == 3 and sv.shape[2] == len(classes) and sv.shape[:2] in [(n_samples, n_features), (n_features, n_samples)]:
            base = ensure_ns_nf(sv[:, :, 0], n_samples, n_features)
            mats = []
            for k in range(len(classes)):
                mats.append(ensure_ns_nf(sv[:, :, k], n_samples, n_features))
            shap_values = mats
        else:
            raise ValueError(f"Unexpected shap_values ndarray shape: {sv.shape}")

    # 1) GLOBAL importance per class
    global_rows = []
    for k, cls in enumerate(classes):
        sv = shap_values[k]  # (n_samples, n_features)
        mean_abs = np.abs(sv).mean(axis=0)
        for f, val in zip(feature_cols, mean_abs):
            global_rows.append({"class": cls, "feature": f, "mean_abs_shap": float(val)})

    global_df = pd.DataFrame(global_rows).sort_values(["class", "mean_abs_shap"], ascending=[True, False])
    global_df["rank_in_class"] = global_df.groupby("class")["mean_abs_shap"].rank(ascending=False, method="dense")
    global_df.to_csv(GLOBAL_OUT, index=False)
    print(f"✅ Saved global SHAP: {GLOBAL_OUT}")

    # 2) Predictions + risk
    probs = pipe.predict_proba(X_test)
    proba_df = pd.DataFrame(probs, columns=[f"proba_{c}" for c in classes])
    proba_df["pred_outcome"] = pipe.predict(X_test)
    proba_df["risk_score"] = 1.0 - proba_df[[f"proba_{c}" for c in classes]].max(axis=1)

    id_df = test_df[[c for c in ID_COLS if c in test_df.columns]].reset_index(drop=True)
    base_pred = pd.concat([id_df, y_test.reset_index(drop=True).rename("true_outcome"), proba_df], axis=1)

    example_idx = choose_example_indices(proba_df, n_each=2)

    # 3) LOCAL explanations
    local_rows = []
    top_k = 7

    for idx in example_idx:
        pred_o = base_pred.loc[idx, "pred_outcome"]
        cls_i = classes.index(pred_o) if pred_o in classes else 0
        sv_row = shap_values[cls_i][idx, :]  # now always valid (n_samples,n_features)

        top_feat_idx = np.argsort(np.abs(sv_row))[::-1][:top_k]
        row_base = base_pred.loc[idx].to_dict()
        row_base["shap_class_explained"] = classes[cls_i]

        for rank, j in enumerate(top_feat_idx, start=1):
            feat = feature_cols[j]
            val = X_test.iloc[idx][feat]
            local_rows.append({
                **row_base,
                "feature_rank": rank,
                "feature": feat,
                "feature_value": float(val) if pd.notna(val) else np.nan,
                "shap_value": float(sv_row[j]),
                "abs_shap_value": float(abs(sv_row[j])),
            })

    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(LOCAL_OUT, index=False)
    print(f"✅ Saved local SHAP examples: {LOCAL_OUT}")

    # 4) Chatbot summary
    summary_rows = []
    for idx in example_idx:
        pred_o = base_pred.loc[idx, "pred_outcome"]
        cls_i = classes.index(pred_o) if pred_o in classes else 0
        sv_row = shap_values[cls_i][idx, :]

        top_feat_idx = np.argsort(np.abs(sv_row))[::-1][:5]
        parts = []
        for j in top_feat_idx:
            feat = feature_cols[j]
            val = X_test.iloc[idx][feat]
            s = sv_row[j]
            direction = "pushes toward" if s > 0 else "pushes away from"
            parts.append(f"{feat}={val} ({direction} {classes[cls_i]})")

        row = base_pred.loc[idx].to_dict()
        row["shap_class_explained"] = classes[cls_i]
        row["top_factors_text"] = " | ".join(parts)
        summary_rows.append(row)

    chatbot_df = pd.DataFrame(summary_rows)
    chatbot_df.to_csv(CHATBOT_OUT, index=False)
    print(f"✅ Saved chatbot SHAP summary: {CHATBOT_OUT}")

    # Print top global
    print("\nTop 10 features per class (mean |SHAP|):")
    for cls in classes:
        top = global_df[global_df["class"] == cls].head(10)[["feature", "mean_abs_shap"]]
        print(f"\n--- {cls} ---")
        print(top.to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()
