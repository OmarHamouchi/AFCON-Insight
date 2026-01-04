import re
import pandas as pd
from pathlib import Path

# =========================================================
# Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MATCHES_IN = DATA_PROCESSED / "afcon_matches_2019_2021_2023.csv"
TEAMS_IN   = DATA_PROCESSED / "afcon_teams_2019_2021_2023.csv"

PREMATCH_OUT = DATA_PROCESSED / "pre_match_dataset.csv"
POSTMATCH_OUT = DATA_PROCESSED / "post_match_drivers.csv"

PREMATCH_TRAIN_OUT = DATA_PROCESSED / "pre_match_train_2019_2021.csv"
PREMATCH_TEST_OUT  = DATA_PROCESSED / "pre_match_test_2023.csv"

POSTMATCH_TRAIN_OUT = DATA_PROCESSED / "post_match_train_2019_2021.csv"
POSTMATCH_TEST_OUT  = DATA_PROCESSED / "post_match_test_2023.csv"

# =========================================================
# Utils
# =========================================================
def normalize_col(c: str) -> str:
    c = c.strip()
    c = c.replace(" ", "_").replace("-", "_")
    c = re.sub(r"__+", "_", c)
    return c

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    # required columns
    for col in ["home_team_goal_count", "away_team_goal_count"]:
        if col not in df.columns:
            raise KeyError(f"Missing column in matches: {col}")

    hg = pd.to_numeric(df["home_team_goal_count"], errors="coerce")
    ag = pd.to_numeric(df["away_team_goal_count"], errors="coerce")

    df["target_outcome"] = "DRAW"
    df.loc[hg > ag, "target_outcome"] = "HOME_WIN"
    df.loc[hg < ag, "target_outcome"] = "AWAY_WIN"
    return df

def season_split(df: pd.DataFrame):
    train = df[df["season"].isin([2019, 2021])].copy()
    test  = df[df["season"] == 2023].copy()
    return train, test

# =========================================================
# Main
# =========================================================
def main():
    if not MATCHES_IN.exists():
        raise FileNotFoundError(f"Missing: {MATCHES_IN}")
    matches = pd.read_csv(MATCHES_IN)

    # normalize columns (important for clean code later)
    matches.columns = [normalize_col(c) for c in matches.columns]

    # ensure season exists
    if "season" not in matches.columns:
        raise KeyError("Column 'season' not found in merged matches file.")

    # create target
    matches = make_target(matches)

    # minimal identifiers (we keep these in both datasets)
    id_cols = [c for c in [
        "season", "timestamp", "match_datetime", "date_gmt",
        "home_team_name", "away_team_name", "stadium_name",
        "referee", "attendance", "game_week"
    ] if c in matches.columns]

    # =====================================================
    # A) PRE-MATCH DATASET (prediction BEFORE match)
    # Keep only pre-match columns + odds + a few neutral fields
    # =====================================================
    pre_cols = []
    for c in matches.columns:
        cl = c.lower()
        if "pre_match" in cl:
            pre_cols.append(c)
        if cl.startswith("odds_"):
            pre_cols.append(c)

    # remove duplicates
    pre_cols = sorted(list(set(pre_cols)))

    # Always keep target + ids
    prematch_cols = id_cols + ["target_outcome"] + pre_cols
    prematch_cols = [c for c in prematch_cols if c in matches.columns]
    prematch = matches[prematch_cols].copy()

    # Drop rows where target missing
    prematch = prematch.dropna(subset=["target_outcome"])

    # =====================================================
    # B) POST-MATCH DRIVERS (analysis AFTER match)
    # Keep match stats that explain the result (shots, possession, corners, cards, fouls, xg…)
    # EXCLUDE target-leak columns like goals (we keep goals only for reference optionally)
    # =====================================================
    candidate_driver_keywords = [
        "shots", "shots_on_target", "shots_off_target",
        "possession", "corners", "yellow_cards", "red_cards",
        "fouls", "xg", "team_a_xg", "team_b_xg"
    ]

    driver_cols = []
    for c in matches.columns:
        cl = c.lower()
        if any(k in cl for k in candidate_driver_keywords):
            driver_cols.append(c)

    # Remove goals columns from drivers (they define the outcome)
    leak_cols = [c for c in driver_cols if "goal_count" in c.lower() or "total_goal" in c.lower()]
    driver_cols = [c for c in driver_cols if c not in leak_cols]

    driver_cols = sorted(list(set(driver_cols)))

    postmatch_cols = id_cols + ["target_outcome"] + driver_cols
    postmatch_cols = [c for c in postmatch_cols if c in matches.columns]
    postmatch = matches[postmatch_cols].copy()
    postmatch = postmatch.dropna(subset=["target_outcome"])

    # =====================================================
    # Save
    # =====================================================
    prematch.to_csv(PREMATCH_OUT, index=False)
    postmatch.to_csv(POSTMATCH_OUT, index=False)

    # Split train/test
    prematch_train, prematch_test = season_split(prematch)
    postmatch_train, postmatch_test = season_split(postmatch)

    prematch_train.to_csv(PREMATCH_TRAIN_OUT, index=False)
    prematch_test.to_csv(PREMATCH_TEST_OUT, index=False)

    postmatch_train.to_csv(POSTMATCH_TRAIN_OUT, index=False)
    postmatch_test.to_csv(POSTMATCH_TEST_OUT, index=False)

    # logs
    print("✅ prepare_data finished")
    print(f"Prematch dataset: {prematch.shape} -> {PREMATCH_OUT.name}")
    print(f"Postmatch drivers: {postmatch.shape} -> {POSTMATCH_OUT.name}")
    print(f"Prematch train/test: {prematch_train.shape} / {prematch_test.shape}")
    print(f"Postmatch train/test: {postmatch_train.shape} / {postmatch_test.shape}")

    # quick sanity check
    print("\nTarget distribution (prematch):")
    print(prematch["target_outcome"].value_counts())

if __name__ == "__main__":
    main()
