import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MATCH_FILES = {
    2019: RAW_DIR / "international-africa-cup-of-nations-matches-2019-to-2019-stats.csv",
    2021: RAW_DIR / "international-africa-cup-of-nations-matches-2021-to-2021-stats.csv",
    2023: RAW_DIR / "international-africa-cup-of-nations-matches-2023-to-2023-stats.csv",
}

TEAM_FILES = {
    2019: RAW_DIR / "international-africa-cup-of-nations-teams-2019-to-2019-stats.csv",
    2021: RAW_DIR / "international-africa-cup-of-nations-teams-2021-to-2021-stats.csv",
    2023: RAW_DIR / "international-africa-cup-of-nations-teams-2023-to-2023-stats.csv",
}

MATCHES_OUT = PROCESSED_DIR / "afcon_matches_2019_2021_2023.csv"
TEAMS_OUT = PROCESSED_DIR / "afcon_teams_2019_2021_2023.csv"

DELETE_OLD_MERGED = True
DELETE_RAW = False  # ⚠️ mets True فقط si tu veux supprimer les raw après merge

# =========================================================
# Helpers
# =========================================================
def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def safe_remove(path: Path) -> None:
    if path.exists():
        path.unlink()

# =========================================================
# Main
# =========================================================
def main():
    # 0) check dirs
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR not found: {RAW_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1) supprimer anciens outputs
    if DELETE_OLD_MERGED:
        safe_remove(MATCHES_OUT)
        safe_remove(TEAMS_OUT)

    # 2) combine matches
    matches_all = []
    for season, path in MATCH_FILES.items():
        df = read_csv_robust(path)
        df["season"] = season
        matches_all.append(df)

    matches = pd.concat(matches_all, ignore_index=True)

    if "timestamp" in matches.columns:
        matches["timestamp"] = pd.to_numeric(matches["timestamp"], errors="coerce")
        matches["match_datetime"] = pd.to_datetime(matches["timestamp"], unit="s", errors="coerce")

    if "attendance" in matches.columns:
        matches["attendance"] = matches["attendance"].replace(["N/A", "NA", ""], pd.NA)

    # dedup
    dedup_cols = [c for c in ["timestamp", "home_team_name", "away_team_name"] if c in matches.columns]
    if len(dedup_cols) == 3:
        before = len(matches)
        matches = matches.drop_duplicates(subset=dedup_cols, keep="first")
        print(f"Matches dedup: {before} -> {len(matches)}")

    # 3) combine teams
    teams_all = []
    for season, path in TEAM_FILES.items():
        df = read_csv_robust(path)
        df["season"] = season
        teams_all.append(df)

    teams = pd.concat(teams_all, ignore_index=True)

    # 4) save
    matches.to_csv(MATCHES_OUT, index=False)
    teams.to_csv(TEAMS_OUT, index=False)

    print("✅ Merge done")
    print(f"RAW_DIR: {RAW_DIR}")
    print(f"Matches: {matches.shape} -> {MATCHES_OUT}")
    print(f"Teams:   {teams.shape} -> {TEAMS_OUT}")

    # 5) delete raw if enabled
    if DELETE_RAW:
        for p in list(MATCH_FILES.values()) + list(TEAM_FILES.values()):
            safe_remove(p)
        print("🗑️ Raw files deleted (DELETE_RAW=True)")

if __name__ == "__main__":
    main()
