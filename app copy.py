# app.py
# =========================================================
# AFCON AI SBI — CAN-style Streamlit App
# 3 pages:
# 1) Analyse interactive (2019 / 2021 / 2023 + focus équipe)
# 2) Prédiction + XAI + LLM (à partir des fichiers processed)
# 3) Chat RAG + XAI (retrieval sur tes CSV + réponse LLM)
# =========================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from typing import Dict, Tuple, Optional

# LLM (OpenRouter via src/llm_explainer.py)
# Must expose: generate_explanation(user_question: str, context: str) -> str
try:
    from src.llm_explainer import generate_explanation
except Exception:
    generate_explanation = None

# RAG (TFIDF retriever + LLM)
# Must expose:
#   build_rag_retriever(project_root: Path) -> retriever
#   answer_with_rag(question: str, retriever, top_k: int) -> dict
try:
    from src.chat_rag import build_rag_retriever, answer_with_rag
except Exception:
    build_rag_retriever = None
    answer_with_rag = None


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="AFCON AI Lab — SBI Challenge",
    page_icon="⚽",
    layout="wide",
)

# =========================================================
# CAN-style theme (Green / Red / Gold)
# =========================================================
st.markdown(
    """
<style>
:root{
  --bg1: #04150d;
  --bg2: #061f14;
  --panel: rgba(255,255,255,0.06);
  --panel2: rgba(255,255,255,0.04);
  --stroke: rgba(255,255,255,0.10);
  --text: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.70);
  --gold: #d4af37;
  --red: #d72638;
  --green: #1fbf75;
}

.stApp{
  background:
    radial-gradient(1200px 700px at 20% 0%, rgba(212,175,55,0.10) 0%, rgba(4,21,13,0.0) 45%),
    radial-gradient(900px 500px at 85% 10%, rgba(215,38,56,0.10) 0%, rgba(4,21,13,0.0) 50%),
    linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 60%, #020b07 100%);
  color: var(--text);
}

.block-container{ padding-top: 1.0rem; padding-bottom: 2.0rem; max-width: 1250px; }
h1,h2,h3{ letter-spacing: .2px; }
small, .muted { color: var(--muted); }

.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: var(--panel2);
  font-size: 0.85rem;
}

.badge.green{ border-color: rgba(31,191,117,.45); }
.badge.red{ border-color: rgba(215,38,56,.45); }
.badge.gold{ border-color: rgba(212,175,55,.45); }

.card{
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 18px 50px rgba(0,0,0,.22);
}

.kpiTitle{ color: var(--muted); font-size: .90rem; margin-bottom: 6px; }
.kpiValue{ font-size: 1.55rem; font-weight: 800; }

.hr{ height:1px; background: var(--stroke); margin: 14px 0 18px; }

.header{
  background: linear-gradient(90deg, rgba(31,191,117,0.18), rgba(212,175,55,0.14), rgba(215,38,56,0.18));
  border: 1px solid var(--stroke);
  border-radius: 22px;
  padding: 16px 18px;
}

.headerTitle{ font-size: 1.35rem; font-weight: 900; }
.headerSub{ color: var(--muted); margin-top: 4px; font-size: .95rem; }

.note{
  background: rgba(212,175,55,0.08);
  border: 1px solid rgba(212,175,55,0.22);
  border-radius: 16px;
  padding: 12px 14px;
}

.stButton>button{
  border-radius: 14px;
  border: 1px solid rgba(212,175,55,0.28);
}

.stTextArea textarea, .stTextInput input{
  border-radius: 14px !important;
}

</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header">
  <div class="headerTitle">⚽ AFCON AI Lab — SBI Challenge</div>
  <div class="headerSub">Analyse • Prédiction expliquée (XAI) • Chat RAG (preuves + LLM) — style “CAN”</div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Paths & loading
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"


RAW_FILES = {
    2019: {
        "matches": RAW / "international-africa-cup-of-nations-matches-2019-to-2019-stats.csv",
        "teams": RAW / "international-africa-cup-of-nations-teams-2019-to-2019-stats.csv",
    },
    2021: {
        "matches": RAW / "international-africa-cup-of-nations-matches-2021-to-2021-stats.csv",
        "teams": RAW / "international-africa-cup-of-nations-teams-2021-to-2021-stats.csv",
    },
    2023: {
        "matches": RAW / "international-africa-cup-of-nations-matches-2023-to-2023-stats.csv",
        "teams": RAW / "international-africa-cup-of-nations-teams-2023-to-2023-stats.csv",
    },
}

PROC_FILES = {
    "pred_pre": PROCESSED / "predictions_prematch_2023.csv",
    "pred_post": PROCESSED / "predictions_postmatch_2023.csv",
    "shap_sum_pre": PROCESSED / "shap_summary_per_match_prematch.csv",
    "shap_sum_post": PROCESSED / "shap_summary_per_match_postmatch.csv",
    "shap_g_pre": PROCESSED / "shap_global_prematch.csv",
    "shap_g_post": PROCESSED / "shap_global_postmatch.csv",
}


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "match_datetime" in df.columns:
        df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
        return df
    if "date_GMT" in df.columns:
        df["match_datetime"] = pd.to_datetime(df["date_GMT"], errors="coerce")
        return df
    if "timestamp" in df.columns:
        # timestamp is usually seconds
        df["match_datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df
    df["match_datetime"] = pd.NaT
    return df


def _compute_outcome(home_goals: float, away_goals: float) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "N/A"
    if home_goals > away_goals:
        return "HOME_WIN"
    if away_goals > home_goals:
        return "AWAY_WIN"
    return "DRAW"


def _outcome_fr(o: str) -> str:
    return {
        "HOME_WIN": "Victoire domicile",
        "AWAY_WIN": "Victoire extérieur",
        "DRAW": "Match nul",
        "N/A": "N/A",
    }.get(o, o)


def _badge_outcome(o: str) -> str:
    if o == "HOME_WIN":
        return '<span class="badge green">Victoire domicile</span>'
    if o == "AWAY_WIN":
        return '<span class="badge red">Victoire extérieur</span>'
    if o == "DRAW":
        return '<span class="badge gold">Match nul</span>'
    return '<span class="badge">N/A</span>'


def _badge_risk(r: float) -> str:
    if pd.isna(r):
        return '<span class="badge">Incertitude: N/A</span>'
    # r = 1 - max proba
    if r <= 0.25:
        return f'<span class="badge green">Incertitude faible • {r:.2f}</span>'
    if r <= 0.45:
        return f'<span class="badge gold">Incertitude moyenne • {r:.2f}</span>'
    return f'<span class="badge red">Incertitude élevée • {r:.2f}</span>'


def kpi(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
<div class="card">
  <div class="kpiTitle">{title}</div>
  <div class="kpiValue">{value}</div>
  <div class="muted" style="margin-top:6px; font-size:.90rem;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_matches(year: int) -> pd.DataFrame:
    path = RAW_FILES[year]["matches"]
    df = load_csv(path)
    if df.empty:
        return df

    df["season"] = year
    df = _parse_datetime(df)

    # Normalize key columns
    if "home_team_goal_count" in df.columns:
        df["home_goals"] = _to_num(df["home_team_goal_count"])
    else:
        df["home_goals"] = np.nan
    if "away_team_goal_count" in df.columns:
        df["away_goals"] = _to_num(df["away_team_goal_count"])
    else:
        df["away_goals"] = np.nan

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["outcome"] = [
        _compute_outcome(h, a) for h, a in zip(df["home_goals"], df["away_goals"])
    ]

    # Clean numerics (optional)
    for c in [
        "attendance",
        "home_team_corner_count",
        "away_team_corner_count",
        "home_team_yellow_cards",
        "away_team_yellow_cards",
        "home_team_red_cards",
        "away_team_red_cards",
        "home_team_shots",
        "away_team_shots",
        "home_team_shots_on_target",
        "away_team_shots_on_target",
        "home_team_possession",
        "away_team_possession",
        "team_a_xg",
        "team_b_xg",
    ]:
        if c in df.columns:
            df[c] = _to_num(df[c])

    # Nice display helpers
    df["score"] = np.where(
        df["outcome"] == "N/A",
        "N/A",
        df["home_goals"].fillna(0).astype(int).astype(str)
        + " - "
        + df["away_goals"].fillna(0).astype(int).astype(str),
    )

    return df


@st.cache_data(show_spinner=False)
def load_teams(year: int) -> pd.DataFrame:
    path = RAW_FILES[year]["teams"]
    df = load_csv(path)
    if df.empty:
        return df
    df["season"] = year
    return df


def compute_points_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Simple points table from all matches (works as an overview)."""
    if matches.empty:
        return pd.DataFrame()

    rows = []
    for _, r in matches.iterrows():
        h = r.get("home_team_name", "")
        a = r.get("away_team_name", "")
        out = r.get("outcome", "N/A")
        hg = r.get("home_goals", np.nan)
        ag = r.get("away_goals", np.nan)

        if not h or not a or out == "N/A":
            continue

        if out == "HOME_WIN":
            rows.append((h, 3, hg, ag))
            rows.append((a, 0, ag, hg))
        elif out == "AWAY_WIN":
            rows.append((h, 0, hg, ag))
            rows.append((a, 3, ag, hg))
        else:
            rows.append((h, 1, hg, ag))
            rows.append((a, 1, ag, hg))

    df = pd.DataFrame(rows, columns=["team", "points", "gf", "ga"])
    g = df.groupby("team", as_index=False).agg(
        points=("points", "sum"),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
        matches=("team", "count"),
    )
    g["goal_diff"] = g["goals_for"] - g["goals_against"]
    g = g.sort_values(["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
    return g


def team_journey(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if matches.empty or not team:
        return pd.DataFrame()

    m = matches[
        (matches["home_team_name"] == team) | (matches["away_team_name"] == team)
    ].copy()

    if m.empty:
        return m

    def _row(r):
        home = r["home_team_name"]
        away = r["away_team_name"]
        is_home = (home == team)
        opponent = away if is_home else home
        hg = r.get("home_goals", np.nan)
        ag = r.get("away_goals", np.nan)

        gf = hg if is_home else ag
        ga = ag if is_home else hg
        out = r.get("outcome", "N/A")

        # points for this team
        pts = 0
        if out != "N/A":
            if out == "DRAW":
                pts = 1
            elif out == "HOME_WIN" and is_home:
                pts = 3
            elif out == "AWAY_WIN" and (not is_home):
                pts = 3

        res = "N/A"
        if out != "N/A":
            if pts == 3:
                res = "Victoire"
            elif pts == 1:
                res = "Nul"
            else:
                res = "Défaite"

        return pd.Series(
            {
                "Date": r.get("match_datetime", pd.NaT),
                "Adversaire": opponent,
                "Lieu": "Domicile" if is_home else "Extérieur",
                "Score": r.get("score", "N/A"),
                "Résultat": res,
                "Points": pts,
                "Buts pour": gf,
                "Buts contre": ga,
            }
        )

    journey = m.apply(_row, axis=1)
    journey = journey.sort_values("Date").reset_index(drop=True)
    journey["Points cumulés"] = journey["Points"].cumsum()
    return journey


# =========================================================
# Load processed (for prediction/XAI/chat)
# =========================================================
pred_pre = load_csv(PROC_FILES["pred_pre"])
pred_post = load_csv(PROC_FILES["pred_post"])
shap_sum_pre = load_csv(PROC_FILES["shap_sum_pre"])
shap_sum_post = load_csv(PROC_FILES["shap_sum_post"])
shap_g_pre = load_csv(PROC_FILES["shap_g_pre"])
shap_g_post = load_csv(PROC_FILES["shap_g_post"])

if not pred_pre.empty:
    pred_pre = _parse_datetime(pred_pre)
if not pred_post.empty:
    pred_post = _parse_datetime(pred_post)


@st.cache_resource(show_spinner=False)
def get_rag_retriever():
    if build_rag_retriever is None:
        return None
    return build_rag_retriever(PROJECT_ROOT)


# =========================================================
# Navigation (3 pages)
# =========================================================
tab1, tab2, tab3 = st.tabs(
    ["📊 Analyse (CAN 2019/2021/2023)", "🎯 Prédiction + Explication", "🤖 Chat RAG (preuves + LLM)"]
)

# =========================================================
# TAB 1 — Analyse interactive
# =========================================================
with tab1:
    st.markdown("## Tableau de bord — Analyse du tournoi")

    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        year = st.selectbox("Choisir l’édition", [2019, 2021, 2023], index=2)
    matches = load_matches(year)
    teams = load_teams(year)

    # Team selector based on matches (most reliable)
    all_teams = sorted(
        set(matches.get("home_team_name", pd.Series(dtype=str)).dropna().unique()).union(
            set(matches.get("away_team_name", pd.Series(dtype=str)).dropna().unique())
        )
    )
    with c2:
        team_focus = st.selectbox("Focus équipe (optionnel)", ["(Tout le tournoi)"] + all_teams, index=0)
    with c3:
        st.markdown(
            f"""
<div class="note">
<b>Idée “CAN 2025”</b><br/>
Compare les styles de jeu (buts, discipline, corners, xG) entre 2019 / 2021 / 2023.<br/>
Puis observe le “parcours” d’une équipe (victoires, buts, points cumulés).
</div>
""",
            unsafe_allow_html=True,
        )

    if matches.empty:
        st.error("Je ne trouve pas les fichiers dans data/raw pour cette édition. Vérifie les noms des CSV.")
        st.stop()

    # Global KPIs
    total_matches = int(matches.shape[0])
    total_goals = float(matches["total_goals"].dropna().sum()) if "total_goals" in matches.columns else 0.0
    avg_goals = float(matches["total_goals"].dropna().mean()) if "total_goals" in matches.columns else np.nan

    # Cards & corners
    corners = np.nan
    if "home_team_corner_count" in matches.columns and "away_team_corner_count" in matches.columns:
        corners = (matches["home_team_corner_count"] + matches["away_team_corner_count"]).dropna().mean()
    cards = np.nan
    if "home_team_yellow_cards" in matches.columns and "away_team_yellow_cards" in matches.columns:
        cards = (matches["home_team_yellow_cards"] + matches["away_team_yellow_cards"]).dropna().mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Matchs", f"{total_matches}", f"Édition {year}")
    with k2:
        kpi("Buts", f"{int(total_goals)}", "Total sur le tournoi")
    with k3:
        kpi("Buts / match", f"{avg_goals:.2f}" if not pd.isna(avg_goals) else "N/A", "Intensité offensive")
    with k4:
        subtitle = []
        if not pd.isna(corners):
            subtitle.append(f"Corners/match: {corners:.2f}")
        if not pd.isna(cards):
            subtitle.append(f"Cartons jaunes/match: {cards:.2f}")
        kpi("Style de match", "📌", " • ".join(subtitle) if subtitle else "Corners / cartons indisponibles")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Team focus view
    if team_focus != "(Tout le tournoi)":
        st.markdown(f"### Parcours — {team_focus} ({year})")
        j = team_journey(matches, team_focus)
        if j.empty:
            st.info("Aucun match trouvé pour cette équipe.")
        else:
            # Summary
            wins = int((j["Résultat"] == "Victoire").sum())
            draws = int((j["Résultat"] == "Nul").sum())
            losses = int((j["Résultat"] == "Défaite").sum())
            gf = int(pd.to_numeric(j["Buts pour"], errors="coerce").fillna(0).sum())
            ga = int(pd.to_numeric(j["Buts contre"], errors="coerce").fillna(0).sum())
            pts = int(pd.to_numeric(j["Points"], errors="coerce").fillna(0).sum())

            a1, a2, a3, a4 = st.columns(4)
            with a1:
                kpi("Bilan", f"{wins}V • {draws}N • {losses}D", "Résultats")
            with a2:
                kpi("Points", f"{pts}", "Total (calcul simplifié)")
            with a3:
                kpi("Buts", f"{gf} pour / {ga} contre", "Attaque & défense")
            with a4:
                kpi("Différence", f"{gf-ga:+d}", "Buts pour − buts contre")

            # Cumulative points chart (interactive)
            ch = (
                alt.Chart(j)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Points cumulés:Q", title="Points cumulés"),
                    tooltip=["Date:T", "Adversaire:N", "Score:N", "Résultat:N", "Points:Q", "Points cumulés:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(ch, use_container_width=True)

            st.markdown("#### Matchs (détails)")
            st.dataframe(
                j[["Date", "Adversaire", "Lieu", "Score", "Résultat", "Points"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Tournament overview charts
    st.markdown("### Statistiques clés du tournoi")

    left, right = st.columns([1, 1])

    with left:
        # Outcome distribution
        dist = matches["outcome"].value_counts(dropna=False).reset_index()
        dist.columns = ["Outcome", "Count"]
        dist["Libellé"] = dist["Outcome"].map(_outcome_fr)

        chart = (
            alt.Chart(dist)
            .mark_bar()
            .encode(
                x=alt.X("Libellé:N", title="Résultat"),
                y=alt.Y("Count:Q", title="Nombre de matchs"),
                tooltip=["Libellé:N", "Count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        # Goals distribution
        gdf = matches[["total_goals"]].dropna().copy()
        if gdf.empty:
            st.info("Distribution des buts indisponible (colonnes buts manquantes).")
        else:
            hist = (
                alt.Chart(gdf)
                .mark_bar()
                .encode(
                    x=alt.X("total_goals:Q", bin=alt.Bin(maxbins=10), title="Buts dans le match"),
                    y=alt.Y("count()", title="Nombre de matchs"),
                    tooltip=[alt.Tooltip("count()", title="Matchs")],
                )
                .properties(height=280)
            )
            st.altair_chart(hist, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    st.markdown("### Équipes — classement simple (points)")
    table = compute_points_table(matches)
    if table.empty:
        st.info("Impossible de calculer le classement (données buts manquantes).")
    else:
        top_n = st.slider("Afficher le Top", 5, 24, 10, 1)
        show = table.head(top_n).copy()
        show["Rang"] = np.arange(1, len(show) + 1)

        st.dataframe(
            show[["Rang", "team", "points", "goal_diff", "goals_for", "goals_against", "matches"]],
            use_container_width=True,
            hide_index=True,
        )

        # Chart
        ch2 = (
            alt.Chart(show)
            .mark_bar()
            .encode(
                x=alt.X("points:Q", title="Points"),
                y=alt.Y("team:N", sort="-x", title="Équipe"),
                tooltip=["team:N", "points:Q", "goal_diff:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(ch2, use_container_width=True)


# =========================================================
# TAB 2 — Prédiction + XAI + LLM
# =========================================================
with tab2:
    st.markdown("## Prédire et expliquer un match")
    st.markdown(
        "<div class='muted'>Ici, on utilise tes fichiers <b>processed</b> (prédictions + explications SHAP). "
        "On affiche des mots simples : pronostic, résultat, incertitude, facteurs clés.</div>",
        unsafe_allow_html=True,
    )

    mode = st.radio(
        "Choisir le mode",
        ["Avant match (prévision)", "Après match (facteurs du résultat)"],
        horizontal=True,
    )

    if mode.startswith("Avant"):
        pred_df = pred_pre.copy()
        shap_sum = shap_sum_pre.copy()
        shap_g = shap_g_pre.copy()
        title_mode = "Avant match"
    else:
        pred_df = pred_post.copy()
        shap_sum = shap_sum_post.copy()
        shap_g = shap_g_post.copy()
        title_mode = "Après match"

    if pred_df.empty:
        st.error(
            "Je ne trouve pas les fichiers de prédiction dans data/processed. "
            "Assure-toi d’avoir généré les CSV (train_model*.py)."
        )
        st.stop()

    # Filters
    all_teams_pred = sorted(
        set(pred_df["home_team_name"].dropna().unique()).union(
            set(pred_df["away_team_name"].dropna().unique())
        )
    )

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        team_filter = st.selectbox("Filtrer par équipe (optionnel)", ["(Toutes)"] + all_teams_pred, index=0)
    with f2:
        max_risk = st.slider("Incertitude max", 0.0, 1.0, 1.0, 0.05)
    with f3:
        only_wrong = st.toggle("Voir seulement les erreurs", value=False)

    df = pred_df.copy()
    if team_filter != "(Toutes)":
        df = df[(df["home_team_name"] == team_filter) | (df["away_team_name"] == team_filter)]
    if "risk_score" in df.columns:
        df = df[df["risk_score"] <= max_risk]
    if only_wrong and "pred_outcome" in df.columns and "true_outcome" in df.columns:
        df = df[df["pred_outcome"] != df["true_outcome"]]

    df = df.reset_index(drop=True)
    df["match_label"] = df["home_team_name"].astype(str) + " vs " + df["away_team_name"].astype(str)

    if df.empty:
        st.info("Aucun match ne correspond à tes filtres.")
        st.stop()

    pick = st.selectbox("Choisir un match", list(range(len(df))), format_func=lambda i: df.loc[i, "match_label"])
    sel = df.loc[pick]

    # Build probabilities table
    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    probs = {}
    if proba_cols:
        for c in proba_cols:
            try:
                probs[c.replace("proba_", "")] = float(sel[c])
            except Exception:
                pass

    pred_out = str(sel.get("pred_outcome", "N/A"))
    true_out = str(sel.get("true_outcome", "N/A"))
    risk = float(sel.get("risk_score", np.nan)) if "risk_score" in sel.index else np.nan

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown(
            f"""
<div class="card">
  <div class="kpiTitle">{title_mode} — Résumé</div>
  <div style="font-size:1.25rem; font-weight:900; margin-top:6px;">
    {sel['home_team_name']} vs {sel['away_team_name']}
  </div>
  <div style="margin-top:10px;">
    Pronostic : {_badge_outcome(pred_out)} &nbsp;&nbsp;
    Résultat : {_badge_outcome(true_out)}
  </div>
  <div style="margin-top:10px;">
    {_badge_risk(risk)}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.markdown("### Probabilités")
        if probs:
            p_df = (
                pd.DataFrame({"Résultat": list(probs.keys()), "Probabilité": list(probs.values())})
                .sort_values("Probabilité", ascending=False)
                .reset_index(drop=True)
            )
            chart = (
                alt.Chart(p_df)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilité:Q", title="Probabilité"),
                    y=alt.Y("Résultat:N", sort="-x", title=""),
                    tooltip=["Résultat:N", alt.Tooltip("Probabilité:Q", format=".2f")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Ce fichier ne contient pas les colonnes de probabilités (proba_*).")

    with right:
        st.markdown("### Pourquoi ce pronostic ? (XAI)")
        # Find SHAP local text if available
        shap_text = ""

        if not shap_sum.empty:
            # Build key based on timestamp + teams (same as your pipeline)
            if "timestamp" in shap_sum.columns:
                shap_sum = shap_sum.copy()
                shap_sum["_key"] = (
                    shap_sum["timestamp"].astype(str)
                    + "|"
                    + shap_sum["home_team_name"].astype(str)
                    + "|"
                    + shap_sum["away_team_name"].astype(str)
                )
                key = str(sel["timestamp"]) + "|" + str(sel["home_team_name"]) + "|" + str(sel["away_team_name"])
                row = shap_sum[shap_sum["_key"] == key]
                if not row.empty:
                    shap_text = str(row.iloc[0].get("top_factors_text", ""))

        if shap_text:
            st.markdown(
                f"""
<div class="card">
  <div class="kpiTitle">Facteurs clés (extraits SHAP)</div>
  <div style="font-size:0.98rem; line-height:1.55;">{shap_text}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Je n’ai pas trouvé l’explication locale SHAP pour ce match (échantillon SHAP).")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

        st.markdown("### Facteurs importants (global)")
        if shap_g.empty or ("class" not in shap_g.columns):
            st.info("Fichier SHAP global manquant pour ce mode.")
            top_global_lines = []
        else:
            top_global = (
                shap_g[shap_g["class"] == pred_out]
                .sort_values("mean_abs_shap", ascending=False)
                .head(8)
            )
            if top_global.empty:
                st.info("Pas de facteurs globaux pour cette classe.")
                top_global_lines = []
            else:
                st.dataframe(
                    top_global[["feature", "mean_abs_shap"]],
                    use_container_width=True,
                    hide_index=True,
                )
                top_global_lines = [
                    f"- {r.feature} (importance={r.mean_abs_shap:.3f})"
                    for r in top_global.itertuples()
                ]

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("## Demander une explication au LLM (texte)")
    if generate_explanation is None:
        st.error("Module LLM indisponible. Vérifie src/llm_explainer.py")
        st.stop()

    quick = st.columns(3)
    with quick[0]:
        q1 = st.button("Explique simplement le pronostic")
    with quick[1]:
        q2 = st.button("Donne 2 recommandations tactiques")
    with quick[2]:
        q3 = st.button("Pourquoi ce match est incertain ?")

    default_q = "Explique le pronostic en te basant uniquement sur les preuves, sans inventer."
    if q1:
        default_q = "Explique simplement le pronostic et les facteurs qui influencent la décision."
    if q2:
        default_q = "Donne 2 recommandations tactiques pour l’équipe la plus favorisée, en restant prudent."
    if q3:
        default_q = "Explique pourquoi ce match est incertain (quels signaux se contredisent)."

    user_question = st.text_area("Ta question", value=default_q)

    # Build evidence
    evidence = []
    evidence.append(f"Mode: {title_mode}")
    evidence.append(f"Match: {sel['home_team_name']} vs {sel['away_team_name']}")
    evidence.append(f"Pronostic: {pred_out} ({_outcome_fr(pred_out)})")
    evidence.append(f"Résultat réel: {true_out} ({_outcome_fr(true_out)})")
    if probs:
        evidence.append("Probabilités: " + ", ".join([f"{k}={v:.2f}" for k, v in probs.items()]))
    if not pd.isna(risk):
        evidence.append(f"Incertitude (risk): {risk:.2f}")
    if shap_text:
        evidence.append("Facteurs clés (SHAP local): " + shap_text)
    if top_global_lines:
        evidence.append("Facteurs importants (SHAP global):\n" + "\n".join(top_global_lines))

    context_text = "\n".join(evidence)

    if st.button("Générer la réponse (LLM)"):
        try:
            with st.spinner("Appel au LLM (OpenRouter)..."):
                ans = generate_explanation(user_question, context_text)
            st.markdown("### Réponse")
            st.write(ans)

            with st.expander("Voir les preuves envoyées au LLM"):
                st.code(context_text)
        except Exception as e:
            st.error(f"Erreur LLM: {e}")


# =========================================================
# TAB 3 — Chat RAG + XAI
# =========================================================
with tab3:
    st.markdown("## Chat RAG — réponses basées sur des preuves")
    st.markdown(
        "<div class='muted'>Tu poses une question. Le système récupère automatiquement des lignes pertinentes "
        "dans tes CSV (prédictions, SHAP), puis le LLM répond <b>uniquement</b> avec ces preuves.</div>",
        unsafe_allow_html=True,
    )

    if build_rag_retriever is None or answer_with_rag is None:
        st.error("Modules RAG indisponibles. Vérifie src/chat_rag.py et src/rag_store.py")
        st.stop()

    if generate_explanation is None:
        st.error("Module LLM indisponible. Vérifie src/llm_explainer.py")
        st.stop()

    retriever = get_rag_retriever()
    if retriever is None:
        st.error("Impossible d’initialiser le retriever RAG.")
        st.stop()

    a, b, c = st.columns([1, 1, 1.2])
    with a:
        top_k = st.slider("Nombre de preuves (Top-K)", 3, 15, 8, 1)
    with b:
        show_evidence = st.toggle("Afficher les preuves (debug)", value=True)
    with c:
        st.markdown(
            """
<div class="note">
<b>Exemples de questions</b><br/>
• Quels facteurs influencent le plus “Victoire domicile” avant match ?<br/>
• Quels matchs sont les plus incertains ?<br/>
• Résume les drivers après match: tirs, possession, xG.
</div>
""",
            unsafe_allow_html=True,
        )

    question = st.text_area(
        "Ta question",
        value="Quels sont les facteurs les plus importants pour prédire une victoire à domicile avant match ?",
        height=120,
    )

    if st.button("Envoyer (RAG)"):
        try:
            with st.spinner("Recherche de preuves + appel LLM..."):
                result = answer_with_rag(question, retriever, top_k=top_k)

            st.markdown("### Réponse")
            st.write(result["answer"])

            if show_evidence:
                st.markdown("### Preuves utilisées")
                st.code(result["evidence"])
        except Exception as e:
            st.error(f"Erreur RAG: {e}")
