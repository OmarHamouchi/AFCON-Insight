from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path

from src.llm_explainer import generate_explanation
from src.chat_rag import build_rag_retriever, answer_with_rag


CAN_GOLD = "#d4af37"
CAN_RED = "#d72638"
CAN_GREEN = "#1fbf75"
CAN_SOFT_GOLD = "#f1d27a"
CAN_DARK = "#04150d"

st.set_page_config(
    page_title="AFCON Insight — SBI Challenge",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
header, [data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer { visibility: hidden; height: 0; }
.block-container { padding-top: 0.7rem; padding-bottom: 2rem; }

:root{
  --bg1: #04150d;
  --bg2: #062016;
  --panel: rgba(255,255,255,0.08);
  --panel2: rgba(255,255,255,0.05);
  --stroke: rgba(255,255,255,0.14);
  --text: rgba(255,255,255,0.94);
  --muted: rgba(255,255,255,0.74);
  --gold: #d4af37;
  --red:  #d72638;
  --green:#1fbf75;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 700px at 20% 0%, rgba(212,175,55,0.14) 0%, rgba(4,21,13,0.0) 45%),
    radial-gradient(900px 500px at 85% 10%, rgba(215,38,56,0.14) 0%, rgba(4,21,13,0.0) 50%),
    linear-gradient(180deg, var(--bg2) 0%, var(--bg1) 65%, #020b07 100%);
}
.stApp { color: var(--text); }

h1,h2,h3,h4 { color: var(--text); }
small, .muted { color: var(--muted); }

.card{
  background: var(--panel);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 16px 16px;
  box-shadow: 0 18px 45px rgba(0,0,0,.30);
}
.kpiTitle{ color: var(--muted); font-size: .90rem; margin-bottom: 6px; }
.kpiValue{ font-size: 1.65rem; font-weight: 900; }
.hr{ height:1px; background: var(--stroke); margin: 14px 0 18px; }

.hero{
  border: 1px solid var(--stroke);
  border-radius: 22px;
  padding: 16px 18px;
  background: linear-gradient(90deg, rgba(31,191,117,0.18), rgba(212,175,55,0.14), rgba(215,38,56,0.18));
}
.heroTitle{ font-size: 1.35rem; font-weight: 900; }
.heroSub{ color: var(--muted); margin-top: 4px; font-size: .95rem; }

.badge{
  display:inline-block;
  padding:6px 10px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: var(--panel2);
  font-size: 0.85rem;
}
.badge.green{ border-color: rgba(31,191,117,.55); }
.badge.red  { border-color: rgba(215,38,56,.55); }
.badge.gold { border-color: rgba(212,175,55,.55); }

.stButton>button{
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(212,175,55,0.40) !important;
  background: rgba(212,175,55,0.14) !important;
  color: var(--text) !important;
  font-weight: 700 !important;
}
.stButton>button:hover{
  border-color: rgba(212,175,55,0.70) !important;
  background: rgba(212,175,55,0.22) !important;
}

.stTabs [data-baseweb="tab-list"]{ gap: 8px; }
.stTabs [data-baseweb="tab"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 10px 12px;
  color: rgba(255,255,255,0.92) !important;
}
.stTabs [aria-selected="true"]{
  background: rgba(212,175,55,0.14) !important;
  border-color: rgba(212,175,55,0.40) !important;
}

.stTextArea textarea, .stTextInput input{ border-radius: 14px !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="heroTitle">⚽ AFCON AI Lab — SBI Challenge</div>
  <div class="heroSub">Analyse interactive • Prédiction expliquée (XAI) • Chat RAG (preuves + LLM)</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# Altair theme (CAN palette + NO BLUE)
# =========================
def _altair_can_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"strokeOpacity": 0},
            "axis": {
                "labelColor": "rgba(255,255,255,0.90)",
                "titleColor": "rgba(255,255,255,0.90)",
                "gridColor": "rgba(255,255,255,0.10)",
                "domainColor": "rgba(255,255,255,0.20)",
                "tickColor": "rgba(255,255,255,0.20)",
            },
            "legend": {
                "labelColor": "rgba(255,255,255,0.90)",
                "titleColor": "rgba(255,255,255,0.90)",
            },
            "title": {"color": "rgba(255,255,255,0.95)"},
            # ✅ palette for categorical colors (no blue)
            "range": {
                "category": [CAN_GOLD, CAN_GREEN, CAN_RED, CAN_SOFT_GOLD, "#74d6a6", "#ff7a86"]
            },
            # ✅ default mark colors
            "bar": {"color": CAN_GOLD},
            "line": {"color": CAN_GOLD},
            "point": {"filled": True, "color": CAN_GOLD},
            "area": {"color": CAN_GOLD, "opacity": 0.35},
        }
    }


alt.themes.register("can_dark", _altair_can_theme)
alt.themes.enable("can_dark")


# =========================
# Paths
# =========================
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


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _parse_dt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "match_datetime" in df.columns:
        df["match_datetime"] = pd.to_datetime(df["match_datetime"], errors="coerce")
        return df
    if "date_GMT" in df.columns:
        df["match_datetime"] = pd.to_datetime(df["date_GMT"], errors="coerce")
        return df
    if "timestamp" in df.columns:
        df["match_datetime"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        return df
    df["match_datetime"] = pd.NaT
    return df


def _outcome(home_goals: float, away_goals: float) -> str:
    if pd.isna(home_goals) or pd.isna(away_goals):
        return "N/A"
    if home_goals > away_goals:
        return "HOME_WIN"
    if away_goals > home_goals:
        return "AWAY_WIN"
    return "DRAW"


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
  <div class="muted" style="margin-top:6px; font-size:.92rem;">{subtitle}</div>
</div>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_matches(year: int) -> pd.DataFrame:
    df = load_csv(RAW_FILES[year]["matches"])
    if df.empty:
        return df
    df["season"] = year
    df = _parse_dt(df)

    df["home_goals"] = _num(df.get("home_team_goal_count"))
    df["away_goals"] = _num(df.get("away_team_goal_count"))
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["outcome"] = [_outcome(h, a) for h, a in zip(df["home_goals"], df["away_goals"])]

    for c in [
        "home_team_corner_count", "away_team_corner_count",
        "home_team_yellow_cards", "away_team_yellow_cards",
        "home_team_shots", "away_team_shots",
        "home_team_shots_on_target", "away_team_shots_on_target",
        "home_team_possession", "away_team_possession",
        "team_a_xg", "team_b_xg",
        "attendance",
    ]:
        if c in df.columns:
            df[c] = _num(df[c])

    df["score"] = np.where(
        df["outcome"] == "N/A",
        "N/A",
        df["home_goals"].fillna(0).astype(int).astype(str) + " - " + df["away_goals"].fillna(0).astype(int).astype(str),
    )
    return df


def compute_points_table(matches: pd.DataFrame) -> pd.DataFrame:
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
            rows += [(h, 3, hg, ag), (a, 0, ag, hg)]
        elif out == "AWAY_WIN":
            rows += [(h, 0, hg, ag), (a, 3, ag, hg)]
        else:
            rows += [(h, 1, hg, ag), (a, 1, ag, hg)]

    df = pd.DataFrame(rows, columns=["team", "points", "gf", "ga"])
    g = df.groupby("team", as_index=False).agg(
        points=("points", "sum"),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
        matches=("team", "count"),
    )
    g["goal_diff"] = g["goals_for"] - g["goals_against"]
    return g.sort_values(["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)


def team_journey(matches: pd.DataFrame, team: str) -> pd.DataFrame:
    if matches.empty or not team:
        return pd.DataFrame()
    m = matches[(matches["home_team_name"] == team) | (matches["away_team_name"] == team)].copy()
    if m.empty:
        return m

    def row(r):
        home = r["home_team_name"]
        away = r["away_team_name"]
        is_home = home == team
        opp = away if is_home else home
        hg = r.get("home_goals", np.nan)
        ag = r.get("away_goals", np.nan)

        gf = hg if is_home else ag
        ga = ag if is_home else hg

        out = r.get("outcome", "N/A")
        pts = 0
        if out == "DRAW":
            pts = 1
        elif out == "HOME_WIN" and is_home:
            pts = 3
        elif out == "AWAY_WIN" and (not is_home):
            pts = 3

        res = "N/A"
        if out != "N/A":
            res = "Victoire" if pts == 3 else ("Nul" if pts == 1 else "Défaite")

        return pd.Series(
            {
                "Date": r.get("match_datetime", pd.NaT),
                "Adversaire": opp,
                "Lieu": "Domicile" if is_home else "Extérieur",
                "Score": r.get("score", "N/A"),
                "Résultat": res,
                "Points": pts,
            }
        )

    j = m.apply(row, axis=1).sort_values("Date").reset_index(drop=True)
    j["Points cumulés"] = j["Points"].cumsum()
    return j


# =========================
# Load processed data
# =========================
pred_pre = _parse_dt(load_csv(PROC_FILES["pred_pre"]))
pred_post = _parse_dt(load_csv(PROC_FILES["pred_post"]))
shap_sum_pre = load_csv(PROC_FILES["shap_sum_pre"])
shap_sum_post = load_csv(PROC_FILES["shap_sum_post"])
shap_g_pre = load_csv(PROC_FILES["shap_g_pre"])
shap_g_post = load_csv(PROC_FILES["shap_g_post"])


@st.cache_resource(show_spinner=False)
def get_retriever():
    return build_rag_retriever(PROJECT_ROOT)


# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(
    ["📊 Analyse", "🎯 Prédire + Expliquer (XAI + LLM)", "🤖 Chat RAG (preuves + LLM)"]
)

# =========================================================
# TAB 1 — Analyse
# =========================================================
with tab1:
    st.markdown("## Analyse interactive du tournoi")
    year = st.selectbox("Choisir l’édition", [2019, 2021, 2023], index=2)

    matches = load_matches(year)
    if matches.empty:
        st.error("Je ne trouve pas le fichier CSV des matchs pour cette édition dans data/raw.")
        st.stop()

    all_teams = sorted(set(matches["home_team_name"].dropna().unique()).union(set(matches["away_team_name"].dropna().unique())))
    team_focus = st.selectbox("Focus équipe (optionnel)", ["(Tout le tournoi)"] + all_teams, index=0)

    total_matches = len(matches)
    total_goals = int(matches["total_goals"].dropna().sum()) if "total_goals" in matches.columns else 0
    avg_goals = float(matches["total_goals"].dropna().mean()) if "total_goals" in matches.columns else np.nan

    corners = np.nan
    if "home_team_corner_count" in matches.columns and "away_team_corner_count" in matches.columns:
        corners = (matches["home_team_corner_count"] + matches["away_team_corner_count"]).dropna().mean()
    yellows = np.nan
    if "home_team_yellow_cards" in matches.columns and "away_team_yellow_cards" in matches.columns:
        yellows = (matches["home_team_yellow_cards"] + matches["away_team_yellow_cards"]).dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Matchs", f"{total_matches}", f"Édition {year}")
    with c2: kpi("Buts", f"{total_goals}", "Total du tournoi")
    with c3: kpi("Buts / match", f"{avg_goals:.2f}" if not np.isnan(avg_goals) else "N/A", "Rythme offensif")
    with c4:
        txt = []
        if not np.isnan(corners): txt.append(f"Corners/match: {corners:.2f}")
        if not np.isnan(yellows): txt.append(f"Cartons jaunes/match: {yellows:.2f}")
        kpi("Style", "📌", " • ".join(txt) if txt else "Données corners/cartons manquantes")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    # ✅ parcours équipe (line gold)
    if team_focus != "(Tout le tournoi)":
        st.markdown(f"### Parcours — {team_focus} ({year})")
        j = team_journey(matches, team_focus)
        if j.empty:
            st.info("Aucun match trouvé pour cette équipe.")
        else:
            chart = (
                alt.Chart(j)
                .mark_line(point=alt.OverlayMarkDef(filled=True, size=70, color=CAN_GOLD), color=CAN_GOLD, strokeWidth=3)
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Points cumulés:Q", title="Points cumulés"),
                    tooltip=["Date:T", "Adversaire:N", "Score:N", "Résultat:N", "Points:Q", "Points cumulés:Q"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(j[["Date", "Adversaire", "Lieu", "Score", "Résultat", "Points"]], use_container_width=True, hide_index=True)

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    # ✅ outcome distribution → donut (vert/gold/rouge)
    with left:
        dist = matches["outcome"].value_counts().reset_index()
        dist.columns = ["Outcome", "Count"]
        dist["Label"] = dist["Outcome"].map({"HOME_WIN": "Victoire domicile", "AWAY_WIN": "Victoire extérieur", "DRAW": "Match nul"}).fillna("N/A")

        color_scale = alt.Scale(
            domain=["Victoire domicile", "Match nul", "Victoire extérieur", "N/A"],
            range=[CAN_GREEN, CAN_GOLD, CAN_RED, "rgba(255,255,255,0.35)"],
        )

        pie = (
            alt.Chart(dist)
            .mark_arc(innerRadius=65, stroke="rgba(255,255,255,0.18)")
            .encode(
                theta=alt.Theta("Count:Q", title=""),
                color=alt.Color("Label:N", scale=color_scale, legend=alt.Legend(title="Résultats")),
                tooltip=["Label:N", "Count:Q"],
            )
            .properties(height=280)
        )
        st.altair_chart(pie, use_container_width=True)

    # ✅ histogram (gold) + bonus (corners or shots)
    with right:
        gdf = matches[["total_goals"]].dropna()
        if gdf.empty:
            st.info("Distribution des buts indisponible.")
        else:
            hist_goals = (
                alt.Chart(gdf)
                .mark_bar(color=CAN_GOLD)
                .encode(
                    x=alt.X("total_goals:Q", bin=alt.Bin(maxbins=10), title="Buts dans le match"),
                    y=alt.Y("count()", title="Nombre de matchs"),
                    tooltip=[alt.Tooltip("count()", title="Matchs")],
                )
                .properties(height=220)
            )
            st.altair_chart(hist_goals, use_container_width=True)

        # optional: corners or shots histogram
        metric = None
        if "home_team_corner_count" in matches.columns and "away_team_corner_count" in matches.columns:
            matches["total_corners"] = (matches["home_team_corner_count"] + matches["away_team_corner_count"])
            metric = "total_corners"
            title = "Corners dans le match"
        elif "home_team_shots" in matches.columns and "away_team_shots" in matches.columns:
            matches["total_shots"] = (matches["home_team_shots"] + matches["away_team_shots"])
            metric = "total_shots"
            title = "Tirs dans le match"

        if metric:
            d = matches[[metric]].dropna()
            hist2 = (
                alt.Chart(d)
                .mark_bar(color=CAN_SOFT_GOLD)
                .encode(
                    x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=12), title=title),
                    y=alt.Y("count()", title="Nombre de matchs"),
                    tooltip=[alt.Tooltip("count()", title="Matchs")],
                )
                .properties(height=220)
            )
            st.altair_chart(hist2, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("### Classement simple (points)")
    table = compute_points_table(matches)
    if table.empty:
        st.info("Impossible de calculer le classement (buts manquants).")
    else:
        top_n = st.slider("Afficher Top", 5, 24, 10, 1)
        show = table.head(top_n).copy()
        show["Rang"] = np.arange(1, len(show) + 1)
        st.dataframe(show[["Rang", "team", "points", "goal_diff", "goals_for", "goals_against", "matches"]], use_container_width=True, hide_index=True)

        chart = (
            alt.Chart(show)
            .mark_bar(color=CAN_GOLD)
            .encode(
                x=alt.X("points:Q", title="Points"),
                y=alt.Y("team:N", sort="-x", title="Équipe"),
                tooltip=["team:N", "points:Q", "goal_diff:Q"],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)


# =========================================================
# TAB 2 — Prediction + XAI + LLM
# =========================================================
with tab2:
    st.markdown("## Prédire et expliquer un match")
    mode = st.radio("Choisir le mode", ["Avant match (prévision)", "Après match (facteurs)"], horizontal=True)

    if mode.startswith("Avant"):
        df = pred_pre.copy()
        shap_sum = shap_sum_pre.copy()
        shap_g = shap_g_pre.copy()
        label_mode = "Avant match"
    else:
        df = pred_post.copy()
        shap_sum = shap_sum_post.copy()
        shap_g = shap_g_post.copy()
        label_mode = "Après match"

    if df.empty:
        st.error("Je ne trouve pas tes fichiers predictions dans data/processed.")
        st.stop()

    teams = sorted(set(df["home_team_name"].dropna().unique()).union(set(df["away_team_name"].dropna().unique())))
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        team_filter = st.selectbox("Filtrer par équipe", ["(Toutes)"] + teams, index=0)
    with col2:
        max_risk = st.slider("Incertitude max", 0.0, 1.0, 1.0, 0.05)
    with col3:
        only_wrong = st.toggle("Voir seulement les erreurs", value=False)

    if team_filter != "(Toutes)":
        df = df[(df["home_team_name"] == team_filter) | (df["away_team_name"] == team_filter)]
    if "risk_score" in df.columns:
        df = df[df["risk_score"] <= max_risk]
    if only_wrong and "true_outcome" in df.columns:
        df = df[df["pred_outcome"] != df["true_outcome"]]

    df = df.reset_index(drop=True)
    if df.empty:
        st.info("Aucun match selon ces filtres.")
        st.stop()

    df["match_label"] = df["home_team_name"].astype(str) + " vs " + df["away_team_name"].astype(str)
    idx = st.selectbox("Choisir un match", list(range(len(df))), format_func=lambda i: df.loc[i, "match_label"])
    sel = df.loc[idx]

    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    probs = {c.replace("proba_", ""): float(sel[c]) for c in proba_cols} if proba_cols else {}

    pred_out = str(sel.get("pred_outcome", "N/A"))
    true_out = str(sel.get("true_outcome", "N/A"))
    risk = float(sel.get("risk_score", np.nan)) if "risk_score" in sel.index else np.nan

    left, right = st.columns([1, 1])

    with left:
        st.markdown(
            f"""
<div class="card">
  <div class="kpiTitle">{label_mode} — résumé</div>
  <div style="font-size:1.25rem; font-weight:900; margin-top:6px;">
    {sel['home_team_name']} vs {sel['away_team_name']}
  </div>
  <div style="margin-top:10px;">
    Pronostic : {_badge_outcome(pred_out)} &nbsp;&nbsp;
    Résultat : {_badge_outcome(true_out)}
  </div>
  <div style="margin-top:10px;">{_badge_risk(risk)}</div>
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
            )
            p_df["Label"] = p_df["Résultat"].map(
                {"HOME_WIN": "Victoire domicile", "DRAW": "Match nul", "AWAY_WIN": "Victoire extérieur"}
            ).fillna(p_df["Résultat"])

            prob_color = alt.Scale(
                domain=["Victoire domicile", "Match nul", "Victoire extérieur"],
                range=[CAN_GREEN, CAN_GOLD, CAN_RED],
            )

            chart = (
                alt.Chart(p_df)
                .mark_bar()
                .encode(
                    x=alt.X("Probabilité:Q", title="Probabilité"),
                    y=alt.Y("Label:N", sort="-x", title=""),
                    color=alt.Color("Label:N", scale=prob_color, legend=None),
                    tooltip=["Label:N", alt.Tooltip("Probabilité:Q", format=".2f")],
                )
                .properties(height=240)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Pas de colonnes proba_* dans ce fichier.")

    with right:
        st.markdown("### Explication XAI (SHAP)")

        shap_text = ""
        if not shap_sum.empty and "timestamp" in shap_sum.columns:
            s = shap_sum.copy()
            s["_key"] = s["timestamp"].astype(str) + "|" + s["home_team_name"].astype(str) + "|" + s["away_team_name"].astype(str)
            key = str(sel["timestamp"]) + "|" + str(sel["home_team_name"]) + "|" + str(sel["away_team_name"])
            row = s[s["_key"] == key]
            if not row.empty:
                shap_text = str(row.iloc[0].get("top_factors_text", ""))

        if shap_text:
            st.markdown(
                f"""
<div class="card">
  <div class="kpiTitle">Facteurs clés (local)</div>
  <div style="font-size:0.98rem; line-height:1.55;">{shap_text}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.info("Explication locale non trouvée pour ce match (échantillon SHAP).")

        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.markdown("### Facteurs importants (global)")
        top_lines = []
        if not shap_g.empty and "class" in shap_g.columns:
            top = shap_g[shap_g["class"] == pred_out].sort_values("mean_abs_shap", ascending=False).head(8)
            if not top.empty:
                st.dataframe(top[["feature", "mean_abs_shap"]], use_container_width=True, hide_index=True)
                top_lines = [f"- {r.feature} (importance={r.mean_abs_shap:.3f})" for r in top.itertuples()]
            else:
                st.info("Pas de SHAP global pour cette classe.")
        else:
            st.info("Fichier SHAP global manquant.")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    st.markdown("## Explication texte par LLM (avec preuves)")
    question = st.text_area("Ta question", value="Explique simplement le pronostic et les raisons principales.")
    if st.button("Générer (LLM)"):
        evidence = [
            f"Mode: {label_mode}",
            f"Match: {sel['home_team_name']} vs {sel['away_team_name']}",
            f"Pronostic: {pred_out}",
            f"Résultat réel: {true_out}",
        ]
        if probs:
            evidence.append("Probabilités: " + ", ".join([f"{k}={v:.2f}" for k, v in probs.items()]))
        if not pd.isna(risk):
            evidence.append(f"Incertitude: {risk:.2f}")
        if shap_text:
            evidence.append("Facteurs clés (SHAP local): " + shap_text)
        if top_lines:
            evidence.append("Facteurs globaux (SHAP):\n" + "\n".join(top_lines))

        context = "\n".join(evidence)
        with st.spinner("Appel OpenRouter..."):
            ans = generate_explanation(question, context)

        st.markdown("### Réponse")
        st.write(ans)
        with st.expander("Voir les preuves envoyées"):
            st.code(context)


# =========================================================
# TAB 3 — Chat RAG
# =========================================================
with tab3:
    st.markdown("## Chat RAG (preuves + LLM)")
    st.markdown(
        "<div class='muted'>Le RAG cherche d’abord des lignes pertinentes dans tes CSV, puis le LLM répond uniquement avec ces preuves.</div>",
        unsafe_allow_html=True,
    )

    retriever = get_retriever()

    a, b, c = st.columns([1, 1, 1])
    with a:
        top_k = st.slider("Top-K preuves", 3, 15, 8, 1)
    with b:
        show_evidence = st.toggle("Afficher preuves", value=True)
    with c:
        st.caption("Astuce : inclure une équipe (ex: Maroc) ou une année (2023) améliore la recherche.")

    q = st.text_area(
        "Ta question",
        value="Quels facteurs influencent le plus la victoire à domicile avant match ?",
        height=120,
    )

    if st.button("Envoyer (RAG)"):
        with st.spinner("Recherche + réponse..."):
            result = answer_with_rag(q, retriever, top_k=top_k)

        st.markdown("### Réponse")
        st.write(result["answer"])

        if show_evidence:
            st.markdown("### Preuves utilisées")
            st.code(result["evidence"])
