from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import pandas as pd

from src.rag_store import CSVTFIDFRetriever, RetrievalResult
from src.llm_explainer import generate_explanation


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _compact_row(row: Dict, max_fields: int = 18) -> str:
    priority = [
        "season", "match_datetime", "date_GMT", "timestamp",
        "home_team_name", "away_team_name",
        "true_outcome", "pred_outcome", "risk_score",
        "class", "feature", "mean_abs_shap",
        "top_factors_text",
    ]
    ordered = [k for k in priority if k in row]
    for k in row.keys():
        if k not in ordered:
            ordered.append(k)
    ordered = ordered[:max_fields]

    parts = []
    for k in ordered:
        v = row.get(k, "")
        if v is None:
            continue
        s = str(v)
        if s and s != "nan":
            if len(s) > 240:
                s = s[:240] + "..."
            parts.append(f"{k}={s}")
    return " | ".join(parts)


def build_rag_retriever(project_root: Path) -> CSVTFIDFRetriever:
    processed = project_root / "data" / "processed"
    raw = project_root / "data" / "raw"

    retriever = CSVTFIDFRetriever()

    # --- PROCESSED (XAI + predictions)
    tables = {
        "pred_pre_2023": processed / "predictions_prematch_2023.csv",
        "pred_post_2023": processed / "predictions_postmatch_2023.csv",
        "shap_local_pre": processed / "shap_summary_per_match_prematch.csv",
        "shap_local_post": processed / "shap_summary_per_match_postmatch.csv",
        "shap_global_pre": processed / "shap_global_prematch.csv",
        "shap_global_post": processed / "shap_global_postmatch.csv",
    }

    for name, path in tables.items():
        df = _load_csv(path)
        if not df.empty:
            retriever.add_table(name, df)

    # --- RAW (tournoi complet 2019/2021/2023)
    raw_tables = [
        ("matches_2019", raw / "international-africa-cup-of-nations-matches-2019-to-2019-stats.csv"),
        ("matches_2021", raw / "international-africa-cup-of-nations-matches-2021-to-2021-stats.csv"),
        ("matches_2023", raw / "international-africa-cup-of-nations-matches-2023-to-2023-stats.csv"),
        ("teams_2019", raw / "international-africa-cup-of-nations-teams-2019-to-2019-stats.csv"),
        ("teams_2021", raw / "international-africa-cup-of-nations-teams-2021-to-2021-stats.csv"),
        ("teams_2023", raw / "international-africa-cup-of-nations-teams-2023-to-2023-stats.csv"),
    ]

    for name, path in raw_tables:
        df = _load_csv(path)
        if not df.empty:
            # add season column if missing
            if "season" not in df.columns:
                if "2019" in name:
                    df["season"] = 2019
                elif "2021" in name:
                    df["season"] = 2021
                elif "2023" in name:
                    df["season"] = 2023
            retriever.add_table(name, df)

    retriever.build()
    return retriever


def answer_with_rag(question: str, retriever: CSVTFIDFRetriever, top_k: int = 8) -> Dict:
    hits: List[RetrievalResult] = retriever.query(question, top_k=top_k)

    if not hits:
        evidence = (
            "Aucune ligne trouvée. "
            "Conseil: ajoute une équipe (ex: Morocco) ou une année (2023) dans la question."
        )
    else:
        lines = []
        for i, h in enumerate(hits, start=1):
            lines.append(f"[{i}] source={h.source} | score={h.score:.3f}\n{_compact_row(h.row)}")
        evidence = "\n\n".join(lines)

    answer = generate_explanation(user_question=question, context=evidence)
    return {"answer": answer, "evidence": evidence, "hits": hits}
