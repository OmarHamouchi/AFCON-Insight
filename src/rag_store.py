from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


@dataclass
class RetrievalResult:
    source: str
    score: float
    row: Dict


class CSVTFIDFRetriever:
    def __init__(self):
        self.tables: Dict[str, pd.DataFrame] = {}
        self.row_texts: Dict[str, List[str]] = {}
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.matrices: Dict[str, np.ndarray] = {}

    @staticmethod
    def _row_to_text(table_name: str, row: pd.Series) -> str:
        # ✅ IMPORTANT: include SHAP global columns
        priority = [
            "season",
            "match_datetime",
            "date_GMT",
            "timestamp",
            "home_team_name",
            "away_team_name",
            "stadium_name",
            "referee",
            "true_outcome",
            "pred_outcome",
            "risk_score",
            "top_factors_text",
            "class",
            "feature",
            "mean_abs_shap",
        ]

        parts = [f"table={table_name}"]

        for k in priority:
            if k in row.index:
                v = _s(row[k]).strip()
                if v:
                    parts.append(f"{k}={v}")

        # Add extra numeric/text columns that help matching (limited)
        extras = []
        for c in row.index:
            if c in priority:
                continue
            lc = c.lower()
            if (
                c.startswith("proba_")
                or "xg" in lc
                or "odds" in lc
                or "shots" in lc
                or "possession" in lc
                or "cards" in lc
                or "corners" in lc
                or "goals" in lc
            ):
                v = _s(row[c]).strip()
                if v:
                    extras.append(f"{c}={v}")

        parts.extend(extras[:25])
        return " | ".join(parts)

    def add_table(self, name: str, df: pd.DataFrame) -> None:
        df = df.copy().reset_index(drop=True)
        self.tables[name] = df
        self.row_texts[name] = [self._row_to_text(name, df.iloc[i]) for i in range(len(df))]

    def build(self) -> None:
        for name, texts in self.row_texts.items():
            vec = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=60000,
            )
            X = vec.fit_transform(texts)
            self.vectorizers[name] = vec
            self.matrices[name] = X

    def query(self, query_text: str, top_k: int = 8, min_score: float = 0.02) -> List[RetrievalResult]:
        q = query_text.strip()
        if not q:
            return []

        results: List[RetrievalResult] = []

        for name in self.tables.keys():
            vec = self.vectorizers.get(name)
            X = self.matrices.get(name)
            if vec is None or X is None:
                continue

            qv = vec.transform([q])
            sims = cosine_similarity(qv, X).flatten()
            if sims.size == 0:
                continue

            idx = np.argsort(sims)[::-1][: max(10, top_k)]
            for i in idx:
                score = float(sims[i])
                if score < min_score:
                    continue
                results.append(
                    RetrievalResult(
                        source=name,
                        score=score,
                        row=self.tables[name].iloc[int(i)].to_dict(),
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
