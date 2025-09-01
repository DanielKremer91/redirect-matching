import base64
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

# Hinweis:
# - sentence_transformers & faiss werden erst bei Bedarf importiert,
#   damit Exact Match ohne diese Dependencies funktioniert.


# =========================
# UI: Seite & Branding
# =========================
st.set_page_config(page_title="ONE Redirector", layout="wide")

st.image(
    "https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png",
    width=250,
)

st.title("ONE Redirector – finde die passenden Redirect-Ziele 🔀")

st.markdown(
    """
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 600px; margin-bottom: 1.5em; line-height: 1.5;">
Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a>
von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a>
&nbsp;|&nbsp; Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</div>
<hr>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
### Was macht der ONE Redirector?
**Ziel:** Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

---

**Vorgehen:** Du hast die Wahl zwischen zwei Matching-Ansätzen:

- **Exact Matching** – 1:1-Abgleich auf Basis identischer Inhalte in ausgewählten Spalten *(z. B. identische H1, Meta Title, etc.)*
- **Semantisches Matching** – Zuordnung auf Basis **inhaltlicher Ähnlichkeit** (Vektor-Embeddings)

---

**Was wird von dir benötigt?** Lade zwei Dateien hoch – jeweils mit den URLs deiner alten und neuen Domain.

✅ CSV **oder** Excel (idealerweise Screaming Frog Crawl-Dateien)

---

**Modelle zur Embedding-Erstellung (optional):**
- `all-MiniLM-L6-v2` (Standard) – sehr schnell, solide Semantik
- `all-MiniLM-L12-v2` – gründlicher, immer noch schnell

Wenn du bereits Embeddings in deinen Dateien bereitstellst, wird **kein Modell geladen**. Wichtig: beide Dateien müssen mit **demselben Modell** erstellt worden sein.

---

**FAISS vs. sklearn (für semantisches Matching)**

| Methode   | Geschwindigkeit | Genauigkeit | Ideal für                          |
|-----------|------------------|-------------|------------------------------------|
| **FAISS** | Sehr hoch        | ~90–95 %    | Große Projekte (ab ~2.000 URLs)    |
| **sklearn** | Langsamer       | Exakt       | Kleine bis mittlere Projekte       |

- FAISS nutzt Approximate Nearest Neighbor Search (schnell, leicht ungenau)
- sklearn berechnet exakte Cosine Similarity

---

**Output:** Eine CSV mit bis zu **5 Redirect-Zielen pro ALT-URL** (inkl. Score). Auch ohne Treffer tauchen ALT-URLs im Ergebnis als "No Match" auf.

**Weitere Features:**
- Flexible Spaltenauswahl für Exact und/oder Semantik
- Einstellbarer **Similarity Threshold**
- Unterstützung vorberechneter Embeddings
- Ergebnis-Spalten tauchen nur auf, wenn sie Inhalte haben
"""
)


# =========================
# Hilfsfunktionen
# =========================
def load_file(uploaded_file) -> pd.DataFrame:
    """Liest CSV oder Excel ein und trimmt Spaltennamen."""
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        # Für Excel benötigt requirements: openpyxl
        df = pd.read_excel(uploaded_file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_embedding_column(df: pd.DataFrame) -> Optional[str]:
    """Sucht nach einer Spalte, die Embeddings enthält (heuristisch über Spaltenname)."""
    candidates = [c for c in df.columns if "embedding" in str(c).lower()]
    return candidates[0] if candidates else None


def parse_embedding_cell(cell) -> Optional[np.ndarray]:
    """
    Versucht, einen Embedding-Vektor aus einer Zelle zu parsen.
    Unterstützt:
      - Kommagetrennte Zahlen: "0.1,0.2,0.3"
      - JSON-ähnliche Liste: "[0.1, 0.2, 0.3]"
    """
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    try:
        if s.startswith("[") and s.endswith("]"):
            # JSON-ähnliche Liste
            s = s[1:-1]
        floats = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
        return np.array(floats, dtype="float32")
    except Exception:
        return None


def build_embeddings_from_columns(
    df_old: pd.DataFrame, df_new: pd.DataFrame, columns: List[str], model_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Erzeugt Satz-Embeddings für ausgewählte Spalten mithilfe sentence-transformers."""
    # Lazy import um Startzeit & Anforderungen zu schonen
    from sentence_transformers import SentenceTransformer

    model_key = model_name.split()[0]  # "all-MiniLM-L6-v2 (..)" -> "all-MiniLM-L6-v2"
    st.write("Erstelle Embeddings mit", model_key)
    model = SentenceTransformer(model_key)

    text_old = df_old[columns].fillna("").agg(" ".join, axis=1).tolist()
    text_new = df_new[columns].fillna("").agg(" ".join, axis=1).tolist()

    emb_old = model.encode(text_old, show_progress_bar=True)
    emb_new = model.encode(text_new, show_progress_bar=True)
    return np.array(emb_old, dtype="float32"), np.array(emb_new, dtype="float32")


def load_embeddings_from_frames(df_old: pd.DataFrame, df_new: pd.DataFrame):
    """Nimmt die erste Spalte mit 'embedding' im Namen (case-insensitive) und parst deren Werte."""
    emb_col_old = next((col for col in df_old.columns if 'embedding' in col.lower()), None)
    emb_col_new = next((col for col in df_new.columns if 'embedding' in col.lower()), None)

    if not emb_col_old or not emb_col_new:
        return None, None

    def parse_values(x):
        # Entfernt Klammern, splittet an Komma und wandelt in Floats
        return np.array([float(v) for v in str(x).replace("[", "").replace("]", "").split(",") if v.strip()])

    emb_old = np.stack(df_old[emb_col_old].dropna().apply(parse_values).values).astype("float32")
    emb_new = np.stack(df_new[emb_col_new].dropna().apply(parse_values).values).astype("float32")

    return emb_old, emb_new


def exact_match(
    df_old: pd.DataFrame, df_new: pd.DataFrame, cols: List[str]
) -> Tuple[List[dict], set]:
    """Führt Exact Match durch. Gibt (results, gematchte_old_urls) zurück."""
    results = []
    matched_old = set()

    for col in cols:
        if col not in df_old.columns or col not in df_new.columns:
            continue

        # Inner Join auf identischen Werten in col
        exact_matches = pd.merge(
            df_old[["Address", col]],
            df_new[["Address", col]],
            on=col,
            how="inner",
            suffixes=("_old", "_new"),
        )

        for _, row in exact_matches.iterrows():
            results.append(
                {
                    "Old URL": row["Address_old"],
                    "Matched URL 1": row["Address_new"],
                    "Match Type": f"Exact Match ({col})",
                    "Cosine Similarity Score 1": 1.0,
                    "Matching Basis (nur für Exact Matching relevant)": f"{col}: {row[col]}",
                }
            )
            matched_old.add(row["Address_old"])

    return results, matched_old


def similarity_match_sklearn(
    emb_old: np.ndarray,
    emb_new: np.ndarray,
    df_old_remaining: pd.DataFrame,
    df_new: pd.DataFrame,
    threshold: float,
    top_k: int = 5,
) -> List[dict]:
    """Exaktes Similarity-Matching mit sklearn (Top-N=5)."""
    sim_matrix = cosine_similarity(emb_old, emb_new)  # (n_old, n_new)

    results = []
    for i in range(sim_matrix.shape[0]):
        row_scores = sim_matrix[i]
        top_indices = np.argsort(row_scores)[::-1][:top_k]

        row = {"Old URL": df_old_remaining["Address"].iloc[i], "Match Type": "Similarity (sklearn)"}
        rank = 1
        for idx in top_indices:
            score = float(row_scores[idx])
            if score < threshold:
                continue

            row[f"Matched URL {rank}"] = df_new["Address"].iloc[idx]
            row[f"Cosine Similarity Score {rank}"] = round(score, 4)
            rank += 1

        if rank > 1:
            results.append(row)

    return results


def similarity_match_faiss(
    emb_old: np.ndarray,
    emb_new: np.ndarray,
    df_old_remaining: pd.DataFrame,
    df_new: pd.DataFrame,
    threshold: float,
) -> List[dict]:
    """
    Approximatives Similarity-Matching mit FAISS.
    Gemäß Vorgabe: Nur Top-1 (nicht Top-5) und Score als Cosine Similarity.
    """
    import faiss  # lazy import

    # Cosine-Similarity durch Normalisierung + Inner Product
    emb_new_n = emb_new / np.linalg.norm(emb_new, axis=1, keepdims=True)
    emb_old_n = emb_old / np.linalg.norm(emb_old, axis=1, keepdims=True)

    dim = emb_new_n.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_new_n.astype("float32"))

    # k=1 (nur bester Treffer)
    sim_matrix, I = index.search(emb_old_n.astype("float32"), k=1)

    results = []
    for i in range(I.shape[0]):
        idx = int(I[i][0])
        score = float(sim_matrix[i][0])
        if idx < 0 or idx >= len(df_new):
            continue
        if score < threshold:
            continue

        row = {
            "Old URL": df_old_remaining["Address"].iloc[i],
            "Matched URL 1": df_new["Address"].iloc[idx],
            "Cosine Similarity Score 1": round(score, 4),
            "Match Type": "Similarity (faiss)",
        }
        results.append(row)

    return results


def prune_empty_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entfernt Spalten 'Matched URL 2..5' / 'Cosine Similarity Score 2..5' usw.,
    wenn sie komplett leer sind. Lässt Pflichtspalten bestehen.
    """
    if df.empty:
        return df

    keep = {"Old URL", "Match Type", "Matched URL 1", "Cosine Similarity Score 1",
            "Matching Basis (nur für Exact Matching relevant)"}

    cols_to_drop = []
    for col in df.columns:
        if col in keep:
            continue
        # Drop, wenn komplett NaN/leer
        if df[col].isna().all():
            cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop)


def add_unmatched_old_urls(df_result: pd.DataFrame, df_old: pd.DataFrame) -> pd.DataFrame:
    """Fügt ALT-URLs ohne Treffer als 'No Match' hinzu."""
    matched_old = set(df_result["Old URL"].dropna().unique())
    unmatched = df_old[~df_old["Address"].isin(matched_old)]

    if unmatched.empty:
        return df_result

    filler = pd.DataFrame(
        [{"Old URL": url, "Match Type": "No Match"} for url in unmatched["Address"].tolist()]
    )
    out = pd.concat([df_result, filler], ignore_index=True)
    return out


# =========================
# Datei-Upload
# =========================
st.subheader("1. Dateien hochladen")
uploaded_old = st.file_uploader(
    "Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)", type=["csv", "xlsx"], key="old"
)
uploaded_new = st.file_uploader(
    "Datei mit den Ziel-URLs (CSV oder Excel)", type=["csv", "xlsx"], key="new"
)

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if "Address" not in df_old.columns or "Address" not in df_new.columns:
        st.error("Beide Dateien müssen eine 'Address'-Spalte enthalten.")
        st.stop()

    # =========================
    # Matching Methode wählen
    # =========================
    st.subheader("2. Matching-Methode wählen")
    matching_method = st.selectbox(
        "Wie möchtest du matchen?",
        [
            "Exact Match",
            "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
            "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)",
        ],
    )

    # =========================
    # Embedding-Quelle (nur für Semantik)
    # =========================
    embedding_choice = None
    model_name = None
    similarity_cols: List[str] = []

    if matching_method != "Exact Match":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio(
            "Stellst du die Embeddings bereit oder sollen sie erzeugt werden?",
            [
                "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
                "Embeddings sind bereits generiert und in Input-Dateien vorhanden",
            ],
        )

        if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
            model_name = st.selectbox(
                "Welches Modell zur Embedding-Generierung soll verwendet werden?",
                sorted(
                    [
                        "all-MiniLM-L12-v2 (schnell, gründlicher)",
                        "all-MiniLM-L6-v2 (sehr schnell, gründlich)",
                    ]
                ),
            )
        else:
            model_name = None

    # =========================
    # Spaltenauswahl (Exact & Semantik)
    # =========================
    st.subheader("4. Spaltenauswahl")
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))

    st.caption(
        "Tipp: Du kannst **nur Exact**, **nur Semantik** oder **beides** nutzen. "
        "Bei beidem hat Exact Match Priorität."
    )

    exact_cols = st.multiselect("Spalten für Exact Match auswählen", common_cols, default=[])

    if matching_method != "Exact Match" and embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
        similarity_cols = st.multiselect(
            "Spalten für semantisches Matching auswählen – diese Inhalte werden vektorisiert",
            common_cols,
            default=[],
        )

    # =========================
    # Threshold
    # =========================
    st.subheader("5. Cosine Similarity Schwelle")
    threshold = st.slider(
        "Minimaler Score (0–1), ab dem ein Treffer in den Output kommt (gilt nur für Semantik)",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    # =========================
    # Start-Button
    # =========================
    if st.button("Let's Go", type="primary"):
        results: List[dict] = []

        # 1) Exact Match
        if exact_cols:
            exact_results, matched_old = exact_match(df_old, df_new, exact_cols)
            results.extend(exact_results)
        else:
            matched_old = set()

        # 2) Semantisches Matching für nicht-exakt-gematchte ALT-URLs
        df_remaining = df_old[~df_old["Address"].isin(matched_old)].reset_index(drop=True)

        if matching_method != "Exact Match" and len(df_remaining) > 0:
            emb_old: Optional[np.ndarray] = None
            emb_new: Optional[np.ndarray] = None

            if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
                if not similarity_cols:
                    st.warning("Du hast Semantik gewählt, aber keine Spalten für die Embedding-Erstellung ausgewählt.")
                else:
                    emb_old, emb_new = build_embeddings_from_columns(
                        df_remaining, df_new, similarity_cols, model_name or "all-MiniLM-L6-v2"
                    )
            else:
                emb_old, emb_new = load_embeddings_from_frames(df_remaining, df_new)
                if emb_old is None or emb_new is None:
                    st.error("Keine gültigen Embedding-Spalten gefunden. "
                             "Erwarte z. B. eine Spalte namens 'Embeddings' mit Kommaliste oder JSON-Liste.")
                    st.stop()

            if emb_old is not None and emb_new is not None:
                if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                    # Top-N=5 NUR für sklearn
                    results.extend(
                        similarity_match_sklearn(
                            emb_old=emb_old,
                            emb_new=emb_new,
                            df_old_remaining=df_remaining,
                            df_new=df_new,
                            threshold=threshold,
                            top_k=5,
                        )
                    )
                else:
                    # FAISS: Top-1
                    results.extend(
                        similarity_match_faiss(
                            emb_old=emb_old,
                            emb_new=emb_new,
                            df_old_remaining=df_remaining,
                            df_new=df_new,
                            threshold=threshold,
                        )
                    )

        # 3) DataFrame & Spalten bereinigen
        df_result = pd.DataFrame(results) if results else pd.DataFrame(columns=["Old URL", "Match Type"])
        df_result = prune_empty_result_columns(df_result)

        # 4) Unmatched ALT-URLs ergänzen
        df_result = add_unmatched_old_urls(df_result, df_old)

        # 5) Ergebnis anzeigen & Download
        st.subheader("🔽 Ergebnisse")
        st.dataframe(df_result, use_container_width=True)

        csv_bytes = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Ergebnisse als CSV herunterladen",
            data=csv_bytes,
            file_name="redirect_mapping_result.csv",
            mime="text/csv",
        )
