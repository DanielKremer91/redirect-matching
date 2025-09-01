import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# ========= Helpers =========

def _cleanup_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def find_column(possible_names, columns):
    # 1) exact
    for name in possible_names:
        if name in columns:
            return name
    # 2) case-insensitive
    lower = {str(c).lower(): c for c in columns}
    for name in possible_names:
        n = str(name).lower()
        if n in lower:
            return lower[n]
    # 3) heuristic tokens
    for c in columns:
        n = str(c).lower().replace("-", " ").replace("_", " ")
        tokens = n.split()
        if any(tok in tokens for tok in ["address","url","urls","page","pages","landing","seite","seiten"]):
            return c
    return None

def parse_embedding_cell(val):
    """
    Parse a cell into a float vector.
    Accepts: "[0.1, 0.2, ...]" or "0.1,0.2,..." or whitespace-separated.
    Returns np.ndarray or None.
    """
    if isinstance(val, (list, np.ndarray)):
        try:
            arr = np.asarray(val, dtype=float)
            return arr if arr.size else None
        except Exception:
            return None
    s = str(val).strip()
    if not s:
        return None
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    parts = s.split(",") if "," in s else s.split()
    try:
        arr = np.array([float(x) for x in parts if str(x).strip() != ""], dtype=float)
        return arr if arr.size else None
    except Exception:
        return None

def looks_like_numeric_embedding_column(df: pd.DataFrame, col: str, sample_rows: int = 200) -> bool:
    """
    Check up to 'sample_rows' non-null rows if they parse to consistent float vectors.
    """
    s = df[col].dropna().astype(str).head(sample_rows)
    if s.empty:
        return False
    parsed = [parse_embedding_cell(v) for v in s]
    valid = [v for v in parsed if isinstance(v, np.ndarray) and v.size > 0 and np.isfinite(v).all()]
    if len(valid) < max(3, int(len(s) * 0.2)):
        return False
    dims = [v.shape[0] for v in valid]
    common_dim = max(set(dims), key=dims.count)
    return dims.count(common_dim) >= max(3, int(0.6 * len(valid)))

def pick_first_numeric_embedding_column(df: pd.DataFrame) -> str | None:
    """
    Return the first column whose name contains 'embedding' AND parses to numeric vectors
    of a consistent dimension. Otherwise None.
    """
    candidates = [c for c in df.columns if "embedding" in str(c).lower()]
    for c in candidates:
        if looks_like_numeric_embedding_column(df, c):
            return c
    return None

# ========= UI & Branding =========

st.set_page_config(page_title="ONE Redirector", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Redirector – finde die passenden Redirect-Ziele 🔀")

st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 600px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</div>
<hr>
""", unsafe_allow_html=True)

st.markdown("""
### Was macht der ONE Redirector?

**Ziel:**  
Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

---

**Vorgehen:**  
- **Exact Matching:** 1:1-Abgleich auf Basis identischer Inhalte in ausgewählten Spalten  
- **Semantisches Matching:** Zuordnung auf Basis **inhaltlicher Ähnlichkeit** (Embeddings)

**Output:**  
CSV mit bis zu **5 passenden Redirect-Zielen** pro alter URL (inkl. Score).  
URLs ohne Treffer: **"No Match"**.
""")

# ========= Datei-Upload =========

st.subheader("1. Dateien hochladen")
uploaded_old = st.file_uploader("Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)", type=["csv", "xlsx"], key="old")
uploaded_new = st.file_uploader("Datei mit den Ziel-URLs (CSV oder Excel)", type=["csv", "xlsx"], key="new")

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)
    return _cleanup_headers(df)

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    # ===== Flexible URL-Spalte erkennen und intern "Address" nennen =====
    URL_CANDIDATES = [
        "Address","URL","Urls","Page","Pages","Landing Page",
        "Seiten-URL","Seiten URL","Landingpage","Adresse"
    ]
    old_url_col = find_column(URL_CANDIDATES, df_old.columns)
    new_url_col = find_column(URL_CANDIDATES, df_new.columns)

    if not old_url_col or not new_url_col:
        st.error("Konnte die URL-Spalte nicht erkennen. Bitte nutze z. B. 'Address', 'URL', 'Page', 'Landing Page' oder 'Seiten-URL'.")
        st.stop()

    df_old = df_old.rename(columns={old_url_col: "Address"})
    df_new = df_new.rename(columns={new_url_col: "Address"})

    # ========= Matching-Einstellungen =========

    st.subheader("2. Matching Methode wählen")
    matching_method = st.selectbox(
        "Wie möchtest du matchen?",
        [
            "Exact Match",
            "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
            "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)"
        ]
    )

    if matching_method != "Exact Match":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio(
            "Embeddings-Quelle",
            [
                "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
                "Embeddings sind bereits generiert und in Input-Dateien vorhanden"
            ]
        )
        # Dezenter Hilfetext direkt unter der Auswahl:
        if embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
            st.caption("Hinweis: **Keine Spaltenauswahl nötig.** Das Tool erkennt automatisch die **erste valide Embedding-Spalte** (Spaltenname enthält „embedding“ und die Zellen enthalten numerische Vektoren).")
        model_name = "all-MiniLM-L6-v2" if embedding_choice.startswith("Embeddings müssen") else None
        if model_name:
            model_label = st.selectbox(
                "Welches Modell zur Embedding-Generierung soll verwendet werden?",
                ["all-MiniLM-L6-v2 (sehr schnell, gründlich)", "all-MiniLM-L12-v2 (schnell, gründlicher)"]
            )
            model_name = model_label.split()[0]
    else:
        embedding_choice = None
        model_name = None

    # ========= Spaltenauswahl =========

    st.subheader("4. Spaltenauswahl")
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))

    if matching_method != "Exact Match":
        st.caption("Optional: Du kannst die Auswahl bei Exact Match leer lassen, wenn du nur semantisches Matching durchführen möchtest.")

    exact_cols = st.multiselect("Spalten für Exact Match auswählen", common_cols)

    if matching_method != "Exact Match":
        if embedding_choice and embedding_choice.startswith("Embeddings müssen"):
            similarity_cols = st.multiselect(
                "Spalten für semantisches Matching – daraus werden die Embeddings erzeugt",
                common_cols
            )
        else:
            # zusätzlicher, dezenter Hinweis direkt bei der Spaltensektion
            st.caption("Hinweis: Bei **„Embeddings sind bereits generiert und in Input-Dateien vorhanden“** musst du **keine** Spalten auswählen – die Embedding-Spalte wird automatisch erkannt.")
            similarity_cols = []
    else:
        similarity_cols = []

    # ========= Threshold =========

    if matching_method != "Exact Match":
        st.subheader("5. Cosine Similarity Schwelle")
        threshold = st.slider(
            "Minimaler Cosine-Similarity-Score, damit ein Ziel in den Output kommt",
            0.0, 1.0, 0.5, 0.01
        )
    else:
        threshold = 0.5

    # ========= Start =========

    if st.button("Let's Go", type="primary"):
        results = []
        matched_old = set()

        # --- Exact Match ---
        for col in exact_cols:
            exact_matches = pd.merge(
                df_old[["Address", col]],
                df_new[["Address", col]],
                on=col,
                how="inner"
            )
            for _, row in exact_matches.iterrows():
                results.append({
                    "Old URL": row["Address_x"],
                    "Matched URL 1": row["Address_y"],
                    "Match Type": f"Exact Match ({col})",
                    "Cosine Similarity Score 1": 1.0,
                    "Matching Basis (nur für Exact Matching relevant)": f"{col}: {row[col]}"
                })
                matched_old.add(row["Address_x"])

        # --- Semantisches Matching ---
        df_remaining = df_old[~df_old['Address'].isin(matched_old)].reset_index(drop=True)

        if matching_method != "Exact Match" and len(df_remaining) > 0:
            df_remaining_used = None
            df_new_used = None
            emb_old = None
            emb_new = None

            if embedding_choice and embedding_choice.startswith("Embeddings müssen") and similarity_cols:
                st.write("Erstelle Embeddings mit", model_name)
                model = SentenceTransformer(model_name)
                df_remaining_used = df_remaining.copy()
                df_new_used = df_new.copy()
                df_remaining_used['__text'] = df_remaining_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                df_new_used['__text'] = df_new_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                emb_old = model.encode(df_remaining_used['__text'].tolist(), show_progress_bar=True)
                emb_new = model.encode(df_new_used['__text'].tolist(), show_progress_bar=True)

            elif embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
                emb_col_old = pick_first_numeric_embedding_column(df_old)
                emb_col_new = pick_first_numeric_embedding_column(df_new)

                if not emb_col_old or not emb_col_new:
                    st.error("Keine **valide** Embedding-Spalte gefunden. Hinweis: Spaltenname muss „embedding“ enthalten **und** numerische Vektoren (z. B. `[0.1, 0.2, ...]`) enthalten.")
                    st.stop()

                # Nur Zeilen verwenden, die tatsächlich parsebar sind
                mask_old = df_remaining[emb_col_old].apply(lambda v: parse_embedding_cell(v) is not None)
                df_remaining_used = df_remaining.loc[mask_old].reset_index(drop=True)

                mask_new = df_new[emb_col_new].apply(lambda v: parse_embedding_cell(v) is not None)
                df_new_used = df_new.loc[mask_new].reset_index(drop=True)

                # Matrizen bauen
                def stack(series: pd.Series):
                    vecs = []
                    dim = None
                    for v in series:
                        arr = parse_embedding_cell(v)
                        if arr is None:
                            continue
                        if dim is None:
                            dim = arr.shape[0]
                        if arr.shape[0] == dim and np.isfinite(arr).all():
                            vecs.append(arr.astype(np.float32, copy=False))
                    return np.vstack(vecs) if vecs else None

                emb_old = stack(df_remaining_used[emb_col_old])
                emb_new = stack(df_new_used[emb_col_new])

                if emb_old is None or emb_new is None:
                    st.error("Konnte Embeddings nicht verarbeiten – bitte Format prüfen (z. B. `[0.1, 0.2, ...]` oder `0.1,0.2,...`).")
                    st.stop()

                if emb_old.shape[1] != emb_new.shape[1]:
                    st.error(f"Embedding-Dimensionen unterschiedlich (old: {emb_old.shape[1]}, new: {emb_new.shape[1]}). Beide Dateien müssen mit **demselben Modell** erzeugt sein.")
                    st.stop()

            # Falls Embeddings vorhanden → Ähnlichkeiten berechnen
            if emb_old is not None and emb_new is not None:
                if 'df_new_used' not in locals() or df_new_used is None:
                    df_new_used = df_new
                if 'df_remaining_used' not in locals() or df_remaining_used is None:
                    df_remaining_used = df_remaining

                if len(df_new_used) == 0 or emb_new.shape[0] == 0:
                    st.warning("Keine Ziel-Embeddings verfügbar – semantisches Matching übersprungen.")
                elif emb_old.shape[0] == 0:
                    st.warning("Keine Quell-Embeddings verfügbar – semantisches Matching übersprungen.")
                else:
                    if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                        sim_matrix = cosine_similarity(emb_old, emb_new)
                        n_rows = sim_matrix.shape[0]
                        index_result = None
                    else:
                        # FAISS (Cosine via Inner Product on unit norm)
                        dim = emb_new.shape[1]
                        index = faiss.IndexFlatIP(dim)
                        emb_new_n = emb_new / np.linalg.norm(emb_new, axis=1, keepdims=True)
                        emb_old_n = emb_old / np.linalg.norm(emb_old, axis=1, keepdims=True)
                        index.add(emb_new_n.astype('float32'))
                        k = min(5, emb_new_n.shape[0])
                        if k == 0:
                            sim_matrix = None
                            index_result = None
                            n_rows = 0
                        else:
                            sim_matrix, index_result = index.search(emb_old_n.astype('float32'), k=k)
                            n_rows = index_result.shape[0]

                    # Ergebnisse einsammeln
                    for i in range(n_rows):
                        if i >= len(df_remaining_used):
                            break

                        row_result = {"Old URL": df_remaining_used['Address'].iloc[i]}

                        if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                            row_scores = sim_matrix[i]
                            top_indices = np.argsort(row_scores)[::-1][:5]
                            idx_iter = [(idx, float(row_scores[idx])) for idx in top_indices]
                        else:
                            idx_iter = [(int(idx), float(sim_matrix[i][j])) for j, idx in enumerate(index_result[i])]

                        rank = 1
                        for idx, score in idx_iter:
                            if idx >= len(df_new_used):
                                continue
                            score = round(score, 4)
                            if score < threshold:
                                continue
                            row_result[f"Matched URL {rank}"] = df_new_used['Address'].iloc[idx]
                            row_result[f"Cosine Similarity Score {rank}"] = score
                            if rank == 1:
                                row_result["Match Type"] = f"Similarity ({'sklearn' if matching_method.startswith('Semantisches Matching mit sklearn') else 'faiss'})"
                            rank += 1

                        if rank > 1:
                            results.append(row_result)

        # --- No Match für übrig gebliebene ALT-URLs ---
        matched_urls_final = set(r["Old URL"] for r in results)
        unmatched = df_old[~df_old['Address'].isin(matched_urls_final)]
        for _, row in unmatched.iterrows():
            results.append({"Old URL": row['Address'], "Match Type": "No Match"})

        # --- Anzeige & Download ---
        df_result = pd.DataFrame(results)
        st.subheader("🔽 Ergebnisse")
        st.dataframe(df_result)

        csv = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Ergebnisse als CSV herunterladen",
            data=csv,
            file_name='redirect_mapping_result.csv',
            mime='text/csv'
        )
