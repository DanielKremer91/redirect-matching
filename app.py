import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import base64
from collections import Counter
import re

# -----------------------------
# Parsing / Utility-Funktionen
# -----------------------------
float_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_series_to_matrix(
    series,
    expected_dim: int,
    allow_padding: bool = True,
    pad_limit_ratio: float = 0.2,
    label: str = ""
):
    """
    - Parsed nur echte Zahlen (ignoriert [], Whitespace, Newlines, Kommas)
    - Bringt alle Zeilen auf expected_dim:
        * kürzer -> Padding mit 0 (wenn allow_padding=True und fehlender Anteil <= pad_limit_ratio)
        * länger -> abschneiden (rechts)
    - Verwirft Zeilen mit Parsing-Fehlern oder wenn das Padding-Limit überschritten wird
    - Gibt (np.ndarray[n_rows, expected_dim], index_liste) zurück
    """
    vecs, idxs = [], []
    dropped_parse = dropped_too_short = dropped_pad_limit = 0

    for idx, val in series.items():
        if pd.isna(val):
            dropped_parse += 1
            continue

        nums = float_re.findall(str(val))
        if not nums:
            dropped_parse += 1
            continue

        try:
            arr = np.array([float(x) for x in nums], dtype="float32")
        except ValueError:
            dropped_parse += 1
            continue

        L = len(arr)
        if L == expected_dim:
            vecs.append(arr); idxs.append(idx)
        elif L > expected_dim:
            vecs.append(arr[:expected_dim]); idxs.append(idx)
        else:
            missing = expected_dim - L
            missing_ratio = missing / expected_dim
            if allow_padding and missing_ratio <= pad_limit_ratio:
                arr = np.pad(arr, (0, missing), 'constant', constant_values=0.0)
                vecs.append(arr); idxs.append(idx)
            else:
                if missing_ratio > pad_limit_ratio:
                    dropped_pad_limit += 1
                else:
                    dropped_too_short += 1

    used = len(vecs)
    total = len(series)
    if used == 0:
        return None, None

    msg = (
        f"📏 Embedding-Parsing {label}: verwendet={used}/{total}, "
        f"expected_dim={expected_dim}, padding={'an' if allow_padding else 'aus'}"
    )
    detail = []
    if dropped_parse:     detail.append(f"Parsing-Fehler/leer: {dropped_parse}")
    if dropped_too_short: detail.append(f"zu kurz (ohne Padding): {dropped_too_short}")
    if dropped_pad_limit: detail.append(f"Padding-Limit überschritten: {dropped_pad_limit}")
    if detail:
        msg += " | verworfen: " + ", ".join(detail)
    st.info(msg)

    return np.vstack(vecs), list(idxs)

def count_dims(series):
    dims = []
    for val in series.dropna():
        nums = float_re.findall(str(val))
        dims.append(len(nums))
    return Counter(dims)

def infer_expected_dim(*series_list):
    # Häufigste Dimension über alle angegebenen Serien (0 ignorieren).
    combined = Counter()
    for s in series_list:
        combined.update(count_dims(s))
    if 0 in combined:
        del combined[0]
    if not combined:
        return None
    mode_count = max(combined.values())
    candidates = [d for d, c in combined.items() if c == mode_count]
    return max(candidates)  # bei Gleichstand nimm die größere

# -----------------------------
# Streamlit Grundlayout
# -----------------------------
st.set_page_config(page_title="ONE Redirector", layout="wide")

# Logos
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px;">
  <img src="https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png" alt="ONE Beyond Search" style="height: 60px;">
</div>
""", unsafe_allow_html=True)

st.title("ONE Redirector – finde die passenden Redirect-Ziele 🔀")

# -----------------------------
# Eingeklappter Erklär-/Hilfetext
# -----------------------------
with st.expander("ℹ️ Was macht das Tool? (Erklärung & Tipps)", expanded=False):
    st.markdown("""
**Ziel:** Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

---

**Vorgehen:** Du hast die Wahl zwischen zwei Matching-Ansätzen:
- **Exact Matching** – 1:1-Abgleich auf Basis identischer Inhalte in ausgewählten Spalten *(z. B. identische H1, Meta Title)*
- **Semantisches Matching** – Zuordnung auf Basis **inhaltlicher Ähnlichkeit**. Grundlage: **Vektor-Embeddings**, die du entweder bereitstellst oder automatisch erstellen lässt.

**Was wird benötigt?** Lade zwei Dateien hoch – jeweils mit den URLs deiner alten und neuen Domain.  
✅ CSV & Excel werden unterstützt (ideal: Screaming Frog Crawls)

💡 Tipp: Mit einem Custom JavaScript kannst du relevanten Seiteninhalt extrahieren und für das semantische Matching nutzen oder (Pro-Tipp) direkt im Screaming Frog basierend auf dem Inhalt die Embeddings berechnen lassen.

---

**Modelle zur Embedding-Erstellung (lokal, ohne API):**
- `all-MiniLM-L6-v2` – sehr schnell, solide Semantik (für große Projekte)
- `all-MiniLM-L12-v2` – gründlicher bei guter Geschwindigkeit
- `all-mpnet-base-v2` – höchste Genauigkeit (für kleinere/mittlere Projekte)

Wenn Embeddings **bereits in deinen Dateien** vorliegen, lädt das Tool **kein Modell**. Wichtig: **Beide Dateien müssen mit demselben Modell** erzeugt worden sein.

---

**FAISS vs. sklearn (semantisches Matching)**

| Methode | Geschwindigkeit | Genauigkeit | Ideal für |
|-------------|------------------|------------------|------------------------|
| **FAISS** | Sehr hoch | ~90–95 % | Große Projekte (ab ca. 2.000 URLs) |
| **sklearn** | Langsamer | 100 % exakt | Kleine bis mittlere Projekte |

- **FAISS IVF Flat** nutzt Approximate Neighbor Search – extrem schnell, aber leicht ungenau.
- **sklearn** berechnet exakte Cosine Similarity – gründlich, aber bei großen Datenmengen langsamer.

**Output:** CSV mit bis zu 5 passenden Redirect-Zielen (inkl. Score).  
Nicht gematchte ALT-URLs werden mit „No Match“ ausgewiesen.

**Weitere Features:**
- Flexible Spaltenauswahl für Exact und/oder semantisches Matching
- Manuell einstellbarer **Similarity Threshold**
- Unterstützung von vorberechneten Embeddings
- Nachvollziehbare Entscheidungen & Scores
    """)

# -----------------------------
# URL-Spaltenerkennung & Normalisierung
# -----------------------------
URL_COL_SYNONYMS = [
    "address", "url", "page", "landing page", "seiten-url", "seite",
    "ziel-url", "ziel", "canonical", "canonical url"
]

def _looks_like_url_series(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    hits = 0
    for v in sample:
        v = v.strip()
        if v.startswith("http://") or v.startswith("https://") or ("/" in v and "." in v):
            hits += 1
    return hits >= max(3, int(len(sample) * 0.2))

def detect_url_column(df: pd.DataFrame) -> str | None:
    cols = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in cols}
    # 1) exakte Synonyme
    for key in URL_COL_SYNONYMS:
        if key in lower_map and _looks_like_url_series(df[lower_map[key]]):
            return lower_map[key]
    # 2) enthält-Suche
    for c in cols:
        if any(k in c.lower() for k in URL_COL_SYNONYMS) and _looks_like_url_series(df[c]):
            return c
    # 3) fallback: erste spalte, die wie URL aussieht
    for c in cols:
        if _looks_like_url_series(df[c]):
            return c
    return None

def normalize_url_column(df: pd.DataFrame) -> pd.DataFrame:
    url_col = detect_url_column(df)
    if url_col and url_col != "Address" and "Address" not in df.columns:
        df = df.rename(columns={url_col: "Address"})
    return df

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    df = normalize_url_column(df)
    return df

# -----------------------------
# Datei-Upload
# -----------------------------
st.subheader("1. Dateien hochladen")
uploaded_old = st.file_uploader(
    "Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)",
    type=["csv", "xlsx"],
    key="old"
)
uploaded_new = st.file_uploader(
    "Datei mit den Ziel-URLs (CSV oder Excel)",
    type=["csv", "xlsx"],
    key="new"
)

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if 'Address' not in df_old.columns or 'Address' not in df_new.columns:
        st.error("Beide Dateien müssen eine 'Address'-Spalte enthalten.")
        st.stop()

    # -----------------------------
    # Matching-Methode
    # -----------------------------
    st.subheader("2. Matching-Methode wählen")
    matching_method = st.selectbox(
        "Wie möchtest du matchen?",
        [
            "Exact Match",
            "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
            "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)"
        ]
    )

    # -----------------------------
    # Embedding-Quelle (nur bei semantisch)
    # -----------------------------
    if matching_method != "Exact Match":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio(
            "Stellst du die Embeddings bereits zur Verfügung oder sollen sie erstellt werden?",
            ["Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
             "Embeddings sind bereits generiert und in Input-Dateien vorhanden"]
        )

        model_name = "all-MiniLM-L6-v2"
        if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
            model_label = st.selectbox(
                "Welches Modell zur Embedding-Generierung soll verwendet werden?",
                [
                    "all-MiniLM-L6-v2 (sehr schnell, solide Semantik)",
                    "all-MiniLM-L12-v2 (schnell, gründlicher)",
                    "all-mpnet-base-v2 (präziser, aber langsamer)"
                ]
            )
            # erster Token = eigentlicher Modellname
            model_name = model_label.split()[0]
        else:
            model_name = None
    else:
        embedding_choice = None
        model_name = None

    # -----------------------------
    # Spaltenauswahl
    # -----------------------------
    st.subheader("4. Spaltenauswahl")
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))
    if matching_method != "Exact Match":
        st.caption("Optional: Du kannst die Auswahl bei Exact Match leer lassen, wenn du nur semantisches Matching durchführen möchtest.")
    exact_cols = st.multiselect("Spalten für Exact Match auswählen", common_cols)

    # Embedding-Spaltenauswahl (falls bereits vorhandene Embeddings)
    if matching_method != "Exact Match" and embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
        st.markdown("#### Embedding-Spaltenauswahl")
        cand_old = [c for c in df_old.columns if 'embed' in c.lower()]
        cand_new = [c for c in df_new.columns if 'embed' in c.lower()]

        if not cand_old or not cand_new:
            st.error("Keine Embedding-Spalte gefunden. Benenne deine Spalten z. B. 'Embeddings'.")
            st.stop()

        emb_col_old = st.selectbox("Embedding-Spalte (OLD)", cand_old, index=0)
        emb_col_new = st.selectbox("Embedding-Spalte (NEW)", cand_new, index=0)

        # Dimension erkennen & Eingabe anbieten
        suggested_dim = infer_expected_dim(df_old[emb_col_old], df_new[emb_col_new])
        if suggested_dim is None:
            st.warning("Konnte keine sinnvolle Embedding-Dimension erkennen. Bitte gib sie manuell an.")
            suggested_dim = 768

        st.caption(f"Erkannte häufigste Dimension: **{suggested_dim}**")
        expected_dim = st.number_input(
            "Expected Embedding Dimension",
            min_value=8, max_value=4096, value=int(suggested_dim), step=8,
            help="Trage hier die Modell-Dimension ein (z. B. 768 für MiniLM)."
        )

        allow_padding = st.checkbox(
            "Fehlende Werte mit 0 auffüllen (Padding) – empfohlen, wenn alle Embeddings aus derselben Pipeline stammen",
            value=True
        )
        pad_limit_ratio = st.slider(
            "Max. Anteil fehlender Werte pro Zeile, der gepaddet werden darf",
            min_value=0.0, max_value=0.9, value=0.2, step=0.05,
            help="Beispiel: 0.2 = 20 % Padding pro Zeile."
        )

        with st.expander("Embedding-Dimensionen anzeigen (Diagnose)"):
            st.write("OLD dims:", dict(count_dims(df_old[emb_col_old])))
            st.write("NEW dims:", dict(count_dims(df_new[emb_col_new])))
    else:
        emb_col_old = None
        emb_col_new = None
        expected_dim = None
        allow_padding = True
        pad_limit_ratio = 0.2

    # Textspalten für On-the-fly-Embeddings
    if matching_method != "Exact Match" and embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
        similarity_cols = st.multiselect(
            "Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen",
            common_cols
        )
    else:
        similarity_cols = []

    # -----------------------------
    # FAISS IVF Einstellungen (nur wenn FAISS gewählt)
    # -----------------------------
    use_faiss_ivf = False
    faiss_nlist = 0
    faiss_nprobe = 0
    if matching_method.startswith("Semantisches Matching mit FAISS"):
        st.markdown("#### FAISS Einstellungen")
        use_faiss_ivf = st.checkbox(
            "IVF Flat verwenden (empfohlen ab ~2.000 Ziel-URLs)",
            value=True,
            help="Approximate Neighbor Search mit Clustering. Liefert sehr schnelle Suche bei großen Korpora."
        )
        # sinnvolle Defaults: ~2*sqrt(N), geclippt
        est_nlist = int(np.clip(int(np.sqrt(max(1, len(df_new))) * 2), 100, 16384))
        faiss_nlist = st.number_input(
            "nlist (Anzahl Cluster)",
            min_value=1, max_value=16384, value=est_nlist, step=1,
            help="Mehr Cluster = feinere Vorselektion, mehr Speicher/Trainingszeit."
        )
        # nprobe-Default ca. 10% von nlist, max 64
        default_nprobe = int(np.clip(max(1, faiss_nlist // 10), 1, 64))
        faiss_nprobe = st.number_input(
            "nprobe (Cluster pro Suche)",
            min_value=1, max_value=max(1, faiss_nlist), value=default_nprobe, step=1,
            help="Mehr nprobe = höherer Recall, aber langsamer."
        )

    # -----------------------------
    # Threshold
    # -----------------------------
    if matching_method != "Exact Match":
        st.subheader("5. Cosine Similarity Schwelle")
        threshold = st.slider(
            "Minimaler Score für semantisches Matching – Empfehlung: mind. 0.75",
            0.0, 1.0, 0.5, 0.01
        )
    else:
        threshold = 0.5  # Fallback

    # -----------------------------
    # Start-Button
    # -----------------------------
    if st.button("Let's Go", type="primary"):
        results = []
        matched_old = set()

        # 1) Exact Matching
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

        # 2) Semantisches Matching (für verbleibende ALT-URLs)
        df_remaining = df_old[~df_old['Address'].isin(matched_old)].reset_index(drop=True)

        if matching_method != "Exact Match" and df_remaining.shape[0] > 0:
            emb_old_mat = None
            emb_new_mat = None
            df_remaining_used = df_remaining.copy()
            df_new_used = df_new.copy()

            if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden" and similarity_cols:
                st.write("Erstelle Embeddings mit", model_name)
                model = SentenceTransformer(model_name)
                expected_dim_gen = model.get_sentence_embedding_dimension()
                st.caption(f"Embedding-Dimension des gewählten Modells: **{expected_dim_gen}**")

                df_remaining_used['text'] = df_remaining_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                df_new_used['text'] = df_new_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                emb_old_mat = model.encode(df_remaining_used['text'].tolist(), show_progress_bar=True)
                emb_new_mat = model.encode(df_new_used['text'].tolist(), show_progress_bar=True)

            elif embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
                if not emb_col_old or not emb_col_new or expected_dim is None:
                    st.error("Bitte oben die Embedding-Spalten & die Dimension auswählen.")
                    st.stop()

                emb_old_mat, rows_old = parse_series_to_matrix(
                    df_remaining[emb_col_old],
                    expected_dim=int(expected_dim),
                    allow_padding=allow_padding,
                    pad_limit_ratio=float(pad_limit_ratio),
                    label="OLD"
                )
                emb_new_mat, rows_new = parse_series_to_matrix(
                    df_new[emb_col_new],
                    expected_dim=int(expected_dim),
                    allow_padding=allow_padding,
                    pad_limit_ratio=float(pad_limit_ratio),
                    label="NEW"
                )

                if emb_old_mat is None or emb_new_mat is None:
                    st.error("Embeddings konnten nicht zuverlässig geparst werden.")
                    st.stop()

                df_remaining_used = df_remaining.iloc[rows_old].reset_index(drop=True)
                df_new_used       = df_new.iloc[rows_new].reset_index(drop=True)

            # --- Ähnlichkeits-Berechnung nur, wenn befüllt ---
            if emb_old_mat is not None and emb_new_mat is not None:
                # sklearn-Zweig (exakt)
                if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                    sim_matrix = cosine_similarity(emb_old_mat, emb_new_mat)

                    def get_topk(i):
                        row_scores = sim_matrix[i]
                        top_indices = np.argsort(row_scores)[::-1][:5]
                        return top_indices, row_scores[top_indices]

                else:
                    # FAISS (IVF oder Flat)
                    def _l2norm(x, eps=1e-12):
                        return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

                    emb_new_norm = _l2norm(emb_new_mat).astype("float32")
                    emb_old_norm = _l2norm(emb_old_mat).astype("float32")
                    dim = emb_new_norm.shape[1]
                    k = min(5, len(df_new_used))

                    index = None
                    used_ivf = False

                    # IVF nur, wenn groß genug (Daumenregel) und genügend Daten für Training
                    if use_faiss_ivf and len(df_new_used) >= max(1000, int(faiss_nlist) * 5):
                        quantizer = faiss.IndexFlatIP(dim)
                        index = faiss.IndexIVFFlat(quantizer, dim, int(faiss_nlist), faiss.METRIC_INNER_PRODUCT)
                        try:
                            index.train(emb_new_norm)
                            index.add(emb_new_norm)
                            index.nprobe = int(min(max(1, int(faiss_nprobe)), int(faiss_nlist)))
                            used_ivf = True
                        except Exception as e:
                            st.warning(f"IVF-Training fehlgeschlagen oder Datensatz zu klein. Fallback auf Flat-IP. Details: {e}")
                            index = None

                    if index is None:
                        index = faiss.IndexFlatIP(dim)
                        index.add(emb_new_norm)

                    faiss_scores, faiss_indices = index.search(emb_old_norm, k=k)

                    def get_topk(i):
                        return faiss_indices[i], faiss_scores[i]

                    # Hinweis, welche Engine aktiv war
                    if used_ivf:
                        st.info(f"FAISS IVF Flat aktiv • nlist={int(faiss_nlist)} • nprobe={int(min(max(1, int(faiss_nprobe)), int(faiss_nlist)))} • Korpus={len(df_new_used)}")
                    else:
                        st.info(f"FAISS Flat (exakt) aktiv • Korpus={len(df_new_used)}")

                # Ergebnisse einsammeln
                for i in range(len(df_remaining_used)):
                    row_result = {"Old URL": df_remaining_used['Address'].iloc[i]}
                    top_indices, top_scores = get_topk(i)

                    rank = 1
                    for idx, score in zip(top_indices, top_scores):
                        if idx >= len(df_new_used):
                            continue
                        score = float(score)
                        if score < threshold:
                            continue

                        row_result[f"Matched URL {rank}"] = df_new_used['Address'].iloc[idx]
                        row_result[f"Cosine Similarity Score {rank}"] = round(score, 4)
                        if rank == 1:
                            if matching_method.startswith("Semantisches Matching mit sklearn"):
                                engine = "sklearn"
                            else:
                                engine = "faiss-ivf" if (use_faiss_ivf and len(df_new_used) >= max(1000, int(faiss_nlist) * 5)) else "faiss-flat"
                            row_result["Match Type"] = f"Similarity ({engine})"
                        rank += 1

                    if rank > 1:
                        results.append(row_result)

        # 3) Nicht gematchte ALT-URLs anhängen
        matched_urls_final = set(r["Old URL"] for r in results)
        unmatched = df_old[~df_old['Address'].isin(matched_urls_final)]
        for _, row in unmatched.iterrows():
            results.append({"Old URL": row['Address'], "Match Type": "No Match"})

        # 4) Ergebnisse anzeigen & Download
        df_result = pd.DataFrame(results)
        st.subheader("🔽 Ergebnisse")
        st.dataframe(df_result)

        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8-sig')

        csv = convert_df(df_result)
        st.download_button(
            label="📥 Ergebnisse als CSV herunterladen",
            data=csv,
            file_name='redirect_mapping_result.csv',
            mime='text/csv'
        )
