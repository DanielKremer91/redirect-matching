import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import base64
from collections import Counter
# --- Robustes Parsing für vorhandene Embeddings ---
import re
float_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_series_to_matrix(
    series,
    expected_dim: int,
    allow_padding: bool = True,
    pad_limit_ratio: float = 0.2,  # max. Anteil fehlender Werte pro Zeile, der gepaddet werden darf
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


# Layout und Branding
st.set_page_config(page_title="ONE Redirector", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Redirector – finde die passenden Redirect-Ziele 🔀")
st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 600px; margin-bottom: 1.5em; line-height: 1.5;">
Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp; Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</div>
<hr>
""", unsafe_allow_html=True)

# Erklärtext
st.markdown("""
### Was macht der ONE Redirector?
**Ziel:** Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

---

**Vorgehen:** Du hast die Wahl zwischen zwei Matching-Ansätzen:
- **Exact Matching** 1:1-Abgleich auf Basis identischer Inhalte in ausgewählten Spalten *(z. B. identische H1, Meta Title, etc.)*
- **Semantisches Matching** Zuordnung auf Basis **inhaltlicher Ähnlichkeit**. Grundlage: **Vektor-Embeddings**, die du entweder bereitstellst oder automatisch erstellen lässt.

---

**Was wird von dir benötigt?** Lade zwei Dateien hoch – jeweils mit den URLs deiner alten und neuen Domain.
✅ Unterstützt werden CSV und Excel
✅ Ideal: **Screaming Frog Crawl-Dateien**

💡 Tipp: Mit einem Custom JavaScript kannst du den für dich relevanten Seiteninhalt extrahieren und für das semantische Matching nutzen oder (Pro-Tipp) direkt im Screaming Frog basierend auf dem extrahierten Content die Embeddings berechnen lassen. Schreib mich bei Fragen hierzu gerne an!

---

**Modelle zur Embedding-Erstellung:**
Wenn du Embeddings **automatisch im Tool erstellen** lässt, stehen dir folgende Modelle zur Auswahl:
- all-MiniLM-L6-v2 (Standard) – sehr schnell, solide Semantik
- all-MiniLM-L12-v2 – gründlicher, aber immer noch schnell

Beide Modelle stammen aus der sentence-transformers-Bibliothek.

**Wenn du bereits Embeddings in deinen Dateien zur Verfügung stellst**, wird **kein Modell im Tool geladen**. Das Matching erfolgt dann direkt auf Basis deiner Vektoren – unabhängig davon, mit welchem Modell du sie erzeugt hast. Wichtig ist nur:
👉 **Beide Dateien müssen mit demselben Modell verarbeitet worden sein** und die Embeddings müssen korrekt formatiert vorliegen.

---

**Unterschied: FAISS vs. sklearn (für semantisches Matching)**

| Methode | Geschwindigkeit | Genauigkeit | Ideal für |
|-------------|------------------|------------------|------------------------|
| **FAISS** | Sehr hoch | ~90–95 % | Große Projekte (ab ca. 2.000 URLs) |
| **sklearn** | Langsamer | 100 % exakt | Kleine bis mittlere Projekte |

- **FAISS** nutzt Approximate Nearest Neighbor Search – extrem schnell, ideal für große Datenmengen, aber leicht ungenau
- **sklearn** berechnet exakte Cosine Similarity – sehr gründlich, aber bei vielen URLs langsam und speicherintensiv

---

**Output:** Du erhältst eine **CSV-Datei** mit bis zu **5 passenden Redirect-Zielen** (inkl. Score)
Auch URLs ohne passenden Treffer werden im Ergebnis mit "No Match" ausgewiesen.

---

**Weitere Features:**
- Flexible Spaltenauswahl für Exact und/oder semantisches Matching
- Manuell einstellbarer **Similarity Threshold**
- Unterstützung von vorberechneten Embeddings
- Keine Blackbox: Alle Entscheidungen und Scores sind im Ergebnis nachvollziehbar
---
""")

# --- Flexible URL-Spaltenerkennung -> normalisiere auf "Address" ---
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

# Datei-Upload
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

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    df = normalize_url_column(df)  # <- NEU: sorgt dafür, dass es "Address" gibt
    return df

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if 'Address' not in df_old.columns or 'Address' not in df_new.columns:
        st.error("Beide Dateien müssen eine 'Address'-Spalte enthalten.")
        st.stop()

    # Matching Methode wählen
    st.subheader("2. Matching Methode wählen")
    matching_method = st.selectbox(
        "Wie möchtest du matchen?",
        [
            "Exact Match",
            "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
            "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)"
        ]
    )

    # Embedding-Quelle nur anzeigen, wenn semantisches Matching
    if matching_method != "Exact Match":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio(
            "Stellst du die Embeddings für das semantische Matching in deinen Input-Dateien bereits zur Verfügung oder müssen diese erst noch generiert werden?",
            ["Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
             "Embeddings sind bereits generiert und in Input-Dateien vorhanden"]
        )

        model_name = "all-MiniLM-L6-v2"
        if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
            model_label = st.selectbox(
                "Welches Modell zur Embedding-Generierung soll verwendet werden?",
                sorted([
                    "all-MiniLM-L6-v2 (sehr schnell, gründlich)",
                    "all-MiniLM-L12-v2 (schnell, gründlicher)"
                ])
            )
            model_name = model_label
        else:
            model_name = None
    else:
        embedding_choice = None
        model_name = None

    # Spaltenauswahl
    st.subheader("4. Spaltenauswahl")
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))
    if matching_method != "Exact Match":
        st.caption("Optional: Du kannst die Auswahl bei Exact Match leer lassen, wenn du nur semantisches Matching durchführen möchtest.")
    exact_cols = st.multiselect("Spalten für Exact Match auswählen", common_cols)
    # --- Embedding-Spaltenauswahl vorziehen, falls bereits vorhandene Embeddings genutzt werden ---
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



    if matching_method != "Exact Match" and embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
        similarity_cols = st.multiselect(
            "Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen",
            common_cols
        )
    else:
        similarity_cols = []

    # Threshold
    if matching_method != "Exact Match":
        st.subheader("5. Cosine Similarity Schwelle")
        threshold = st.slider(
            "Minimaler Score für semantisches Matching – welchen Schwellenwert an Cosinus Similarity muss eine URL erreichen, um als potentielles Weiterleitungsziel in den Output aufgenommen zu werden? Interpretation der Zahlenwerte: Cosine Similarity von 0 = keine Ähnlichkeit, die URLs sind sich absolut unähnlich; 1 = die URLs sind sich identisch. Empfehlung: Mindestens 0.75 auswählen.",
            0.0, 1.0, 0.5, 0.01
        )
    else:
        threshold = 0.5  # Fallback

    if st.button("Let's Go", type="primary"):
        results = []
        matched_old = set()

        # 1. Exact Matching
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

        # 2. Similarity Matching
        df_remaining = df_old[~df_old['Address'].isin(matched_old)].reset_index(drop=True)
        
        if matching_method != "Exact Match" and df_remaining.shape[0] > 0:
            emb_old_mat = None
            emb_new_mat = None
            df_remaining_used = df_remaining
            df_new_used = df_new
        
            if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden" and similarity_cols:
                st.write("Erstelle Embeddings mit", model_name)
                model = SentenceTransformer(model_name.split()[0])
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
                if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                    sim_matrix = cosine_similarity(emb_old_mat, emb_new_mat)
                    # top-k bestimmen per Sortierung
                    def get_topk(i):
                        row_scores = sim_matrix[i]
                        top_indices = np.argsort(row_scores)[::-1][:5]
                        return row_scores, top_indices
                else:
                    dim = emb_new_mat.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    emb_new_norm = emb_new_mat / np.linalg.norm(emb_new_mat, axis=1, keepdims=True)
                    emb_old_norm = emb_old_mat / np.linalg.norm(emb_old_mat, axis=1, keepdims=True)
                    index.add(emb_new_norm.astype('float32'))
                    k = min(5, len(df_new_used))
                    faiss_scores, I = index.search(emb_old_norm.astype('float32'), k=k)
                    def get_topk(i):
                        return faiss_scores[i], I[i]
        
                # Ergebnisse einsammeln (nutze *used-DataFrames*)
                for i in range(len(df_remaining_used)):
                    row_result = {"Old URL": df_remaining_used['Address'].iloc[i]}
                    row_scores, top_indices = get_topk(i)
        
                    rank = 1
                    for j, idx in enumerate(top_indices):
                        if idx >= len(df_new_used):
                            continue
                        # Score-Auswahl: sklearn vs. faiss
                        if matching_method.startswith("Semantisches Matching mit sklearn"):
                            score = float(row_scores[idx])
                        else:
                            score = float(row_scores[j])
        
                        if score < threshold:
                            continue
        
                        row_result[f"Matched URL {rank}"] = df_new_used['Address'].iloc[idx]
                        row_result[f"Cosine Similarity Score {rank}"] = round(score, 4)
                        if rank == 1:
                            row_result["Match Type"] = f"Similarity ({'sklearn' if matching_method.startswith('Semantisches Matching mit sklearn') else 'faiss'})"
                        rank += 1
        
                    if rank > 1:
                        results.append(row_result)


        # 3. Nicht gematchte ALT-URLs ergänzen
        matched_urls_final = set(r["Old URL"] for r in results)
        unmatched = df_old[~df_old['Address'].isin(matched_urls_final)]
        for _, row in unmatched.iterrows():
            results.append({"Old URL": row['Address'], "Match Type": "No Match"})

        # 4. Ergebnis anzeigen und bereitstellen
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
