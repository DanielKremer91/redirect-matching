import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import base64

def parse_embedding_cell(val):
    if isinstance(val, (list, np.ndarray)):
        arr = np.asarray(val, dtype=float)
        return arr
    s = str(val).strip()
    if not s:
        return None
    # Klammern entfernen, falls vorhanden
    if s[0] == "[" and s[-1] == "]":
        s = s[1:-1]
    # Komma- oder Leerzeichen-getrennt zulassen
    if "," in s:
        parts = s.split(",")
    else:
        parts = s.split()
    try:
        arr = np.array([float(x) for x in parts if str(x).strip() != ""], dtype=float)
        return arr if arr.size else None
    except Exception:
        return None

def stack_embedding_column(series: pd.Series):
    vecs = []
    dim = None
    for v in series:
        arr = parse_embedding_cell(v)
        if arr is None:
            continue
        if dim is None:
            dim = arr.shape[0]
        # ggf. trimmen/padden (hier: trimmen, um hart zu sein)
        if arr.shape[0] == dim:
            vecs.append(arr)
    if not vecs:
        return None
    return np.vstack(vecs)

def load_csv_robust(uploaded_file):
    import io
    # Datei in Bytes lesen, um mehrfach zu parsen
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    encodings = ["utf-8-sig","utf-8","latin-1","cp1252","UTF-16"]
    seps = [",",";","\t","|",":"]
    for enc in encodings:
        # 1) Auto-Delimiter detection
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
        # 2) harte Delimiter durchprobieren
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                if df.shape[1] > 0:
                    return df
            except Exception:
                pass
    # letzter Versuch: Standardparser
    return pd.read_csv(io.BytesIO(raw))

# ---------- Column detection helpers (URL) ----------
def _cleanup_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df

def find_column(possible_names, columns):
    # 1) exakte Treffer
    for name in possible_names:
        if name in columns:
            return name
    # 2) case-insensitive Treffer
    lower = {str(c).lower(): c for c in columns}
    for name in possible_names:
        n = str(name).lower()
        if n in lower:
            return lower[n]
    # 3) heuristisch über Tokens
    for c in columns:
        n = str(c).lower().replace("-", " ").replace("_", " ")
        tokens = n.split()
        if any(tok in tokens for tok in ["address","url","urls","page","pages","landing","seite","seiten"]):
            return c
    return None

def pick_embedding_column(df: pd.DataFrame):
    """
    Wählt unter allen Spalten, deren Name 'embedding' enthält, diejenige aus,
    die am zuverlässigsten als numerischer Vektor parsebar ist.
    """
    candidates = [c for c in df.columns if "embedding" in str(c).lower()]
    best_col, best_rate, best_dim = None, 0.0, 0
    for c in candidates:
        s = df[c].dropna().astype(str).head(200)
        if s.empty:
            continue
        parsed = [parse_embedding_cell(v) for v in s]
        valid = [v for v in parsed if isinstance(v, np.ndarray) and v.size > 0]
        if not valid:
            continue
        # Konsistenz der Dimension grob prüfen
        dims = [v.shape[0] for v in valid]
        dim = max(set(dims), key=dims.count)
        rate = len(valid) / len(s)
        # Beste Spalte: höchste Valid-Rate, bei Gleichstand höhere Dim bevorzugen
        if (rate > best_rate) or (abs(rate - best_rate) < 1e-9 and dim > best_dim):
            best_col, best_rate, best_dim = c, rate, dim
    return best_col

# Layout und Branding
st.set_page_config(page_title="ONE Redirector", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("ONE Redirector – finde die passenden Redirect-Ziele 🔀")

st.markdown("""
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 600px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mich auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</div>
<hr>
""", unsafe_allow_html=True)

# Erklärtext
st.markdown("""
### Was macht der ONE Redirector?

**Ziel:**  
Dieses Tool hilft dir dabei, bei **Relaunches** oder **Domain-Migrationen** passende Redirect-Ziele auf Knopfdruck zu finden.

---

**Vorgehen:**  
Du hast die Wahl zwischen zwei Matching-Ansätzen:

- **Exact Matching**  
  1:1-Abgleich auf Basis identischer Inhalte in ausgewählten Spalten  
  *(z. B. identische H1, Meta Title, etc.)*

- **Semantisches Matching**  
  Zuordnung auf Basis **inhaltlicher Ähnlichkeit**.  
  Grundlage: **Vektor-Embeddings**, die du entweder bereitstellst oder automatisch erstellen lässt.

---

**Was wird von dir benötigt?**  
Lade zwei Dateien hoch – jeweils mit den URLs deiner alten und neuen Domain.  
✅ Unterstützt werden CSV und Excel  
✅ Ideal: **Screaming Frog Crawl-Dateien**  
💡 Tipp: Mit einem Custom JavaScript kannst du den für dich relevanten Seiteninhalt extrahieren und für das semantische Matching nutzen. Sprich mich gerne an, wenn du das Skript haben möchtest!

---

**Modelle zur Embedding-Erstellung:**  
Wenn du Embeddings **automatisch im Tool erstellen** lässt, stehen dir folgende Modelle zur Auswahl:

- `all-MiniLM-L6-v2` (Standard) – sehr schnell, solide Semantik  
- `all-MiniLM-L12-v2` – gründlicher, aber immer noch schnell  

Beide Modelle stammen aus der `sentence-transformers`-Bibliothek.

**Wenn du bereits Embeddings in deinen Dateien zur Verfügung stellst**, wird **kein Modell im Tool geladen**. Das Matching erfolgt dann direkt auf Basis deiner Vektoren – unabhängig davon, mit welchem Modell du sie erzeugt hast. Wichtig ist nur:  
👉 **Beide Dateien müssen mit demselben Modell verarbeitet worden sein** und die Embeddings müssen korrekt formatiert vorliegen.

---

**Unterschied: FAISS vs. sklearn (für semantisches Matching)**

| Methode     | Geschwindigkeit | Genauigkeit     | Ideal für             |
|-------------|------------------|------------------|------------------------|
| **FAISS**   | Sehr hoch        | ~90–95 %         | Große Projekte (ab ca. 2.000 URLs) |
| **sklearn** | Langsamer        | 100 % exakt      | Kleine bis mittlere Projekte        |

- **FAISS** nutzt Approximate Nearest Neighbor Search – extrem schnell, ideal für große Datenmengen, aber leicht ungenau  
- **sklearn** berechnet exakte Cosine Similarity – sehr gründlich, aber bei vielen URLs langsam und speicherintensiv

---

**Output:**  
Du erhältst eine **CSV-Datei** mit bis zu **5 passenden Redirect-Zielen** (inkl. Score)  
Auch URLs ohne passenden Treffer werden im Ergebnis mit `"No Match"` ausgewiesen.

---

**Weitere Features:**

- Flexible Spaltenauswahl für Exact und/oder semantisches Matching  
- Manuell einstellbarer **Similarity Threshold**  
- Unterstützung von vorberechneten Embeddings  
- Keine Blackbox: Alle Entscheidungen und Scores sind im Ergebnis nachvollziehbar

---
""")

# Datei-Upload
st.subheader("1. Dateien hochladen")
uploaded_old = st.file_uploader("Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)", type=["csv", "xlsx"], key="old")
uploaded_new = st.file_uploader("Datei mit den Ziel-URLs (CSV oder Excel)", type=["csv", "xlsx"], key="new")

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = load_csv_robust(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return _cleanup_headers(df)

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    URL_CANDIDATES = [
        "Address","URL","Urls","Page","Pages","Landing Page",
        "Seiten-URL","Seiten URL","Landingpage","Adresse"
    ]
    old_url_col = find_column(URL_CANDIDATES, df_old.columns)
    new_url_col = find_column(URL_CANDIDATES, df_new.columns)

    if not old_url_col or not new_url_col:
        st.error("Konnte die URL-Spalte nicht erkennen. Bitte nutze z. B. 'Address', 'URL', 'Page', 'Landing Page' oder 'Seiten-URL'.")
        st.stop()

    # intern auf 'Address' vereinheitlichen – ab hier bleibt dein Code unverändert
    df_old = df_old.rename(columns={old_url_col: "Address"})
    df_new = df_new.rename(columns={new_url_col: "Address"})

    # Matching Methode wählen
    st.subheader("2. Matching Methode wählen")
    matching_method = st.selectbox("Wie möchtest du matchen?", [
        "Exact Match",
        "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
        "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)"
    ])

    # Embedding-Quelle nur anzeigen, wenn semantisches Matching
    if matching_method != "Exact Match":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio(
            "Stellst du die Embeddings für das semantische Matching in deinen Input-Dateien bereits zur Verfügung oder müssen diese erst noch generiert werden?",
            ["Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden", "Embeddings sind bereits generiert und in Input-Dateien vorhanden"]
        )

        model_name = "all-MiniLM-L6-v2"
        if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
            model_label = st.selectbox("Welches Modell zur Embedding-Generierung soll verwendet werden?", sorted([
                "all-MiniLM-L6-v2 (sehr schnell, gründlich)",
                "all-MiniLM-L12-v2 (schnell, gründlicher)"
            ]))
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
    
    if matching_method != "Exact Match":
        if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
            similarity_cols = st.multiselect(
                "Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen",
                common_cols
            )
        elif embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
            st.caption("Hinweis: Wenn du die Option Embeddings sind bereits generiert und in Input-Dateien vorhanden ausgewählt hast und kein Exact Match durchführen möchtest, musst du keine Spaltenauswahl vornehmen – das Tool erkennt die Embedding-Spalten automatisch, wenn sie im Spaltennamen 'embedding' enthalten.")
            similarity_cols = []
        else:
            similarity_cols = []
    else:
        similarity_cols = []

    # Threshold
    if matching_method != "Exact Match":
        st.subheader("5. Cosine Similarity Schwelle")
        threshold = st.slider("Minimaler Score für semantisches Matching – welchen Schwellenwert an Cosinus Similarity muss eine URL erreichen, um als potentielles Weiterleitungsziel in den Output aufgenommen zu werden", 0.0, 1.0, 0.5, 0.01)
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
            if embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden" and similarity_cols:
                st.write("Erstelle Embeddings mit", model_name)
                model = SentenceTransformer(model_name.split()[0])
                df_remaining['text'] = df_remaining[similarity_cols].fillna('').agg(' '.join, axis=1)
                df_new['text'] = df_new[similarity_cols].fillna('').agg(' '.join, axis=1)
                emb_old = model.encode(df_remaining['text'].tolist(), show_progress_bar=True)
                emb_new = model.encode(df_new['text'].tolist(), show_progress_bar=True)

                # Wichtig: diese DataFrames für die Ergebnis-Schleife setzen
                df_remaining_used = df_remaining
                df_new_used = df_new

            elif embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
                # Spalten mit Embeddings finden
                emb_col_old = pick_embedding_column(df_old)
                emb_col_new = pick_embedding_column(df_new)
                if not emb_col_old or not emb_col_new:
                    st.error("Keine gültige Embedding-Spalte gefunden (suche Spaltennamen mit 'embedding').")
                    st.stop()

                # --- OLD: nur die noch nicht gematchten Zeilen verwenden ---
                mask_old = df_remaining[emb_col_old].apply(lambda v: parse_embedding_cell(v) is not None)
                df_remaining_used = df_remaining.loc[mask_old].reset_index(drop=True)

                # --- NEW: alle Zeilen mit gültigen Embeddings verwenden ---
                mask_new = df_new[emb_col_new].apply(lambda v: parse_embedding_cell(v) is not None)
                df_new_used = df_new.loc[mask_new].reset_index(drop=True)

                # Matrizen bauen
                emb_old_mat = stack_embedding_column(df_remaining_used[emb_col_old])
                emb_new_mat = stack_embedding_column(df_new_used[emb_col_new])

                if emb_old_mat is None or emb_new_mat is None:
                    st.error("Konnte Embeddings nicht verarbeiten – bitte Format prüfen (z. B. '[0.1, 0.2, ...]' oder '0.1,0.2,...').")
                    st.stop()

                if emb_old_mat.shape[1] != emb_new_mat.shape[1]:
                    st.error(f"Embedding-Dimensionen unterscheiden sich (old: {emb_old_mat.shape[1]}, new: {emb_new_mat.shape[1]}). Beide Dateien müssen mit demselben Modell erzeugt sein.")
                    st.stop()

                emb_old = emb_old_mat
                emb_new = emb_new_mat
            else:
                emb_old, emb_new = None, None

            if emb_old is not None and emb_new is not None:
                # Guards: keine Ziele oder keine Queries -> nichts tun
                if emb_new.shape[0] == 0 or (isinstance(df_new_used, pd.DataFrame) and len(df_new_used) == 0):
                    st.warning("Es sind keine Ziel-Embeddings verfügbar – semantisches Matching übersprungen.")
                elif emb_old.shape[0] == 0:
                    st.warning("Es sind keine Quell-Embeddings verfügbar – semantisches Matching übersprungen.")
                else:
                    if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                        sim_matrix = cosine_similarity(emb_old, emb_new)
                        n_rows = sim_matrix.shape[0]
                    else:
                        dim = emb_new.shape[1]
                        index = faiss.IndexFlatIP(dim)
                        # unit-norm für Cosinus-Ähnlichkeit via Inner Product
                        emb_new = emb_new / np.linalg.norm(emb_new, axis=1, keepdims=True)
                        emb_old = emb_old / np.linalg.norm(emb_old, axis=1, keepdims=True)
                        index.add(emb_new.astype('float32'))
            
                        k = min(5, emb_new.shape[0])  # safe: k <= Anzahl Zielvektoren
                        if k == 0:
                            st.warning("Keine Zielvektoren vorhanden – semantisches Matching übersprungen.")
                            sim_matrix = None
                            I = None
                            n_rows = 0
                        else:
                            sim_matrix, I = index.search(emb_old.astype('float32'), k=k)
                            n_rows = I.shape[0]
            
                    # Ergebnis-Schleife nur laufen lassen, wenn wir Scores haben
                    for i in range(n_rows):
                        # Safety: falls df_remaining_used kürzer ist (sollte nicht passieren, aber sicher ist sicher)
                        if i >= len(df_remaining_used):
                            break
            
                        row_result = {"Old URL": df_remaining_used['Address'].iloc[i]}
            
                        if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                            row_scores = sim_matrix[i]
                            top_indices = np.argsort(row_scores)[::-1][:5]
                        else:
                            top_indices = I[i]
                            row_scores = sim_matrix[i]
            
                        rank = 1
                        for j, idx in enumerate(top_indices):
                            # Safety: idx gegen df_new_used-Grenze prüfen
                            if idx >= len(df_new_used):
                                continue
            
                            try:
                                # sklearn: row_scores ist 1D über alle NEW-Indices
                                score = float(row_scores[idx]) if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)" else float(row_scores[j])
                            except (IndexError, ValueError):
                                continue
            
                            score = round(score, 4)
                            if score < threshold:
                                continue
            
                            row_result[f"Matched URL {rank}"] = df_new_used['Address'].iloc[idx]
                            row_result[f"Cosine Similarity Score {rank}"] = score
            
                            if rank == 1:
                                row_result["Match Type"] = f"Similarity ({'sklearn' if matching_method == 'Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)' else 'faiss'})"
            
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
