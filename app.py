import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import base64

# Layout und Branding
st.set_page_config(page_title="ONE Redirector", layout="wide")
st.image("https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png", width=250)
st.title("🔀 ONE Redirector – finde die passenden Redirect-Ziele")

st.markdown("""
<small>
Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a>  
<br>
Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a>
</small>
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
  *(z. B. identische H1, Meta Title, etc.)*

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
| **FAISS**   | Sehr hoch        | ~90–95 %         | Große Projekte (ab ca. 2.000 URLs) |
| **sklearn** | Langsamer        | 100 % exakt      | Kleine bis mittlere Projekte        |

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
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    else:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    return df

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if 'Address' not in df_old.columns or 'Address' not in df_new.columns:
        st.error("Beide Dateien müssen eine 'Address'-Spalte enthalten.")
        st.stop()

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

    if matching_method != "Exact Match" and embedding_choice == "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden":
        similarity_cols = st.multiselect("Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen", common_cols)
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
            elif embedding_choice == "Embeddings sind bereits generiert und in Input-Dateien vorhanden":
                emb_col_old = next((col for col in df_old.columns if 'embedding' in col.lower()), None)
                emb_col_new = next((col for col in df_new.columns if 'embedding' in col.lower()), None)
                if not emb_col_old or not emb_col_new:
                    st.error("Keine gültige Embedding-Spalte gefunden.")
                    st.stop()
                emb_old = np.stack(df_remaining[emb_col_old].dropna().apply(lambda x: np.array([float(v) for v in str(x).split(',')])).values)
                emb_new = np.stack(df_new[emb_col_new].dropna().apply(lambda x: np.array([float(v) for v in str(x).split(',')])).values)
            else:
                emb_old, emb_new = None, None

            if emb_old is not None and emb_new is not None:
                if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                    sim_matrix = cosine_similarity(emb_old, emb_new)
                else:
                    dim = emb_new.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    emb_new = emb_new / np.linalg.norm(emb_new, axis=1, keepdims=True)
                    emb_old = emb_old / np.linalg.norm(emb_old, axis=1, keepdims=True)
                    index.add(emb_new.astype('float32'))
                    k = min(5, len(df_new))
                    sim_matrix, I = index.search(emb_old.astype('float32'), k=k)

                for i in range(len(df_remaining)):
                    row_result = {"Old URL": df_remaining['Address'].iloc[i]}
                    
                    if matching_method == "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)":
                        row_scores = sim_matrix[i]
                        top_indices = np.argsort(row_scores)[::-1][:5]
                    else:
                        top_indices = I[i]
                        row_scores = sim_matrix[i]

                    rank = 1
                    for j, idx in enumerate(top_indices):
                        if idx >= len(df_new):
                            continue

                        try:
                            score = float(row_scores[j]) if matching_method != "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)" else float(row_scores[idx])
                        except (IndexError, ValueError):
                            continue

                        score = round(score, 4)

                        if score < threshold:
                            continue

                        row_result[f"Matched URL {rank}"] = df_new['Address'].iloc[idx]
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
