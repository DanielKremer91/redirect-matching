import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import base64

st.set_page_config(page_title="ONE Redirector", layout="wide")
st.title("🔀 ONE Redirector – finde die passenden Redirect-Ziele")

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
    matching_method = st.selectbox("Wie möchtest du matchen?", ["Nur Exact Match verwenden", "Schnell (FAISS)", "Gründlich (sklearn cosine similarity)"])

    # Embedding Quelle nur zeigen, wenn nicht nur Exact Matching
    if matching_method != "Nur Exact Match verwenden":
        st.subheader("3. Embedding-Quelle")
        embedding_choice = st.radio("Stellst du die Embeddings in deinen Input-Dateien bereits zur Verfügung?", ["Nein, Embeddings automatisch erstellen", "Ja, Embeddings aus Dateien verwenden"])

        model_name = "all-MiniLM-L6-v2"
        if embedding_choice == "Nein, Embeddings automatisch erstellen":
            model_label = st.selectbox("Welches Modell soll verwendet werden?", [
                "all-MiniLM-L6-v2 (sehr schnell, gründlich)",
                "all-MiniLM-L12-v2 (schnell, gründlicher)"
            ])
            model_name = model_label
        else:
            model_name = None
    else:
        embedding_choice = None
        model_name = None

    # Spaltenauswahl
    st.subheader("4. Spaltenauswahl")
    common_cols = list(set(df_old.columns) & set(df_new.columns))

    if matching_method != "Nur Exact Match verwenden":
        st.caption("Optional: Du kannst diese Auswahl leer lassen, wenn du nur semantisches Matching durchführen möchtest.")

    exact_cols = st.multiselect("Spalten für Exact Match auswählen", common_cols)



    if matching_method != "Nur Exact Match verwenden" and embedding_choice == "Nein, Embeddings automatisch erstellen":
        similarity_cols = st.multiselect("Spalten für semantisches Matching auswählen", common_cols)
    else:
        similarity_cols = []

    # Threshold
    if matching_method != "Nur Exact Match verwenden":
        st.subheader("5. Cosine Similarity Schwelle")
        threshold = st.slider("Minimaler Score für semantisches Matching", 0.0, 1.0, 0.5, 0.01)
    else:
        threshold = 0.5  # Fallback

    if st.button("Let's Go"):
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
        if matching_method != "Nur Exact Match verwenden" and df_remaining.shape[0] > 0:
            if embedding_choice == "Nein, Embeddings automatisch erstellen" and similarity_cols:
                st.write("Erstelle Embeddings mit", model_name)
                model = SentenceTransformer(model_name.split()[0])
                df_remaining['text'] = df_remaining[similarity_cols].fillna('').agg(' '.join, axis=1)
                df_new['text'] = df_new[similarity_cols].fillna('').agg(' '.join, axis=1)
                emb_old = model.encode(df_remaining['text'].tolist(), show_progress_bar=True)
                emb_new = model.encode(df_new['text'].tolist(), show_progress_bar=True)
            elif embedding_choice == "Ja, Embeddings aus Dateien verwenden":
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
                if matching_method == "Gründlich (sklearn cosine similarity)":
                    sim_matrix = cosine_similarity(emb_old, emb_new)
                else:
                    dim = emb_new.shape[1]
                    index = faiss.IndexFlatIP(dim)
                    emb_new = emb_new / np.linalg.norm(emb_new, axis=1, keepdims=True)
                    emb_old = emb_old / np.linalg.norm(emb_old, axis=1, keepdims=True)
                    index.add(emb_new.astype('float32'))
                    sim_matrix, I = index.search(emb_old.astype('float32'), k=5)

                for i in range(len(df_remaining)):
                    row_result = {"Old URL": df_remaining['Address'].iloc[i]}
                    row_scores = sim_matrix[i] if matching_method == "Gründlich (sklearn cosine similarity)" else sim_matrix[i]
                    row_indices = np.argsort(row_scores)[::-1] if matching_method == "Gründlich (sklearn cosine similarity)" else I[i]
                    rank = 1
                    for idx in row_indices:
                        score = round(float(row_scores[idx]), 4)
                        if score < threshold:
                            continue
                        row_result[f"Matched URL {rank}"] = df_new['Address'].iloc[idx]
                        row_result[f"Cosine Similarity Score {rank}"] = score
                        if rank == 1:
                            row_result["Match Type"] = f"Similarity ({'sklearn' if matching_method == 'Gründlich (sklearn cosine similarity)' else 'faiss'})"
                        rank += 1
                        if rank > 5:
                            break
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