import streamlit as st
import pandas as pd

st.set_page_config(page_title="Semantic Redirect Mapping", layout="centered")

st.title("🔁 Semantic Redirect Mapping")

st.markdown("Lade zwei Dateien hoch – ALT-Domain & ZIEL-Domain – im Format CSV oder Excel.")

uploaded_old = st.file_uploader("📂 Datei der ALT-Domain", type=["csv", "xls", "xlsx"])
uploaded_new = st.file_uploader("📂 Datei der ZIEL-Domain", type=["csv", "xls", "xlsx"])

if uploaded_old and uploaded_new:
    try:
        df_old = pd.read_excel(uploaded_old) if uploaded_old.name.endswith(('xls', 'xlsx')) else pd.read_csv(uploaded_old)
        df_new = pd.read_excel(uploaded_new) if uploaded_new.name.endswith(('xls', 'xlsx')) else pd.read_csv(uploaded_new)

        st.success("✅ Beide Dateien erfolgreich geladen.")
        st.write("🔍 Vorschau ALT-Domain:", df_old.head())
        st.write("🔍 Vorschau ZIEL-Domain:", df_new.head())
    except Exception as e:
        st.error(f"Fehler beim Einlesen der Dateien: {e}")
