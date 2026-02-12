import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from collections import Counter
import re
from typing import Optional

# ============================================================
# i18n (DE/EN)
# ============================================================
def init_lang():
    if "lang" not in st.session_state:
        st.session_state["lang"] = "de"

init_lang()

I18N = {
    "de": {
        # UI
        "lang_toggle": "English",
        "page_title": "ONE Redirector",
        "title": "ONE Redirector – finde die passenden Redirect-Ziele 🔀",
        "help_expander": "ℹ️ Was macht das Tool? (Erklärung & Tipps)",

        "step_upload": "1. Dateien hochladen",
        "upload_old": "Datei mit den URLs, die weitergeleitet werden sollen (CSV oder Excel)",
        "upload_new": "Datei mit den Ziel-URLs (CSV oder Excel)",
        "need_address": "Beide Dateien müssen eine 'Address'-Spalte enthalten.",

        "step_method": "2. Matching-Methode wählen",
        "method_label": "Wie möchtest du matchen?",
        "method_exact": "Exact Match",
        "method_faiss": "Semantisches Matching mit FAISS (Schneller, für große Datenmengen geeignet)",
        "method_sklearn": "Semantisches Matching mit sklearn (Arbeitet gründlicher, aber langsamer)",

        "step_embed_source": "3. Embedding-Quelle",
        "embed_source_label": "Stellst du die Embeddings bereits zur Verfügung oder sollen sie erstellt werden?",
        "embed_create": "Embeddings müssen basierend auf meinen Input-Dateien erst noch erstellt werden",
        "embed_existing": "Embeddings sind bereits generiert und in Input-Dateien vorhanden",

        "model_label": "Welches Modell zur Embedding-Generierung soll verwendet werden?",
        "model_l6": "all-MiniLM-L6-v2 (sehr schnell, solide Semantik)",
        "model_l12": "all-MiniLM-L12-v2 (schnell, gründlicher)",
        "model_mpnet": "all-mpnet-base-v2 (präziser, aber langsamer)",

        "step_cols": "4. Spaltenauswahl",
        "exact_cols": "Spalten für Exact Match auswählen",
        "exact_optional_hint": "Optional: Du kannst die Auswahl bei Exact Match leer lassen, wenn du nur semantisches Matching durchführen möchtest.",

        "embed_col_header": "#### Embedding-Spaltenauswahl",
        "no_embed_col": "Keine Embedding-Spalte gefunden. Benenne deine Spalten z. B. 'Embeddings'.",
        "embed_col_old": "Embedding-Spalte (OLD)",
        "embed_col_new": "Embedding-Spalte (NEW)",
        "dim_warn": "Konnte keine sinnvolle Embedding-Dimension erkennen. Bitte gib sie manuell an.",
        "dim_detected": "Erkannte häufigste Dimension: **{dim}**",
        "dim_input": "Expected Embedding Dimension",
        "dim_help": "Trage hier die Modell-Dimension ein (z. B. 768 für MiniLM).",
        "padding_label": "Fehlende Werte mit 0 auffüllen (Padding) – empfohlen, wenn alle Embeddings aus derselben Pipeline stammen",
        "padlimit_label": "Max. Anteil fehlender Werte pro Zeile, der gepaddet werden darf",
        "padlimit_help": "Beispiel: 0.2 = 20 % Padding pro Zeile.",
        "dims_diag": "Embedding-Dimensionen anzeigen (Diagnose)",
        "dims_old": "OLD dims:",
        "dims_new": "NEW dims:",

        "similarity_cols": "Spalten für semantisches Matching auswählen – auf Basis dieser Inhalte werden die Embeddings erstellt und verglichen",

        "faiss_settings": "#### FAISS Einstellungen",
        "faiss_ivf": "IVF Flat verwenden (empfohlen ab ~2.000 Ziel-URLs)",
        "faiss_ivf_help": "Approximate Neighbor Search mit Clustering. Liefert sehr schnelle Suche bei großen Korpora.",
        "faiss_nlist": "nlist (Anzahl Cluster)",
        "faiss_nlist_help": "Mehr Cluster = feinere Vorselektion, mehr Speicher/Trainingszeit.",
        "faiss_nprobe": "nprobe (Cluster pro Suche)",
        "faiss_nprobe_help": "Mehr nprobe = höherer Recall, aber langsamer.",

        "step_threshold": "5. Cosine Similarity Schwelle",
        "threshold_label": "Minimaler Score für semantisches Matching – Empfehlung: mind. 0.75",

        "go": "Let's Go",

        "creating_embeddings": "Erstelle Embeddings mit",
        "model_dim": "Embedding-Dimension des gewählten Modells: **{dim}**",
        "need_embed_selection": "Bitte oben die Embedding-Spalten & die Dimension auswählen.",
        "embed_parse_failed": "Embeddings konnten nicht zuverlässig geparst werden.",
        "ivf_fail_fallback": "IVF-Training fehlgeschlagen oder Datensatz zu klein. Fallback auf Flat-IP. Details: {err}",
        "faiss_ivf_active": "FAISS IVF Flat aktiv • nlist={nlist} • nprobe={nprobe} • Korpus={n}",
        "faiss_flat_active": "FAISS Flat (exakt) aktiv • Korpus={n}",

        "parse_info": "📏 Embedding-Parsing {label}: verwendet={used}/{total}, expected_dim={dim}, padding={padding}",
        "parse_dropped": " | verworfen: {detail}",
        "parse_drop_parse": "Parsing-Fehler/leer: {n}",
        "parse_drop_short": "zu kurz (ohne Padding): {n}",
        "parse_drop_padlimit": "Padding-Limit überschritten: {n}",

        "results_header": "🔽 Ergebnisse",
        "download": "📥 Ergebnisse als CSV herunterladen",
        "download_name": "redirect_mapping_result.csv",

        # OUTPUT HEADERS (DE)
        "out_old_url": "Alte URL",
        "out_match_type": "Match-Typ",
        "out_match_basis": "Matching-Basis",
        "out_matched_url": "Ziel-URL {rank}",
        "out_score": "Cosine-Score {rank}",
        "out_no_match": "Kein Match",
        "out_exact_prefix": "Exact Match ({col})",
        "out_similarity": "Similarity ({engine})",
    },
    "en": {
        # UI
        "lang_toggle": "English",
        "page_title": "ONE Redirector",
        "title": "ONE Redirector – find the best redirect targets 🔀",
        "help_expander": "ℹ️ What does this tool do? (Explanation & tips)",

        "step_upload": "1. Upload files",
        "upload_old": "File with URLs to redirect (CSV or Excel)",
        "upload_new": "File with target URLs (CSV or Excel)",
        "need_address": "Both files must contain an 'Address' column.",

        "step_method": "2. Choose matching method",
        "method_label": "How would you like to match?",
        "method_exact": "Exact Match",
        "method_faiss": "Semantic matching with FAISS (faster, best for large datasets)",
        "method_sklearn": "Semantic matching with sklearn (more thorough, but slower)",

        "step_embed_source": "3. Embedding source",
        "embed_source_label": "Are embeddings already available, or should the tool create them?",
        "embed_create": "Embeddings should be created from my input files",
        "embed_existing": "Embeddings already exist inside my input files",

        "model_label": "Which model should be used to generate embeddings?",
        "model_l6": "all-MiniLM-L6-v2 (very fast, solid semantics)",
        "model_l12": "all-MiniLM-L12-v2 (fast, more thorough)",
        "model_mpnet": "all-mpnet-base-v2 (more accurate, but slower)",

        "step_cols": "4. Column selection",
        "exact_cols": "Select columns for Exact Match",
        "exact_optional_hint": "Optional: You can leave Exact Match empty if you only want semantic matching.",

        "embed_col_header": "#### Embedding column selection",
        "no_embed_col": "No embedding column found. Please name your column e.g. 'Embeddings'.",
        "embed_col_old": "Embedding column (OLD)",
        "embed_col_new": "Embedding column (NEW)",
        "dim_warn": "Could not detect a reliable embedding dimension. Please enter it manually.",
        "dim_detected": "Most common detected dimension: **{dim}**",
        "dim_input": "Expected embedding dimension",
        "dim_help": "Enter the model dimension (e.g., 768 for MiniLM).",
        "padding_label": "Pad missing values with zeros (recommended if embeddings come from the same pipeline)",
        "padlimit_label": "Max share of missing values per row that may be padded",
        "padlimit_help": "Example: 0.2 = up to 20% padding per row.",
        "dims_diag": "Show embedding dimensions (diagnostics)",
        "dims_old": "OLD dims:",
        "dims_new": "NEW dims:",

        "similarity_cols": "Select columns for semantic matching – embeddings will be created from these fields and compared",

        "faiss_settings": "#### FAISS settings",
        "faiss_ivf": "Use IVF Flat (recommended above ~2,000 target URLs)",
        "faiss_ivf_help": "Approximate nearest neighbor search with clustering. Very fast on large corpora.",
        "faiss_nlist": "nlist (number of clusters)",
        "faiss_nlist_help": "More clusters = finer preselection, more memory/training time.",
        "faiss_nprobe": "nprobe (clusters per query)",
        "faiss_nprobe_help": "Higher nprobe = better recall, but slower.",

        "step_threshold": "5. Cosine similarity threshold",
        "threshold_label": "Minimum score for semantic matching – recommendation: at least 0.75",

        "go": "Let's Go",

        "creating_embeddings": "Creating embeddings with",
        "model_dim": "Embedding dimension of the selected model: **{dim}**",
        "need_embed_selection": "Please select embedding columns and the expected dimension above.",
        "embed_parse_failed": "Embeddings could not be parsed reliably.",
        "ivf_fail_fallback": "IVF training failed or dataset too small. Falling back to Flat-IP. Details: {err}",
        "faiss_ivf_active": "FAISS IVF Flat active • nlist={nlist} • nprobe={nprobe} • corpus={n}",
        "faiss_flat_active": "FAISS Flat (exact) active • corpus={n}",

        "parse_info": "📏 Embedding parsing {label}: used={used}/{total}, expected_dim={dim}, padding={padding}",
        "parse_dropped": " | dropped: {detail}",
        "parse_drop_parse": "Parse errors/empty: {n}",
        "parse_drop_short": "too short (no padding): {n}",
        "parse_drop_padlimit": "padding limit exceeded: {n}",

        "results_header": "🔽 Results",
        "download": "📥 Download results as CSV",
        "download_name": "redirect_mapping_result.csv",

        # OUTPUT HEADERS (EN)
        "out_old_url": "Old URL",
        "out_match_type": "Match Type",
        "out_match_basis": "Match Basis",
        "out_matched_url": "Matched URL {rank}",
        "out_score": "Cosine Similarity Score {rank}",
        "out_no_match": "No Match",
        "out_exact_prefix": "Exact Match ({col})",
        "out_similarity": "Similarity ({engine})",
    },
}

HELP_MD = {
    "de": """
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
Nicht gematchte ALT-URLs werden mit „Kein Match“ ausgewiesen.

**Weitere Features:**
- Flexible Spaltenauswahl für Exact und/oder semantisches Matching
- Manuell einstellbarer **Similarity Threshold**
- Unterstützung von vorberechneten Embeddings
- Nachvollziehbare Entscheidungen & Scores
""",
    "en": """
**Goal:** This tool helps you quickly find good redirect targets for **relaunches** or **domain migrations**.

---

**How it works:** Choose between two approaches:
- **Exact matching** – 1:1 matching based on identical values in selected columns *(e.g., same H1, Meta Title)*
- **Semantic matching** – mapping based on **content similarity** using **vector embeddings** (either provided by you or generated by the tool)

**What you need:** Upload two files – one with old URLs and one with new target URLs.  
✅ CSV & Excel are supported (Screaming Frog exports work great)

💡 Tip: You can extract relevant page text with a custom JavaScript, or (pro tip) generate embeddings directly in Screaming Frog based on page content.

---

**Embedding models (local, no API):**
- `all-MiniLM-L6-v2` – very fast, solid semantics (good for large projects)
- `all-MiniLM-L12-v2` – more thorough at good speed
- `all-mpnet-base-v2` – highest accuracy (best for small/medium projects)

If embeddings already exist in your files, the tool **won’t load any model**. Important: **Both files must be generated with the same model**.

---

**FAISS vs. sklearn (semantic matching)**

| Method | Speed | Accuracy | Best for |
|-------------|------------------|------------------|------------------------|
| **FAISS** | Very high | ~90–95% | Large projects (≈ 2,000+ URLs) |
| **sklearn** | Slower | Exact | Small to medium projects |

- **FAISS IVF Flat** uses approximate nearest neighbor search – extremely fast, slightly less precise.
- **sklearn** computes exact cosine similarity – thorough, but slower on large datasets.

**Output:** CSV with up to 5 redirect suggestions per old URL (incl. score).  
Unmatched old URLs are marked as “No Match”.

**More features:**
- Flexible column selection for exact and/or semantic matching
- Manual **similarity threshold**
- Supports precomputed embeddings
- Transparent scores & decisions
"""
}

def t(key: str, **kwargs) -> str:
    lang = st.session_state.get("lang", "de")
    s = I18N[lang].get(key, key)
    return s.format(**kwargs)

# ============================================================
# Parsing / Utility
# ============================================================
float_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

def parse_series_to_matrix(
    series: pd.Series,
    expected_dim: int,
    allow_padding: bool = True,
    pad_limit_ratio: float = 0.2,
    label: str = ""
):
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

    if st.session_state["lang"] == "de":
        padding_txt = "an" if allow_padding else "aus"
    else:
        padding_txt = "on" if allow_padding else "off"

    msg = t("parse_info", label=label, used=used, total=total, dim=expected_dim, padding=padding_txt)
    detail = []
    if dropped_parse:     detail.append(t("parse_drop_parse", n=dropped_parse))
    if dropped_too_short: detail.append(t("parse_drop_short", n=dropped_too_short))
    if dropped_pad_limit: detail.append(t("parse_drop_padlimit", n=dropped_pad_limit))
    if detail:
        msg += t("parse_dropped", detail=", ".join(detail))
    st.info(msg)

    return np.vstack(vecs), list(idxs)

def count_dims(series: pd.Series):
    dims = []
    for val in series.dropna():
        nums = float_re.findall(str(val))
        dims.append(len(nums))
    return Counter(dims)

def infer_expected_dim(*series_list: pd.Series) -> Optional[int]:
    combined = Counter()
    for s in series_list:
        combined.update(count_dims(s))
    if 0 in combined:
        del combined[0]
    if not combined:
        return None
    mode_count = max(combined.values())
    candidates = [d for d, c in combined.items() if c == mode_count]
    return max(candidates)

# ============================================================
# Streamlit Layout
# ============================================================
st.set_page_config(page_title=t("page_title"), layout="wide")

# Language toggle
colA, colB = st.columns([1, 6])
with colA:
    is_en = st.toggle(t("lang_toggle"), value=(st.session_state["lang"] == "en"))
    st.session_state["lang"] = "en" if is_en else "de"

# Logo
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px;">
  <img src="https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png" alt="ONE Beyond Search" style="height: 60px;">
</div>
""", unsafe_allow_html=True)

st.title(t("title"))

with st.expander(t("help_expander"), expanded=False):
    st.markdown(HELP_MD[st.session_state["lang"]])

# ============================================================
# URL detection / normalization
# ============================================================
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

def detect_url_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in cols}
    for key in URL_COL_SYNONYMS:
        if key in lower_map and _looks_like_url_series(df[lower_map[key]]):
            return lower_map[key]
    for c in cols:
        if any(k in c.lower() for k in URL_COL_SYNONYMS) and _looks_like_url_series(df[c]):
            return c
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

# ============================================================
# File upload
# ============================================================
st.subheader(t("step_upload"))
uploaded_old = st.file_uploader(t("upload_old"), type=["csv", "xlsx"], key="old")
uploaded_new = st.file_uploader(t("upload_new"), type=["csv", "xlsx"], key="new")

if uploaded_old and uploaded_new:
    df_old = load_file(uploaded_old)
    df_new = load_file(uploaded_new)

    if 'Address' not in df_old.columns or 'Address' not in df_new.columns:
        st.error(t("need_address"))
        st.stop()

    # ============================================================
    # Matching method
    # ============================================================
    st.subheader(t("step_method"))
    matching_method = st.selectbox(
        t("method_label"),
        [t("method_exact"), t("method_faiss"), t("method_sklearn")]
    )

    # ============================================================
    # Embedding source
    # ============================================================
    if matching_method != t("method_exact"):
        st.subheader(t("step_embed_source"))
        embedding_choice = st.radio(
            t("embed_source_label"),
            [t("embed_create"), t("embed_existing")]
        )

        model_name = "all-MiniLM-L6-v2"
        if embedding_choice == t("embed_create"):
            model_label = st.selectbox(
                t("model_label"),
                [t("model_l6"), t("model_l12"), t("model_mpnet")]
            )
            model_name = model_label.split()[0]
        else:
            model_name = None
    else:
        embedding_choice = None
        model_name = None

    # ============================================================
    # Column selection
    # ============================================================
    st.subheader(t("step_cols"))
    common_cols = sorted(list(set(df_old.columns) & set(df_new.columns)))

    if matching_method != t("method_exact"):
        st.caption(t("exact_optional_hint"))

    exact_cols = st.multiselect(t("exact_cols"), common_cols)

    # Embedding columns (existing embeddings)
    if matching_method != t("method_exact") and embedding_choice == t("embed_existing"):
        st.markdown(t("embed_col_header"))
        cand_old = [c for c in df_old.columns if 'embed' in c.lower()]
        cand_new = [c for c in df_new.columns if 'embed' in c.lower()]

        if not cand_old or not cand_new:
            st.error(t("no_embed_col"))
            st.stop()

        emb_col_old = st.selectbox(t("embed_col_old"), cand_old, index=0)
        emb_col_new = st.selectbox(t("embed_col_new"), cand_new, index=0)

        suggested_dim = infer_expected_dim(df_old[emb_col_old], df_new[emb_col_new])
        if suggested_dim is None:
            st.warning(t("dim_warn"))
            suggested_dim = 768

        st.caption(t("dim_detected", dim=suggested_dim))
        expected_dim = st.number_input(
            t("dim_input"),
            min_value=8, max_value=4096, value=int(suggested_dim), step=8,
            help=t("dim_help")
        )

        allow_padding = st.checkbox(t("padding_label"), value=True)
        pad_limit_ratio = st.slider(
            t("padlimit_label"),
            min_value=0.0, max_value=0.9, value=0.2, step=0.05,
            help=t("padlimit_help")
        )

        with st.expander(t("dims_diag")):
            st.write(t("dims_old"), dict(count_dims(df_old[emb_col_old])))
            st.write(t("dims_new"), dict(count_dims(df_new[emb_col_new])))
    else:
        emb_col_old = None
        emb_col_new = None
        expected_dim = None
        allow_padding = True
        pad_limit_ratio = 0.2

    # Text columns for on-the-fly embeddings
    if matching_method != t("method_exact") and embedding_choice == t("embed_create"):
        similarity_cols = st.multiselect(t("similarity_cols"), common_cols)
    else:
        similarity_cols = []

    # ============================================================
    # FAISS IVF settings
    # ============================================================
    use_faiss_ivf = False
    faiss_nlist = 0
    faiss_nprobe = 0
    if matching_method == t("method_faiss"):
        st.markdown(t("faiss_settings"))
        use_faiss_ivf = st.checkbox(t("faiss_ivf"), value=True, help=t("faiss_ivf_help"))
        est_nlist = int(np.clip(int(np.sqrt(max(1, len(df_new))) * 2), 100, 16384))
        faiss_nlist = st.number_input(
            t("faiss_nlist"),
            min_value=1, max_value=16384, value=est_nlist, step=1,
            help=t("faiss_nlist_help")
        )
        default_nprobe = int(np.clip(max(1, faiss_nlist // 10), 1, 64))
        faiss_nprobe = st.number_input(
            t("faiss_nprobe"),
            min_value=1, max_value=max(1, faiss_nlist), value=default_nprobe, step=1,
            help=t("faiss_nprobe_help")
        )

    # ============================================================
    # Threshold
    # ============================================================
    if matching_method != t("method_exact"):
        st.subheader(t("step_threshold"))
        threshold = st.slider(t("threshold_label"), 0.0, 1.0, 0.5, 0.01)
    else:
        threshold = 0.5

    # ============================================================
    # Start
    # ============================================================
    if st.button(t("go"), type="primary"):
        results = []
        matched_old = set()

        # Output column names (language-specific)
        COL_OLD = t("out_old_url")
        COL_TYPE = t("out_match_type")
        COL_BASIS = t("out_match_basis")

        def col_matched(rank: int) -> str:
            return t("out_matched_url", rank=rank)

        def col_score(rank: int) -> str:
            return t("out_score", rank=rank)

        # 1) Exact matching
        for col in exact_cols:
            exact_matches = pd.merge(
                df_old[["Address", col]],
                df_new[["Address", col]],
                on=col,
                how="inner"
            )
            for _, row in exact_matches.iterrows():
                results.append({
                    COL_OLD: row["Address_x"],
                    col_matched(1): row["Address_y"],
                    COL_TYPE: t("out_exact_prefix", col=col),
                    col_score(1): 1.0,
                    COL_BASIS: f"{col}: {row[col]}",
                })
                matched_old.add(row["Address_x"])

        # 2) Semantic matching for remaining old URLs
        df_remaining = df_old[~df_old['Address'].isin(matched_old)].reset_index(drop=True)

        if matching_method != t("method_exact") and df_remaining.shape[0] > 0:
            emb_old_mat = None
            emb_new_mat = None
            df_remaining_used = df_remaining.copy()
            df_new_used = df_new.copy()

            if embedding_choice == t("embed_create") and similarity_cols:
                st.write(t("creating_embeddings"), model_name)
                model = SentenceTransformer(model_name)
                expected_dim_gen = model.get_sentence_embedding_dimension()
                st.caption(t("model_dim", dim=expected_dim_gen))

                df_remaining_used['text'] = df_remaining_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                df_new_used['text'] = df_new_used[similarity_cols].fillna('').agg(' '.join, axis=1)
                emb_old_mat = model.encode(df_remaining_used['text'].tolist(), show_progress_bar=True)
                emb_new_mat = model.encode(df_new_used['text'].tolist(), show_progress_bar=True)

            elif embedding_choice == t("embed_existing"):
                if not emb_col_old or not emb_col_new or expected_dim is None:
                    st.error(t("need_embed_selection"))
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
                    st.error(t("embed_parse_failed"))
                    st.stop()

                df_remaining_used = df_remaining.iloc[rows_old].reset_index(drop=True)
                df_new_used = df_new.iloc[rows_new].reset_index(drop=True)

            # Similarity computation
            if emb_old_mat is not None and emb_new_mat is not None:
                # sklearn (exact)
                if matching_method == t("method_sklearn"):
                    sim_matrix = cosine_similarity(emb_old_mat, emb_new_mat)

                    def get_topk(i):
                        row_scores = sim_matrix[i]
                        top_indices = np.argsort(row_scores)[::-1][:5]
                        return top_indices, row_scores[top_indices]

                    engine_name_for_label = "sklearn"
                    engine_for_type = "sklearn"

                else:
                    # FAISS (IVF or Flat) with cosine via L2 norm + inner product
                    def _l2norm(x, eps=1e-12):
                        return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

                    emb_new_norm = _l2norm(emb_new_mat).astype("float32")
                    emb_old_norm = _l2norm(emb_old_mat).astype("float32")
                    dim = emb_new_norm.shape[1]
                    k = min(5, len(df_new_used))

                    index = None
                    used_ivf = False

                    if use_faiss_ivf and len(df_new_used) >= max(1000, int(faiss_nlist) * 5):
                        quantizer = faiss.IndexFlatIP(dim)
                        index = faiss.IndexIVFFlat(quantizer, dim, int(faiss_nlist), faiss.METRIC_INNER_PRODUCT)
                        try:
                            index.train(emb_new_norm)
                            index.add(emb_new_norm)
                            index.nprobe = int(min(max(1, int(faiss_nprobe)), int(faiss_nlist)))
                            used_ivf = True
                        except Exception as e:
                            st.warning(t("ivf_fail_fallback", err=e))
                            index = None

                    if index is None:
                        index = faiss.IndexFlatIP(dim)
                        index.add(emb_new_norm)

                    faiss_scores, faiss_indices = index.search(emb_old_norm, k=k)

                    def get_topk(i):
                        return faiss_indices[i], faiss_scores[i]

                    if used_ivf:
                        st.info(t("faiss_ivf_active",
                                  nlist=int(faiss_nlist),
                                  nprobe=int(min(max(1, int(faiss_nprobe)), int(faiss_nlist))),
                                  n=len(df_new_used)))
                        engine_for_type = "faiss-ivf"
                    else:
                        st.info(t("faiss_flat_active", n=len(df_new_used)))
                        engine_for_type = "faiss-flat"

                    engine_name_for_label = "faiss"

                # Collect results (Top-5)
                for i in range(len(df_remaining_used)):
                    row_result = {COL_OLD: df_remaining_used['Address'].iloc[i]}
                    top_indices, top_scores = get_topk(i)

                    rank = 1
                    for idx, score in zip(top_indices, top_scores):
                        if idx >= len(df_new_used):
                            continue
                        score = float(score)
                        if score < threshold:
                            continue

                        row_result[col_matched(rank)] = df_new_used['Address'].iloc[idx]
                        row_result[col_score(rank)] = round(score, 4)
                        if rank == 1:
                            row_result[COL_TYPE] = t("out_similarity", engine=engine_for_type)
                        rank += 1

                    if rank > 1:
                        results.append(row_result)

        # 3) Add unmatched old URLs
        matched_urls_final = set(r.get(COL_OLD) for r in results if r.get(COL_OLD) is not None)
        unmatched = df_old[~df_old['Address'].isin(matched_urls_final)]
        for _, row in unmatched.iterrows():
            results.append({COL_OLD: row['Address'], COL_TYPE: t("out_no_match")})

        # 4) Build dataframe
        df_result = pd.DataFrame(results)

        # Drop "Matching Basis" column if it has no content anywhere
        if COL_BASIS in df_result.columns:
            ser = df_result[COL_BASIS].astype(str).str.strip()
            ser = ser.replace("nan", "")
            if not ser.ne("").any():
                df_result = df_result.drop(columns=[COL_BASIS])

        # Optional: drop empty "Matched URL {rank}" / "Score {rank}" columns (only if fully empty)
        # This keeps output tidy when fewer than 5 matches exist globally.
        for r in range(5, 1, -1):
            mu = col_matched(r)
            sc = col_score(r)
            if mu in df_result.columns:
                s = df_result[mu].astype(str).str.strip().replace("nan", "")
                if not s.ne("").any():
                    df_result = df_result.drop(columns=[mu])
            if sc in df_result.columns:
                s = df_result[sc].astype(str).str.strip().replace("nan", "")
                if not s.ne("").any():
                    df_result = df_result.drop(columns=[sc])

        # 5) Display + download
        st.subheader(t("results_header"))
        st.dataframe(df_result)

        csv = df_result.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=t("download"),
            data=csv,
            file_name=t("download_name"),
            mime='text/csv'
        )
